#!/usr/bin/env python3
# This file is part of the Luau programming language and is licensed under MIT License; see LICENSE.txt for details

# Differential fuzzer for @gpu kernels: generates random kernels in the GPU subset, runs each one with gpu.run and by
# calling it from Luau through luau-gpu, and reports every output that differs.
#
# usage: difffuzz.py [--luau-gpu build/luau-gpu] [--seed N] [--rounds R] [--kernels K] [--family int|float|both]
#
# Two families of kernels:
#   int    only integer sources, literals and operations: outputs must be bit-exact
#   float  floats too: f32 outputs are compared with a relative tolerance, since the GPU computes in f32 and Luau in
#          f64; integer outputs derived from floats may then differ by rounding, which is reported separately as
#          "precision" and not counted as a failure
#
# A kernel gets one buffer: 64 integer inputs, 20 float inputs, then its outputs, SLOTS words per index.

import argparse
import math
import os
import random
import re
import subprocess
import sys
import tempfile

# integer literals of the int family stay within the 32-bit range: larger ones are f32 on the GPU, like any float
INT_LITERALS = ["0", "1", "2", "3", "7", "255", "65535", "2147483647", "2147483648", "4294967295", "-1", "-7",
                "-2147483648", "0x7fffffff", "0x80000000", "0xffffffff"]
FLOAT_LITERALS = ["0.5", "-0.5", "1.5", "-2.25", "0.1", "1e-3", "3e9", "-3e9", "1e20", "-0", "-0.0", "2.5e-8",
                  "65536.5", "4294967295.5", "4294967296", "-2147483649", "1e10", "-1e10"]

# interesting 32-bit words for the integer input buffer
U32_INPUTS = [0, 1, 2, 3, 7, 255, 0x7fff, 0x8000, 0xffff, 0x7fffffff, 0x80000000, 0x80000001, 0xfffffffe, 0xffffffff,
              123456789, 3000000000, 4000000000, 16777216, 16777217]

F32_INPUTS = ["0", "-0.0", "0.5", "-0.5", "1", "-1", "1.5", "-2.75", "3e9", "-3e9", "2147483648", "-2147483648",
              "4294967296", "1e10", "-1e10", "16777217", "0.1", "1e-30", "123.456", "-0.999"]

# the "tame" profile keeps values small so that kernels run on the GPU; there, a failed run is suspicious (a check that
# fires although the value fits) and is reported
TAME_INT_LITERALS = ["0", "1", "2", "3", "5", "7", "10", "100", "255", "-1", "-3", "-10", "1000"]
TAME_FLOAT_LITERALS = ["0.5", "-0.5", "1.5", "-2.25", "0.25", "3.75", "10.5"]
TAME_U32_INPUTS = [0, 1, 2, 3, 5, 7, 9, 12, 100, 255, 1000, 4095]
TAME_F32_INPUTS = ["0", "0.5", "-0.5", "1", "-1", "1.5", "-2.75", "3.25", "10", "-12.5"]

SLOTS = 8

# where the parts of the buffer start
FSRC = 64 * 4
OUT = FSRC + 20 * 4

# every generated loop counts its iterations against this; it behaves the same on the GPU and in Luau
GUARD = "guard += 1 if guard > 4096 then break end"
N = 64


class Gen:
    def __init__(self, rng, family, tame=False):
        self.rng = rng
        self.family = family
        self.tame = tame
        self.vars = ["i", "a", "b", "c", "d"]
        self.counter = 0

    def leaf(self):
        r = self.rng.random()
        if r < 0.45:
            return self.rng.choice(self.vars)
        if self.family == "float" and r < 0.65:
            return self.rng.choice(TAME_FLOAT_LITERALS if self.tame else FLOAT_LITERALS)
        return self.rng.choice(TAME_INT_LITERALS if self.tame else INT_LITERALS)

    def expr(self, depth=0):
        if depth >= 3 or self.rng.random() < 0.3:
            return self.leaf()

        ops = ["+", "-", "*", "//", "%", "neg", "if", "floor", "ceil", "round", "abs", "min", "max", "clamp", "band",
               "bor", "bxor", "bnot", "lshift", "rshift", "arshift", "extract", "btest", "read", "helper"]
        if self.tame:
            # products of products and bnot leave the small range quickly
            ops = [op for op in ops if op not in ("bnot",)] + ["+", "-", "//", "%", "if", "min", "max"]
        if self.family == "float":
            ops += ["/", "/", "sqrt", "^", "fmod"]

        op = self.rng.choice(ops)
        e = lambda: self.expr(depth + 1)

        if op in ("//", "%") and self.tame:
            return f"({e()} {op} {self.rng.choice(['3', '7', '-2', '10', '2.5', '-0.75'] if self.family == 'float' else ['3', '7', '-2', '10'])})"
        if op in ("+", "-", "*", "//", "%", "/", "^"):
            return f"({e()} {op} {e()})"
        if op == "neg":
            return f"(-({e()}))"
        if op == "if":
            cmp = self.rng.choice(["<", "<=", ">", ">=", "==", "~="])
            return f"(if {e()} {cmp} {e()} then {e()} else {e()})"
        if op in ("floor", "ceil", "round", "abs"):
            return f"math.{op}({e()})"
        if op in ("min", "max"):
            return f"math.{op}({e()}, {e()})"
        if op == "clamp":
            lo, hi = sorted([self.rng.randint(-100000, 100000), self.rng.randint(-100000, 100000)])
            return f"math.clamp({e()}, {lo}, {hi})"
        if op == "sqrt":
            return f"math.sqrt(math.abs({e()}))"
        if op == "fmod":
            return f"math.fmod({e()}, {self.rng.choice(['3', '-2.5', '7', '0.75'])})"
        if op in ("band", "bor", "bxor"):
            return f"bit32.{op}({e()}, {e()})"
        if op == "bnot":
            return f"bit32.bnot({e()})"
        if op in ("lshift", "rshift", "arshift"):
            if self.tame:
                amount = str(self.rng.randint(-6, 6))
            else:
                amount = self.rng.choice([str(self.rng.randint(-40, 40)), f"({e()}) % 40", "-3", "31", "32"])
            return f"bit32.{op}({e()}, {amount})"
        if op == "extract":
            f = self.rng.randint(0, 31)
            w = self.rng.randint(1, 32 - f)
            return f"bit32.extract({e()}, {f}, {w})"
        if op == "btest":
            return f"(if bit32.btest({e()}, {e()}) then 1 else 0)"
        if op == "helper":
            # helpers are specialized per argument types, so mixing int, uint and float arguments matters
            return self.rng.choice([f"h1({e()}, {e()})", f"h2({e()})"])
        if op == "read":
            return f"buffer.readu32(buf, (bit32.band({e()}, 63)) * 4)"
        raise AssertionError(op)

    def statement(self, out, indent):
        r = self.rng.random()
        pad = "    " * indent
        if r < 0.4:
            name = f"v{self.counter}"
            self.counter += 1
            out.append(f"{pad}local {name} = {self.expr()}")
            self.vars.append(name)
        elif r < 0.6:
            var = self.rng.choice(self.vars[1:])
            op = self.rng.choice(["+=", "-=", "*=", "//=", "%="] + (["/="] if self.family == "float" else []))
            value = self.rng.choice(["3", "7", "-2"]) if self.tame and op in ("//=", "%=", "*=") else self.expr(1)
            out.append(f"{pad}{var} {op} {value}")
        elif r < 0.8 and indent < 3:
            var = self.rng.choice(self.vars[1:])
            out.append(f"{pad}if {self.expr(1)} < {self.expr(1)} then")
            out.append(f"{pad}    {var} = {self.expr(1)}")
            out.append(f"{pad}else")
            out.append(f"{pad}    {var} = {self.expr(1)}")
            out.append(f"{pad}end")
        elif indent < 3 and r < 0.86:
            var = self.rng.choice(self.vars[1:])
            n = f"n{self.counter}"
            self.counter += 1
            if self.rng.random() < 0.5:
                out.append(f"{pad}local {n} = 0")
                out.append(f"{pad}while {n} < i % 6 do")
                out.append(f"{pad}    {n} += 1")
                out.append(f"{pad}    {GUARD}")
                out.append(f"{pad}    {var} += {n} * {self.expr(2)}")
                out.append(f"{pad}end")
            else:
                out.append(f"{pad}local {n} = 0")
                out.append(f"{pad}repeat")
                out.append(f"{pad}    {n} += 1")
                out.append(f"{pad}    {GUARD}")
                out.append(f"{pad}    {var} -= {self.expr(2)}")
                out.append(f"{pad}until {n} >= i % 4")
        elif indent < 3 and r < 0.9 and self.family == "float":
            var = self.rng.choice(self.vars[1:])
            # starts are literals: a start computed from d can be -inf, which loops forever in Luau too
            out.append(f"{pad}for x = {self.rng.choice(['0.5', '-1.25', '0', '-3.5'])}, {self.rng.choice(['3', '2.5', 'i % 4'])}, {self.rng.choice(['0.5', '0.25', '1', '0.75'])} do")
            out.append(f"{pad}    {GUARD}")
            out.append(f"{pad}    {var} += x")
            out.append(f"{pad}end")
        elif indent < 3:
            var = self.rng.choice(self.vars[1:])
            lo = self.rng.choice(["1", "0", "-2", "i % 5"])
            hi = self.rng.choice(["3", "i % 7", "-1", "5"])
            step = self.rng.choice(["", "", ", 2", ", -1", ", -2"])
            out.append(f"{pad}for k = {lo}, {hi}{step} do")
            out.append(f"{pad}    {GUARD}")
            out.append(f"{pad}    {var} += k * {self.expr(2)}")
            out.append(f"{pad}end")
        else:
            var = self.rng.choice(self.vars[1:])
            out.append(f"{pad}{var} = {self.expr(1)}")

    def kernel(self, index):
        self.vars = ["i", "a", "b", "c", "d"]
        self.counter = 0
        lines = [
            "@gpu",
            f"local function k{index}(buf: buffer, i: number)",
            "    local a = buffer.readu32(buf, (i % 19) * 4)",
            "    local b = buffer.readi32(buf, ((i * 7 + 3) % 19) * 4)",
            "    local c = i - 32",
            "    local guard = 0",
        ]
        if self.family == "float":
            lines.append(f"    local d = buffer.readf32(buf, {FSRC} + (i % 20) * 4)")
        else:
            lines.append("    local d = buffer.readu32(buf, ((i + 11) % 19) * 4) // 3")
        for _ in range(self.rng.randint(2, 8)):
            self.statement(lines, 1)
        writes = []
        for s in range(SLOTS):
            kind = self.rng.choice(["u32", "i32"] + (["f32", "f32"] if self.family == "float" else []))
            value = self.rng.choice(self.vars) if self.rng.random() < 0.7 else self.expr(1)
            writes.append(kind)
            lines.append(f"    buffer.write{kind}(buf, {OUT} + (i * {SLOTS} + {s}) * 4, {value})")
        lines.append("end")
        return lines, writes


VEC_OPS = ["add", "sub", "mul", "div", "scale", "neg", "dot", "cross", "normalize", "magnitude", "floor", "ceil", "abs",
           "min", "max", "if"]


class VecGen:
    """kernels over vectors: Luau computes vector math in f32 too, so results should agree closely"""

    def __init__(self, rng):
        self.rng = rng

    def vec(self, depth=0):
        if depth >= 3 or self.rng.random() < 0.3:
            return self.rng.choice(["p", "q", "r", "vector.create(1, -2, 0.5)", "vector.create(s, t, 1)"])
        op = self.rng.choice(VEC_OPS)
        v = lambda: self.vec(depth + 1)
        if op in ("add", "sub", "mul", "div"):
            return f"({v()} {dict(add='+', sub='-', mul='*', div='/')[op]} {v()})"
        if op == "scale":
            return f"({v()} * {self.num(depth + 1)})"
        if op == "neg":
            return f"(-({v()}))"
        if op in ("dot", "magnitude"):
            return f"vector.create(vector.{op}({v()}{', ' + v() if op == 'dot' else ''}), 0, 0)"
        if op == "if":
            return f"(if {self.num(depth + 1)} < {self.num(depth + 1)} then {v()} else {v()})"
        if op in ("min", "max", "cross"):
            return f"vector.{op}({v()}, {v()})"
        return f"vector.{op}({v()})"

    def num(self, depth=0):
        if depth >= 3 or self.rng.random() < 0.4:
            return self.rng.choice(["s", "t", "0.5", "-2", "3"])
        return self.rng.choice([f"{self.vec(depth + 1)}.x", f"{self.vec(depth + 1)}.y", f"vector.dot({self.vec(depth + 1)}, p)",
                                f"vector.magnitude({self.vec(depth + 1)})", f"({self.num(depth + 1)} * {self.num(depth + 1)})"])

    def kernel(self, index):
        lines = [
            "@gpu",
            f"local function k{index}(buf: buffer, i: number)",
            f"    local o = {FSRC} + (i % 20) * 4",
            "    local s = buffer.readf32(buf, o)",
            f"    local t = buffer.readf32(buf, {FSRC} + ((i * 3 + 1) % 20) * 4)",
            "    local p = vector.create(s, t, s * t)",
            "    local q = vector.create(t, 1.5, -s)",
            "    local r = vector.create(0.25, s - t, 2)",
        ]
        writes = []
        for slot in range(0, SLOTS, 4):
            name = f"w{slot}"
            lines.append(f"    local {name} = {self.vec()}")
            for c, comp in enumerate(["x", "y", "z"]):
                lines.append(f"    buffer.writef32(buf, {OUT} + (i * {SLOTS} + {slot + c}) * 4, {name}.{comp})")
            lines.append(f"    buffer.writef32(buf, {OUT} + (i * {SLOTS} + {slot + 3}) * 4, {self.num()})")
            writes += ["f32"] * 4
        lines.append("end")
        return lines, writes


HELPERS = """@gpu
local function h1(x: number, y: number)
    if x < y then
        return x - y
    end
    return x + y * 2
end

@gpu
local function h2(x: number)
    local r = bit32.band(x, 255)
    for k = 1, 3 do
        r += k
    end
    return r + x // 3
end
"""

CHECK = """
-- reasons and errors can span lines (shader compiler messages); the output is one case per line
local function oneLine(s: any): string
    return (string.gsub(tostring(s), "%s+", " "))
end

-- a campaign restarted after a crash skips the cases before "round:kernel" given as the first argument
local startRound, startKernel = 0, 0
do
    local resume = (arg :: {{ string }})[1]
    if resume then
        local r, k = string.match(resume, "(%d+):(%d+)")
        startRound, startKernel = tonumber(r) :: number, tonumber(k) :: number
    end
end

-- runs each kernel with gpu.run and by calling it from Luau, on copies of the inputs, and prints the words that differ; a
-- run that returns false must leave its buffer as it was
local function check(round: number, kernels: {{ any }})
    for j, k in kernels do
        if round < startRound or (round == startRound and j - 1 < startKernel) then
            continue
        end

        local onGpu, inLuau = buffer.create({size}), buffer.create({size})
        buffer.copy(onGpu, 0, inputs)
        buffer.copy(inLuau, 0, inputs)

        local okGpu, errGpu = gpu.run(k, {{ onGpu }}, {n})
        local okLuau, errLuau = pcall(function()
            for i = 0, {n} - 1 do
                k(inLuau, i)
            end
        end)

        if not okGpu and buffer.tostring(onGpu) ~= buffer.tostring(inputs) then
            print("TOUCHED", round, j - 1, oneLine(errGpu))
        elseif errGpu == "Function cannot run in GPU" then
            print("REFUSED", round, j - 1, oneLine(errGpu))
        elseif okGpu and not okLuau then
            print("ERRDIFF", round, j - 1, "ran on the GPU", oneLine(errLuau))
        elseif not okLuau then
            print("BOTHERR", round, j - 1, oneLine(errLuau))
        elseif not okGpu then
            print("RESULT", round, j - 1, "failed", "")
        else
            local diffs = {{}}
            for o = 0, {words} - 1 do
                local p = {out} + o * 4
                local x, y = buffer.readu32(onGpu, p), buffer.readu32(inLuau, p)
                if x ~= y then
                    table.insert(diffs, string.format("%d:%08x:%08x:%.17g:%.17g", o, x, y, buffer.readf32(onGpu, p), buffer.readf32(inLuau, p)))
                end
            end
            print("RESULT", round, j - 1, "gpu", table.concat(diffs, " "))
        end
    end
end
"""


def generate(seed, family, count, tame):
    rng = random.Random(seed)
    gen = VecGen(rng) if family == "vector" else Gen(rng, family, tame)
    return [gen.kernel(j) for j in range(count)]


def build_campaign(rounds, tame):
    """one script for the whole campaign: each round's kernels live in their own block, so one luau-gpu process with
    one Vulkan device runs every kernel, the way a game would"""
    u32 = ", ".join(str(v) for v in (TAME_U32_INPUTS if tame else U32_INPUTS))
    f32 = ", ".join(TAME_F32_INPUTS if tame else F32_INPUTS)
    out = ["-- generated by Gpu/fuzz/difffuzz.py", "", HELPERS]
    out.append(f"""
-- 64 integer words, cycling through the inputs, so that any index masked with 63 is valid, then 20 floats
local integers = {{ {u32} }}
local inputs = buffer.create({OUT} + {N * SLOTS * 4})
for j = 0, 63 do
    buffer.writeu32(inputs, j * 4, integers[j % #integers + 1])
end
for j, v in {{ {f32} }} do
    buffer.writef32(inputs, {FSRC} + (j - 1) * 4, v)
end
""")
    out.append(CHECK.format(words=N * SLOTS, n=N, out=OUT, size=OUT + N * SLOTS * 4))
    for index, (seed, family, kernels) in enumerate(rounds):
        out.append(f"do -- round {index}: seed {seed}, {family}")
        for lines, _ in kernels:
            out += ["    " + line if line else line for line in lines]
            out.append("")
        out.append(f"    check({index}, {{ {', '.join(f'k{j}' for j in range(len(kernels)))} }})")
        out.append("end")
        out.append("")
    return "\n".join(out)


def float_close(x, y):
    if math.isnan(x) and math.isnan(y):
        return True
    if math.isinf(x) or math.isinf(y):
        return x == y
    return abs(x - y) <= 1e-4 * max(1.0, abs(y))


def classify(family, lines, writes, diffs):
    """splits differing words into exact mismatches and the accepted precision differences"""
    hard, soft = [], []
    text = "\n".join(lines)
    for d in diffs:
        o, xg, xv, fg, fv = d.split(":")
        kind = writes[int(o) % SLOTS]
        # a NaN on one side only isn't rounding: the GPU lost Luau's NaN semantics (or produced a NaN Luau doesn't)
        if kind == "f32" and math.isnan(float(fg)) != math.isnan(float(fv)):
            hard.append(d)
            continue
        if family == "vector":
            if float_close(float(fg), float(fv)):
                continue
            # GPU division and sqrt may be off by an ulp (Vulkan doesn't require them to be correctly rounded),
            # which floor, ceil, comparisons and divisions by a near-zero value turn into larger differences
            if any(op in text for op in ("vector.floor(", "vector.ceil(", " / ", "if ", "normalize", "magnitude")):
                soft.append(d)
            else:
                hard.append(d)
        elif family == "float":
            if kind == "f32" and float_close(float(fg), float(fv)):
                continue
            soft.append(d)
        else:
            hard.append(d)
    return hard, soft


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--luau-gpu", default="build/luau-gpu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--kernels", type=int, default=100)
    parser.add_argument("--family", default="both", choices=["int", "float", "vector", "both", "all"])
    parser.add_argument("--profile", default="extreme", choices=["extreme", "tame"])
    parser.add_argument("--workdir", default=tempfile.gettempdir())
    parser.add_argument("--show", type=int, default=3, help="number of failing kernels to print")
    parser.add_argument("--timeout", type=int, default=1800, help="seconds after which the campaign counts as hung")
    parser.add_argument("--show-precision", action="store_true", help="print kernels whose outputs only differ by f32 rounding")
    parser.add_argument("--show-failed", type=int, default=0, help="tame profile: print this many kernels whose run failed")
    args = parser.parse_args()

    tame = args.profile == "tame"
    families = {"both": ["int", "float"], "all": ["int", "float", "vector"]}.get(args.family, [args.family])
    rounds = [(args.seed + r, family, generate(args.seed + r, family, args.kernels, tame)) for r in range(args.rounds) for family in families]

    with tempfile.NamedTemporaryFile("w", suffix=".luau", delete=False, dir=args.workdir) as f:
        f.write(build_campaign(rounds, tame))
        path = f.name

    stats = {"cases": 0, "gpu": 0, "failed": 0, "refused": 0, "bad": 0, "precision": 0, "both_error": 0}
    reasons = {}
    reports = []
    failed = []

    # the output is line buffered, so a crash (a shader compiler bug can take the process down) points at the case
    # after the last one printed; that case is reported and the campaign resumes after it in a new process
    crashes = []
    output = []
    resume = None
    returncode = 0
    stderr = ""
    while True:
        command = ["stdbuf", "-oL", args.luau_gpu, path] + ([f"{resume[0]}:{resume[1]}"] if resume else [])
        try:
            proc = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout)
        except subprocess.TimeoutExpired as e:
            # kernels bound their loops, so this means something else is stuck; the partial output says where
            partial = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
            last = [l for l in partial.splitlines() if l.split("\t")[0] in ("RESULT", "REFUSED", "ERRDIFF", "BOTHERR", "TOUCHED")]
            print(f"campaign didn't finish within {args.timeout} s; last case: {last[-1][:80] if last else 'none'}; script: {path}")
            sys.exit(1)

        lines = proc.stdout.splitlines()
        output += lines
        if proc.returncode >= 0 or len(crashes) >= 20:
            returncode, stderr = proc.returncode, proc.stderr
            break

        cases = [l.split("\t") for l in lines if l.split("\t")[0] in ("RESULT", "REFUSED", "ERRDIFF", "BOTHERR", "TOUCHED")]
        r, j = (int(cases[-1][1]), int(cases[-1][2]) + 1) if cases else (resume or (0, 0))
        if j >= len(rounds[r][2]):
            r, j = r + 1, 0
        if r >= len(rounds):
            break
        crashes.append((r, j, -proc.returncode))
        resume = (r, j + 1) if j + 1 < len(rounds[r][2]) else (r + 1, 0)
        if resume[0] >= len(rounds):
            break

    for line in output:
        parts = line.split("\t", 4) if line.startswith("RESULT") else line.split("\t", 3)
        tag = parts[0]
        if tag not in ("RESULT", "REFUSED", "ERRDIFF", "BOTHERR", "TOUCHED"):
            continue
        r, j = int(parts[1]), int(parts[2])
        seed, family, kernels = rounds[r]
        lines, writes = kernels[j]
        stats["cases"] += 1

        if tag == "REFUSED":
            stats["refused"] += 1
            reason = re.sub(r"^line \d+: ", "", parts[3])
            reasons[reason] = reasons.get(reason, 0) + 1
        elif tag == "BOTHERR":
            stats["both_error"] += 1
        elif tag in ("ERRDIFF", "TOUCHED"):
            # gpu.run succeeded where Luau raises an error, or changed a buffer although it returned false
            stats["bad"] += 1
            reports.append((r, j, line))
        else:
            stats[parts[3]] += 1
            if parts[3] == "failed" and tame:
                failed.append((r, j))
            hard, soft = classify(family, lines, writes, parts[4].split() if len(parts) > 4 else [])
            if hard:
                stats["bad"] += 1
                reports.append((r, j, f"{len(hard)} exact mismatches, first: {hard[:3]}"))
            elif soft:
                stats["precision"] += 1
                if args.show_precision:
                    reports.append((r, j, f"precision only: {soft[:3]}"))

    for r, j, sig in crashes:
        stats["bad"] += 1
        stats["crashes"] = stats.get("crashes", 0) + 1
        reports.insert(0, (r, j, f"luau-gpu died with signal {sig} on this kernel"))

    if returncode != 0:
        stats["bad"] += 1
        reports.append((None, None, "luau-gpu failed: " + stderr[-1000:]))

    shown = reports[: args.show] + [(r, j, "gpu.run failed") for r, j in failed[: args.show_failed]]
    for r, j, what in shown:
        if r is None:
            print(what)
            continue
        seed, family, kernels = rounds[r]
        print(f"seed {seed} ({family}) kernel {j}: {what}")
        print("\n".join("    " + l for l in kernels[j][0]))

    if stats["bad"]:
        print(f"script kept: {path}")
    else:
        os.unlink(path)
    print("totals:", stats)
    for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"  refused {count}x: {reason}")
    sys.exit(1 if stats["bad"] else 0)


if __name__ == "__main__":
    main()
