#!/usr/bin/env python3
# This file is part of the Luau programming language and is licensed under MIT License; see LICENSE.txt for details

# Differential fuzzer for @gpu kernels: generates random kernels in the GPU subset, runs each one with gpu.run, and
# compares the results with a reference computed by luau-gpu in the VM, and reports every output that differs.
#
# usage: difffuzz.py [--luau-gpu build/luau-gpu] [--seed N] [--rounds R] [--kernels K] [--family int|float|both]
#
# Kernels compute with f32 numbers, each operation rounded as Luau's would be if numbers were f32. The reference is the same
# kernel written for the VM with every operation's result rounded to f32, by F, which stores it with buffer.writef32 and
# reads it back: Luau computes the operation in f64, which for +, -, *, / and sqrt of f32 operands rounds to the same f32 as
# the f32 operation; // and % are written as the VM computes them, floor(a / b) and a - floor(a / b) * b, each step
# rounded. Outputs must then be bit-exact, except for ^, which neither side computes exactly: those differences are
# reported separately as "precision" and not counted as failures.
#
# Two families of kernels: int (integer literals and inputs) and float (floats too), and vector kernels.
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


# a variable or a read in an expression, which keeps the compiler from folding it
VARIABLE = re.compile(r"\b(i|a|b|c|d|k|x|buf|v\d+|n\d+)\b")


class Gen:
    def __init__(self, rng, family, tame=False):
        self.rng = rng
        self.family = family
        self.tame = tame
        self.vars = ["i", "a", "b", "c", "d"]
        self.counter = 0

    # every expression is a pair: the kernel's code, and the reference's
    def leaf(self):
        r = self.rng.random()
        if r < 0.45:
            v = self.rng.choice(self.vars)
            return v, v
        if self.family == "float" and r < 0.65:
            lit = self.rng.choice(TAME_FLOAT_LITERALS if self.tame else FLOAT_LITERALS)
        else:
            lit = self.rng.choice(TAME_INT_LITERALS if self.tame else INT_LITERALS)
        # a negative literal is parenthesized as F(...) is: -0.5 ^ c is -(0.5 ^ c)
        return (f"({lit})" if lit.startswith("-") else lit), f"F({lit})"

    def expr(self, depth=0):
        # the compiler folds expressions of constants in f64, where the kernel would compute in f32: they're avoided
        g, r = self.compound(depth)
        if (r.startswith("F(") and r.count("F(") == 1) or VARIABLE.search(g):
            return g, r
        v = self.rng.choice(self.vars)
        return v, v

    def compound(self, depth):
        if depth >= 3 or self.rng.random() < 0.3:
            return self.leaf()

        ops = ["+", "-", "*", "//", "%", "neg", "if", "floor", "ceil", "round", "abs", "min", "max", "clamp", "band",
               "bor", "bxor", "bnot", "lshift", "rshift", "arshift", "extract", "btest", "read", "helper"]
        if self.tame:
            ops = [op for op in ops if op not in ("bnot",)] + ["+", "-", "//", "%", "if", "min", "max"]
        if self.family == "float":
            ops += ["/", "/", "sqrt", "^", "fmod"]

        op = self.rng.choice(ops)
        e = lambda: self.expr(depth + 1)

        def binary(a, o, b):
            (ga, ra), (gb, rb) = a, b
            g = f"({ga} {o} {gb})"
            if o in ("+", "-", "*", "/", "^"):
                return g, f"F({ra} {o} {rb})"
            if o == "//":
                return g, f"F(math.floor(F({ra} / {rb})))"
            return g, f"F({ra} - F(math.floor(F({ra} / {rb})) * {rb}))"

        if op in ("//", "%") and self.tame:
            k = self.rng.choice(['3', '7', '-2', '10', '2.5', '-0.75'] if self.family == 'float' else ['3', '7', '-2', '10'])
            return binary(e(), op, (k, f"F({k})"))
        if op in ("+", "-", "*", "//", "%", "/", "^"):
            return binary(e(), op, e())
        if op == "neg":
            g, r = e()
            return f"(-({g}))", f"(-({r}))"
        if op == "if":
            cmp = self.rng.choice(["<", "<=", ">", ">=", "==", "~="])
            (g1, r1), (g2, r2), (g3, r3), (g4, r4) = e(), e(), e(), e()
            return f"(if {g1} {cmp} {g2} then {g3} else {g4})", f"(if {r1} {cmp} {r2} then {r3} else {r4})"
        if op in ("floor", "ceil", "round", "abs"):
            g, r = e()
            return f"math.{op}({g})", f"math.{op}({r})"
        if op in ("min", "max"):
            (g1, r1), (g2, r2) = e(), e()
            return f"math.{op}({g1}, {g2})", f"math.{op}({r1}, {r2})"
        if op == "clamp":
            lo, hi = sorted([self.rng.randint(-100000, 100000), self.rng.randint(-100000, 100000)])
            g, r = e()
            return f"math.clamp({g}, {lo}, {hi})", f"math.clamp({r}, {lo}, {hi})"
        if op == "sqrt":
            g, r = e()
            return f"math.sqrt(math.abs({g}))", f"F(math.sqrt(math.abs({r})))"
        if op == "fmod":
            k = self.rng.choice(['3', '-2.5', '7', '0.75'])
            g, r = e()
            return f"math.fmod({g}, {k})", f"F(math.fmod({r}, {k}))"
        if op in ("band", "bor", "bxor"):
            (g1, r1), (g2, r2) = e(), e()
            return f"bit32.{op}({g1}, {g2})", f"F(bit32.{op}({r1}, {r2}))"
        if op == "bnot":
            g, r = e()
            return f"bit32.bnot({g})", f"F(bit32.bnot({r}))"
        if op in ("lshift", "rshift", "arshift"):
            if self.tame:
                amount = str(self.rng.randint(-6, 6))
                ga, ra = amount, amount
            else:
                choice = self.rng.random()
                if choice < 0.6:
                    ga = ra = self.rng.choice([str(self.rng.randint(-40, 40)), "-3", "31", "32"])
                else:
                    gx, rx = e()
                    if not VARIABLE.search(gx):
                        gx = rx = self.rng.choice(self.vars)
                    ga, ra = f"({gx}) % 40", f"F({rx} - F(math.floor(F({rx} / 40)) * 40))"
            g, r = e()
            return f"bit32.{op}({g}, {ga})", f"F(bit32.{op}({r}, {ra}))"
        if op == "extract":
            f = self.rng.randint(0, 31)
            w = self.rng.randint(1, 32 - f)
            g, r = e()
            return f"bit32.extract({g}, {f}, {w})", f"F(bit32.extract({r}, {f}, {w}))"
        if op == "btest":
            (g1, r1), (g2, r2) = e(), e()
            return f"(if bit32.btest({g1}, {g2}) then 1 else 0)", f"(if bit32.btest({r1}, {r2}) then 1 else 0)"
        if op == "helper":
            if self.rng.random() < 0.5:
                (g1, r1), (g2, r2) = e(), e()
                return f"h1({g1}, {g2})", f"r1({r1}, {r2})"
            g, r = e()
            return f"h2({g})", f"r2({r})"
        if op == "read":
            g, r = e()
            if not VARIABLE.search(g):
                g = r = self.rng.choice(self.vars)
            # words, or 8 and 16-bit values at any offset of the integer inputs (a 16-bit one can span two words)
            kind = self.rng.choice(["u32", "u32", "u8", "i8", "u16", "i16"])
            if kind == "u32":
                return f"buffer.readu32(buf, (bit32.band({g}, 63)) * 4)", f"F(buffer.readu32(buf, (bit32.band({r}, 63)) * 4))"
            return f"buffer.read{kind}(buf, bit32.band({g}, 253))", f"buffer.read{kind}(buf, bit32.band({r}, 253))"
        raise AssertionError(op)

    def statement(self, out, ref, indent):
        r = self.rng.random()
        pad = "    " * indent

        def both(g, rr=None):
            out.append(pad + g)
            ref.append(pad + (g if rr is None else rr))

        def update(var, op, value):
            g, rv = value
            o = op[:-1]
            if o in ("+", "-", "*", "/"):
                both(f"{var} {op} {g}", f"{var} = F({var} {o} ({rv}))")
            elif o == "//":
                both(f"{var} {op} {g}", f"{var} = F(math.floor(F({var} / ({rv}))))")
            else:
                both(f"{var} {op} {g}", f"{var} = F({var} - F(math.floor(F({var} / ({rv}))) * ({rv})))")

        if r < 0.4:
            name = f"v{self.counter}"
            self.counter += 1
            g, rv = self.expr()
            # a local that's never assigned again is a constant the compiler folds, in f64
            if not VARIABLE.search(g):
                g = rv = self.rng.choice(self.vars)
            both(f"local {name} = {g}", f"local {name} = {rv}")
            self.vars.append(name)
        elif r < 0.6:
            var = self.rng.choice(self.vars[1:])
            op = self.rng.choice(["+=", "-=", "*=", "//=", "%="] + (["/="] if self.family == "float" else []))
            if self.tame and op in ("//=", "%=", "*="):
                k = self.rng.choice(["3", "7", "-2"])
                value = (k, k)
            else:
                value = self.expr(1)
            update(var, op, value)
        elif r < 0.8 and indent < 3:
            var = self.rng.choice(self.vars[1:])
            (g1, r1), (g2, r2), (g3, r3), (g4, r4) = self.expr(1), self.expr(1), self.expr(1), self.expr(1)
            both(f"if {g1} < {g2} then", f"if {r1} < {r2} then")
            both(f"    {var} = {g3}", f"    {var} = {r3}")
            both("else")
            both(f"    {var} = {g4}", f"    {var} = {r4}")
            both("end")
        elif indent < 3 and r < 0.86:
            var = self.rng.choice(self.vars[1:])
            n = f"n{self.counter}"
            self.counter += 1
            g, rv = self.expr(2)
            both(f"local {n} = 0")
            if self.rng.random() < 0.5:
                both(f"while {n} < i % 6 do")
                both(f"    {n} += 1")
                both(f"    {GUARD}")
                both(f"    {var} += {n} * {g}", f"    {var} = F({var} + F({n} * ({rv})))")
                both("end")
            else:
                both("repeat")
                both(f"    {n} += 1")
                both(f"    {GUARD}")
                both(f"    {var} -= {g}", f"    {var} = F({var} - ({rv}))")
                both(f"until {n} >= i % 4")
        elif indent < 3 and r < 0.9 and self.family == "float":
            var = self.rng.choice(self.vars[1:])
            # starts are literals: a start computed from d can be -inf, which loops forever in Luau too; the counters stay
            # exact in f32, as their steps are
            both(f"for x = {self.rng.choice(['0.5', '-1.25', '0', '-3.5'])}, {self.rng.choice(['3', '2.5', 'i % 4'])}, {self.rng.choice(['0.5', '0.25', '1', '0.75'])} do")
            both(f"    {GUARD}")
            both(f"    {var} += x", f"    {var} = F({var} + x)")
            both("end")
        elif indent < 3:
            var = self.rng.choice(self.vars[1:])
            lo = self.rng.choice(["1", "0", "-2", "i % 5"])
            hi = self.rng.choice(["3", "i % 7", "-1", "5"])
            step = self.rng.choice(["", "", ", 2", ", -1", ", -2"])
            g, rv = self.expr(2)
            both(f"for k = {lo}, {hi}{step} do")
            both(f"    {GUARD}")
            both(f"    {var} += k * {g}", f"    {var} = F({var} + F(k * ({rv})))")
            both("end")
        else:
            var = self.rng.choice(self.vars[1:])
            g, rv = self.expr(1)
            both(f"{var} = {g}", f"{var} = {rv}")

    def kernel(self, index):
        self.vars = ["i", "a", "b", "c", "d"]
        self.counter = 0
        head = [
            "    local a = buffer.readu32(buf, (i % 19) * 4)",
            "    local b = buffer.readi32(buf, ((i * 7 + 3) % 19) * 4)",
            "    local c = i - 32",
            "    local guard = 0",
        ]
        refHead = [
            "    local a = F(buffer.readu32(buf, (i % 19) * 4))",
            "    local b = F(buffer.readi32(buf, ((i * 7 + 3) % 19) * 4))",
            "    local c = i - 32",
            "    local guard = 0",
        ]
        if self.family == "float":
            head.append(f"    local d = buffer.readf32(buf, {FSRC} + (i % 20) * 4)")
            refHead.append(f"    local d = buffer.readf32(buf, {FSRC} + (i % 20) * 4)")
        else:
            head.append("    local d = buffer.readu32(buf, ((i + 11) % 19) * 4) // 3")
            refHead.append("    local d = F(math.floor(F(F(buffer.readu32(buf, ((i + 11) % 19) * 4)) / 3)))")
        lines = ["@gpu", f"local function k{index}(buf: buffer, i: number)"] + head
        ref = [f"local function k{index}ref(buf: buffer, i: number)"] + refHead
        for _ in range(self.rng.randint(2, 8)):
            self.statement(lines, ref, 1)
        writes = []
        for s in range(SLOTS):
            kind = self.rng.choice(["u32", "i32"] + (["f32", "f32"] if self.family == "float" else []))
            if self.rng.random() < 0.7:
                v = self.rng.choice(self.vars)
                g, rv = v, v
            else:
                g, rv = self.expr(1)
            writes.append(kind)
            lines.append(f"    buffer.write{kind}(buf, {OUT} + (i * {SLOTS} + {s}) * 4, {g})")
            ref.append(f"    buffer.write{kind}(buf, {OUT} + (i * {SLOTS} + {s}) * 4, {rv})")
        lines.append("end")
        ref.append("end")
        return lines + [""] + ref, writes


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
                                f"vector.magnitude({self.vec(depth + 1)})", f"F({self.num(depth + 1)} * {self.num(depth + 1)})"])

    def kernel(self, index):
        lines = [
            "@gpu",
            f"local function k{index}(buf: buffer, i: number)",
            f"    local o = {FSRC} + (i % 20) * 4",
            "    local s = buffer.readf32(buf, o)",
            f"    local t = buffer.readf32(buf, {FSRC} + ((i * 3 + 1) % 20) * 4)",
            "    local p = vector.create(s, t, F(s * t))",
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
        # the reference rounds the products of numbers, which the kernel computes in f32 without F
        ref = [l.replace(f"local function k{index}(", f"local function k{index}ref(") for l in lines if l != "@gpu"]
        return [l.replace("F(", "(") for l in lines] + [""] + ref, writes


HELPERS = """-- rounds a result of the VM to f32, as kernels compute
local roundBuffer = buffer.create(4)
local function F(x: number): number
    buffer.writef32(roundBuffer, 0, x)
    return buffer.readf32(roundBuffer, 0)
end

@gpu
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

local function r1(x: number, y: number)
    if x < y then
        return F(x - y)
    end
    return F(x + F(y * 2))
end

local function r2(x: number)
    local r = F(bit32.band(x, 255))
    for k = 1, 3 do
        r = F(r + k)
    end
    return F(r + F(math.floor(F(x / 3))))
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

-- runs each kernel with gpu.run and its reference from Luau, on copies of the inputs, and prints the words that differ; a
-- run that returns false must leave its buffer as it was
local function check(round: number, kernels: {{ any }})
    for j, pair in kernels do
        local k, ref = pair[1], pair[2]
        if round < startRound or (round == startRound and j - 1 < startKernel) then
            continue
        end

        local onGpu, inLuau = buffer.create({size}), buffer.create({size})
        buffer.copy(onGpu, 0, inputs)
        buffer.copy(inLuau, 0, inputs)

        local okGpu, errGpu = gpu.run(k, {{ onGpu }}, {n})
        local okLuau, errLuau = pcall(function()
            for i = 0, {n} - 1 do
                ref(inLuau, i)
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
            print("RESULT", round, j - 1, "failed", oneLine(errGpu))
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
        out.append("    local cases = {}")
        # each kernel and its reference in a block of their own, which keeps the locals of a round within the limit
        for j, (lines, _) in enumerate(kernels):
            out.append("    do")
            out += ["        " + line if line else line for line in lines]
            out.append(f"        cases[{j + 1}] = {{ k{j}, k{j}ref }}")
            out.append("    end")
        out.append(f"    check({index}, cases)")
        out.append("end")
        out.append("")
    return "\n".join(out)


def classify(family, lines, writes, diffs):
    """splits differing words into exact mismatches and the accepted precision differences: only ^ isn't exact"""
    hard, soft = [], []
    inexact = any(" ^ " in l for l in lines)
    for d in diffs:
        o, xg, xv, fg, fv = d.split(":")
        kind = writes[int(o) % SLOTS]
        # NaN payloads aren't specified, but a NaN on one side only isn't rounding
        if kind == "f32" and math.isnan(float(fg)) and math.isnan(float(fv)):
            continue
        if inexact and not (kind == "f32" and math.isnan(float(fg)) != math.isnan(float(fv))):
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
        # the fuzzer compares results, so runs wait for their kernel instead of returning while it compiles
        command = ["stdbuf", "-oL", args.luau_gpu, "--wait-for-pipelines", path] + ([f"{resume[0]}:{resume[1]}"] if resume else [])
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
            reason = "refused: " + re.sub(r"^line \d+: ", "", parts[3])
            reasons[reason] = reasons.get(reason, 0) + 1
        elif tag == "BOTHERR":
            stats["both_error"] += 1
        elif tag in ("ERRDIFF", "TOUCHED"):
            # gpu.run succeeded where Luau raises an error, or changed a buffer although it returned false
            stats["bad"] += 1
            reports.append((r, j, line))
        else:
            stats[parts[3]] += 1
            if parts[3] == "failed":
                # numbers are f32 as in a VM computing in f32: kernels only fail on what fails there too, or on what the GPU
                # doesn't do (unaligned accesses, the loop budget)
                reason = re.sub(r" at line \d+", "", parts[4] if len(parts) > 4 else "")
                reasons["failed: " + reason] = reasons.get("failed: " + reason, 0) + 1
                if tame:
                    failed.append((r, j))
                continue
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
        print(f"  {count}x: {reason}")
    sys.exit(1 if stats["bad"] else 0)


if __name__ == "__main__":
    main()
