-- Runs every kernel with gpu.run and from Luau on copies of the same buffer, and checks that both give the same results.
-- usage: luau-gpu Gpu/examples/selftest.luau

local N = 1000

local function copy(b: buffer): buffer
    local c = buffer.create(buffer.len(b))
    buffer.copy(c, 0, b)
    return c
end

local failures = 0

-- the first runs of a kernel return false while the driver creates its pipeline, which each check waits for
local COMPILING = "the kernel is being compiled for the GPU"
local function runReady(fn, buffers: { buffer }, runs: number)
    local ok, err = gpu.run(fn, buffers, runs)
    while not ok and err == COMPILING do
        ok, err = gpu.run(fn, buffers, runs)
    end
    return ok, err
end

local function fail(name: string, problem: string)
    print(string.format("FAIL %-22s %s", name, problem))
    failures += 1
end

-- runs fn(b, i) for i = 0 .. runs-1 with gpu.run on a copy of the buffer, and from Luau on another; floats are compared
-- with a relative tolerance. Numbers are f32 on the GPU: the kernels compared exactly keep to integers f32 holds, below 2^24
local function check(name: string, fn, runs: number, input: buffer, floatTolerance: number?)
    local onGpu, inLuau = copy(input), copy(input)

    local ok, err = runReady(fn, { onGpu }, runs)
    if not ok then
        return fail(name, `didn't run on the GPU: {err}`)
    end

    for i = 0, runs - 1 do
        fn(inLuau, i)
    end

    for o = 0, buffer.len(onGpu) - 4, 4 do
        local x, y = buffer.readu32(onGpu, o), buffer.readu32(inLuau, o)
        if x ~= y then
            local close = false
            if floatTolerance then
                local fx, fy = buffer.readf32(onGpu, o), buffer.readf32(inLuau, o)
                close = math.abs(fx - fy) <= floatTolerance * math.max(1, math.abs(fy))
            end
            if not close then
                return fail(
                    name,
                    string.format("offset %d: gpu %08x luau %08x (%g vs %g)", o, x, y, buffer.readf32(onGpu, o), buffer.readf32(inLuau, o))
                )
            end
        end
    end

    print(string.format("ok   %s", name))
end

-- runs fn with gpu.run, which must return false and why, with the buffer left byte for byte as it was
local function fails(name: string, fn, runs: number, input: buffer, expected: string?)
    local onGpu = copy(input)
    local ok, err = runReady(fn, { onGpu }, runs)
    if ok then
        return fail(name, "ran on the GPU")
    elseif expected and err ~= expected then
        return fail(name, `failed with '{err}'`)
    elseif buffer.tostring(onGpu) ~= buffer.tostring(input) then
        return fail(name, "changed the buffer")
    end
    print(string.format("ok   %s: %s", name, err))
end

-- a buffer of n words written by f, followed by 'extra' zero bytes for results
local function filled(n: number, f: (number) -> number, float: boolean?, extra: number?): buffer
    local b = buffer.create(n * 4 + (extra or 0))
    for i = 0, n - 1 do
        if float then
            buffer.writef32(b, i * 4, f(i))
        else
            buffer.writeu32(b, i * 4, f(i))
        end
    end
    return b
end

local SCALE = 3

@gpu
local function intArith(b: buffer, i: number)
    local x = i - 500
    local d = (i % 7) - 3
    local o = i * 16
    buffer.writei32(b, o, x * SCALE + d)
    buffer.writei32(b, o + 4, if d ~= 0 then x // d else 0)
    buffer.writei32(b, o + 8, if d ~= 0 then x % d else 0)
    buffer.writei32(b, o + 12, -x)
end
check("integer arithmetic", intArith, N, buffer.create(N * 16))

@gpu
local function floatMath(b: buffer, i: number)
    local x = (i - 500) / 37
    local o = i * 32
    buffer.writef32(b, o, math.sin(x) * math.cos(x * 2))
    buffer.writef32(b, o + 4, math.sqrt(math.abs(x)))
    buffer.writef32(b, o + 8, x % 1.5)
    buffer.writef32(b, o + 12, x // 0.25)
    buffer.writef32(b, o + 16, x ^ 2)
    buffer.writef32(b, o + 20, math.clamp(x, -2, 2.5))
    buffer.writef32(b, o + 24, math.max(x, 1, -x))
    buffer.writei32(b, o + 28, math.floor(x) + math.round(x * 3))
end
check("float math", floatMath, N, buffer.create(N * 32), 1e-5)

-- inputs in the first N words, results after them
@gpu
local function bits(b: buffer, i: number)
    local v = buffer.readu32(b, i * 4)
    local o = N * 4 + i * 16
    buffer.writeu32(b, o, bit32.bor(bit32.lshift(v, 3), bit32.band(v, 0xff)))
    buffer.writeu32(b, o + 4, bit32.bxor(v, 0xbeef))
    buffer.writeu32(b, o + 8, bit32.rshift(v, i % 32))
    buffer.writeu32(b, o + 12, bit32.bxor(v, 0xffffff))
end
check("bit32 on 21-bit words", bits, N, filled(N, function(i) return (i * 2654435761) % 2097152 end, false, N * 16))

-- above 2^24, numbers are the nearest f32: what the reads and bit32 give rounds, as any result does
@gpu
local function wideNumbers(b: buffer, i: number)
    local v = buffer.readu32(b, 0) -- 0xffffffff
    local w = buffer.readu32(b, 4) -- 0xdeadbeef
    buffer.writef32(b, 8, v)
    buffer.writeu32(b, 12, v)
    buffer.writeu32(b, 16, bit32.bxor(w, 0x100))
    buffer.writef32(b, 20, i + 16777217)
end
do
    local b = buffer.create(24)
    buffer.writeu32(b, 0, 0xffffffff)
    buffer.writeu32(b, 4, 0xdeadbeef)
    local ok, err = runReady(wideNumbers, { b }, 1)
    if not ok then
        fail("numbers above 2^24", `didn't run on the GPU: {err}`)
    elseif
        buffer.readf32(b, 8) ~= 4294967296 -- 0xffffffff rounds to 2^32
        or buffer.readu32(b, 12) ~= 0 -- which writes as 0, modulo 2^32
        or buffer.readu32(b, 16) ~= 0xdeadbe00 -- 0xdeadbeef reads as 0xdeadbf00
        or buffer.readf32(b, 20) ~= 16777216 -- 2^24 + 1 rounds to even
    then
        fail("numbers above 2^24", string.format("%g %08x %08x %g", buffer.readf32(b, 8), buffer.readu32(b, 12), buffer.readu32(b, 16), buffer.readf32(b, 20)))
    else
        print("ok   numbers above 2^24")
    end
end

@gpu
local function control(b: buffer, i: number)
    local sum = 0
    for k = 1, i % 50 do
        if k % 3 == 0 then
            continue
        end
        sum += k
        if sum > 300 then
            break
        end
    end
    local j = 0
    while j * j < i do
        j += 1
    end
    local r = 0
    repeat
        r += 2
    until r >= i % 9
    local acc
    if i % 2 == 0 then
        acc = 1.5
    else
        acc = 2
    end
    for k = 10, 1, -3 do
        acc += k
    end
    buffer.writei32(b, i * 16, sum)
    buffer.writei32(b, i * 16 + 4, j)
    buffer.writei32(b, i * 16 + 8, r)
    buffer.writef32(b, i * 16 + 12, acc)
end
check("control flow", control, N, buffer.create(N * 16))

@gpu
local function edge(ax: number, ay: number, bx: number, by: number, px: number, py: number)
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)
end

@gpu
local function storePair(b: buffer, o: number, x: number, y: number)
    buffer.writef32(b, o, x)
    buffer.writef32(b, o + 4, y)
end

@gpu
local function helpers(b: buffer, i: number)
    local px, py = i % 40, i // 40
    storePair(b, i * 12, edge(0, 0, 40, 25, px, py), edge(3, 1, 7, 2, px + 0.5, py + 0.5))
    buffer.writei32(b, i * 12 + 8, edge(1, 2, 3, 4, px, py))
end
check("helper functions", helpers, N, buffer.create(N * 12), 1e-6)

-- points in the first 3N words, results after them
@gpu
local function vectors(b: buffer, i: number)
    local o = i * 12
    local p = vector.create(buffer.readf32(b, o), buffer.readf32(b, o + 4), buffer.readf32(b, o + 8))
    local q = vector.normalize(vector.cross(p, vector.create(0, 1, 0)) + p * 0.5)
    local d = vector.dot(q, p) / (vector.magnitude(p) + 1)
    local r = N * 12 + i * 16
    buffer.writef32(b, r, q.x)
    buffer.writef32(b, r + 4, q.y)
    buffer.writef32(b, r + 8, q.z)
    buffer.writef32(b, r + 12, d)
end
check("vectors", vectors, N, filled(N * 3, function(i) return math.sin(i) * 10 end, true, N * 16), 1e-5)

-- values in the first 10N words, then 16 bins and the minimum, which every invocation shares and updates with atomics
@gpu
local function histogram(b: buffer, i: number)
    local v = buffer.readu32(b, i * 4)
    gpu.atomicadd(b, N * 40 + (v % 16) * 4, 1)
    gpu.atomicmin(b, N * 40 + 64, v)
end
do
    local input = filled(N * 10, function(i) return (i * 7919) % 100003 + 3 end, false, 68)
    buffer.writeu32(input, N * 40 + 64, 0xffffffff)
    check("atomics", histogram, N * 10, input)
end

-- f32 holds every index up to 2^24, as many runs as gpu.run allows
@gpu
local function largeIndex(b: buffer, i: number)
    buffer.writeu32(b, i * 4, i)
end
check("2^24 runs", largeIndex, 16_777_216, buffer.create(16_777_216 * 4))
do
    local ok, err = pcall(gpu.run, largeIndex, { buffer.create(4) }, 16_777_217)
    if ok then
        fail("more runs", "didn't raise an error")
    else
        print(`ok   more runs: {err}`)
    end
end

---------------------------------------------------------------------------------------------------
-- safety: whatever a kernel does, gpu.run either gives Luau's results, or returns false and leaves the buffer as it was

@gpu
local function outOfBounds(b: buffer, i: number)
    buffer.writeu32(b, i * 4 + 4, i)
end
fails("out of bounds", outOfBounds, 10, buffer.create(40))

-- functions outside the subset can't run
local CANNOT_RUN = "Function cannot run in GPU"

@gpu
local function usesTable(b: buffer, i: number)
    local t = { 1, 2 }
    buffer.writeu32(b, 0, t[1])
end
fails("unsupported code", usesTable, 1, buffer.create(4), CANNOT_RUN)

local function notGpu(b: buffer, i: number)
    buffer.writeu32(b, i * 4, i)
end
fails("not @gpu", notGpu, 1, buffer.create(4), CANNOT_RUN)

-- a loop far longer than a frame: the GPU stops once the loop budget is spent (2^28 iterations per invocation)
local LONG_LOOP = 268435456

@gpu
local function longLoop(b: buffer, i: number)
    local acc = 0
    for k = 1, LONG_LOOP do
        acc += 1
    end
    buffer.writei32(b, i * 4, acc + i)
end
fails("long loop", longLoop, 4, buffer.create(16))

-- integer edge cases where the GPU's arithmetic and Luau's could disagree, all held by f32; inputs in the first 3 words
@gpu
local function integerEdges(b: buffer, i: number)
    local a = buffer.readi32(b, 0) -- -2147483648
    local m = buffer.readi32(b, 4) -- -1
    local s = buffer.readi32(b, 8) -- -7
    local o = 12 + i * 48
    buffer.writei32(b, o, s // 2)
    buffer.writei32(b, o + 4, s % 2)
    buffer.writei32(b, o + 8, 7 % (s + 5))
    buffer.writei32(b, o + 12, s % (s + 5))
    buffer.writei32(b, o + 16, math.floor(s / 14))
    buffer.writei32(b, o + 20, math.round(s / 2.8))
    buffer.writeu32(b, o + 24, bit32.rshift(0xffffff00, 31 - i))
    buffer.writeu32(b, o + 28, bit32.lshift(1, 31))
    buffer.writeu32(b, o + 32, bit32.arshift(0x80000000, 7))
    buffer.writeu32(b, o + 36, bit32.lshift(3, 30))
    buffer.writei32(b, o + 40, a - m * 128)
    buffer.writei32(b, o + 44, a * m - 256)
end
do
    local input = buffer.create(12 + 96)
    buffer.writei32(input, 0, -2147483648)
    buffer.writei32(input, 4, -1)
    buffer.writei32(input, 8, -7)
    check("integer edge cases", integerEdges, 2, input)
end

-- like native code, shifts by 32 or more leave to the VM builtin, which the GPU can't do
@gpu
local function wideShifts(b: buffer, i: number)
    buffer.writeu32(b, i * 8, bit32.rshift(0xffffffff, 32 + i))
    buffer.writeu32(b, i * 8 + 4, bit32.lshift(1, 40 - i))
end
fails("shifts by 32 or more", wideShifts, 2, buffer.create(16))

-- numbers don't overflow as 32-bit integers would: results beyond what f32 holds round, as Luau's writef32 rounds them
@gpu
local function beyond32Bits(b: buffer, i: number)
    local a = buffer.readi32(b, 0) -- -2147483648
    local m = buffer.readi32(b, 4) -- -1
    buffer.writef32(b, 8 + i * 8, a - 1 - i)
    buffer.writef32(b, 12 + i * 8, a // m)
end
do
    local input = buffer.create(24)
    buffer.writei32(input, 0, -2147483648)
    buffer.writei32(input, 4, -1)
    check("beyond 32-bit integers", beyond32Bits, 2, input)
end

-- each invocation reads and writes its own word
@gpu
local function inPlace(b: buffer, i: number)
    buffer.writeu32(b, i * 4, buffer.readu32(b, i * 4) * 2 + 1)
end
check("in place", inPlace, N, filled(N, function(i) return i end))

-- zero length buffers: running on them is fine, any access to them is out of bounds
@gpu
local function lengthOnly(b: buffer, i: number)
    if buffer.len(b) > 0 then
        buffer.writeu32(b, 0, 1)
    end
end
check("empty buffer", lengthOnly, 8, buffer.create(0))

@gpu
local function readsEmpty(b: buffer, i: number)
    buffer.writeu32(b, 0, buffer.readu32(b, 0))
end
fails("reading an empty buffer", readsEmpty, 8, buffer.create(0))

-- kernels can't reach anything but their buffer: other APIs can't run on the GPU
@gpu
local function runsAgain(b: buffer, i: number)
    gpu.run(runsAgain, { b }, 0)
end

@gpu
local function readsClock(b: buffer, i: number)
    buffer.writef32(b, i * 4, os.clock())
end

@gpu
local function prints(b: buffer, i: number)
    print(i)
end

for _, case in { { "runs again", runsAgain }, { "reads the clock", readsClock }, { "prints", prints } } do
    fails(case[1], case[2], 1, buffer.create(4), CANNOT_RUN)
end
do
    -- and still run from Luau
    local out = buffer.create(4)
    readsClock(out, 0)
    assert(buffer.readf32(out, 0) > 0)
end

-- 2^24 invocations contending on 1024 counters, checked against the exact counts (Luau would take seconds)
@gpu
local function manyInvocations(b: buffer, i: number)
    gpu.atomicadd(b, (i % 1024) * 4, 1)
end
do
    local INVOCATIONS = 16_777_216
    local bins = buffer.create(1024 * 4)
    local ok, err = runReady(manyInvocations, { bins }, INVOCATIONS)
    for b = 0, 1023 do
        local expected = INVOCATIONS // 1024 + (if b < INVOCATIONS % 1024 then 1 else 0)
        ok = ok and buffer.readu32(bins, b * 4) == expected
    end
    if ok then
        print("ok   2^24 invocations")
    else
        fail("2^24 invocations", tostring(err))
    end
end

print(if failures == 0 then "all tests passed" else `{failures} failures`)
if failures > 0 then
    error("selftest failed")
end
