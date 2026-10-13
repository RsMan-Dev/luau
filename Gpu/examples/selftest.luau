-- Runs every kernel twice, on the GPU and in the VM, and checks that both produce the same buffers.
-- usage: luau-gpu Gpu/examples/selftest.luau

local N = 1000

local function copy(b: buffer): buffer
    local c = buffer.create(buffer.len(b))
    buffer.copy(c, 0, b)
    return c
end

local failures = 0

-- runs fn on copies of the inputs on both backends; floats are compared with a relative tolerance (GPU math is f32)
local function check(name: string, fn, n: number, inputs: { buffer }, floatTolerance: number?)
    local status, reason = gpu.status(fn)
    if status ~= "gpu" then
        print(string.format("FAIL %-22s not translated: %s", name, tostring(reason)))
        failures += 1
        return
    end

    local onGpu, inVm = {}, {}
    for i, b in inputs do
        onGpu[i] = copy(b)
        inVm[i] = copy(b)
    end

    local before = host.stats().gpuDispatches
    host.setgpu(true)
    gpu.dispatch(fn, n, table.unpack(onGpu))
    local ran = host.stats().gpuDispatches > before
    host.setgpu(false)
    gpu.dispatch(fn, n, table.unpack(inVm))
    host.setgpu(true)

    if not ran then
        print(string.format("FAIL %-22s did not run on the GPU", name))
        failures += 1
        return
    end

    for i = 1, #inputs do
        local a, b = onGpu[i], inVm[i]
        for o = 0, buffer.len(a) - 4, 4 do
            local x, y = buffer.readu32(a, o), buffer.readu32(b, o)
            if x ~= y then
                local ok = false
                if floatTolerance then
                    local fx, fy = buffer.readf32(a, o), buffer.readf32(b, o)
                    ok = math.abs(fx - fy) <= floatTolerance * math.max(1, math.abs(fy))
                end
                if not ok then
                    print(string.format("FAIL %-22s buffer %d offset %d: gpu %08x vm %08x (%g vs %g)", name, i, o, x, y,
                        buffer.readf32(a, o), buffer.readf32(b, o)))
                    failures += 1
                    return
                end
            end
        end
    end

    print(string.format("ok   %s", name))
end

local function filled(n: number, f: (number) -> number, float: boolean?): buffer
    local b = buffer.create(n * 4)
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
local function intArith(i: number, out: buffer)
    local a = i - 500
    local b = (i % 7) - 3
    local o = i * 16
    buffer.writei32(out, o, a * SCALE + b)
    buffer.writei32(out, o + 4, if b ~= 0 then a // b else 0)
    buffer.writei32(out, o + 8, if b ~= 0 then a % b else 0)
    buffer.writei32(out, o + 12, -a)
end
check("integer arithmetic", intArith, N, { buffer.create(N * 16) })

@gpu
local function floatMath(i: number, out: buffer)
    local x = (i - 500) / 37
    local o = i * 32
    buffer.writef32(out, o, math.sin(x) * math.cos(x * 2))
    buffer.writef32(out, o + 4, math.sqrt(math.abs(x)))
    buffer.writef32(out, o + 8, x % 1.5)
    buffer.writef32(out, o + 12, x // 0.25)
    buffer.writef32(out, o + 16, x ^ 2)
    buffer.writef32(out, o + 20, math.clamp(x, -2, 2.5))
    buffer.writef32(out, o + 24, math.max(x, 1, -x))
    buffer.writei32(out, o + 28, math.floor(x) + math.round(x * 3))
end
check("float math", floatMath, N, { buffer.create(N * 32) }, 1e-5)

@gpu
local function bits(i: number, src: buffer, out: buffer)
    local v = buffer.readu32(src, i * 4)
    local o = i * 16
    buffer.writeu32(out, o, bit32.bor(bit32.lshift(v, 3), bit32.band(v, 0xff)))
    buffer.writeu32(out, o + 4, bit32.bxor(v, 0xdeadbeef))
    buffer.writeu32(out, o + 8, bit32.rshift(v, i % 40))
    buffer.writeu32(out, o + 12, bit32.bnot(v))
end
check("bit32 on u32", bits, N, { filled(N, function(i) return (i * 2654435761) % 4294967296 end), buffer.create(N * 16) })

@gpu
local function control(i: number, out: buffer)
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
    buffer.writei32(out, i * 16, sum)
    buffer.writei32(out, i * 16 + 4, j)
    buffer.writei32(out, i * 16 + 8, r)
    buffer.writef32(out, i * 16 + 12, acc)
end
check("control flow", control, N, { buffer.create(N * 16) })

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
local function helpers(i: number, out: buffer)
    local px, py = i % 40, i // 40
    storePair(out, i * 12, edge(0, 0, 40, 25, px, py), edge(3, 1, 7, 2, px + 0.5, py + 0.5))
    buffer.writei32(out, i * 12 + 8, edge(1, 2, 3, 4, px, py))
end
check("helper functions", helpers, N, { buffer.create(N * 12) }, 1e-6)

@gpu
local function vectors(i: number, src: buffer, out: buffer)
    local o = i * 12
    local p = vector.create(buffer.readf32(src, o), buffer.readf32(src, o + 4), buffer.readf32(src, o + 8))
    local q = vector.normalize(vector.cross(p, vector.create(0, 1, 0)) + p * 0.5)
    local d = vector.dot(q, p) / (vector.magnitude(p) + 1)
    buffer.writef32(out, i * 16, q.x)
    buffer.writef32(out, i * 16 + 4, q.y)
    buffer.writef32(out, i * 16 + 8, q.z)
    buffer.writef32(out, i * 16 + 12, d)
end
check("vectors", vectors, N, { filled(N * 3, function(i) return math.sin(i) * 10 end, true), buffer.create(N * 16) }, 1e-5)

@gpu
local function histogram(i: number, src: buffer, hist: buffer, extremes: buffer)
    local v = buffer.readu32(src, i * 4)
    gpu.atomicadd(hist, (v % 16) * 4, 1)
    gpu.atomicmin(extremes, 0, v)
    gpu.atomicmax(extremes, 4, v)
end
local ext = buffer.create(8)
buffer.writeu32(ext, 0, 0xffffffff)
check("atomics", histogram, N * 10, { filled(N * 10, function(i) return (i * 7919) % 100003 end), buffer.create(64), ext })

@gpu
local function largeIndex(i: number, out: buffer)
    buffer.writeu32(out, i * 4, i)
end
check("20M invocations", largeIndex, 20_000_000, { buffer.create(20_000_000 * 4) })

-- errors on the GPU make the dispatch rerun in the VM, which raises the Luau error
@gpu
local function outOfBounds(i: number, out: buffer)
    buffer.writeu32(out, i * 4 + 4, i)
end
do
    local before = host.stats().fallbacks
    local ok, err = pcall(gpu.dispatch, outOfBounds, 10, buffer.create(40))
    local fellBack = host.stats().fallbacks > before
    if not ok and fellBack and string.find(err, "out of bounds") then
        print("ok   out of bounds falls back to the VM error")
    else
        print("FAIL out of bounds", ok, err, fellBack)
        failures += 1
    end
end

-- functions outside the subset report why they stay in the VM
@gpu
local function usesTable(i: number, out: buffer)
    local t = { 1, 2 }
    buffer.writeu32(out, 0, t[1])
end
do
    local where, why = gpu.status(usesTable)
    if where == "vm" and why and string.find(why, "tables") then
        print("ok   unsupported code reports: " .. why)
    else
        print("FAIL unsupported code", where, why)
        failures += 1
    end
    local b = buffer.create(4)
    gpu.dispatch(usesTable, 1, b)
    assert(buffer.readu32(b, 0) == 1)
end

---------------------------------------------------------------------------------------------------
-- safety: whatever a kernel does, the dispatch either completes on the GPU with the VM's results, or is discarded and
-- rerun in the VM

-- runs fn on both paths and checks the results are identical; 'path' is "gpu" if the dispatch must complete on the GPU,
-- "fallback" if it must be discarded and rerun in the VM; with 'alias', the first buffer is passed twice
local function samePaths(name: string, path: "gpu" | "fallback", fn, n: number, inputs: { buffer }, alias: boolean?)
    local onGpu, inVm = {}, {}
    for i, b in inputs do
        onGpu[i] = copy(b)
        inVm[i] = copy(b)
    end

    local function args(list: { buffer })
        if alias then
            return list[1], list[1], table.unpack(list, 2)
        end
        return table.unpack(list)
    end

    local before = host.stats()
    host.setgpu(true)
    local okGpu, errGpu = pcall(gpu.dispatch, fn, n, args(onGpu))
    local after = host.stats()
    host.setgpu(false)
    local okVm, errVm = pcall(gpu.dispatch, fn, n, args(inVm))
    host.setgpu(true)

    local ranOnGpu = after.gpuDispatches > before.gpuDispatches
    local fellBack = after.fallbacks > before.fallbacks

    local problem = nil
    if path == "gpu" and not ranOnGpu then
        problem = "did not complete on the GPU"
    elseif path == "fallback" and not fellBack then
        problem = "was not discarded and rerun in the VM"
    elseif okGpu ~= okVm then
        problem = `errors differ: {tostring(errGpu)} / {tostring(errVm)}`
    else
        for i = 1, #inputs do
            if buffer.tostring(onGpu[i]) ~= buffer.tostring(inVm[i]) then
                problem = `buffer {i} differs from the VM`
                break
            end
        end
    end

    if problem then
        print(string.format("FAIL %-22s %s", name, problem))
        failures += 1
    else
        print(string.format("ok   %s", name))
    end
end

-- a loop far longer than a frame: the GPU stops once the loop budget is spent and the VM reruns the dispatch
-- (2^28 iterations per invocation; the VM finishes in under a second, and so does the GPU when there is no budget)
local LONG_LOOP = 268435456

@gpu
local function longLoop(i: number, out: buffer)
    local acc = 0
    for k = 1, LONG_LOOP do
        acc += 1
    end
    buffer.writei32(out, i * 4, acc + i)
end
samePaths("long loop is discarded", "fallback", longLoop, 4, { buffer.create(16) })

-- integer edge cases where 32-bit GPU arithmetic and Luau's doubles could disagree
@gpu
local function integerEdges(i: number, src: buffer, out: buffer)
    local a = buffer.readi32(src, 0) -- -2147483648
    local m = buffer.readi32(src, 4) -- -1
    local s = buffer.readi32(src, 8) -- -7
    local o = i * 48
    buffer.writei32(out, o, s // 2)
    buffer.writei32(out, o + 4, s % 2)
    buffer.writei32(out, o + 8, 7 % (s + 5))
    buffer.writei32(out, o + 12, s % (s + 5))
    buffer.writei32(out, o + 16, math.floor(s / 14))
    buffer.writei32(out, o + 20, math.round(s / 2.8))
    buffer.writeu32(out, o + 24, bit32.rshift(0xffffffff, 32 + i))
    buffer.writeu32(out, o + 28, bit32.lshift(1, 31))
    buffer.writeu32(out, o + 32, bit32.arshift(0x80000000, 31))
    buffer.writeu32(out, o + 36, bit32.lshift(1, 40))
    buffer.writei32(out, o + 40, a - m)
    buffer.writei32(out, o + 44, a + s * m)
end
do
    local src = buffer.create(12)
    buffer.writei32(src, 0, -2147483648)
    buffer.writei32(src, 4, -1)
    buffer.writei32(src, 8, -7)
    samePaths("integer edge cases", "gpu", integerEdges, 2, { src, buffer.create(96) })
end

-- -2147483648 // -1 doesn't fit 32 bits: the GPU must not produce its own answer
@gpu
local function overflowingDivision(i: number, src: buffer, out: buffer)
    buffer.writef32(out, i * 4, buffer.readi32(src, 0) // buffer.readi32(src, 4))
end
do
    local src = buffer.create(8)
    buffer.writei32(src, 0, -2147483648)
    buffer.writei32(src, 4, -1)
    samePaths("INT_MIN // -1", "fallback", overflowingDivision, 1, { src, buffer.create(4) })
end

-- the same buffer passed twice, read through one parameter and written through the other
@gpu
local function inPlace(i: number, src: buffer, dst: buffer)
    buffer.writeu32(dst, i * 4, buffer.readu32(src, i * 4) * 2 + 1)
end
samePaths("aliased buffers", "gpu", inPlace, N, { filled(N, function(i) return i end) }, true)

-- zero length buffers: unused ones are fine, any access to them is out of bounds
@gpu
local function ignoresEmpty(i: number, empty: buffer, out: buffer)
    buffer.writeu32(out, i * 4, buffer.len(empty) + i)
end
samePaths("unused empty buffer", "gpu", ignoresEmpty, 8, { buffer.create(0), buffer.create(32) })

@gpu
local function readsEmpty(i: number, empty: buffer, out: buffer)
    buffer.writeu32(out, i * 4, buffer.readu32(empty, 0))
end
samePaths("reading an empty buffer", "fallback", readsEmpty, 8, { buffer.create(0), buffer.create(32) })

-- kernels can't reach anything but their buffers: other APIs keep them in the VM
@gpu
local function dispatchesAgain(i: number, out: buffer)
    gpu.dispatch(dispatchesAgain, 0, out)
end

@gpu
local function readsClock(i: number, out: buffer)
    buffer.writef32(out, i * 4, os.clock())
end

@gpu
local function prints(i: number, out: buffer)
    print(i)
end

for _, case in
    {
        { dispatchesAgain, "gpu.dispatch can't be called from a kernel" },
        { readsClock, "os.clock is not supported" },
        { prints, "'print' is not a @gpu function" },
    }
do
    local where, why = gpu.status(case[1])
    if where == "vm" and why and string.find(why, case[2], 1, true) then
        print("ok   stays in the VM: " .. why)
    else
        print("FAIL expected the VM", where, why)
        failures += 1
    end
end
do
    -- and still runs there
    local out = buffer.create(4)
    gpu.dispatch(readsClock, 1, out)
    assert(buffer.readf32(out, 0) > 0)
end

-- 50M invocations contending on 1024 counters, checked against the exact counts (the VM would take seconds)
@gpu
local function manyInvocations(i: number, bins: buffer)
    gpu.atomicadd(bins, (i % 1024) * 4, 1)
end
do
    local INVOCATIONS = 50_000_000
    local bins = buffer.create(1024 * 4)
    local before = host.stats().gpuDispatches
    gpu.dispatch(manyInvocations, INVOCATIONS, bins)
    local ok = host.stats().gpuDispatches > before
    for b = 0, 1023 do
        local expected = INVOCATIONS // 1024 + (if b < INVOCATIONS % 1024 then 1 else 0)
        ok = ok and buffer.readu32(bins, b * 4) == expected
    end
    if ok then
        print("ok   50M invocations")
    else
        print("FAIL 50M invocations")
        failures += 1
    end
end

print(if failures == 0 then "all tests passed" else `{failures} failures`)
if failures > 0 then
    error("selftest failed")
end
