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

print(if failures == 0 then "all tests passed" else `{failures} failures`)
if failures > 0 then
    error("selftest failed")
end
