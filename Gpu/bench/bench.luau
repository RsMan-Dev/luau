--!native
-- Compares gpu.dispatch on the GPU against the same kernels running as native Luau code.
--
-- usage: luau-gpu Gpu/bench/bench.luau [filter]
--   filter: only run scenarios whose name contains this text
--
-- Each scenario runs on copies of the same inputs in three modes:
--   loop      the kernels called from a plain Luau loop, compiled to native code: what a script would do without gpu.dispatch
--   vm        gpu.dispatch without a GPU backend, which calls the kernel from C for every invocation
--   gpu       gpu.dispatch on the GPU
-- GPU timings include every transfer a dispatch makes (uploads of changed buffers, downloads of written buffers), since
-- that is what a script pays. Outputs are compared bit for bit when the computation is exact, and with a tolerance when
-- the GPU's f32 math can differ from Luau's f64.

local filter = (arg :: { string })[1]

if not host.gpuname() then
    error("no GPU backend; run with luau-gpu on a machine with a Vulkan device")
end

local function copy(b: buffer): buffer
    local c = buffer.create(buffer.len(b))
    buffer.copy(c, 0, b)
    return c
end

local function copyAll(list: { buffer }): { buffer }
    local result = {}
    for i, b in list do
        result[i] = copy(b)
    end
    return result
end

-- deterministic pseudo random numbers in [0, 1), so both modes and every run see the same inputs
local seed = 12345
local function random(): number
    seed = (seed * 1103515245 + 12345) % 2147483648
    return seed / 2147483648
end

type Output = {
    index: number, -- which input buffer holds the result
    kind: "u32" | "f32" | "approx", -- exact words, floats with a relative tolerance, or words that may differ in a few places
    tolerance: number?, -- relative tolerance for "f32", fraction of differing words allowed for "approx"
}

type Scenario = {
    name: string,
    size: string,
    kernels: { (...any) -> () },
    dispatches: number, -- gpu.dispatch calls per run
    inputs: () -> { buffer },
    run: ({ buffer }) -> (),
    outputs: { Output },
}

local scenarios: { Scenario } = {}

-- scenarios dispatch through this, so the 'loop' mode can replace gpu.dispatch with a Luau loop
local loopMode = false

local function dispatch(fn: (...any) -> (), n: number, ...: buffer)
    if loopMode then
        -- forwarding '...' on every call would cost more than some of the kernels themselves
        local a, b, c, d = ...
        for i = 0, n - 1 do
            fn(i, a, b, c, d)
        end
    else
        gpu.dispatch(fn, n, ...)
    end
end

local function scenario(s: Scenario)
    table.insert(scenarios, s)
end

---------------------------------------------------------------------------------------------------
-- saxpy: y = a * x + y, one multiply-add per 8 bytes moved; limited by transfers

@gpu
local function saxpy(i: number, params: buffer, x: buffer, y: buffer)
    local o = i * 4
    buffer.writef32(y, o, buffer.readf32(params, 0) * buffer.readf32(x, o) + buffer.readf32(y, o))
end

local function saxpyScenario(name: string, n: number)
    scenario({
        name = name,
        size = `{n} floats`,
        kernels = { saxpy },
        dispatches = 1,
        inputs = function()
            local params = buffer.create(4)
            buffer.writef32(params, 0, 2.5)
            local x, y = buffer.create(n * 4), buffer.create(n * 4)
            for i = 0, n - 1 do
                buffer.writef32(x, i * 4, random() * 100)
                buffer.writef32(y, i * 4, random() * 100)
            end
            return { params, x, y }
        end,
        run = function(b)
            dispatch(saxpy, n, b[1], b[2], b[3])
        end,
        outputs = { { index = 3, kind = "f32", tolerance = 1e-6 } },
    })
end

saxpyScenario("saxpy", 1_000_000)

-- the same kernel on 64 elements: the fixed cost of a dispatch dominates, the GPU is expected to lose
saxpyScenario("tiny saxpy", 64)

---------------------------------------------------------------------------------------------------
-- Mandelbrot: up to 256 iterations per pixel, almost no memory traffic

local MANDEL_SIZE = 1024
local MANDEL_ITERATIONS = 256

@gpu
local function mandelbrot(i: number, out: buffer)
    local cx = (i % MANDEL_SIZE) / MANDEL_SIZE * 3 - 2.25
    local cy = (i // MANDEL_SIZE) / MANDEL_SIZE * 3 - 1.5
    local x, y = 0, 0
    local k = 0
    while k < MANDEL_ITERATIONS and x * x + y * y < 4 do
        x, y = x * x - y * y + cx, 2 * x * y + cy
        k += 1
    end
    buffer.writeu32(out, i * 4, k)
end

scenario({
    name = "mandelbrot",
    size = `{MANDEL_SIZE}x{MANDEL_SIZE}, {MANDEL_ITERATIONS} iterations`,
    kernels = { mandelbrot },
    dispatches = 1,
    inputs = function()
        return { buffer.create(MANDEL_SIZE * MANDEL_SIZE * 4) }
    end,
    run = function(b)
        dispatch(mandelbrot, MANDEL_SIZE * MANDEL_SIZE, b[1])
    end,
    -- iteration counts near the set's boundary depend on rounding, f32 and f64 escape at different iterations there
    outputs = { { index = 1, kind = "approx", tolerance = 0.01 } },
})

---------------------------------------------------------------------------------------------------
-- n-body: every body reads every other body, O(n^2) work for O(n) data

local BODIES = 4096

@gpu
local function nbody(i: number, pos: buffer, acc: buffer)
    local o = i * 12
    local p = vector.create(buffer.readf32(pos, o), buffer.readf32(pos, o + 4), buffer.readf32(pos, o + 8))
    local a = vector.create(0, 0, 0)
    for j = 0, BODIES - 1 do
        local q = j * 12
        local d = vector.create(buffer.readf32(pos, q), buffer.readf32(pos, q + 4), buffer.readf32(pos, q + 8)) - p
        local r2 = vector.dot(d, d) + 0.01
        a += d / (r2 * math.sqrt(r2))
    end
    buffer.writef32(acc, o, a.x)
    buffer.writef32(acc, o + 4, a.y)
    buffer.writef32(acc, o + 8, a.z)
end

scenario({
    name = "n-body",
    size = `{BODIES} bodies`,
    kernels = { nbody },
    dispatches = 1,
    inputs = function()
        local pos = buffer.create(BODIES * 12)
        for i = 0, BODIES * 3 - 1 do
            buffer.writef32(pos, i * 4, random() * 20 - 10)
        end
        return { pos, buffer.create(BODIES * 12) }
    end,
    run = function(b)
        dispatch(nbody, BODIES, b[1], b[2])
    end,
    -- sums of 4096 terms; vector math is f32 in both modes, the scalar parts are f64 in the VM
    outputs = { { index = 2, kind = "f32", tolerance = 1e-3 } },
})

---------------------------------------------------------------------------------------------------
-- Game of Life: 10 generations on a wrapping grid, double buffered (src -> dst, then swap)

local LIFE_SIZE = 1024
local LIFE_STEPS = 10

@gpu
local function cell(grid: buffer, x: number, y: number)
    return buffer.readu32(grid, (((y + LIFE_SIZE) % LIFE_SIZE) * LIFE_SIZE + (x + LIFE_SIZE) % LIFE_SIZE) * 4)
end

@gpu
local function life(i: number, src: buffer, dst: buffer)
    local x, y = i % LIFE_SIZE, i // LIFE_SIZE
    local n = cell(src, x - 1, y - 1) + cell(src, x, y - 1) + cell(src, x + 1, y - 1) + cell(src, x - 1, y) + cell(src, x + 1, y)
        + cell(src, x - 1, y + 1) + cell(src, x, y + 1) + cell(src, x + 1, y + 1)
    local alive = cell(src, x, y)
    buffer.writeu32(dst, i * 4, if n == 3 or (n == 2 and alive == 1) then 1 else 0)
end

scenario({
    name = "game of life",
    size = `{LIFE_SIZE}x{LIFE_SIZE}, {LIFE_STEPS} steps`,
    kernels = { life },
    dispatches = LIFE_STEPS,
    inputs = function()
        local grid = buffer.create(LIFE_SIZE * LIFE_SIZE * 4)
        for i = 0, LIFE_SIZE * LIFE_SIZE - 1 do
            buffer.writeu32(grid, i * 4, if random() < 0.3 then 1 else 0)
        end
        return { grid, buffer.create(LIFE_SIZE * LIFE_SIZE * 4) }
    end,
    run = function(b)
        for _ = 1, LIFE_STEPS / 2 do
            dispatch(life, LIFE_SIZE * LIFE_SIZE, b[1], b[2])
            dispatch(life, LIFE_SIZE * LIFE_SIZE, b[2], b[1])
        end
    end,
    outputs = { { index = 1, kind = "u32" } },
})

---------------------------------------------------------------------------------------------------
-- histogram: 4M values into 256 bins with atomics, many invocations contend on the same words

local HISTOGRAM_VALUES = 4_000_000

@gpu
local function histogram(i: number, values: buffer, bins: buffer)
    local v = buffer.readu32(values, i * 4)
    gpu.atomicadd(bins, bit32.band(bit32.bxor(v, bit32.rshift(v, 13)), 255) * 4, 1)
end

scenario({
    name = "histogram",
    size = `{HISTOGRAM_VALUES} values, 256 bins`,
    kernels = { histogram },
    dispatches = 1,
    inputs = function()
        local values = buffer.create(HISTOGRAM_VALUES * 4)
        for i = 0, HISTOGRAM_VALUES - 1 do
            buffer.writeu32(values, i * 4, (i * 2654435761) % 4294967296)
        end
        return { values, buffer.create(256 * 4) }
    end,
    run = function(b)
        dispatch(histogram, HISTOGRAM_VALUES, b[1], b[2])
    end,
    outputs = { { index = 2, kind = "u32" } },
})

---------------------------------------------------------------------------------------------------
-- particles: one simulation step with gravity, drag and a bouncing floor, vector math

local PARTICLES = 1_000_000

@gpu
local function particles(i: number, params: buffer, state: buffer)
    local o = i * 24
    local dt = buffer.readf32(params, 0)
    local p = vector.create(buffer.readf32(state, o), buffer.readf32(state, o + 4), buffer.readf32(state, o + 8))
    local v = vector.create(buffer.readf32(state, o + 12), buffer.readf32(state, o + 16), buffer.readf32(state, o + 20))
    v = (v + vector.create(0, -9.81, 0) * dt) * 0.999
    p += v * dt
    if p.y < 0 then
        p = vector.create(p.x, -p.y, p.z)
        v = vector.create(v.x, -v.y * 0.8, v.z)
    end
    buffer.writef32(state, o, p.x)
    buffer.writef32(state, o + 4, p.y)
    buffer.writef32(state, o + 8, p.z)
    buffer.writef32(state, o + 12, v.x)
    buffer.writef32(state, o + 16, v.y)
    buffer.writef32(state, o + 20, v.z)
end

scenario({
    name = "particles",
    size = `{PARTICLES} particles`,
    kernels = { particles },
    dispatches = 1,
    inputs = function()
        local params = buffer.create(4)
        buffer.writef32(params, 0, 1 / 60)
        local state = buffer.create(PARTICLES * 24)
        for i = 0, PARTICLES * 6 - 1 do
            buffer.writef32(state, i * 4, random() * 10 - 2)
        end
        return { params, state }
    end,
    run = function(b)
        dispatch(particles, PARTICLES, b[1], b[2])
    end,
    outputs = { { index = 2, kind = "f32", tolerance = 1e-5 } },
})

---------------------------------------------------------------------------------------------------
-- blur: a 5x5 box blur over a 1024x1024 float image, the shape of a post-processing pass

local IMAGE_SIZE = 1024

@gpu
local function blur(i: number, src: buffer, dst: buffer)
    local x, y = i % IMAGE_SIZE, i // IMAGE_SIZE
    local sum, count = 0, 0
    for dy = -2, 2 do
        local yy = y + dy
        if yy >= 0 and yy < IMAGE_SIZE then
            for dx = -2, 2 do
                local xx = x + dx
                if xx >= 0 and xx < IMAGE_SIZE then
                    sum += buffer.readf32(src, (yy * IMAGE_SIZE + xx) * 4)
                    count += 1
                end
            end
        end
    end
    buffer.writef32(dst, i * 4, sum / count)
end

scenario({
    name = "blur 5x5",
    size = `{IMAGE_SIZE}x{IMAGE_SIZE}`,
    kernels = { blur },
    dispatches = 1,
    inputs = function()
        local src = buffer.create(IMAGE_SIZE * IMAGE_SIZE * 4)
        for i = 0, IMAGE_SIZE * IMAGE_SIZE - 1 do
            buffer.writef32(src, i * 4, random())
        end
        return { src, buffer.create(IMAGE_SIZE * IMAGE_SIZE * 4) }
    end,
    run = function(b)
        dispatch(blur, IMAGE_SIZE * IMAGE_SIZE, b[1], b[2])
    end,
    outputs = { { index = 2, kind = "f32", tolerance = 1e-5 } },
})

---------------------------------------------------------------------------------------------------
-- runner

-- returns an error message if the outputs differ more than allowed
local function compare(s: Scenario, onGpu: { buffer }, native: { buffer }): string?
    for _, out in s.outputs do
        local a, b = onGpu[out.index], native[out.index]
        local words = buffer.len(a) // 4
        local differing = 0

        for w = 0, words - 1 do
            local o = w * 4
            local x, y = buffer.readu32(a, o), buffer.readu32(b, o)
            if x ~= y then
                if out.kind == "u32" then
                    return string.format("buffer %d word %d: gpu %08x, native %08x", out.index, w, x, y)
                elseif out.kind == "f32" then
                    local fx, fy = buffer.readf32(a, o), buffer.readf32(b, o)
                    if not (math.abs(fx - fy) <= (out.tolerance :: number) * math.max(1, math.abs(fy))) then
                        return string.format("buffer %d word %d: gpu %.9g, native %.9g", out.index, w, fx, fy)
                    end
                else
                    differing += 1
                end
            end
        end

        if out.kind == "approx" and differing > (out.tolerance :: number) * words then
            return string.format("buffer %d: %d of %d words differ", out.index, differing, words)
        end
    end

    return nil
end

-- runs the scenario on fresh copies until it has 'runs' timings or exceeds the time budget; returns the sorted timings
local function measure(s: Scenario, inputs: { buffer }, runs: number, budget: number): ({ number }, { buffer })
    local times = {}
    local firstResult = nil
    local started = host.clock()

    while #times < runs and (#times == 0 or host.clock() - started < budget) do
        local b = copyAll(inputs)
        local t = host.clock()
        s.run(b)
        table.insert(times, host.clock() - t)
        firstResult = firstResult or b
    end

    table.sort(times)
    return times, firstResult :: { buffer }
end

local results = {}
local failures = 0

print(string.format("GPU: %s, native code generation for the VM side", host.gpuname() :: string))

for _, s in scenarios do
    if filter and not string.find(s.name, filter, 1, true) then
        continue
    end

    local reason = nil
    for _, k in s.kernels do
        local where, why = gpu.status(k)
        if where ~= "gpu" then
            reason = why
        end
    end

    local inputs = s.inputs()

    host.setgpu(false)
    loopMode = true
    local loopTimes, native = measure(s, inputs, 3, 2)
    loopMode = false
    local vmTimes, inVm = measure(s, inputs, 3, 2)

    host.setgpu(true)
    -- the first GPU run also uploads every input and creates the device buffers; a warm-up keeps it out of the timings
    s.run(copyAll(inputs))
    local before = host.stats()
    local gpuTimes, onGpu = measure(s, inputs, 10, 2)
    local after = host.stats()

    local expected = s.dispatches * #gpuTimes
    local ranOnGpu = after.gpuDispatches - before.gpuDispatches
    local mismatch = compare(s, onGpu, native) or compare(s, inVm, native)

    local status = "ok"
    if reason then
        status = "not translated: " .. tostring(reason)
    elseif ranOnGpu ~= expected then
        status = string.format("only %d of %d dispatches ran on the GPU", ranOnGpu, expected)
    elseif mismatch then
        status = "MISMATCH " .. mismatch
    end
    if status ~= "ok" then
        failures += 1
    end

    local nativeMs, gpuMs = loopTimes[1] * 1000, gpuTimes[1] * 1000
    table.insert(results, {
        name = s.name,
        size = s.size,
        native = nativeMs,
        vm = vmTimes[1] * 1000,
        gpu = gpuMs,
        gpuMedian = gpuTimes[(#gpuTimes + 1) // 2] * 1000,
        status = status,
    })
    print(string.format("%-14s loop %10.3f ms   vm %10.3f ms   gpu %8.3f ms   %s", s.name, nativeMs, vmTimes[1] * 1000, gpuMs, status))
end

print()
print(string.format("| %-14s | %-28s | %14s | %12s | %10s | %13s | %10s | %s", "scenario", "size", "native loop ms", "vm ms", "gpu ms",
    "gpu median ms", "speedup", "match"))
print(string.format("|%s|%s|%s|%s|%s|%s|%s|%s", string.rep("-", 16), string.rep("-", 30), string.rep("-", 16), string.rep("-", 14),
    string.rep("-", 12), string.rep("-", 15), string.rep("-", 12), string.rep("-", 7)))
for _, r in results do
    local speedup = r.native / r.gpu
    local speedupText = if speedup >= 1 then string.format("%.1fx", speedup) else string.format("%.3fx", speedup)
    print(string.format("| %-14s | %-28s | %14.3f | %12.3f | %10.3f | %13.3f | %10s | %s", r.name, r.size, r.native, r.vm, r.gpu, r.gpuMedian,
        speedupText, if r.status == "ok" then "yes" else r.status))
end

local st = host.stats()
print()
print(string.format("dispatches on the GPU %d / %d, fallbacks %d", st.gpuDispatches, st.dispatches, st.fallbacks))

if failures > 0 then
    error(`{failures} scenarios failed`)
end
