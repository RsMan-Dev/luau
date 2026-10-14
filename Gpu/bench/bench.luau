--!native
-- Compares gpu.run on the GPU against the same kernels called from a native Luau loop.
--
-- usage: luau-gpu Gpu/bench/bench.luau [filter]
--   filter: only run scenarios whose name contains this text
--
-- Each scenario runs on copies of the same buffer in two modes:
--   loop      the kernel called from a plain Luau loop, compiled to native code: what a script would do without gpu.run
--   gpu       gpu.run on the GPU
-- A kernel only gets its buffer, so each scenario keeps its inputs and results in one. GPU timings include every transfer
-- a run makes (uploads of the parts that changed, downloads of the parts written), since that is what a script pays. Results
-- are compared bit for bit when the computation is exact, and with a tolerance when the GPU's f32 math can differ from
-- Luau's f64.

local filter = (arg :: { string })[1]

if not host.gpuname() then
    error("no GPU backend; run with luau-gpu on a machine with a Vulkan device")
end

local function copy(b: buffer): buffer
    local c = buffer.create(buffer.len(b))
    buffer.copy(c, 0, b)
    return c
end

-- deterministic pseudo random numbers in [0, 1), so both modes and every run see the same inputs
local seed = 12345
local function random(): number
    seed = (seed * 1103515245 + 12345) % 2147483648
    return seed / 2147483648
end

type Output = {
    offset: number, -- where the results are in the buffer
    size: number, -- in bytes
    kind: "u32" | "f32" | "approx", -- exact words, floats with a relative tolerance, or words that may differ in a few places
    tolerance: number?, -- relative tolerance for "f32", fraction of differing words allowed for "approx"
}

type Scenario = {
    name: string,
    size: string,
    runs: number, -- gpu.run calls per run of the scenario
    input: () -> buffer,
    run: (buffer) -> (),
    output: Output,
}

local scenarios: { Scenario } = {}

-- scenarios run kernels through this, so the 'loop' mode can replace gpu.run with a Luau loop
local loopMode = false

local function run(fn: (buffer, number) -> (), n: number, b: buffer)
    if loopMode then
        for i = 0, n - 1 do
            fn(b, i)
        end
    else
        local ok, err = gpu.run(fn, { b }, n)
        if not ok then
            error(err)
        end
    end
end

local function scenario(s: Scenario)
    table.insert(scenarios, s)
end

---------------------------------------------------------------------------------------------------
-- saxpy: y = a * x + y, one multiply-add per 8 bytes moved; limited by transfers
-- buffer: a, the offset of y, x, y

@gpu
local function saxpy(b: buffer, i: number)
    local o = i * 4
    local y = buffer.readu32(b, 4) + o
    buffer.writef32(b, y, buffer.readf32(b, 0) * buffer.readf32(b, 16 + o) + buffer.readf32(b, y))
end

local function saxpyScenario(name: string, n: number)
    scenario({
        name = name,
        size = `{n} floats`,
        runs = 1,
        input = function()
            local b = buffer.create(16 + n * 8)
            buffer.writef32(b, 0, 2.5)
            buffer.writeu32(b, 4, 16 + n * 4)
            for i = 0, n * 2 - 1 do
                buffer.writef32(b, 16 + i * 4, random() * 100)
            end
            return b
        end,
        run = function(b)
            run(saxpy, n, b)
        end,
        output = { offset = 16 + n * 4, size = n * 4, kind = "f32", tolerance = 1e-6 },
    })
end

saxpyScenario("saxpy", 1_000_000)

-- the same kernel on 64 elements: the fixed cost of a run dominates, the GPU is expected to lose
saxpyScenario("tiny saxpy", 64)

---------------------------------------------------------------------------------------------------
-- Mandelbrot: up to 256 iterations per pixel, almost no memory traffic

local MANDEL_SIZE = 1024
local MANDEL_ITERATIONS = 256

@gpu
local function mandelbrot(b: buffer, i: number)
    local cx = (i % MANDEL_SIZE) / MANDEL_SIZE * 3 - 2.25
    local cy = (i // MANDEL_SIZE) / MANDEL_SIZE * 3 - 1.5
    local x, y = 0, 0
    local k = 0
    while k < MANDEL_ITERATIONS and x * x + y * y < 4 do
        x, y = x * x - y * y + cx, 2 * x * y + cy
        k += 1
    end
    buffer.writeu32(b, i * 4, k)
end

scenario({
    name = "mandelbrot",
    size = `{MANDEL_SIZE}x{MANDEL_SIZE}, {MANDEL_ITERATIONS} iterations`,
    runs = 1,
    input = function()
        return buffer.create(MANDEL_SIZE * MANDEL_SIZE * 4)
    end,
    run = function(b)
        run(mandelbrot, MANDEL_SIZE * MANDEL_SIZE, b)
    end,
    -- iteration counts near the set's boundary depend on rounding, f32 and f64 escape at different iterations there
    output = { offset = 0, size = MANDEL_SIZE * MANDEL_SIZE * 4, kind = "approx", tolerance = 0.01 },
})

---------------------------------------------------------------------------------------------------
-- n-body: every body reads every other body, O(n^2) work for O(n) data
-- buffer: positions, accelerations

local BODIES = 4096

@gpu
local function nbody(b: buffer, i: number)
    local o = i * 12
    local p = vector.create(buffer.readf32(b, o), buffer.readf32(b, o + 4), buffer.readf32(b, o + 8))
    local a = vector.create(0, 0, 0)
    for j = 0, BODIES - 1 do
        local q = j * 12
        local d = vector.create(buffer.readf32(b, q), buffer.readf32(b, q + 4), buffer.readf32(b, q + 8)) - p
        local r2 = vector.dot(d, d) + 0.01
        a += d / (r2 * math.sqrt(r2))
    end
    local out = BODIES * 12 + o
    buffer.writef32(b, out, a.x)
    buffer.writef32(b, out + 4, a.y)
    buffer.writef32(b, out + 8, a.z)
end

scenario({
    name = "n-body",
    size = `{BODIES} bodies`,
    runs = 1,
    input = function()
        local b = buffer.create(BODIES * 24)
        for i = 0, BODIES * 3 - 1 do
            buffer.writef32(b, i * 4, random() * 20 - 10)
        end
        return b
    end,
    run = function(b)
        run(nbody, BODIES, b)
    end,
    -- sums of 4096 terms; vector math is f32 in both modes, the scalar parts are f64 in Luau
    output = { offset = BODIES * 12, size = BODIES * 12, kind = "f32", tolerance = 1e-3 },
})

---------------------------------------------------------------------------------------------------
-- Game of Life: 10 generations on a wrapping grid, double buffered
-- buffer: which grid is the current one, the two grids; Luau switches them between runs

local LIFE_SIZE = 1024
local LIFE_STEPS = 10
local LIFE_GRID = LIFE_SIZE * LIFE_SIZE * 4

@gpu
local function cell(b: buffer, grid: number, x: number, y: number)
    return buffer.readu32(b, grid + (((y + LIFE_SIZE) % LIFE_SIZE) * LIFE_SIZE + (x + LIFE_SIZE) % LIFE_SIZE) * 4)
end

@gpu
local function life(b: buffer, i: number)
    local src = 4 + buffer.readu32(b, 0) * LIFE_GRID
    local dst = 4 + (1 - buffer.readu32(b, 0)) * LIFE_GRID
    local x, y = i % LIFE_SIZE, i // LIFE_SIZE
    local n = cell(b, src, x - 1, y - 1) + cell(b, src, x, y - 1) + cell(b, src, x + 1, y - 1) + cell(b, src, x - 1, y)
        + cell(b, src, x + 1, y) + cell(b, src, x - 1, y + 1) + cell(b, src, x, y + 1) + cell(b, src, x + 1, y + 1)
    local alive = cell(b, src, x, y)
    buffer.writeu32(b, dst + i * 4, if n == 3 or (n == 2 and alive == 1) then 1 else 0)
end

scenario({
    name = "game of life",
    size = `{LIFE_SIZE}x{LIFE_SIZE}, {LIFE_STEPS} steps`,
    runs = LIFE_STEPS,
    input = function()
        local b = buffer.create(4 + LIFE_GRID * 2)
        for i = 0, LIFE_SIZE * LIFE_SIZE - 1 do
            buffer.writeu32(b, 4 + i * 4, if random() < 0.3 then 1 else 0)
        end
        return b
    end,
    run = function(b)
        for step = 0, LIFE_STEPS - 1 do
            buffer.writeu32(b, 0, step % 2)
            run(life, LIFE_SIZE * LIFE_SIZE, b)
        end
    end,
    output = { offset = 4, size = LIFE_GRID, kind = "u32" },
})

---------------------------------------------------------------------------------------------------
-- histogram: 4M values into 256 bins with atomics, many invocations contend on the same words
-- buffer: the bins, the values

local HISTOGRAM_VALUES = 4_000_000

@gpu
local function histogram(b: buffer, i: number)
    local v = buffer.readu32(b, 1024 + i * 4)
    gpu.atomicadd(b, bit32.band(bit32.bxor(v, bit32.rshift(v, 13)), 255) * 4, 1)
end

scenario({
    name = "histogram",
    size = `{HISTOGRAM_VALUES} values, 256 bins`,
    runs = 1,
    input = function()
        local b = buffer.create(1024 + HISTOGRAM_VALUES * 4)
        for i = 0, HISTOGRAM_VALUES - 1 do
            buffer.writeu32(b, 1024 + i * 4, (i * 2654435761) % 4294967296)
        end
        return b
    end,
    run = function(b)
        run(histogram, HISTOGRAM_VALUES, b)
    end,
    output = { offset = 0, size = 1024, kind = "u32" },
})

---------------------------------------------------------------------------------------------------
-- particles: one simulation step with gravity, drag and a bouncing floor, vector math
-- buffer: the time step, the particles

local PARTICLES = 1_000_000

@gpu
local function particles(b: buffer, i: number)
    local o = 16 + i * 24
    local dt = buffer.readf32(b, 0)
    local p = vector.create(buffer.readf32(b, o), buffer.readf32(b, o + 4), buffer.readf32(b, o + 8))
    local v = vector.create(buffer.readf32(b, o + 12), buffer.readf32(b, o + 16), buffer.readf32(b, o + 20))
    v = (v + vector.create(0, -9.81, 0) * dt) * 0.999
    p += v * dt
    if p.y < 0 then
        p = vector.create(p.x, -p.y, p.z)
        v = vector.create(v.x, -v.y * 0.8, v.z)
    end
    buffer.writef32(b, o, p.x)
    buffer.writef32(b, o + 4, p.y)
    buffer.writef32(b, o + 8, p.z)
    buffer.writef32(b, o + 12, v.x)
    buffer.writef32(b, o + 16, v.y)
    buffer.writef32(b, o + 20, v.z)
end

scenario({
    name = "particles",
    size = `{PARTICLES} particles`,
    runs = 1,
    input = function()
        local b = buffer.create(16 + PARTICLES * 24)
        buffer.writef32(b, 0, 1 / 60)
        for i = 0, PARTICLES * 6 - 1 do
            buffer.writef32(b, 16 + i * 4, random() * 10 - 2)
        end
        return b
    end,
    run = function(b)
        run(particles, PARTICLES, b)
    end,
    output = { offset = 16, size = PARTICLES * 24, kind = "f32", tolerance = 1e-5 },
})

---------------------------------------------------------------------------------------------------
-- blur: a 5x5 box blur over a 1024x1024 float image, the shape of a post-processing pass
-- buffer: the image, the blurred image

local IMAGE_SIZE = 1024
local IMAGE_BYTES = IMAGE_SIZE * IMAGE_SIZE * 4

@gpu
local function blur(b: buffer, i: number)
    local x, y = i % IMAGE_SIZE, i // IMAGE_SIZE
    local sum, count = 0, 0
    for dy = -2, 2 do
        local yy = y + dy
        if yy >= 0 and yy < IMAGE_SIZE then
            for dx = -2, 2 do
                local xx = x + dx
                if xx >= 0 and xx < IMAGE_SIZE then
                    sum += buffer.readf32(b, (yy * IMAGE_SIZE + xx) * 4)
                    count += 1
                end
            end
        end
    end
    buffer.writef32(b, IMAGE_BYTES + i * 4, sum / count)
end

scenario({
    name = "blur 5x5",
    size = `{IMAGE_SIZE}x{IMAGE_SIZE}`,
    runs = 1,
    input = function()
        local b = buffer.create(IMAGE_BYTES * 2)
        for i = 0, IMAGE_SIZE * IMAGE_SIZE - 1 do
            buffer.writef32(b, i * 4, random())
        end
        return b
    end,
    run = function(b)
        run(blur, IMAGE_SIZE * IMAGE_SIZE, b)
    end,
    output = { offset = IMAGE_BYTES, size = IMAGE_BYTES, kind = "f32", tolerance = 1e-5 },
})

---------------------------------------------------------------------------------------------------
-- runner

-- returns an error message if the results differ more than allowed
local function compare(s: Scenario, onGpu: buffer, native: buffer): string?
    local out = s.output
    local words = out.size // 4
    local differing = 0

    for w = 0, words - 1 do
        local o = out.offset + w * 4
        local x, y = buffer.readu32(onGpu, o), buffer.readu32(native, o)
        if x ~= y then
            if out.kind == "u32" then
                return string.format("word %d: gpu %08x, native %08x", w, x, y)
            elseif out.kind == "f32" then
                local fx, fy = buffer.readf32(onGpu, o), buffer.readf32(native, o)
                if not (math.abs(fx - fy) <= (out.tolerance :: number) * math.max(1, math.abs(fy))) then
                    return string.format("word %d: gpu %.9g, native %.9g", w, fx, fy)
                end
            else
                differing += 1
            end
        end
    end

    if out.kind == "approx" and differing > (out.tolerance :: number) * words then
        return string.format("%d of %d words differ", differing, words)
    end

    return nil
end

-- runs the scenario on fresh copies until it has 'runs' timings or exceeds the time budget; returns the sorted timings
local function measure(s: Scenario, input: buffer, runs: number, budget: number): ({ number }, buffer)
    local times = {}
    local firstResult = nil
    local started = host.clock()

    while #times < runs and (#times == 0 or host.clock() - started < budget) do
        local b = copy(input)
        local t = host.clock()
        s.run(b)
        table.insert(times, host.clock() - t)
        firstResult = firstResult or b
    end

    table.sort(times)
    return times, firstResult :: buffer
end

local results = {}
local failures = 0

print(string.format("GPU: %s, native code generation for the loops", host.gpuname() :: string))

for _, s in scenarios do
    if filter and not string.find(s.name, filter, 1, true) then
        continue
    end

    local input = s.input()

    loopMode = true
    local loopTimes, native = measure(s, input, 3, 2)
    loopMode = false

    -- the first GPU run also lowers the kernel and creates the device buffer; a warm-up keeps it out of the timings
    local status = "ok"
    local ok, err = pcall(s.run, copy(input))
    local gpuTimes, onGpu = { 0 }, input
    if ok then
        ok, err = pcall(function()
            gpuTimes, onGpu = measure(s, input, 10, 2)
        end)
    end

    if not ok then
        status = "FAILED " .. tostring(err)
    else
        local mismatch = compare(s, onGpu, native)
        if mismatch then
            status = "MISMATCH " .. mismatch
        end
    end
    if status ~= "ok" then
        failures += 1
    end

    local nativeMs, gpuMs = loopTimes[1] * 1000, gpuTimes[1] * 1000
    table.insert(results, {
        name = s.name,
        size = s.size,
        native = nativeMs,
        gpu = gpuMs,
        gpuMedian = gpuTimes[(#gpuTimes + 1) // 2] * 1000,
        status = status,
    })
    print(string.format("%-14s loop %10.3f ms   gpu %8.3f ms   %s", s.name, nativeMs, gpuMs, status))
end

print()
print(string.format("| %-14s | %-28s | %14s | %10s | %13s | %10s | %s", "scenario", "size", "native loop ms", "gpu ms", "gpu median ms",
    "speedup", "match"))
print(string.format("|%s|%s|%s|%s|%s|%s|%s", string.rep("-", 16), string.rep("-", 30), string.rep("-", 16), string.rep("-", 12),
    string.rep("-", 15), string.rep("-", 12), string.rep("-", 7)))
for _, r in results do
    local speedup = r.native / r.gpu
    local speedupText = if speedup >= 1 then string.format("%.1fx", speedup) else string.format("%.3fx", speedup)
    print(string.format("| %-14s | %-28s | %14.3f | %10.3f | %13.3f | %10s | %s", r.name, r.size, r.native, r.gpu, r.gpuMedian, speedupText,
        if r.status == "ok" then "yes" else r.status))
end

local st = host.stats()
print()
print(string.format("gpu.run on the GPU %d / %d, failures %d", st.succeeded, st.runs, st.failures))

if failures > 0 then
    error(`{failures} scenarios failed`)
end
