-- Soak test: every frame creates fresh buffers and a fresh kernel closure, like a script that rebuilds its frame data
-- instead of reusing it. Device memory must stay bounded by the buffers that are alive, and the cost of a frame must not
-- grow over time.
--
-- usage: luau-gpu Gpu/examples/soak.luau [seconds]

local seconds = tonumber((arg :: { string })[1]) or 60

local frames = {}
local maxCached, maxMemory, maxKernels = 0, 0, 0
local start = host.clock()
local frame = 0

-- the first frame waits for the driver to create the kernel's pipeline, which gpu.run returns false during
local function runReady(fn, buffers: { buffer }, runs: number)
    local ok, err = gpu.run(fn, buffers, runs)
    while not ok and err == "the kernel is being compiled for the GPU" do
        ok, err = gpu.run(fn, buffers, runs)
    end
    return ok, err
end

while host.clock() - start < seconds do
    frame += 1

    -- a packet of 256 words, then the image; a different size every frame, so no two consecutive frames could share an
    -- allocation
    local data = buffer.create(1024 + 512 * 1024 + (frame % 13) * 4096)
    for o = 0, 1020, 4 do
        buffer.writeu32(data, o, frame + o)
    end

    @gpu
    local function shade(b: buffer, i: number)
        local v = buffer.readu32(b, (i % 256) * 4)
        buffer.writeu32(b, 1024 + i * 4, bit32.bxor(v, i))
    end

    local t = host.clock()
    assert(runReady(shade, { data }, 128 * 1024))
    frames[frame] = host.clock() - t
    assert(buffer.readu32(data, 1024 + 4) == bit32.bxor(frame + 4, 1))

    if frame % 100 == 0 then
        host.collect()
    end

    local stats = host.stats()
    maxCached = math.max(maxCached, stats.cachedBytes)
    maxMemory = math.max(maxMemory, stats.memoryBytes)
    maxKernels = math.max(maxKernels, stats.kernels)
end

local function mean(a: number, b: number)
    local s = 0
    for k = a, b do
        s += frames[k]
    end
    return s / (b - a + 1) * 1000
end

local tenth = frame // 10
local stats = host.stats()
print(string.format(
    "%d frames in %.0fs: first 10%% %.3f ms, last 10%% %.3f ms per run; max cached %.1f MB, max memory %.1f MB, max kernels %d; succeeded %d, failures %d",
    frame,
    seconds,
    mean(1, tenth),
    mean(frame - tenth + 1, frame),
    maxCached / 1e6,
    maxMemory / 1e6,
    maxKernels,
    stats.succeeded,
    stats.failures
))
