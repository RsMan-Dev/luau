-- Soak test: every frame creates fresh buffers and a fresh kernel closure, like a script that rebuilds its frame data
-- instead of reusing it. Device memory must stay bounded by the buffers that are alive, and the cost of a frame must not
-- grow over time.
--
-- usage: luau-gpu Gpu/examples/soak.luau [seconds]

local seconds = tonumber((arg :: { string })[1]) or 60

local frames = {}
local maxCached, maxKernels = 0, 0
local start = host.clock()
local frame = 0

while host.clock() - start < seconds do
    frame += 1

    -- a different size every frame, so no two consecutive frames could share an allocation
    local packet = buffer.create(256 * 1024 + (frame % 13) * 4096)
    local image = buffer.create(512 * 1024)
    for o = 0, 1020, 4 do
        buffer.writeu32(packet, o, frame + o)
    end

    @gpu
    local function shade(i: number, src: buffer, dst: buffer)
        local v = buffer.readu32(src, (i % 256) * 4)
        buffer.writeu32(dst, i * 4, bit32.bxor(v, i))
    end

    local t = host.clock()
    gpu.dispatch({ { shade, 512 * 256, packet, image } }, { image })
    frames[frame] = host.clock() - t
    assert(buffer.readu32(image, 4) == bit32.bxor(frame + 4, 1))

    if frame % 100 == 0 then
        host.collect()
    end

    local stats = host.stats()
    maxCached = math.max(maxCached, stats.cachedBytes)
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
    "%d frames in %.0fs: first 10%% %.3f ms, last 10%% %.3f ms per dispatch; max cached %.1f MB, max kernels %d; gpu %d, fallbacks %d",
    frame, seconds, mean(1, tenth), mean(frame - tenth + 1, frame), maxCached / 1e6, maxKernels, stats.gpuDispatches, stats.fallbacks
))
