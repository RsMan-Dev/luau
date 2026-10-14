-- Software rasterizer written in Luau, running on the GPU through gpu.run.
--
-- usage: luau-gpu Gpu/examples/rasterizer.luau [segments] [--frames N --save path.raw]
--   window controls: drag to rotate, wheel to zoom, space to pause, G to switch between the GPU and Luau
--
-- Everything is in one buffer, the scene, which every kernel gets: a header with the view and where the other parts start,
-- the vertices, the triangles, the projected vertices, and the depth, ids and image of the pixels. Every frame runs four
-- kernels on it:
--   1. transform  one invocation per vertex: rotate, project to the screen
--   2. depth      one invocation per triangle: rasterize, keep the nearest depth per pixel (gpu.atomicmin)
--   3. resolve    one invocation per triangle: of the triangles at the nearest depth, the first claims the pixel
--                 (gpu.atomicmin)
--   4. shade      one invocation per pixel: light the owning triangle, clear depth and ids for the next frame

local W = 1280
local H = 720
local FOCAL = 900
local NEAR = 0.5
-- depth keys are integers, view z * scale, which floats hold exactly below 2^24: views end before z = 32
local DEPTH_SCALE = 524288
local CLEAR_DEPTH = 0x7f7f7f7f
-- numbers are f32 on the GPU: words above 2^24 would round, so the ids and colors stay below
local NO_ID = 0xffffff

-- the header: the rotation matrix and the distance, then the offsets of the other parts
local VERTS, TRIS, PROJ, DEPTH, IDS, IMAGE = 40, 44, 48, 52, 56, 60
local HEADER = 64

local args = arg :: { string }
local segments = tonumber(args[1]) or 2000
local framesToRun, savePath = nil, nil
for i, a in args do
    if a == "--frames" then
        framesToRun = tonumber(args[i + 1])
    elseif a == "--save" then
        savePath = args[i + 1]
    end
end

---------------------------------------------------------------------------------------------------
-- an unnecessarily heavy model: a (3, 7) torus knot tube covered in bumps

local function knot(t: number): vector
    local r = 2 + 0.8 * math.cos(7 * t)
    return vector.create(r * math.cos(3 * t), r * math.sin(3 * t), 0.8 * math.sin(7 * t)) * 0.9
end

local U = segments -- along the knot
local V = math.max(8, segments // 8) -- around the tube
local vertexCount = U * V
local triangleCount = U * V * 2

local t0 = os.clock()

local offsets = {}
local size = HEADER
for i, part in { vertexCount * 12, triangleCount * 12, vertexCount * 12, W * H * 4, W * H * 4, W * H * 4 } do
    offsets[i] = size
    size += part
end

-- word offsets are exact below 2^26, where f32 numbers are 4 apart
assert(size <= 2 ^ 26, "the scene is over 64 MB, where offsets would round: use fewer segments")
local scene = buffer.create(size)
for i, offset in offsets do
    buffer.writeu32(scene, VERTS + (i - 1) * 4, offset)
end

local verts = offsets[1]
for u = 0, U - 1 do
    local t = u / U * math.pi * 2
    local c = knot(t)
    local ahead = knot(t + 0.001)
    local behind = knot(t - 0.001)
    local tangent = vector.normalize(ahead - behind)
    local normal = vector.normalize(ahead + behind - c * 2)
    local binormal = vector.cross(tangent, normal)

    for v = 0, V - 1 do
        local a = v / V * math.pi * 2
        local bumps = 0.04 * math.sin(a * 12) * math.sin(t * 150) + 0.015 * math.sin(a * 40 + t * 400)
        local radius = 0.32 + bumps
        local p = c + (normal * math.cos(a) + binormal * math.sin(a)) * radius
        local o = verts + (u * V + v) * 12
        buffer.writef32(scene, o, p.x)
        buffer.writef32(scene, o + 4, p.y)
        buffer.writef32(scene, o + 8, p.z)
    end
end

local tris = offsets[2]
for u = 0, U - 1 do
    local u1 = (u + 1) % U
    for v = 0, V - 1 do
        local v1 = (v + 1) % V
        local a, b, c, d = u * V + v, u1 * V + v, u1 * V + v1, u * V + v1
        local o = tris + (u * V + v) * 24
        buffer.writeu32(scene, o, a)
        buffer.writeu32(scene, o + 4, b)
        buffer.writeu32(scene, o + 8, c)
        buffer.writeu32(scene, o + 12, a)
        buffer.writeu32(scene, o + 16, c)
        buffer.writeu32(scene, o + 20, d)
    end
end

print(string.format("model: %d vertices, %d triangles, built in %.2fs", vertexCount, triangleCount, os.clock() - t0))

---------------------------------------------------------------------------------------------------
-- kernels

@gpu
local function edge(ax: number, ay: number, bx: number, by: number, px: number, py: number)
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)
end

@gpu
local function rotate(s: buffer, x: number, y: number, z: number): vector
    return vector.create(
        buffer.readf32(s, 0) * x + buffer.readf32(s, 4) * y + buffer.readf32(s, 8) * z,
        buffer.readf32(s, 12) * x + buffer.readf32(s, 16) * y + buffer.readf32(s, 20) * z,
        buffer.readf32(s, 24) * x + buffer.readf32(s, 28) * y + buffer.readf32(s, 32) * z
    )
end

@gpu
local function transform(s: buffer, i: number)
    local o = i * 12
    local v, out = buffer.readu32(s, VERTS) + o, buffer.readu32(s, PROJ) + o
    local p = rotate(s, buffer.readf32(s, v), buffer.readf32(s, v + 4), buffer.readf32(s, v + 8))
    local z = p.z + buffer.readf32(s, 36)
    buffer.writef32(s, out, W / 2 + FOCAL * p.x / z)
    buffer.writef32(s, out + 4, H / 2 - FOCAL * p.y / z)
    buffer.writef32(s, out + 8, z)
end

-- shared by the depth and resolve passes so both compute exactly the same coverage and depth keys
@gpu
local function rasterize(s: buffer, t: number, resolve: boolean)
    local o = buffer.readu32(s, TRIS) + t * 12
    local proj = buffer.readu32(s, PROJ)
    local i0 = proj + buffer.readu32(s, o) * 12
    local i1 = proj + buffer.readu32(s, o + 4) * 12
    local i2 = proj + buffer.readu32(s, o + 8) * 12

    local z0, z1, z2 = buffer.readf32(s, i0 + 8), buffer.readf32(s, i1 + 8), buffer.readf32(s, i2 + 8)
    if z0 < NEAR or z1 < NEAR or z2 < NEAR then
        return
    end

    local x0, y0 = buffer.readf32(s, i0), buffer.readf32(s, i0 + 4)
    local x1, y1 = buffer.readf32(s, i1), buffer.readf32(s, i1 + 4)
    local x2, y2 = buffer.readf32(s, i2), buffer.readf32(s, i2 + 4)
    local depth, ids = buffer.readu32(s, DEPTH), buffer.readu32(s, IDS)

    local area = edge(x0, y0, x1, y1, x2, y2)
    if area <= 0 then
        return -- back face
    end

    local minx = math.max(0, math.floor(math.min(x0, x1, x2)))
    local maxx = math.min(W - 1, math.floor(math.max(x0, x1, x2)))
    local miny = math.max(0, math.floor(math.min(y0, y1, y2)))
    local maxy = math.min(H - 1, math.floor(math.max(y0, y1, y2)))

    for y = miny, maxy do
        local py = y + 0.5
        for x = minx, maxx do
            local px = x + 0.5
            local w0 = edge(x1, y1, x2, y2, px, py)
            local w1 = edge(x2, y2, x0, y0, px, py)
            local w2 = edge(x0, y0, x1, y1, px, py)
            if w0 >= 0 and w1 >= 0 and w2 >= 0 then
                local key = math.floor((w0 * z0 + w1 * z1 + w2 * z2) / area * DEPTH_SCALE)
                local p = (y * W + x) * 4
                if resolve then
                    if buffer.readu32(s, depth + p) == key then
                        gpu.atomicmin(s, ids + p, t)
                    end
                else
                    gpu.atomicmin(s, depth + p, key)
                end
            end
        end
    end
end

@gpu
local function depthPass(s: buffer, t: number)
    rasterize(s, t, false)
end

@gpu
local function resolvePass(s: buffer, t: number)
    rasterize(s, t, true)
end

@gpu
local function channel(v: number)
    return math.clamp(math.floor(v * 255), 0, 255)
end

-- without the alpha byte, which the window ignores, and which would put the word above 2^24
@gpu
local function rgb(r: number, g: number, b: number)
    return bit32.bor(bit32.lshift(channel(r), 16), bit32.lshift(channel(g), 8), channel(b))
end

@gpu
local function shade(s: buffer, p: number)
    local o = p * 4
    local ids, image = buffer.readu32(s, IDS) + o, buffer.readu32(s, IMAGE) + o
    local id = buffer.readu32(s, ids)
    buffer.writeu32(s, ids, NO_ID)
    buffer.writeu32(s, buffer.readu32(s, DEPTH) + o, CLEAR_DEPTH)

    if id == NO_ID then
        local y = p // W
        local g = 0.08 + 0.12 * (1 - y / H)
        buffer.writeu32(s, image, rgb(g * 0.6, g * 0.7, g))
        return
    end

    local verts = buffer.readu32(s, VERTS)
    local t = buffer.readu32(s, TRIS) + id * 12
    local a = verts + buffer.readu32(s, t) * 12
    local b = verts + buffer.readu32(s, t + 4) * 12
    local c = verts + buffer.readu32(s, t + 8) * 12
    local pa = vector.create(buffer.readf32(s, a), buffer.readf32(s, a + 4), buffer.readf32(s, a + 8))
    local pb = vector.create(buffer.readf32(s, b), buffer.readf32(s, b + 4), buffer.readf32(s, b + 8))
    local pc = vector.create(buffer.readf32(s, c), buffer.readf32(s, c + 4), buffer.readf32(s, c + 8))

    local n = vector.cross(pb - pa, pc - pa)
    n = rotate(s, n.x, n.y, n.z)
    n = vector.normalize(n)

    local light = vector.normalize(vector.create(-0.4, 0.6, -0.7))
    local diffuse = math.max(0, -vector.dot(n, light))
    local h = vector.normalize(light + vector.create(0, 0, 1))
    local specular = math.max(0, -vector.dot(n, h)) ^ 40
    local rim = (1 - math.abs(n.z)) ^ 3

    -- color bands from the position along the knot
    local hue = math.atan2(pa.y, pa.x)
    local r = 0.55 + 0.45 * math.sin(hue * 3)
    local g = 0.55 + 0.45 * math.sin(hue * 3 + 2.1)
    local bl = 0.55 + 0.45 * math.sin(hue * 3 + 4.2)

    local k = 0.12 + 0.88 * diffuse
    buffer.writeu32(s, image, rgb(r * k + specular + rim * 0.3, g * k + specular + rim * 0.35, bl * k + specular + rim * 0.5))
end

---------------------------------------------------------------------------------------------------
-- frame loop

buffer.fill(scene, offsets[4], 0x7f, W * H * 4)
for o = offsets[5], offsets[6] - 4, 4 do
    buffer.writeu32(scene, o, NO_ID)
end

local yaw, pitch, distance = 0.6, 0.35, 7.5
local paused = false
local useGpu = host.gpuname() ~= nil

-- runs a pass on the GPU, or calls it from Luau

-- the first runs of a kernel return false while the driver creates its pipeline, which the first frame waits for
local COMPILING = "the kernel is being compiled for the GPU"
local function runReady(fn, buffers: { buffer }, runs: number)
    local ok, err = gpu.run(fn, buffers, runs)
    while not ok and err == COMPILING do
        ok, err = gpu.run(fn, buffers, runs)
    end
    return ok, err
end
local function pass(kernel, runs: number)
    if not useGpu then
        for i = 0, runs - 1 do
            kernel(scene, i)
        end
        return
    end

    local ok, err = runReady(kernel, { scene }, runs)
    if not ok then
        error("the GPU can't render: " .. tostring(err))
    end
end

local function setRotation()
    local cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    -- rotation around y, then x
    local m = { cy, 0, sy, sp * sy, cp, -sp * cy, -cp * sy, sp, cp * cy }
    for i, v in m do
        buffer.writef32(scene, (i - 1) * 4, v)
    end
    buffer.writef32(scene, 36, distance)
end

local headless = framesToRun ~= nil
if not headless then
    window.open(W, H, "luau-gpu rasterizer")
end

local frame = 0
local runStart = host.clock()
local last = host.clock()
local timeKernels = { 0, 0, 0, 0 }
local fpsFrames, fpsStart = 0, host.clock()

while true do
    local keys, dx, dy, wheel
    if headless then
        keys, dx, dy, wheel = {}, 0, 0, 0
    else
        keys, dx, dy, wheel = window.poll()
        if not keys then
            break
        end
    end

    for _, key in keys do
        if key == "G" then
            useGpu = not useGpu and host.gpuname() ~= nil
        elseif key == "Space" then
            paused = not paused
        end
    end

    local now = host.clock()
    if headless then
        yaw += 0.01 -- fixed steps, so the GPU and Luau render the same frames
    elseif not paused and dx == 0 and dy == 0 then
        yaw += (now - last) * 0.4
    end
    last = now
    yaw += dx * 0.01
    pitch = math.clamp(pitch + dy * 0.01, -1.5, 1.5)
    distance = math.clamp(distance - wheel * 0.4, 3, 30)
    setRotation()

    local s = host.clock()
    pass(transform, vertexCount)
    local s1 = host.clock()
    pass(depthPass, triangleCount)
    local s2 = host.clock()
    pass(resolvePass, triangleCount)
    local s3 = host.clock()
    pass(shade, W * H)
    local s4 = host.clock()
    timeKernels = { s1 - s, s2 - s1, s3 - s2, s4 - s3 }

    if not headless then
        window.present(scene, offsets[6])
    end

    frame += 1
    fpsFrames += 1
    local elapsed = host.clock() - fpsStart
    if elapsed > 0.5 or not useGpu then
        local mode = if useGpu then "GPU " .. host.gpuname() else "Luau"
        local msg = string.format("%s | %d triangles | %.1f fps | transform %.1f  depth %.1f  resolve %.1f  shade %.1f ms",
            mode, triangleCount, fpsFrames / elapsed,
            timeKernels[1] * 1000, timeKernels[2] * 1000, timeKernels[3] * 1000, timeKernels[4] * 1000)
        if headless then
            print(msg)
        else
            window.title(msg)
        end
        fpsFrames, fpsStart = 0, host.clock()
    end

    if framesToRun and frame >= framesToRun then
        break
    end
end

if headless then
    local total = host.clock() - runStart
    print(string.format("%d frames in %.2fs: %.1f fps, %.1f ms per frame (transform %.1f, depth %.1f, resolve %.1f, shade %.1f ms on the last frame)",
        frame, total, frame / total, total / frame * 1000,
        timeKernels[1] * 1000, timeKernels[2] * 1000, timeKernels[3] * 1000, timeKernels[4] * 1000))
end

if savePath then
    local image = buffer.create(W * H * 4)
    buffer.copy(image, 0, scene, offsets[6], W * H * 4)
    host.writefile(savePath, image)
    print("saved " .. savePath)
end

local st = host.stats()
print(string.format("gpu runs %d / %d, failures %d, uploaded %.1f MB, downloaded %.1f MB",
    st.succeeded or 0, st.runs or 0, st.failures or 0, (st.uploadedBytes or 0) / 1e6, (st.downloadedBytes or 0) / 1e6))
