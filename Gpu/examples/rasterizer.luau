-- Software rasterizer written in Luau, running on the GPU through gpu.dispatch.
--
-- usage: luau-gpu Gpu/examples/rasterizer.luau [segments] [--frames N --save path.raw]
--   window controls: drag to rotate, wheel to zoom, space to pause, G to switch between GPU and VM
--
-- Every frame runs four kernels:
--   1. transform  one invocation per vertex: rotate, project to the screen
--   2. depth      one invocation per triangle: rasterize, keep the nearest depth per pixel (gpu.atomicmin)
--   3. resolve    one invocation per triangle: the triangle that owns the nearest depth claims the pixel (gpu.atomicmax)
--   4. shade      one invocation per pixel: light the owning triangle, clear depth and ids for the next frame

local W = 1280
local H = 720
local FOCAL = 900
local NEAR = 0.5
local DEPTH_SCALE = 4194304 -- depth keys are integers: view z * scale
local CLEAR_DEPTH = 0x7f7f7f7f

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

local verts = buffer.create(vertexCount * 12)
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
        local o = (u * V + v) * 12
        buffer.writef32(verts, o, p.x)
        buffer.writef32(verts, o + 4, p.y)
        buffer.writef32(verts, o + 8, p.z)
    end
end

local idx = buffer.create(triangleCount * 12)
for u = 0, U - 1 do
    local u1 = (u + 1) % U
    for v = 0, V - 1 do
        local v1 = (v + 1) % V
        local a, b, c, d = u * V + v, u1 * V + v, u1 * V + v1, u * V + v1
        local o = (u * V + v) * 24
        buffer.writeu32(idx, o, a)
        buffer.writeu32(idx, o + 4, b)
        buffer.writeu32(idx, o + 8, c)
        buffer.writeu32(idx, o + 12, a)
        buffer.writeu32(idx, o + 16, c)
        buffer.writeu32(idx, o + 20, d)
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
local function rotate(params: buffer, x: number, y: number, z: number): vector
    return vector.create(
        buffer.readf32(params, 0) * x + buffer.readf32(params, 4) * y + buffer.readf32(params, 8) * z,
        buffer.readf32(params, 12) * x + buffer.readf32(params, 16) * y + buffer.readf32(params, 20) * z,
        buffer.readf32(params, 24) * x + buffer.readf32(params, 28) * y + buffer.readf32(params, 32) * z
    )
end

@gpu
local function transform(i: number, verts: buffer, params: buffer, proj: buffer)
    local o = i * 12
    local p = rotate(params, buffer.readf32(verts, o), buffer.readf32(verts, o + 4), buffer.readf32(verts, o + 8))
    local z = p.z + buffer.readf32(params, 36)
    buffer.writef32(proj, o, W / 2 + FOCAL * p.x / z)
    buffer.writef32(proj, o + 4, H / 2 - FOCAL * p.y / z)
    buffer.writef32(proj, o + 8, z)
end

-- shared by the depth and resolve passes so both compute exactly the same coverage and depth keys
@gpu
local function rasterize(t: number, idx: buffer, proj: buffer, depth: buffer, ids: buffer, resolve: boolean)
    local o = t * 12
    local i0 = buffer.readu32(idx, o) * 12
    local i1 = buffer.readu32(idx, o + 4) * 12
    local i2 = buffer.readu32(idx, o + 8) * 12

    local z0, z1, z2 = buffer.readf32(proj, i0 + 8), buffer.readf32(proj, i1 + 8), buffer.readf32(proj, i2 + 8)
    if z0 < NEAR or z1 < NEAR or z2 < NEAR then
        return
    end

    local x0, y0 = buffer.readf32(proj, i0), buffer.readf32(proj, i0 + 4)
    local x1, y1 = buffer.readf32(proj, i1), buffer.readf32(proj, i1 + 4)
    local x2, y2 = buffer.readf32(proj, i2), buffer.readf32(proj, i2 + 4)

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
                    if buffer.readu32(depth, p) == key then
                        gpu.atomicmax(ids, p, t + 1)
                    end
                else
                    gpu.atomicmin(depth, p, key)
                end
            end
        end
    end
end

@gpu
local function depthPass(t: number, idx: buffer, proj: buffer, depth: buffer, ids: buffer)
    rasterize(t, idx, proj, depth, ids, false)
end

@gpu
local function resolvePass(t: number, idx: buffer, proj: buffer, depth: buffer, ids: buffer)
    rasterize(t, idx, proj, depth, ids, true)
end

@gpu
local function channel(v: number)
    return math.clamp(math.floor(v * 255), 0, 255)
end

@gpu
local function rgb(r: number, g: number, b: number)
    return bit32.bor(0xff000000, bit32.lshift(channel(r), 16), bit32.lshift(channel(g), 8), channel(b))
end

@gpu
local function shade(p: number, ids: buffer, depth: buffer, verts: buffer, idx: buffer, params: buffer, image: buffer)
    local o = p * 4
    local id = buffer.readu32(ids, o)
    buffer.writeu32(ids, o, 0)
    buffer.writeu32(depth, o, CLEAR_DEPTH)

    if id == 0 then
        local y = p // W
        local g = 0.08 + 0.12 * (1 - y / H)
        buffer.writeu32(image, o, rgb(g * 0.6, g * 0.7, g))
        return
    end

    local t = (id - 1) * 12
    local a, b, c = buffer.readu32(idx, t) * 12, buffer.readu32(idx, t + 4) * 12, buffer.readu32(idx, t + 8) * 12
    local pa = vector.create(buffer.readf32(verts, a), buffer.readf32(verts, a + 4), buffer.readf32(verts, a + 8))
    local pb = vector.create(buffer.readf32(verts, b), buffer.readf32(verts, b + 4), buffer.readf32(verts, b + 8))
    local pc = vector.create(buffer.readf32(verts, c), buffer.readf32(verts, c + 4), buffer.readf32(verts, c + 8))

    local n = vector.cross(pb - pa, pc - pa)
    n = rotate(params, n.x, n.y, n.z)
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
    buffer.writeu32(image, o, rgb(r * k + specular + rim * 0.3, g * k + specular + rim * 0.35, bl * k + specular + rim * 0.5))
end

---------------------------------------------------------------------------------------------------
-- frame loop

for _, k in { transform, depthPass, resolvePass, shade } do
    local where, why = gpu.status(k)
    if where ~= "gpu" then
        print("kernel stays in the VM: " .. tostring(why))
    end
end

local proj = buffer.create(vertexCount * 12)
local depth = buffer.create(W * H * 4)
local ids = buffer.create(W * H * 4)
local image = buffer.create(W * H * 4)
local params = buffer.create(64)
buffer.fill(depth, 0, 0x7f)

local yaw, pitch, distance = 0.6, 0.35, 7.5
local paused = false
local useGpu = host.gpuname() ~= nil
host.setgpu(useGpu)

local function setRotation()
    local cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    -- rotation around y, then x
    local m = { cy, 0, sy, sp * sy, cp, -sp * cy, -cp * sy, sp, cp * cy }
    for i, v in m do
        buffer.writef32(params, (i - 1) * 4, v)
    end
    buffer.writef32(params, 36, distance)
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
            useGpu = host.setgpu(not useGpu)
        elseif key == "Space" then
            paused = not paused
        end
    end

    local now = host.clock()
    if headless then
        yaw += 0.01 -- fixed steps, so GPU and VM runs render the same frames
    elseif not paused and dx == 0 and dy == 0 then
        yaw += (now - last) * 0.4
    end
    last = now
    yaw += dx * 0.01
    pitch = math.clamp(pitch + dy * 0.01, -1.5, 1.5)
    distance = math.clamp(distance - wheel * 0.4, 3, 30)
    setRotation()

    local s = host.clock()
    gpu.dispatch(transform, vertexCount, verts, params, proj)
    local s1 = host.clock()
    gpu.dispatch(depthPass, triangleCount, idx, proj, depth, ids)
    local s2 = host.clock()
    gpu.dispatch(resolvePass, triangleCount, idx, proj, depth, ids)
    local s3 = host.clock()
    gpu.dispatch(shade, W * H, ids, depth, verts, idx, params, image)
    local s4 = host.clock()
    timeKernels = { s1 - s, s2 - s1, s3 - s2, s4 - s3 }

    if not headless then
        window.present(image)
    end

    frame += 1
    fpsFrames += 1
    local elapsed = host.clock() - fpsStart
    if elapsed > 0.5 or not useGpu then
        local mode = if useGpu then "GPU " .. host.gpuname() else "VM"
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
    host.writefile(savePath, image)
    print("saved " .. savePath)
end

local st = host.stats()
print(string.format("gpu dispatches %d / %d, fallbacks %d, uploaded %.1f MB, downloaded %.1f MB",
    st.gpuDispatches or 0, st.dispatches or 0, st.fallbacks or 0, (st.uploadedBytes or 0) / 1e6, (st.downloadedBytes or 0) / 1e6))
