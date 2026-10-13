-- This file is part of the Luau programming language and is licensed under MIT License; see LICENSE.txt for details
print("testing gpu library")

-- without a backend, dispatch is a loop over 0..n-1
do
    local b = buffer.create(40)
    gpu.dispatch(function(i, out)
        buffer.writeu32(out, i * 4, i * i)
    end, 10, b)
    for i = 0, 9 do
        assert(buffer.readu32(b, i * 4) == i * i)
    end
end

-- several buffers are passed in order after the index
do
    local a = buffer.create(16)
    local b = buffer.create(16)
    for i = 0, 3 do
        buffer.writef32(a, i * 4, i + 0.5)
    end
    gpu.dispatch(function(i, src, dst)
        buffer.writef32(dst, i * 4, buffer.readf32(src, i * 4) * 2)
    end, 4, a, b)
    for i = 0, 3 do
        assert(buffer.readf32(b, i * 4) == (i + 0.5) * 2)
    end
end

-- zero invocations is allowed
do
    local called = false
    gpu.dispatch(function()
        called = true
    end, 0)
    assert(not called)
end

-- argument validation
assert(not pcall(gpu.dispatch, 1, 1))
assert(not pcall(gpu.dispatch, function() end, -1))
assert(not pcall(gpu.dispatch, function() end, 1, {}))
assert(not pcall(gpu.dispatch, function() end, 1, buffer.create(4), "x"))

-- errors in the kernel propagate
assert(not pcall(gpu.dispatch, function(i, b)
    buffer.writeu32(b, 8, 1)
end, 1, buffer.create(4)))

-- atomics return the previous value
do
    local b = buffer.create(8)
    assert(gpu.atomicadd(b, 0, 5) == 0)
    assert(gpu.atomicadd(b, 0, 7) == 5)
    assert(buffer.readu32(b, 0) == 12)
    assert(gpu.atomicadd(b, 0, 0xffffffff) == 12)
    assert(buffer.readu32(b, 0) == 11)

    buffer.writeu32(b, 4, 100)
    assert(gpu.atomicmin(b, 4, 200) == 100)
    assert(buffer.readu32(b, 4) == 100)
    assert(gpu.atomicmin(b, 4, 50) == 100)
    assert(buffer.readu32(b, 4) == 50)
    assert(gpu.atomicmax(b, 4, 0xfffffff0) == 50)
    assert(buffer.readu32(b, 4) == 0xfffffff0)

    assert(not pcall(gpu.atomicadd, b, 6, 1))
    assert(not pcall(gpu.atomicmax, b, -1, 1))
end

-- histogram through atomics gives the same result in any order
do
    local data = buffer.create(400)
    for i = 0, 99 do
        buffer.writeu32(data, i * 4, i % 7)
    end
    local hist = buffer.create(28)
    gpu.dispatch(function(i, src, h)
        gpu.atomicadd(h, buffer.readu32(src, i * 4) * 4, 1)
    end, 100, data, hist)
    local total = 0
    for k = 0, 6 do
        total += buffer.readu32(hist, k * 4)
    end
    assert(total == 100)
    assert(buffer.readu32(hist, 0) == 15)
end

-- status without a backend
do
    local where, why = gpu.status(function() end)
    assert(where == "vm" and why == "no GPU backend")
end

return "OK"
