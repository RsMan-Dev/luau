// This file is part of the Luau programming language and is licensed under MIT License; see LICENSE.txt for details
#include "lualib.h"

#include "lcommon.h"
#include "lbuffer.h"

#include <string.h>

#define GPU_BACKEND_KEY "_GPUBACKEND"

static const lua_GpuBackend* getbackend(lua_State* L)
{
    lua_rawgetfield(L, LUA_REGISTRYINDEX, GPU_BACKEND_KEY);
    const lua_GpuBackend* backend = (const lua_GpuBackend*)lua_tolightuserdata(L, -1);
    lua_pop(L, 1);
    return backend;
}

void lua_setgpubackend(lua_State* L, const lua_GpuBackend* backend)
{
    if (backend)
        lua_pushlightuserdata(L, (void*)backend);
    else
        lua_pushnil(L);
    lua_rawsetfield(L, LUA_REGISTRYINDEX, GPU_BACKEND_KEY);
}

// gpu.dispatch(fn, n, ...buffers) has the semantics of `for i = 0, n - 1 do fn(i, ...) end` with an unspecified order
static int gpu_dispatch(lua_State* L)
{
    luaL_checktype(L, 1, LUA_TFUNCTION);
    int n = luaL_checkinteger(L, 2);
    luaL_argcheck(L, n >= 0, 2, "invocation count must be non-negative");

    int nargs = lua_gettop(L) - 2;
    for (int i = 0; i < nargs; ++i)
        luaL_checkbuffer(L, 3 + i, NULL);

    const lua_GpuBackend* backend = getbackend(L);
    if (backend && backend->dispatch(backend->context, L, 1, n, 3, nargs))
        return 0;

    luaL_checkstack(L, nargs + 2, "too many arguments");

    for (int i = 0; i < n; ++i)
    {
        lua_pushvalue(L, 1);
        lua_pushinteger(L, i);
        for (int a = 0; a < nargs; ++a)
            lua_pushvalue(L, 3 + a);
        lua_call(L, nargs + 1, 0);
    }

    return 0;
}

// gpu.status(fn) returns "gpu" if dispatches of fn run on the GPU, or "vm" and the reason why they don't
static int gpu_status(lua_State* L)
{
    luaL_checktype(L, 1, LUA_TFUNCTION);

    const lua_GpuBackend* backend = getbackend(L);
    if (!backend)
    {
        lua_pushstring(L, "vm");
        lua_pushstring(L, "no GPU backend");
        return 2;
    }

    backend->status(backend->context, L, 1);

    if (lua_isnil(L, -1))
    {
        lua_pushstring(L, "gpu");
        return 1;
    }

    lua_pushstring(L, "vm");
    lua_insert(L, -2);
    return 2;
}

static uint32_t atomic_add(uint32_t a, uint32_t b)
{
    return a + b;
}

static uint32_t atomic_min(uint32_t a, uint32_t b)
{
    return b < a ? b : a;
}

static uint32_t atomic_max(uint32_t a, uint32_t b)
{
    return b > a ? b : a;
}

// atomics operate on unsigned 32-bit integers and return the previous value
template<uint32_t (*op)(uint32_t, uint32_t)>
static int gpu_atomic(lua_State* L)
{
    size_t len = 0;
    void* buf = luaL_checkbuffer(L, 1, &len);
    int offset = luaL_checkinteger(L, 2);
    unsigned value = luaL_checkunsigned(L, 3);

    if (isoutofbounds(offset, len, sizeof(uint32_t)))
        luaL_error(L, "buffer access out of bounds");

    uint32_t old;
    memcpy(&old, (char*)buf + offset, sizeof(old));

    uint32_t res = op(old, value);
    memcpy((char*)buf + offset, &res, sizeof(res));

    lua_pushunsigned(L, old);
    return 1;
}

static const luaL_Reg gpulib[] = {
    {"dispatch", gpu_dispatch},
    {"status", gpu_status},
    {"atomicadd", gpu_atomic<atomic_add>},
    {"atomicmin", gpu_atomic<atomic_min>},
    {"atomicmax", gpu_atomic<atomic_max>},
    {NULL, NULL},
};

int luaopen_gpu(lua_State* L)
{
    luaL_register(L, LUA_GPULIBNAME, gpulib);

    return 1;
}
