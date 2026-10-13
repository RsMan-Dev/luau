// This file is part of the Luau programming language and is licensed under MIT License; see LICENSE.txt for details
#include "lualib.h"

#include "lcommon.h"
#include "lbuffer.h"

#include <limits.h>
#include <string.h>

LUAU_FASTFLAGVARIABLE(LuauGpuLibrary)

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

// numbers are checked before they are converted to integers, since out of range conversions are undefined in C
static int checkrange(lua_State* L, int narg, double lo, double hi, const char* msg)
{
    double v = luaL_checknumber(L, narg);
    luaL_argcheck(L, v >= lo && v < hi, narg, msg);
    return int(v);
}

// gpu.dispatch(fn, n, ...buffers) has the semantics of `for i = 0, n - 1 do fn(i, ...) end` with an unspecified order
static int gpu_dispatch(lua_State* L)
{
    luaL_checktype(L, 1, LUA_TFUNCTION);
    // like numeric for loops, a fractional count is rounded down
    int n = checkrange(L, 2, 0, double(INT_MAX) + 1, "invocation count must be between 0 and 2147483647");

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

// atomics operate on unsigned 32-bit integers and return the previous value; like bit32, values wrap modulo 2^32, but only
// from integers that are in range for a 32-bit signed or unsigned integer, so that the GPU converts them the same way
template<uint32_t (*op)(uint32_t, uint32_t)>
static int gpu_atomic(lua_State* L)
{
    size_t len = 0;
    void* buf = luaL_checkbuffer(L, 1, &len);

    double off = luaL_checknumber(L, 2);
    // offsets are rounded toward zero as in the buffer library, so the range check happens before the conversion
    if (!(off > -1 && off < double(INT_MAX) + 1))
        luaL_error(L, "buffer access out of bounds");
    int offset = int(off);

    double v = luaL_checknumber(L, 3);
    luaL_argcheck(L, v > double(INT_MIN) - 1 && v < 4294967296.0, 3, "value must be between -2147483648 and 4294967295");
    uint32_t value = v < 0 ? uint32_t(int32_t(v)) : uint32_t(v);

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
