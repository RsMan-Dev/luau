// This file is part of the Luau programming language and is licensed under MIT License; see LICENSE.txt for details
#include "Luau/GpuLibrary.h"

#include "Luau/GpuVulkan.h"

#include "lua.h"
#include "lualib.h"

#include <limits.h>
#include <math.h>
#include <string.h>

namespace Luau
{
namespace Gpu
{

// arguments of the wrong type or out of range are mistakes in the script, which raise errors as in other libraries; false
// and a reason are for what only shows when the run is attempted, which a script can handle
static int gpu_run(lua_State* L)
{
    luaL_checktype(L, 1, LUA_TFUNCTION);
    luaL_checktype(L, 2, LUA_TTABLE);
    // kernels compute with f32 numbers, which hold the index exactly up to 2^24; larger works are split across buffers
    double runs = luaL_checknumber(L, 3);
    luaL_argcheck(L, runs >= 0 && runs <= 16777216 && runs == floor(runs), 3, "must be an integer between 0 and 16777216");

    for (int i = 1, n = lua_objlen(L, 2); i <= n; ++i)
    {
        lua_rawgeti(L, 2, i);
        if (!lua_isbuffer(L, -1))
            luaL_error(L, "invalid argument #2 to 'run' (buffer expected at index %d, got %s)", i, luaL_typename(L, -1));
        lua_pop(L, 1);
    }

    VulkanBackend* backend = static_cast<VulkanBackend*>(lua_tolightuserdata(L, lua_upvalueindex(1)));
    std::string reason = backend ? backend->run(L, 1, 2, uint32_t(runs)) : lua_tostring(L, lua_upvalueindex(2));

    lua_pushboolean(L, reason.empty());
    if (reason.empty())
        return 1;

    lua_pushlstring(L, reason.data(), reason.size());
    return 2;
}

static uint32_t atomicAdd(uint32_t a, uint32_t b)
{
    return a + b;
}

static uint32_t atomicMin(uint32_t a, uint32_t b)
{
    return b < a ? b : a;
}

// atomics operate on unsigned 32-bit integers and return the previous value; offsets and values convert as in the buffer
// library: offsets as integers, values as unsigned integers wrapping modulo 2^32, as bit32's do
template<uint32_t (*op)(uint32_t, uint32_t)>
static int gpu_atomic(lua_State* L)
{
    size_t len = 0;
    void* buf = luaL_checkbuffer(L, 1, &len);

    int offset = luaL_checkinteger(L, 2);
    if (uint64_t(unsigned(offset)) + sizeof(uint32_t) > uint64_t(len))
        luaL_error(L, "buffer access out of bounds");

    uint32_t value = luaL_checkunsigned(L, 3);

    uint32_t old;
    memcpy(&old, (char*)buf + offset, sizeof(old));

    uint32_t res = op(old, value);
    memcpy((char*)buf + offset, &res, sizeof(res));

    lua_pushunsigned(L, old);
    return 1;
}

void openGpuLibrary(lua_State* L, VulkanBackend* backend, const std::string& unavailable)
{
    lua_createtable(L, 0, 3);

    lua_pushlightuserdata(L, backend);
    lua_pushlstring(L, unavailable.data(), unavailable.size());
    lua_pushcclosure(L, gpu_run, "run", 2);
    lua_setfield(L, -2, "run");

    lua_pushcfunction(L, gpu_atomic<atomicAdd>, "atomicadd");
    lua_setfield(L, -2, "atomicadd");
    lua_pushcfunction(L, gpu_atomic<atomicMin>, "atomicmin");
    lua_setfield(L, -2, "atomicmin");

    lua_setglobal(L, "gpu");
}

} // namespace Gpu
} // namespace Luau
