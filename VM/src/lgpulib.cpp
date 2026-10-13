// This file is part of the Luau programming language and is licensed under MIT License; see LICENSE.txt for details
#include "lualib.h"

#include "lcommon.h"
#include "lapi.h"
#include "lbuffer.h"
#include "lobject.h"
#include "lstate.h"

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

// a function can run on the GPU only if the builtins it uses can't have been replaced, as for native code
static bool gpusafe(lua_State* L, int idx)
{
    const TValue* fn = luaA_toobject(L, idx);
    return !clvalue(fn)->isC && clvalue(fn)->env->safeenv;
}

static void runstep(lua_State* L, const lua_GpuStep* step)
{
    luaL_checkstack(L, step->nargs + 2, "too many arguments");

    for (int i = 0; i < step->count; ++i)
    {
        lua_pushvalue(L, step->fn);
        lua_pushinteger(L, i);
        for (int a = 0; a < step->nargs; ++a)
            lua_pushvalue(L, step->args + a);
        lua_call(L, step->nargs + 1, 0);
    }
}

static void checkbuffers(lua_State* L, int narg)
{
    if (lua_isnoneornil(L, narg))
        return;

    luaL_checktype(L, narg, LUA_TTABLE);
    for (int i = 1, n = lua_objlen(L, narg); i <= n; ++i)
    {
        lua_rawgeti(L, narg, i);
        if (!lua_isbuffer(L, -1))
            luaL_argerror(L, narg, "expected a table of buffers");
        lua_pop(L, 1);
    }
}

// checks a step whose function is at stack index fn, followed by its count and nargs buffers
static void checkstep(lua_State* L, lua_GpuStep* step, int fn, int nargs)
{
    luaL_checktype(L, fn, LUA_TFUNCTION);
    step->fn = fn;
    // like numeric for loops, a fractional count is rounded down
    step->count = checkrange(L, fn + 1, 0, double(INT_MAX) + 1, "invocation count must be between 0 and 2147483647");
    step->args = fn + 2;
    step->nargs = nargs > 0 ? nargs : 0;

    for (int i = 0; i < step->nargs; ++i)
        luaL_checkbuffer(L, step->args + i, NULL);

    step->safe = gpusafe(L, fn);
}

// gpu.dispatch(fn, n, ...buffers) has the semantics of `for i = 0, n - 1 do fn(i, ...) end` with an unspecified order
//
// gpu.dispatch(steps, outputs, changed) runs each step { fn, n, ...buffers } of the list like gpu.dispatch(fn, n, ...buffers),
// in order, with two relaxations that let a GPU keep buffers in its memory:
// - after the call, only the buffers in the outputs list have up to date Luau contents; other buffers the call writes have
//   unspecified contents until they are in the outputs of a later call, but later calls see what this one wrote
// - a buffer modified by Luau code since a call used it must be in the changed list of the next call that uses it, otherwise
//   that call may see its former contents
static int gpu_dispatch(lua_State* L)
{
    const lua_GpuBackend* backend = getbackend(L);
    bool batch = lua_istable(L, 1);
    lua_GpuStep* steps;
    int nsteps;

    if (batch)
    {
        checkbuffers(L, 2);
        checkbuffers(L, 3);
        lua_settop(L, 3);

        nsteps = lua_objlen(L, 1);
        steps = (lua_GpuStep*)lua_newuserdata(L, sizeof(lua_GpuStep) * (nsteps > 0 ? nsteps : 1));

        // the steps are copied to the stack, so they can't change while they run
        for (int s = 0; s < nsteps; ++s)
        {
            lua_rawgeti(L, 1, s + 1);
            if (!lua_istable(L, -1))
                luaL_error(L, "step %d must be a table { fn, count, ...buffers }", s + 1);

            int t = lua_gettop(L);
            int n = lua_objlen(L, t);
            luaL_checkstack(L, n + 1, "too many steps");
            for (int i = 1; i <= n; ++i)
                lua_rawgeti(L, t, i);
            lua_remove(L, t);

            checkstep(L, &steps[s], lua_gettop(L) - n + 1, n - 2);
        }
    }
    else
    {
        nsteps = 1;
        int nargs = lua_gettop(L) - 2;
        steps = (lua_GpuStep*)lua_newuserdata(L, sizeof(lua_GpuStep));
        checkstep(L, steps, 1, nargs);
    }

    // the backend is called even for steps that must run in the VM, so that it can bring the buffers it holds up to date
    if (backend && backend->dispatch(backend->context, L, steps, nsteps, batch ? 2 : 0, batch ? 3 : 0))
        return 0;

    for (int s = 0; s < nsteps; ++s)
        runstep(L, &steps[s]);

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
