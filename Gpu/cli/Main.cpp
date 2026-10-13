// This file is part of the Luau programming language and is licensed under MIT License; see LICENSE.txt for details

// luau-gpu: runs a script with the gpu library backed by Vulkan, plus a small 'host' and 'window' library for demos and tests
#include "Luau/GpuVulkan.h"

#include "lua.h"
#include "lualib.h"
#include "luacode.h"
#include "Luau/CodeGen.h"

#include "Luau/Common.h"

#include <SDL.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <chrono>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>

LUAU_FASTFLAG(LuauGpuKernels)
LUAU_FASTFLAG(LuauGpuLibrary)

static std::unique_ptr<Luau::Gpu::VulkanBackend> gBackend;

struct Window
{
    SDL_Window* window = nullptr;
    SDL_Renderer* renderer = nullptr;
    SDL_Texture* texture = nullptr;
    int width = 0;
    int height = 0;
};

static Window gWindow;

static int host_setgpu(lua_State* L)
{
    bool enabled = lua_toboolean(L, 1);
    if (enabled && gBackend)
        gBackend->install(L);
    else
        lua_setgpubackend(L, nullptr);
    lua_pushboolean(L, enabled && gBackend);
    return 1;
}

static int host_gpuname(lua_State* L)
{
    if (gBackend)
        lua_pushstring(L, gBackend->deviceName().c_str());
    else
        lua_pushnil(L);
    return 1;
}

static int host_clock(lua_State* L)
{
    using namespace std::chrono;
    static const steady_clock::time_point start = steady_clock::now();
    lua_pushnumber(L, duration<double>(steady_clock::now() - start).count());
    return 1;
}

static int host_stats(lua_State* L)
{
    lua_createtable(L, 0, 5);
    if (gBackend)
    {
        Luau::Gpu::DispatchStats& s = gBackend->stats();
        lua_pushnumber(L, double(s.dispatches));
        lua_setfield(L, -2, "dispatches");
        lua_pushnumber(L, double(s.gpuDispatches));
        lua_setfield(L, -2, "gpuDispatches");
        lua_pushnumber(L, double(s.fallbacks));
        lua_setfield(L, -2, "fallbacks");
        lua_pushnumber(L, double(s.uploadedBytes));
        lua_setfield(L, -2, "uploadedBytes");
        lua_pushnumber(L, double(s.downloadedBytes));
        lua_setfield(L, -2, "downloadedBytes");
    }
    return 1;
}

static int host_writefile(lua_State* L)
{
    const char* path = luaL_checkstring(L, 1);
    size_t len = 0;
    void* data = luaL_checkbuffer(L, 2, &len);

    FILE* f = fopen(path, "wb");
    if (!f)
        luaL_error(L, "can't open %s", path);
    fwrite(data, 1, len, f);
    fclose(f);
    return 0;
}

static const luaL_Reg hostlib[] = {
    {"writefile", host_writefile},
    {"setgpu", host_setgpu},
    {"gpuname", host_gpuname},
    {"clock", host_clock},
    {"stats", host_stats},
    {NULL, NULL},
};

static int window_open(lua_State* L)
{
    int w = luaL_checkinteger(L, 1);
    int h = luaL_checkinteger(L, 2);
    const char* title = luaL_optstring(L, 3, "luau-gpu");

    if (gWindow.window)
        luaL_error(L, "window is already open");

    if (SDL_Init(SDL_INIT_VIDEO) != 0)
        luaL_error(L, "SDL_Init failed: %s", SDL_GetError());

    gWindow.window = SDL_CreateWindow(title, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, w, h, SDL_WINDOW_SHOWN);
    if (!gWindow.window)
        luaL_error(L, "SDL_CreateWindow failed: %s", SDL_GetError());

    gWindow.renderer = SDL_CreateRenderer(gWindow.window, -1, SDL_RENDERER_ACCELERATED);
    if (!gWindow.renderer)
        gWindow.renderer = SDL_CreateRenderer(gWindow.window, -1, SDL_RENDERER_SOFTWARE);
    if (!gWindow.renderer)
        luaL_error(L, "SDL_CreateRenderer failed: %s", SDL_GetError());

    gWindow.texture = SDL_CreateTexture(gWindow.renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, w, h);
    gWindow.width = w;
    gWindow.height = h;
    return 0;
}

static int window_present(lua_State* L)
{
    size_t len = 0;
    void* data = luaL_checkbuffer(L, 1, &len);

    if (!gWindow.window)
        luaL_error(L, "window is not open");
    if (len < size_t(gWindow.width) * gWindow.height * 4)
        luaL_error(L, "buffer is smaller than the window (%d x %d x 4 bytes)", gWindow.width, gWindow.height);

    SDL_UpdateTexture(gWindow.texture, nullptr, data, gWindow.width * 4);
    SDL_RenderClear(gWindow.renderer);
    SDL_RenderCopy(gWindow.renderer, gWindow.texture, nullptr, nullptr);
    SDL_RenderPresent(gWindow.renderer);
    return 0;
}

static int window_title(lua_State* L)
{
    if (gWindow.window)
        SDL_SetWindowTitle(gWindow.window, luaL_checkstring(L, 1));
    return 0;
}

// returns nil once the window is closed, otherwise the keys pressed since the last poll and the mouse drag/wheel motion
static int window_poll(lua_State* L)
{
    lua_newtable(L);
    int keys = 0;
    int dx = 0, dy = 0, wheel = 0;

    SDL_Event e;
    while (SDL_PollEvent(&e))
    {
        switch (e.type)
        {
        case SDL_QUIT:
            lua_pushnil(L);
            return 1;
        case SDL_KEYDOWN:
            if (e.key.keysym.sym == SDLK_ESCAPE)
            {
                lua_pushnil(L);
                return 1;
            }
            lua_pushstring(L, SDL_GetKeyName(e.key.keysym.sym));
            lua_rawseti(L, -2, ++keys);
            break;
        case SDL_MOUSEMOTION:
            if (e.motion.state & SDL_BUTTON_LMASK)
            {
                dx += e.motion.xrel;
                dy += e.motion.yrel;
            }
            break;
        case SDL_MOUSEWHEEL:
            wheel += e.wheel.y;
            break;
        }
    }

    lua_pushinteger(L, dx);
    lua_pushinteger(L, dy);
    lua_pushinteger(L, wheel);
    return 4;
}

static const luaL_Reg windowlib[] = {
    {"open", window_open},
    {"present", window_present},
    {"title", window_title},
    {"poll", window_poll},
    {NULL, NULL},
};

static int traceback(lua_State* L)
{
    const char* msg = lua_tostring(L, 1);
    luaL_traceback(L, L, msg ? msg : "error", 1);
    return 1;
}

int main(int argc, char** argv)
{
    FFlag::LuauGpuKernels.value = true;
    FFlag::LuauGpuLibrary.value = true;

    bool useGpu = true;
    bool useNative = true;
    const char* path = nullptr;
    int scriptArg = argc;

    for (int i = 1; i < argc; ++i)
    {
        if (strcmp(argv[i], "--vm") == 0)
            useGpu = false;
        else if (strcmp(argv[i], "--no-native") == 0)
            useNative = false;
        else
        {
            path = argv[i];
            scriptArg = i + 1;
            break;
        }
    }

    if (!path)
    {
        fprintf(stderr, "usage: luau-gpu [--vm] [--no-native] script.luau [args...]\n");
        return 1;
    }

    std::ifstream file(path, std::ios::binary);
    if (!file)
    {
        fprintf(stderr, "can't open %s\n", path);
        return 1;
    }
    std::stringstream ss;
    ss << file.rdbuf();
    std::string source = ss.str();
    std::string chunkname = std::string("@") + path;

    lua_State* L = luaL_newstate();
    luaL_openlibs(L);
    luaL_register(L, "host", hostlib);
    lua_pop(L, 1);
    luaL_register(L, "window", windowlib);
    lua_pop(L, 1);

    lua_newtable(L);
    for (int i = scriptArg; i < argc; ++i)
    {
        lua_pushstring(L, argv[i]);
        lua_rawseti(L, -2, i - scriptArg + 1);
    }
    lua_setglobal(L, "arg");

    if (useGpu)
    {
        std::string error;
        gBackend = Luau::Gpu::VulkanBackend::create(error);
        if (!gBackend)
            fprintf(stderr, "luau-gpu: no GPU backend (%s), kernels run in the VM\n", error.c_str());
    }

    if (gBackend)
    {
        gBackend->setFallbackHandler(
            [](const std::string& message)
            {
                fprintf(stderr, "luau-gpu: %s\n", message.c_str());
            }
        );
        gBackend->load(chunkname, source.data(), source.size());
        gBackend->install(L);
    }

    // like the luau REPL, run the script in a sandboxed thread; besides isolating it, a sandboxed environment lets the VM
    // use its fast paths for builtins, which kernels running in the VM depend on
    luaL_sandbox(L);
    lua_State* T = lua_newthread(L);
    luaL_sandboxthread(T);

    lua_CompileOptions options = {};
    options.optimizationLevel = 1;
    options.debugLevel = 1;
    options.typeInfoLevel = 1; // lets native code use type annotations, as in the luau REPL
    size_t bytecodeSize = 0;
    char* bytecode = luau_compile(source.data(), source.size(), &options, &bytecodeSize);
    int status = luau_load(T, chunkname.c_str(), bytecode, bytecodeSize, 0);
    free(bytecode);

    if (status != 0)
    {
        fprintf(stderr, "%s\n", lua_tostring(T, -1));
        return 1;
    }

    // kernels usually have no loops, so cold functions are compiled too
    if (useNative && Luau::CodeGen::isSupported())
    {
        Luau::CodeGen::create(L);
        Luau::CodeGen::CompilationOptions nativeOptions;
        nativeOptions.flags = Luau::CodeGen::CodeGen_ColdFunctions;
        Luau::CodeGen::compile(T, -1, nativeOptions);
    }

    lua_pushcfunction(T, traceback, "traceback");
    lua_insert(T, -2);
    status = lua_pcall(T, 0, 0, -2);

    int code = 0;
    if (status != 0)
    {
        fprintf(stderr, "%s\n", lua_tostring(T, -1));
        code = 1;
    }

    lua_close(L);
    gBackend.reset();

    if (gWindow.window)
    {
        SDL_DestroyTexture(gWindow.texture);
        SDL_DestroyRenderer(gWindow.renderer);
        SDL_DestroyWindow(gWindow.window);
        SDL_Quit();
    }

    return code;
}
