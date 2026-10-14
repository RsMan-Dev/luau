// This file is part of the Luau programming language and is licensed under MIT License; see LICENSE.txt for details

// luau-gpu: runs a script with the gpu library backed by Vulkan, plus a small 'host' and 'window' library for demos and tests
#include "Luau/GpuLibrary.h"
#include "Luau/GpuShaderCompiler.h"
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
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

LUAU_FASTFLAG(LuauGpuKernels)

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

static int host_gpuname(lua_State* L)
{
    if (gBackend)
        lua_pushstring(L, gBackend->deviceName().c_str());
    else
        lua_pushnil(L);
    return 1;
}

// runs a full garbage collection cycle, like collectgarbage("collect") in the luau REPL
static int host_collect(lua_State* L)
{
    lua_gc(L, LUA_GCCOLLECT, 0);
    return 0;
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
    lua_createtable(L, 0, 8);
    if (gBackend)
    {
        Luau::Gpu::RunStats s = gBackend->stats();
        lua_pushnumber(L, double(s.runs));
        lua_setfield(L, -2, "runs");
        lua_pushnumber(L, double(s.succeeded));
        lua_setfield(L, -2, "succeeded");
        lua_pushnumber(L, double(s.failures));
        lua_setfield(L, -2, "failures");
        lua_pushnumber(L, double(s.uploadedBytes));
        lua_setfield(L, -2, "uploadedBytes");
        lua_pushnumber(L, double(s.downloadedBytes));
        lua_setfield(L, -2, "downloadedBytes");
        lua_pushnumber(L, double(s.cachedBytes));
        lua_setfield(L, -2, "cachedBytes");
        lua_pushnumber(L, double(s.memoryBytes));
        lua_setfield(L, -2, "memoryBytes");
        lua_pushnumber(L, double(s.kernels));
        lua_setfield(L, -2, "kernels");
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
    {"collect", host_collect},
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
    // the window can be shown larger than the image, which is scaled to fit
    int scale = luaL_optinteger(L, 4, 1);
    // "argb" (default) takes 32-bit 0xAARRGGBB words, "rgba" takes the bytes R, G, B, A as EditableImage buffers store them
    const char* format = luaL_optstring(L, 5, "argb");
    bool rgba = strcmp(format, "rgba") == 0;
    luaL_argcheck(L, rgba || strcmp(format, "argb") == 0, 5, "format must be 'argb' or 'rgba'");

    luaL_argcheck(L, w > 0 && w <= 16384, 1, "width must be between 1 and 16384");
    luaL_argcheck(L, h > 0 && h <= 16384, 2, "height must be between 1 and 16384");
    luaL_argcheck(L, scale >= 1 && scale <= 8 && w * scale <= 16384 && h * scale <= 16384, 4, "scale must be between 1 and 8");

    if (gWindow.window)
        luaL_error(L, "window is already open");

    if (SDL_Init(SDL_INIT_VIDEO) != 0)
        luaL_error(L, "SDL_Init failed: %s", SDL_GetError());

    gWindow.window =
        SDL_CreateWindow(title, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, w * scale, h * scale, SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);
    if (!gWindow.window)
        luaL_error(L, "SDL_CreateWindow failed: %s", SDL_GetError());

    gWindow.renderer = SDL_CreateRenderer(gWindow.window, -1, SDL_RENDERER_ACCELERATED);
    if (!gWindow.renderer)
        gWindow.renderer = SDL_CreateRenderer(gWindow.window, -1, SDL_RENDERER_SOFTWARE);
    if (!gWindow.renderer)
        luaL_error(L, "SDL_CreateRenderer failed: %s", SDL_GetError());

    gWindow.texture =
        SDL_CreateTexture(gWindow.renderer, rgba ? SDL_PIXELFORMAT_RGBA32 : SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, w, h);
    if (!gWindow.texture)
        luaL_error(L, "SDL_CreateTexture failed: %s", SDL_GetError());
    gWindow.width = w;
    gWindow.height = h;
    return 0;
}

static int window_present(lua_State* L)
{
    size_t len = 0;
    void* data = luaL_checkbuffer(L, 1, &len);
    // the image may be a part of the buffer
    size_t offset = size_t(luaL_optunsigned(L, 2, 0));

    if (!gWindow.window)
        luaL_error(L, "window is not open");
    if (offset > len || len - offset < size_t(gWindow.width) * gWindow.height * 4)
        luaL_error(L, "buffer is smaller than the window (%d x %d x 4 bytes)", gWindow.width, gWindow.height);

    SDL_UpdateTexture(gWindow.texture, nullptr, (char*)data + offset, gWindow.width * 4);
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

// prints the shader of every @gpu function of the loaded chunk at the top of the stack for a target, without running it
static int emit(lua_State* L, const char* target)
{
    using Luau::Gpu::ShaderTarget;

    static const std::pair<const char*, ShaderTarget> kTargets[] = {
        {"spirv", ShaderTarget::SpirvVulkan},
        {"msl-ios", ShaderTarget::MslIos},
        {"msl-macos", ShaderTarget::MslMacos},
        {"gles", ShaderTarget::GlslEs},
    };

    bool glsl = strcmp(target, "glsl") == 0;
    ShaderTarget shaderTarget = ShaderTarget::SpirvVulkan;
    bool known = glsl;
    for (const auto& [name, t] : kTargets)
    {
        if (strcmp(target, name) == 0)
        {
            shaderTarget = t;
            known = true;
        }
    }

    if (!known)
    {
        fprintf(stderr, "unknown target '%s'; expected glsl, spirv, msl-ios, msl-macos or gles\n", target);
        return 1;
    }

    Luau::Gpu::ShaderCompiler compiler;
    int code = 0;

    for (const Luau::CodeGen::GpuKernel& kernel : Luau::CodeGen::compileGpuModule(L, -1))
    {
        printf("// @gpu %s (line %d)\n", kernel.name.c_str(), kernel.line);

        if (!kernel.reason.empty())
        {
            printf("// runs in the VM: %s\n\n", kernel.reason.c_str());
            continue;
        }

        std::string output = kernel.glsl;
        std::string error;
        std::vector<uint32_t> spirv;
        bool ok =
            glsl || (compiler.compileSpirv(kernel.glsl, kernel.name, spirv, error) && compiler.crossCompile(spirv, shaderTarget, output, error));

        if (ok)
        {
            printf("%s\n", output.c_str());
        }
        else
        {
            printf("// internal error: %s\n\n", error.c_str());
            code = 1;
        }
    }

    return code;
}

static int traceback(lua_State* L)
{
    const char* msg = lua_tostring(L, 1);
    luaL_traceback(L, L, msg ? msg : "error", 1);
    return 1;
}

int main(int argc, char** argv)
{
    FFlag::LuauGpuKernels.value = true;

    bool useGpu = true;
    bool useNative = true;
    Luau::Gpu::VulkanOptions backendOptions;
    bool pipelineCache = true;
    const char* emitTarget = nullptr;
    const char* path = nullptr;
    int scriptArg = argc;

    for (int i = 1; i < argc; ++i)
    {
        if (strcmp(argv[i], "--vm") == 0)
            useGpu = false;
        else if (strcmp(argv[i], "--no-native") == 0)
            useNative = false;
        else if (strncmp(argv[i], "--emit=", 7) == 0)
            emitTarget = argv[i] + 7;
        else if (strcmp(argv[i], "--wait-for-pipelines") == 0)
            backendOptions.waitForPipelines = true;
        else if (strcmp(argv[i], "--no-pipeline-cache") == 0)
            pipelineCache = false;
        else
        {
            path = argv[i];
            scriptArg = i + 1;
            break;
        }
    }

    if (!path)
    {
        fprintf(stderr, "usage: luau-gpu [--vm] [--no-native] [--wait-for-pipelines] [--no-pipeline-cache] script.luau [args...]\n");
        fprintf(stderr, "         --wait-for-pipelines: gpu.run waits for the pipeline of a kernel instead of returning false while it's\n");
        fprintf(stderr, "         created; --no-pipeline-cache: doesn't keep the pipelines the driver compiled in ~/.cache/luau-gpu\n");
        fprintf(stderr, "       luau-gpu --emit=<glsl|spirv|msl-ios|msl-macos|gles> script.luau\n");
        fprintf(stderr, "         prints the shader of every @gpu function for that target to stdout, without running the script\n");
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

    std::string unavailable = "the GPU is disabled (--vm)";
    if (useGpu && !emitTarget)
    {
        // the driver's compiled pipelines are kept between runs of scripts, since large kernels take it minutes
        const char* cacheHome = getenv("XDG_CACHE_HOME");
        const char* home = getenv("HOME");
        if (pipelineCache && (cacheHome || home))
        {
            std::filesystem::path dir = cacheHome ? std::filesystem::path(cacheHome) : std::filesystem::path(home) / ".cache";
            dir /= "luau-gpu";
            std::error_code ec;
            std::filesystem::create_directories(dir, ec);
            if (!ec)
                backendOptions.pipelineCachePath = (dir / "pipelines.bin").string();
        }

        gBackend = Luau::Gpu::VulkanBackend::create(unavailable, backendOptions);
        if (!gBackend)
            fprintf(stderr, "luau-gpu: no GPU backend (%s), gpu.run returns false\n", unavailable.c_str());
    }

    Luau::Gpu::openGpuLibrary(L, gBackend.get(), unavailable);

    // like the luau REPL, run the script in a sandboxed thread; besides isolating it, a sandboxed environment is what kernels
    // need, since their builtins must be the real ones
    luaL_sandbox(L);
    lua_State* T = lua_newthread(L);
    luaL_sandboxthread(T);

    lua_CompileOptions options = {};
    options.optimizationLevel = 2; // helper functions of kernels must be inlined for the GPU
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

    if (emitTarget)
        return emit(T, emitTarget);

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
