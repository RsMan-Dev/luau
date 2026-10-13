// This file is part of the Luau programming language and is licensed under MIT License; see LICENSE.txt for details
#include "lua.h"
#include "lualib.h"

#include "Luau/CodeGenGpu.h"
#include "Luau/Compiler.h"

#if defined(LUAU_GPU_BACKEND_TESTS)
#include "Luau/GpuShaderCompiler.h"
#include "Luau/GpuVulkan.h"

#include <math.h>
#include <string.h>
#endif

#include "doctest.h"
#include "ScopedFlags.h"

#include <memory>

LUAU_FASTFLAG(LuauGpuKernels)
LUAU_FASTFLAG(LuauGpuLibrary)

using namespace Luau::CodeGen;

// compiles a chunk that returns a function and lowers that function for the GPU
static GpuKernel lowerGpu(const char* source, bool sandbox = true)
{
    ScopedFastFlag luauGpuKernels{FFlag::LuauGpuKernels, true};

    Luau::CompileOptions options;
    options.optimizationLevel = 2;
    std::string bytecode = Luau::compile(source, options);

    std::unique_ptr<lua_State, void (*)(lua_State*)> state(luaL_newstate(), lua_close);
    lua_State* L = state.get();

    luaL_openlibs(L);
    if (sandbox)
        luaL_sandbox(L);

    REQUIRE(luau_load(L, "=gpu", bytecode.data(), bytecode.size(), 0) == 0);
    REQUIRE(lua_pcall(L, 0, 1, 0) == 0);

    GpuKernel kernel = compileGpu(L, -1);

#if defined(LUAU_GPU_BACKEND_TESTS)
    // every shader the lowering produces must be valid for Vulkan
    if (kernel.reason.empty())
    {
        Luau::Gpu::Kernel shader;
        shader.name = "kernel";
        shader.glsl = kernel.glsl;
        shader.bufferWritten = kernel.bufferWritten;

        Luau::Gpu::ShaderCompiler compiler;
        std::vector<uint32_t> spirv;
        std::string error;
        CHECK_MESSAGE(compiler.compileSpirv(shader, spirv, error), error << "\n" << kernel.glsl);
    }
#endif

    return kernel;
}

static bool contains(const std::string& s, const char* part)
{
    return s.find(part) != std::string::npos;
}

TEST_SUITE_BEGIN("CodeGenGpu");

TEST_CASE("GpuKernelBasics")
{
    GpuKernel kernel = lowerGpu(R"(
@gpu
local function scale(i: number, src: buffer, dst: buffer)
    local o = i * 4
    buffer.writef32(dst, o, buffer.readf32(src, o) * 2)
end
return scale
)");

    REQUIRE_MESSAGE(kernel.reason.empty(), kernel.reason);
    CHECK(kernel.result == CodeGenCompilationResult::Success);
    CHECK(kernel.bufferWritten == std::vector<bool>{false, true});
    CHECK(contains(kernel.glsl, "layout(std430, set = 0, binding = 2) buffer BS { uint err; } st;"));
    CHECK(contains(kernel.glsl, "uint budget;"));

    // offsets computed from the invocation index stay exact integers
    CHECK(contains(kernel.glsl, "l_muli("));
}

TEST_CASE("GpuKernelControlFlowAndBuiltins")
{
    GpuKernel kernel = lowerGpu(R"(
@gpu
local function k(i: number, b: buffer)
    local s = 0
    for j = 1, i % 10 do
        if j % 3 == 0 then
            continue
        end
        s += j
    end
    local x = math.sqrt(i) + math.floor(i / 3) + math.max(i, 2) + math.abs(i - 5)
    local u = bit32.bxor(buffer.readu32(b, 0), bit32.lshift(i, 2))
    while s > 100 do
        s //= 2
    end
    buffer.writef32(b, i * 4, x + s)
    buffer.writeu32(b, i * 4 + 4, bit32.band(u, 0xff00ff00))
end
return k
)");

    REQUIRE_MESSAGE(kernel.reason.empty(), kernel.reason);
    CHECK(kernel.bufferWritten == std::vector<bool>{true});

    // loops count against the budget where native code checks for interrupts
    CHECK(contains(kernel.glsl, "++l_steps > pc.budget"));
    CHECK(contains(kernel.glsl, "l_sqrt("));
    CHECK(contains(kernel.glsl, "l_idivi("));
}

TEST_CASE("GpuKernelGuardsFailTheDispatch")
{
    // the result of the helper isn't known to be a number, so the IR guards it with a tag check and a fallback
    GpuKernel kernel = lowerGpu(R"(
@gpu
local function k(i: number, b: buffer)
    local x = if i > 3 then i else 1.5
    buffer.writef32(b, 0, x * 2)
end
return k
)");

    REQUIRE_MESSAGE(kernel.reason.empty(), kernel.reason);
    CHECK(contains(kernel.glsl, "{ l_fail(); return; }"));
}

TEST_CASE("GpuKernelExactIntegersInFloatRegisters")
{
    // 'x' holds a float on one path and an integer offset on the other: the integer must reach the offset exactly
    GpuKernel kernel = lowerGpu(R"(
@gpu
local function k(i: number, b: buffer)
    local x = i / 2
    if i > 3 then
        x = i * 4
    end
    buffer.writeu32(b, x, 1)
end
return k
)");

    REQUIRE_MESSAGE(kernel.reason.empty(), kernel.reason);
    CHECK(contains(kernel.glsl, "l_i2f("));
    CHECK(contains(kernel.glsl, "l_f2i("));
}

TEST_CASE("GpuKernelRejections")
{
    CHECK(
        lowerGpu(R"(
local function f(i: number) return i end
return f
)")
            .reason == "the function doesn't have the gpu attribute"
    );

    CHECK(
        lowerGpu(
            R"(
@gpu
local function k(i: number, b: buffer) buffer.writeu32(b, 0, i) end
return k
)",
            /* sandbox */ false
        )
            .reason == "the function's environment isn't safe (sandbox the state and don't modify builtin libraries)"
    );

    GpuKernel table = lowerGpu(R"(
@gpu
local function k(i: number, b: buffer)
    local t = { i }
    buffer.writeu32(b, 0, t[1])
end
return k
)");
    CHECK(table.result == CodeGenCompilationResult::CodeGenLoweringFailure);
    CHECK(contains(table.reason, "is not supported on the GPU (line 4)"));

    GpuKernel global = lowerGpu(R"(
@gpu
local function k(i: number, b: buffer)
    print(i)
end
return k
)");
    CHECK(global.reason == "GET_CACHED_IMPORT is not supported on the GPU (line 4)");

    GpuKernel call = lowerGpu(R"(
local function fib(n: number): number
    return if n < 2 then n else fib(n - 1) + fib(n - 2)
end

@gpu
local function k(i: number, b: buffer)
    buffer.writeu32(b, 0, fib(i))
end
return k
)");
    // the first level of the recursion is inlined, the recursive reference to 'fib' isn't
    CHECK(call.reason == "GET_UPVALUE is not supported on the GPU (line 3)");

    GpuKernel atomic = lowerGpu(R"(
@gpu
local function k(i: number, b: buffer)
    buffer.writeu8(b, i, 1)
end
return k
)");
    CHECK(contains(atomic.reason, "BUFFER_WRITEI8 is not supported on the GPU (line 4)"));

    CHECK(
        lowerGpu(R"(
@gpu
local function k(...) end
return k
)")
            .reason == "a variadic function is not supported on the GPU (line 2)"
    );
}

TEST_CASE("GpuKernelInlinedHelpers")
{
    // at -O2, local helpers are inlined into the kernel like they are for native code
    GpuKernel kernel = lowerGpu(R"(
local function edge(ax: number, ay: number, bx: number, by: number, px: number, py: number)
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)
end

@gpu
local function k(i: number, b: buffer)
    buffer.writef32(b, i * 4, edge(0, 0, 4, 3, i % 7, i // 7))
end
return k
)");

    REQUIRE_MESSAGE(kernel.reason.empty(), kernel.reason);
}

#if defined(LUAU_GPU_BACKEND_TESTS)

// Runs a kernel lowered from IR on the GPU and in the VM over the same input and checks that the buffers match
// Returns false when there is no GPU; 'fallbacks' is the number of dispatches the backend discarded and reran in the VM
static bool runOnGpuAndVm(const char* source, int count, int size, bool floats, uint64_t& fallbacks, std::string& vmError)
{
    ScopedFastFlag flags[] = {{FFlag::LuauGpuKernels, true}, {FFlag::LuauGpuLibrary, true}};

    std::string error;
    std::unique_ptr<Luau::Gpu::VulkanBackend> backend = Luau::Gpu::VulkanBackend::create(error);
    if (!backend)
    {
        MESSAGE("skipped, no Vulkan device: " << error);
        return false;
    }

    Luau::CompileOptions options;
    options.optimizationLevel = 2;
    std::string bytecode = Luau::compile(source, options);

    std::unique_ptr<lua_State, void (*)(lua_State*)> state(luaL_newstate(), lua_close);
    lua_State* L = state.get();
    luaL_openlibs(L);
    luaL_sandbox(L);

    REQUIRE(luau_load(L, "=gpu", bytecode.data(), bytecode.size(), 0) == 0);
    REQUIRE(lua_pcall(L, 0, 1, 0) == 0);

    GpuKernel lowered = compileGpu(L, -1);
    REQUIRE_MESSAGE(lowered.reason.empty(), lowered.reason);

    lua_Debug ar = {};
    lua_getinfo(L, -1, "s", &ar);

    std::vector<Luau::Gpu::Kernel> kernels(1);
    kernels[0].name = "kernel";
    kernels[0].line = ar.linedefined;
    kernels[0].glsl = lowered.glsl;
    kernels[0].bufferWritten = lowered.bufferWritten;
    backend->loadKernels("=gpu", kernels);
    REQUIRE_MESSAGE(kernels[0].reason.empty(), kernels[0].reason);

    // the same input for both runs: a pattern of small signed and large unsigned words
    std::vector<uint32_t> input(size / 4);
    for (size_t i = 0; i < input.size(); i++)
        input[i] = uint32_t(i * 2654435761u) ^ uint32_t(i % 7 == 0 ? 0x80000000u : 0);

    std::vector<uint8_t> results[2];
    for (int run = 0; run < 2; run++)
    {
        if (run == 0)
            backend->install(L);
        else
            lua_setgpubackend(L, nullptr);

        lua_getglobal(L, "gpu");
        lua_getfield(L, -1, "dispatch");
        lua_pushvalue(L, -3);
        lua_pushinteger(L, count);
        void* data = lua_newbuffer(L, size);
        memcpy(data, input.data(), size);

        if (lua_pcall(L, 3, 0, 0) != 0)
        {
            vmError = lua_tostring(L, -1);
            lua_pop(L, 1);
        }

        lua_pop(L, 1);
        results[run].assign((uint8_t*)data, (uint8_t*)data + size);
    }

    fallbacks = backend->stats().fallbacks;

    for (int o = 0; o < size; o += 4)
    {
        uint32_t a, b;
        memcpy(&a, &results[0][o], 4);
        memcpy(&b, &results[1][o], 4);
        if (a == b)
            continue;

        float fa, fb;
        memcpy(&fa, &a, 4);
        memcpy(&fb, &b, 4);
        bool close = floats && fabsf(fa - fb) <= 1e-5f * fmaxf(1.0f, fabsf(fb));
        CHECK_MESSAGE(close, "offset " << o << ": gpu " << a << " (" << fa << ") vm " << b << " (" << fb << ")");
        if (!close)
            break;
    }

    return true;
}

TEST_CASE("GpuKernelMatchesTheVm")
{
    uint64_t fallbacks = 0;
    std::string vmError;

    if (!runOnGpuAndVm(
            R"(
@gpu
local function k(i: number, b: buffer)
    local a = i - 500
    local d = (i % 7) - 3
    local o = i * 16
    local q = if d ~= 0 then a // d else 0
    local r = if d ~= 0 then a % d else 0
    buffer.writei32(b, o, q)
    buffer.writei32(b, o + 4, r)
    buffer.writei32(b, o + 8, math.max(a, d) * 3 - math.abs(a))
    buffer.writeu32(b, o + 12, bit32.bor(bit32.rshift(buffer.readu32(b, o + 12), i % 32), bit32.lshift(i, 3)))
end
return k
)",
            1000,
            16000,
            false,
            fallbacks,
            vmError
        ))
        return;

    CHECK(fallbacks == 0);
    CHECK(vmError.empty());

    runOnGpuAndVm(
        R"(
local function edge(ax: number, ay: number, bx: number, by: number, px: number, py: number)
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)
end

@gpu
local function k(i: number, b: buffer)
    local x = (i - 500) / 37
    local s = 0
    for j = 1, i % 20 do
        if j % 3 == 0 then
            continue
        end
        s += j * x
    end
    local o = i * 16
    buffer.writef32(b, o, math.sqrt(math.abs(x)) + math.floor(x) + math.round(x * 3))
    buffer.writef32(b, o + 4, s)
    buffer.writef32(b, o + 8, edge(0, 0, 40, 25, i % 40, i // 40))
    buffer.writef32(b, o + 12, if x > 1 then x // 0.25 else x % 1.5)
end
return k
)",
        1000,
        16000,
        true,
        fallbacks,
        vmError
    );

    CHECK(fallbacks == 0);
}

TEST_CASE("GpuKernelFailuresRerunInTheVm")
{
    uint64_t fallbacks = 0;
    std::string vmError;

    // an integer overflow on the GPU: the VM computes the exact result instead
    if (!runOnGpuAndVm(
            R"(
@gpu
local function k(i: number, b: buffer)
    buffer.writef32(b, i * 4, i * 1073741824 * 4)
end
return k
)",
            16,
            64,
            true,
            fallbacks,
            vmError
        ))
        return;

    CHECK(fallbacks == 1);
    CHECK(vmError.empty());

    // shifting by 32 or more exits to the VM builtin in native code, so it fails the dispatch here
    runOnGpuAndVm(
        R"(
@gpu
local function k(i: number, b: buffer)
    buffer.writeu32(b, i * 4, bit32.rshift(0xffffffff, i))
end
return k
)",
        40,
        160,
        false,
        fallbacks,
        vmError
    );

    CHECK(fallbacks == 1);

    // an out of bounds write: the VM raises the error
    runOnGpuAndVm(
        R"(
@gpu
local function k(i: number, b: buffer)
    buffer.writeu32(b, i * 4 + 4, i)
end
return k
)",
        16,
        64,
        false,
        fallbacks,
        vmError
    );

    CHECK(fallbacks == 1);
    CHECK(vmError.find("buffer access out of bounds") != std::string::npos);

    // a loop over the budget, which the VM then finishes
    vmError.clear();
    runOnGpuAndVm(
        R"(
@gpu
local function k(i: number, b: buffer)
    local n = 0
    while n < 20000000 do
        n += 1
    end
    buffer.writeu32(b, i * 4, n + i)
end
return k
)",
        2,
        8,
        false,
        fallbacks,
        vmError
    );

    CHECK(fallbacks == 1);
    CHECK(vmError.empty());
}

#endif

TEST_SUITE_END();
