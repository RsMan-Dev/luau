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
        Luau::Gpu::ShaderCompiler compiler;
        std::vector<uint32_t> spirv;
        std::string error;
        CHECK_MESSAGE(compiler.compileSpirv(kernel.glsl, kernel.name, spirv, error), error << "\n" << kernel.glsl);
    }
#endif

    return kernel;
}

static bool contains(const std::string& s, const char* part)
{
    return s.find(part) != std::string::npos;
}

static int count(const std::string& s, const char* part)
{
    int n = 0;
    for (size_t at = s.find(part); at != std::string::npos; at = s.find(part, at + 1))
        n++;
    return n;
}

TEST_SUITE_BEGIN("CodeGenGpu");

TEST_CASE("GpuKernelBasics")
{
    GpuKernel kernel = lowerGpu(R"(
@gpu
local function scale(b: buffer, i: number)
    local o = i * 4
    buffer.writef32(b, o, buffer.readf32(b, o) * 2)
end
return scale
)");

    REQUIRE_MESSAGE(kernel.reason.empty(), kernel.reason);
    CHECK(kernel.result == CodeGenCompilationResult::Success);
    CHECK(kernel.bufferWritten == std::vector<bool>{true});
    CHECK(contains(kernel.glsl, "layout(std430, set = 0, binding = 1) coherent volatile buffer BS { uint err; int stop; uint started; } st;"));
    CHECK(contains(kernel.glsl, "uint budget;"));

    // offsets computed from the index stay exact integers, with an overflow check
    CHECK(contains(kernel.glsl, "r1a = index;"));
    CHECK(contains(kernel.glsl, "l_chk("));
}

TEST_CASE("GpuKernelControlFlowAndBuiltins")
{
    GpuKernel kernel = lowerGpu(R"(
@gpu
local function k(b: buffer, i: number)
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
    CHECK(contains(kernel.glsl, "l_fsqrt("));
}

TEST_CASE("GpuKernelGuardsFailTheRun")
{
    // the buffer write is guarded by a bounds check whose fallback fails the run, telling why
    GpuKernel kernel = lowerGpu(R"(
@gpu
local function k(b: buffer, i: number)
    local x = if i > 3 then i else 1.5
    buffer.writef32(b, 0, x * 2)
end
return k
)");

    REQUIRE_MESSAGE(kernel.reason.empty(), kernel.reason);
    CHECK(contains(kernel.glsl, "{ l_fail(1u); return; }"));
}

TEST_CASE("GpuKernelChecksOnlyWhatIsntKnown")
{
    // drivers compile kernels in a time that grows faster than their code, with each check adding to it: accesses at a
    // multiple of 4 from an offset an earlier access checked are aligned, and constants a class holds are written in it
    GpuKernel kernel = lowerGpu(R"(
@gpu
local function k(b: buffer, i: number)
    local o = buffer.readu32(b, 0) * 4
    local x = buffer.readf32(b, o) + buffer.readf32(b, o + 4) + buffer.readf32(b, o + 8)
    buffer.writef32(b, o + 12, x * 8192)
end
return k
)");

    REQUIRE_MESSAGE(kernel.reason.empty(), kernel.reason);
    CHECK(count(kernel.glsl, "l_rd0(") == 2);  // the definition and the read at o
    CHECK(count(kernel.glsl, "l_rda0(") == 4); // the definition and the reads at 0, o + 4 and o + 8
    CHECK(contains(kernel.glsl, "l_wra0("));
    CHECK(!contains(kernel.glsl, "l_i2f((8192))"));
}

TEST_CASE("GpuKernelExactIntegersInFloatRegisters")
{
    // 'x' holds a float on one path and an integer offset on the other: the integer must reach the offset exactly
    GpuKernel kernel = lowerGpu(R"(
@gpu
local function k(b: buffer, i: number)
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
local function k(b: buffer, i: number) buffer.writeu32(b, 0, i) end
return k
)",
            /* sandbox */ false
        )
            .reason == "the function's environment isn't safe (sandbox the state and don't modify builtin libraries)"
    );

    GpuKernel table = lowerGpu(R"(
@gpu
local function k(b: buffer, i: number)
    local t = { i }
    buffer.writeu32(b, 0, t[1])
end
return k
)");
    CHECK(table.result == CodeGenCompilationResult::CodeGenLoweringFailure);
    CHECK(contains(table.reason, "is not supported on the GPU (line 4)"));

    GpuKernel global = lowerGpu(R"(
@gpu
local function k(b: buffer, i: number)
    print(i)
end
return k
)");
    CHECK(global.reason == "a call that wasn't inlined is not supported on the GPU (line 4)");

    GpuKernel call = lowerGpu(R"(
local function fib(n: number): number
    return if n < 2 then n else fib(n - 1) + fib(n - 2)
end

@gpu
local function k(b: buffer, i: number)
    buffer.writeu32(b, 0, fib(i))
end
return k
)");
    // the first level of the recursion is inlined, the recursive reference to 'fib' isn't
    CHECK(call.reason == "GET_UPVALUE is not supported on the GPU (line 3)");

    GpuKernel atomic = lowerGpu(R"(
@gpu
local function k(b: buffer, i: number)
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

    CHECK(
        lowerGpu(R"(
@gpu
local function k(b: buffer, i: number, j: number) end
return k
)")
            .reason == "a function with parameters other than the buffer and the index is not supported on the GPU"
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
local function k(b: buffer, i: number)
    buffer.writef32(b, i * 4, edge(0, 0, 4, 3, i % 7, i // 7))
end
return k
)");

    REQUIRE_MESSAGE(kernel.reason.empty(), kernel.reason);
}

#if defined(LUAU_GPU_BACKEND_TESTS)

// Runs a kernel lowered from IR on the GPU with gpu.run's backend, and calls it from Luau, for index 0 .. count-1 on the same
// input, and checks that the buffers match. Returns false when there is no GPU; 'gpuError' is why the run failed, in which
// case the buffer must be untouched, and 'vmError' the error the calls from Luau raised
static bool runOnGpuAndVm(const char* source, int count, int size, bool floats, std::string& gpuError, std::string& vmError)
{
    ScopedFastFlag flag{FFlag::LuauGpuKernels, true};

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

    // the backend lowers the function itself on its first run
    GpuKernel lowered = compileGpu(L, -1);
    REQUIRE_MESSAGE(lowered.reason.empty(), lowered.reason);

    // the same input for both: a pattern of small signed and large unsigned words
    std::vector<uint32_t> input(size / 4);
    for (size_t i = 0; i < input.size(); i++)
        input[i] = uint32_t(i * 2654435761u) ^ uint32_t(i % 7 == 0 ? 0x80000000u : 0);

    std::vector<uint8_t> results[2];

    void* data = lua_newbuffer(L, size);
    memcpy(data, input.data(), size);
    lua_createtable(L, 1, 0);
    lua_pushvalue(L, -2);
    lua_rawseti(L, -2, 1);
    gpuError = backend->run(L, -3, -1, count);
    lua_pop(L, 2);
    results[0].assign((uint8_t*)data, (uint8_t*)data + size);

    data = lua_newbuffer(L, size);
    memcpy(data, input.data(), size);
    for (int i = 0; i < count && vmError.empty(); i++)
    {
        lua_pushvalue(L, -2);
        lua_pushvalue(L, -2);
        lua_pushinteger(L, i);
        if (lua_pcall(L, 2, 0, 0) != 0)
        {
            vmError = lua_tostring(L, -1);
            lua_pop(L, 1);
        }
    }
    results[1].assign((uint8_t*)data, (uint8_t*)data + size);
    lua_pop(L, 1);

    // a failed run leaves the buffer as it was
    if (!gpuError.empty())
    {
        CHECK(memcmp(results[0].data(), input.data(), size) == 0);
        return true;
    }

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
    std::string gpuError, vmError;

    if (!runOnGpuAndVm(
            R"(
@gpu
local function k(b: buffer, i: number)
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
            gpuError,
            vmError
        ))
        return;

    CHECK(gpuError.empty());
    CHECK(vmError.empty());

    runOnGpuAndVm(
        R"(
local function edge(ax: number, ay: number, bx: number, by: number, px: number, py: number)
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)
end

@gpu
local function k(b: buffer, i: number)
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
        gpuError,
        vmError
    );

    CHECK(gpuError.empty());
}

TEST_CASE("GpuKernelFailuresLeaveTheBufferUntouched")
{
    std::string gpuError, vmError;

    // an integer overflow on the GPU, where the VM computes the exact result
    if (!runOnGpuAndVm(
            R"(
@gpu
local function k(b: buffer, i: number)
    buffer.writef32(b, i * 4, i * 1073741824 * 4)
end
return k
)",
            16,
            64,
            true,
            gpuError,
            vmError
        ))
        return;

    CHECK(!gpuError.empty());
    CHECK(vmError.empty());

    // shifting by 32 or more exits to the VM builtin in native code, so it fails the run here
    gpuError.clear();
    runOnGpuAndVm(
        R"(
@gpu
local function k(b: buffer, i: number)
    buffer.writeu32(b, i * 4, bit32.rshift(0xffffffff, i))
end
return k
)",
        40,
        160,
        false,
        gpuError,
        vmError
    );

    CHECK(!gpuError.empty());

    // an out of bounds write, for which the VM raises an error
    gpuError.clear();
    runOnGpuAndVm(
        R"(
@gpu
local function k(b: buffer, i: number)
    buffer.writeu32(b, i * 4 + 4, i)
end
return k
)",
        16,
        64,
        false,
        gpuError,
        vmError
    );

    CHECK(!gpuError.empty());
    CHECK(vmError.find("buffer access out of bounds") != std::string::npos);

    // a loop over the budget, which the VM finishes
    gpuError.clear();
    vmError.clear();
    runOnGpuAndVm(
        R"(
@gpu
local function k(b: buffer, i: number)
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
        gpuError,
        vmError
    );

    CHECK(!gpuError.empty());
    CHECK(vmError.empty());
}

#endif

TEST_SUITE_END();
