// This file is part of the Luau programming language and is licensed under MIT License; see LICENSE.txt for details

// only built with LUAU_BUILD_GPU, which provides the Vulkan backend and the shader compilers
#if defined(LUAU_GPU_BACKEND_TESTS)

#include "Luau/GpuLibrary.h"
#include "Luau/GpuShaderCompiler.h"
#include "Luau/GpuVulkan.h"

#include "lua.h"
#include "luacode.h"
#include "lualib.h"

#include "ScopedFlags.h"

#include "doctest.h"

#include <stdlib.h>
#include <string.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <thread>

using namespace Luau;
using namespace Luau::Gpu;

LUAU_FASTFLAG(LuauGpuKernels)

namespace
{

const char* const kChunkName = "=gpu";

std::string readFile(const std::filesystem::path& path)
{
    std::ifstream file(path, std::ios::binary);
    std::stringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

std::unique_ptr<VulkanBackend> createBackend(const VulkanOptions& options = VulkanOptions())
{
    std::string error;
    std::unique_ptr<VulkanBackend> backend = VulkanBackend::create(error, options);
    if (!backend)
        MESSAGE("skipped, no Vulkan device: " << error);
    return backend;
}

// stops the script after this many interrupts, like a host enforcing a script timeout
int interruptsLeft = 0;

// runs the chunk with the gpu library using the backend, like a sandboxed script compiled at -O2, and returns the error it
// raised; without a backend, gpu.run returns false and "no device"
std::string run(VulkanBackend* backend, const std::string& source, int interruptLimit = 0)
{
    ScopedFastFlag flag{FFlag::LuauGpuKernels, true};

    lua_State* L = luaL_newstate();
    luaL_openlibs(L);
    openGpuLibrary(L, backend, "no device");

    // backendstats() returns the device memory and the kernels the backend keeps, and the memory it allocated for buffers;
    // collect() runs a full GC cycle
    lua_pushlightuserdata(L, backend);
    lua_pushcclosure(
        L,
        [](lua_State* L)
        {
            RunStats stats = static_cast<VulkanBackend*>(lua_tolightuserdata(L, lua_upvalueindex(1)))->stats();
            lua_pushnumber(L, double(stats.cachedBytes));
            lua_pushnumber(L, double(stats.kernels));
            lua_pushnumber(L, double(stats.memoryBytes));
            return 3;
        },
        "backendstats",
        1
    );
    lua_setglobal(L, "backendstats");
    lua_pushcfunction(
        L,
        [](lua_State* L)
        {
            lua_gc(L, LUA_GCCOLLECT, 0);
            return 0;
        },
        "collect"
    );
    lua_setglobal(L, "collect");

    luaL_sandbox(L);

    if (interruptLimit > 0)
    {
        interruptsLeft = interruptLimit;
        lua_callbacks(L)->interrupt = [](lua_State* L, int gc)
        {
            if (gc < 0 && --interruptsLeft <= 0)
                luaL_error(L, "script timed out");
        };
    }

    lua_State* T = lua_newthread(L);
    luaL_sandboxthread(T);

    lua_CompileOptions options = {};
    options.optimizationLevel = 2;
    options.debugLevel = 1;
    size_t bytecodeSize = 0;
    char* bytecode = luau_compile(source.data(), source.size(), &options, &bytecodeSize);
    int status = luau_load(T, kChunkName, bytecode, bytecodeSize, 0);
    free(bytecode);

    if (status == 0)
        status = lua_pcall(T, 0, 0, 0);

    std::string error = status == 0 ? "" : lua_tostring(T, -1);
    lua_close(L);
    return error;
}

std::string run(VulkanBackend& backend, const std::string& source, int interruptLimit = 0)
{
    return run(&backend, source, interruptLimit);
}

// square(b, i) writes i * i at word i; same(a, b) compares two buffers byte for byte
const char* const kSquare = R"(
@gpu
local function square(b: buffer, i: number)
    buffer.writeu32(b, i * 4, i * i)
    return b
end

local function same(a: buffer, b: buffer)
    return buffer.len(a) == buffer.len(b) and buffer.readstring(a, 0, buffer.len(a)) == buffer.readstring(b, 0, buffer.len(b))
end

local function copy(b: buffer)
    local c = buffer.create(buffer.len(b))
    buffer.copy(c, 0, b)
    return c
end
)";

} // namespace

TEST_SUITE_BEGIN("GpuBackend");

TEST_CASE("examples_compile_for_every_target")
{
    ScopedFastFlag flag{FFlag::LuauGpuKernels, true};
    ShaderCompiler compiler;
    int compiled = 0;

    for (const auto& file : std::filesystem::directory_iterator(LUAU_GPU_EXAMPLES_DIR))
    {
        if (file.path().extension() != ".luau")
            continue;

        std::string source = readFile(file.path());

        lua_State* L = luaL_newstate();
        luaL_openlibs(L);
        openGpuLibrary(L, nullptr);
        luaL_sandbox(L);

        lua_CompileOptions options = {};
        options.optimizationLevel = 2;
        size_t bytecodeSize = 0;
        char* bytecode = luau_compile(source.data(), source.size(), &options, &bytecodeSize);
        REQUIRE(luau_load(L, kChunkName, bytecode, bytecodeSize, 0) == 0);
        free(bytecode);

        std::vector<CodeGen::GpuKernel> kernels = CodeGen::compileGpuModule(L, -1);
        lua_close(L);

        for (const CodeGen::GpuKernel& kernel : kernels)
        {
            if (!kernel.reason.empty())
                continue;

            INFO(file.path().filename().string() << ": " << kernel.name);

            std::vector<uint32_t> spirv;
            std::string error;
            REQUIRE_MESSAGE(compiler.compileSpirv(kernel.glsl, kernel.name, spirv, error), error);

            std::string text;
            CHECK_MESSAGE(compiler.crossCompile(spirv, ShaderTarget::SpirvVulkan, text, error), error);
            CHECK(text.find("OpEntryPoint GLCompute") != std::string::npos);

            for (ShaderTarget target : {ShaderTarget::MslIos, ShaderTarget::MslMacos})
            {
                std::string msl;
                CHECK_MESSAGE(compiler.crossCompile(spirv, target, msl, error), error);
                CHECK(msl.find("kernel void main0(") != std::string::npos);
                CHECK(msl.find("constant PC& pc [[buffer(" + std::to_string(kMslPushConstantBuffer) + ")]]") != std::string::npos);
            }

            std::string gles;
            CHECK_MESSAGE(compiler.crossCompile(spirv, ShaderTarget::GlslEs, gles, error), error);
            CHECK(gles.find("#version 310 es") != std::string::npos);
            CHECK(gles.find("uniform PC pc;") != std::string::npos);

            compiled++;
        }
    }

    // the examples define several kernels each; this catches a directory that wasn't found
    CHECK(compiled >= 10);
}

TEST_CASE("invalid_spirv_input_is_rejected")
{
    ShaderCompiler compiler;
    std::string source, error;
    CHECK(!compiler.crossCompile({0x07230203, 0x00010000, 0, 1, 0}, ShaderTarget::MslIos, source, error));
    CHECK(!error.empty());

    for (ShaderTarget target : {ShaderTarget::SpirvVulkan, ShaderTarget::MslIos, ShaderTarget::MslMacos, ShaderTarget::GlslEs})
    {
        error.clear();
        CHECK(!compiler.crossCompile({}, target, source, error));
        CHECK(!error.empty());
    }
}

TEST_CASE("run_calls_the_function_on_each_buffer_for_each_index")
{
    std::unique_ptr<VulkanBackend> backend = createBackend();
    if (!backend)
        return;

    // the index is 0-based, the value returned is ignored, and buffers of different sizes run in one call
    CHECK(run(*backend, std::string(kSquare) + R"(
local a, b = buffer.create(40), buffer.create(16)
local ok, err = gpu.run(square, { a, b }, 4)
assert(ok == true and err == nil, err)
for i = 0, 3 do
    assert(buffer.readu32(a, i * 4) == i * i and buffer.readu32(b, i * 4) == i * i)
end
assert(buffer.readu32(a, 16) == 0)

-- nothing to run
assert(gpu.run(square, {}, 4) == true)
assert(gpu.run(square, { a }, 0) == true)
)") == "");

    CHECK(backend->stats().succeeded == 1);
}

TEST_CASE("large_indices_are_exact_integers")
{
    VulkanOptions options;
    options.minimumLimits = true;
    std::unique_ptr<VulkanBackend> backend = createBackend(options);
    if (!backend)
        return;

    // 2^24 + 1000 runs, over what f32 counts exactly and over the 65535 workgroups guaranteed along x; the last runs write
    // their index, and its neighbors would collide if it were rounded
    CHECK(run(*backend, R"(
@gpu
local function last(b: buffer, i: number)
    local base = 16777216
    if i >= base then
        buffer.writeu32(b, (i - base) * 4, i)
    end
end

local b = buffer.create(1000 * 4)
assert(gpu.run(last, { b }, 16777216 + 1000))
for k = 0, 999 do
    assert(buffer.readu32(b, k * 4) == 16777216 + k, k)
end
)") == "");
}

TEST_CASE("functions_that_cant_run_on_the_gpu_return_false_and_leave_buffers_untouched")
{
    std::unique_ptr<VulkanBackend> backend = createBackend();
    if (!backend)
        return;

    CHECK(run(*backend, std::string(kSquare) + R"(
local b = buffer.create(16)
buffer.writeu32(b, 0, 7)
local before = copy(b)

local function plain(b: buffer, i: number) buffer.writeu32(b, i * 4, 1) end
@gpu local function usesTable(b: buffer, i: number) local t = { i } buffer.writeu32(b, i * 4, t[1]) end
@gpu local function tooManyParameters(b: buffer, i: number, j: number) buffer.writeu32(b, i * 4, j) end
local up = 3
up += 1
@gpu local function usesUpvalue(b: buffer, i: number) buffer.writeu32(b, i * 4, up) end

for _, f in { plain, usesTable, tooManyParameters, usesUpvalue, print } do
    local ok, err = gpu.run(f, { b }, 4)
    assert(ok == false and err == "Function cannot run in GPU", err)
    assert(same(b, before))
end

-- calling them from Luau still works
usesTable(b, 1)
assert(buffer.readu32(b, 4) == 1)
)") == "");

    CHECK(backend->stats().succeeded == 0);
}

TEST_CASE("functions_whose_shader_does_not_compile_cant_run")
{
    std::unique_ptr<VulkanBackend> backend = createBackend();
    if (!backend)
        return;

    backend->setLowering(
        [](lua_State* L, int idx)
        {
            CodeGen::GpuKernel kernel = CodeGen::compileGpu(L, idx);
            kernel.glsl = "#version 450\nvoid main() { this is not glsl; }\n";
            return kernel;
        }
    );

    CHECK(run(*backend, std::string(kSquare) + R"(
local b = buffer.create(16)
local ok, err = gpu.run(square, { b }, 4)
assert(not ok and err == "Function cannot run in GPU", err)
assert(buffer.readu32(b, 12) == 0)
)") == "");
}

TEST_CASE("functions_whose_environment_isnt_safe_cant_run")
{
    std::unique_ptr<VulkanBackend> backend = createBackend();
    if (!backend)
        return;

    // getfenv marks the environment unsafe, so the builtins a kernel uses can't be trusted anymore
    CHECK(run(*backend, std::string(kSquare) + R"(
local b = buffer.create(16)
assert(gpu.run(square, { b }, 4))
getfenv(1)
buffer.fill(b, 0, 0)
local ok, err = gpu.run(square, { b }, 4)
assert(not ok and err == "Function cannot run in GPU", err)
assert(buffer.readu32(b, 12) == 0)
)") == "");

    CHECK(backend->stats().succeeded == 1);
}

TEST_CASE("invalid_arguments_raise_errors")
{
    std::unique_ptr<VulkanBackend> backend = createBackend();
    if (!backend)
        return;

    CHECK(run(*backend, std::string(kSquare) + R"(
local b = buffer.create(16)
local function fails(...)
    local ok, err = pcall(gpu.run, ...)
    assert(not ok, "expected an error")
    return err
end
assert(string.find(fails(nil, { b }, 4), "function expected"))
assert(string.find(fails(square, b, 4), "table expected"))
assert(string.find(fails(square, { b, "b" }, 4), "buffer expected at index 2, got string"))
assert(string.find(fails(square, { b }, -1), "between 0 and 2147483647"))
assert(string.find(fails(square, { b }, 1.5), "between 0 and 2147483647"))
assert(string.find(fails(square, { b }, 2^31), "between 0 and 2147483647"))
assert(string.find(fails(square, { b }), "number expected"))
assert(buffer.readu32(b, 12) == 0)
)") == "");
}

TEST_CASE("without_a_device_run_returns_false_and_why")
{
    CHECK(run(nullptr, std::string(kSquare) + R"(
local b = buffer.create(16)
local ok, err = gpu.run(square, { b }, 4)
assert(ok == false and err == "no device", err)
assert(buffer.readu32(b, 12) == 0)
)") == "");
}

TEST_CASE("errors_on_the_gpu_return_false_and_leave_every_buffer_untouched")
{
    std::unique_ptr<VulkanBackend> backend = createBackend();
    if (!backend)
        return;

    // the second buffer is too small for the last index: no buffer of the call changes, and a later run starts from the
    // Luau contents again
    CHECK(run(*backend, std::string(kSquare) + R"(
@gpu
local function inc(b: buffer, i: number)
    buffer.writeu32(b, i * 4, buffer.readu32(b, i * 4) + 1)
end

local a, b = buffer.create(16), buffer.create(12)
buffer.writeu32(a, 0, 5)
buffer.writeu32(b, 8, 9)
local beforeA, beforeB = copy(a), copy(b)

local ok, err = gpu.run(inc, { a, b }, 4)
assert(not ok and string.find(err, "failed on the GPU", 1, true), err)
assert(same(a, beforeA) and same(b, beforeB))

assert(gpu.run(inc, { a }, 4))
assert(buffer.readu32(a, 0) == 6 and buffer.readu32(a, 12) == 1)
)") == "");

    RunStats stats = backend->stats();
    CHECK(stats.failures == 1);
    CHECK(stats.succeeded == 1);
}

TEST_CASE("integer_overflows_and_inexact_conversions_fail_the_run")
{
    std::unique_ptr<VulkanBackend> backend = createBackend();
    if (!backend)
        return;

    // the VM computes these exactly in doubles, which 32-bit integers on the GPU can't
    CHECK(run(*backend, std::string(kSquare) + R"(
@gpu
local function overflow(b: buffer, i: number)
    buffer.writeu32(b, 0, (buffer.readu32(b, 0) + i) * 65536 * 65536 // 65536 // 65536)
end

local b = buffer.create(4)
buffer.writeu32(b, 0, 3)
local ok, err = gpu.run(overflow, { b }, 1)
assert(not ok and string.find(err, "failed on the GPU", 1, true), err)
assert(buffer.readu32(b, 0) == 3)
)") == "");
}

TEST_CASE("infinite_loops_fail_the_run")
{
    std::unique_ptr<VulkanBackend> backend = createBackend();
    if (!backend)
        return;

    CHECK(run(*backend, R"(
@gpu
local function spin(b: buffer, i: number)
    local n = 0
    while true do
        n += 1
        buffer.writeu32(b, i * 4, n)
    end
end

local b = buffer.create(256)
local ok, err = gpu.run(spin, { b }, 64)
assert(not ok and string.find(err, "failed on the GPU", 1, true), err)
assert(buffer.readu32(b, 0) == 0)
)") == "");

    CHECK(backend->stats().failures == 1);
}

TEST_CASE("a_runaway_run_gives_the_gpu_back_quickly")
{
    // no loop budget: only the request to stop bounds the GPU time
    VulkanOptions options;
    options.loopBudget = 0;
    std::unique_ptr<VulkanBackend> backend = createBackend(options);
    if (!backend)
        return;

    auto start = std::chrono::steady_clock::now();
    CHECK(run(*backend, R"(
@gpu
local function spin(b: buffer, i: number)
    local n = 0
    while true do
        n += 1
        buffer.writeu32(b, (i % 64) * 4, n)
    end
end

local ok, err = gpu.run(spin, { buffer.create(256) }, 1000000)
assert(not ok and string.find(err, "failed on the GPU", 1, true), err)
)") == "");
    double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

    // the run stops shortly after abortMs
    MESSAGE("runaway run took " << seconds * 1000 << " ms");
    CHECK(seconds < 0.25);

    // the backend is still usable
    CHECK(run(*backend, std::string(kSquare) + R"(
local b = buffer.create(16)
assert(gpu.run(square, { b }, 4))
assert(buffer.readu32(b, 12) == 9)
)") == "");
}

TEST_CASE("buffers_over_the_storage_range_fail_the_run")
{
    VulkanOptions options;
    options.minimumLimits = true;
    std::unique_ptr<VulkanBackend> backend = createBackend(options);
    if (!backend)
        return;

    // 128 MiB + 4 is over the minimum guaranteed storage buffer range
    CHECK(run(*backend, std::string(kSquare) + R"(
local b = buffer.create(128 * 1024 * 1024 + 4)
local ok, err = gpu.run(square, { buffer.create(16), b }, 4)
assert(not ok and string.find(err, "storage buffer range", 1, true), err)
assert(buffer.readu32(b, 12) == 0)
)") == "");
}

TEST_CASE("a_run_over_the_timeout_disables_the_backend_without_copying_results")
{
    VulkanOptions options;
    options.timeoutMs = 1;
    options.loopBudget = 0;
    std::unique_ptr<VulkanBackend> backend = createBackend(options);
    if (!backend)
        return;

    // takes tens of milliseconds on a fast GPU, far over the timeout; short enough not to keep the GPU busy for the tests
    // that follow
    CHECK(run(*backend, R"(
@gpu
local function heavy(b: buffer, i: number)
    local x = 0
    for k = 1, 20000 do
        x = (x * 1664525 + k) % 65536
    end
    buffer.writeu32(b, (i % 1024) * 4, x)
end

local b = buffer.create(4096)
local ok, err = gpu.run(heavy, { b }, 65536)
assert(not ok and string.find(err, "disabled", 1, true), err)
assert(buffer.readu32(b, 0) == 0)
)") == "");

    RunStats stats = backend->stats();
    CHECK(stats.succeeded == 0);
    CHECK(stats.downloadedBytes == 0);

    // the backend stays disabled, including for functions never run before, and nothing touches the busy device
    CHECK(run(*backend, std::string(kSquare) + R"(
local b = buffer.create(16)
local ok, err = gpu.run(square, { b }, 4)
assert(not ok and string.find(err, "disabled", 1, true), err)
)") == "");
    CHECK(backend->stats().succeeded == 0);
}

TEST_CASE("kernels_come_from_the_function_itself")
{
    std::unique_ptr<VulkanBackend> backend = createBackend();
    if (!backend)
        return;

    // two kernels on one line, and closures of the same function
    CHECK(run(*backend, R"(
local a, b = @gpu function(o: buffer, i: number) buffer.writeu32(o, i * 4, 1) end, @gpu function(o: buffer, i: number) buffer.writeu32(o, i * 4, 2) end

local out = buffer.create(16)
assert(gpu.run(a, { out }, 4))
assert(buffer.readu32(out, 12) == 1)
assert(gpu.run(b, { out }, 4))
assert(buffer.readu32(out, 12) == 2)

for k = 1, 3 do
    local f = @gpu function(o: buffer, i: number) buffer.writeu32(o, i * 4, i + 10) end
    assert(gpu.run(f, { out }, 4))
    assert(buffer.readu32(out, 12) == 13)
end
)") == "");

    CHECK(backend->stats().succeeded == 5);
}

TEST_CASE("each_run_mutates_its_buffer_in_the_order_of_the_list")
{
    std::unique_ptr<VulkanBackend> backend = createBackend();
    if (!backend)
        return;

    // a buffer listed twice runs twice, the second time on the results of the first
    CHECK(run(*backend, R"(
@gpu
local function double(b: buffer, i: number)
    buffer.writeu32(b, i * 4, buffer.readu32(b, i * 4) * 2 + 1)
end

local a, b = buffer.create(16), buffer.create(16)
assert(gpu.run(double, { a, b, a }, 4))
assert(buffer.readu32(a, 12) == 3 and buffer.readu32(b, 12) == 1)
)") == "");
}

TEST_CASE("buffers_stay_on_the_gpu_and_are_uploaded_again_only_when_luau_changes_them")
{
    std::unique_ptr<VulkanBackend> backend = createBackend();
    if (!backend)
        return;

    CHECK(run(*backend, R"(
@gpu
local function inc(b: buffer, i: number)
    buffer.writeu32(b, i * 4, buffer.readu32(b, i * 4) + i)
end

local b = buffer.create(64)
for _ = 1, 5 do
    assert(gpu.run(inc, { b }, 16))
end
assert(buffer.readu32(b, 60) == 75)

-- Luau changes the buffer: the next run sees it
buffer.writeu32(b, 60, 100)
assert(gpu.run(inc, { b }, 16))
assert(buffer.readu32(b, 60) == 115)
)") == "");

    RunStats stats = backend->stats();
    CHECK(stats.succeeded == 6);
    // once at first, then again after Luau changed it; every run downloads its results
    CHECK(stats.uploadedBytes == 64 * 2);
    CHECK(stats.downloadedBytes == 64 * 6);
}

TEST_CASE("only_the_parts_kernels_write_are_downloaded_and_the_others_match_luau")
{
    std::unique_ptr<VulkanBackend> backend = createBackend();
    if (!backend)
        return;

    // buffers come back in parts of 64 KB: this one has three whole parts and a last one of 10 bytes. The kernel writes the
    // last word of part 0, the first word of part 1 and a word of the last part, never part 2
    CHECK(run(*backend, std::string(kSquare) + R"(
local PART = 65536
local SIZE = 3 * PART + 10

@gpu
local function sparse(b: buffer, i: number)
    local o = if i == 0 then 65532 elseif i == 1 then 65536 else 196612
    buffer.writeu32(b, o, bit32.bxor(buffer.readu32(b, o), 0x5a5a5a5a + i))
end

-- the same calls from Luau, on another copy
local function reference(b: buffer, times: number)
    for _ = 1, times do
        for i = 0, 2 do
            sparse(b, i)
        end
    end
end

local b = buffer.create(SIZE)
for o = 0, SIZE - 1 do
    buffer.writeu8(b, o, o % 251)
end
local expected = copy(b)

local function runs(list, times)
    assert(gpu.run(sparse, list, 3))
    reference(expected, times)
    assert(same(b, expected))
end

runs({ b }, 1)

-- Luau changes a part the kernel writes and one it doesn't: both changes reach the GPU, and stay
buffer.writeu32(b, 2 * PART + 100, 0xdeadbeef)
buffer.writeu32(b, PART + 8, 0x12345678)
buffer.writeu8(b, SIZE - 1, 7)
buffer.copy(expected, 0, b)
runs({ b }, 1)

-- listed twice, the second run sees what the first wrote
runs({ b, b }, 2)
)") == "");

    // three runs of the kernel's three parts, the last one 10 bytes: part 2 never comes back
    CHECK(backend->stats().downloadedBytes == 3 * (65536 * 2 + 10));

    // a run that fails after writing parts of the buffers leaves them all as they were
    CHECK(run(*backend, std::string(kSquare) + R"(
@gpu
local function partly(b: buffer, i: number)
    buffer.writeu32(b, i * 4 * 16384, 1)
    if i == 3 and buffer.len(b) < 4 * 65536 then
        buffer.writeu32(b, buffer.len(b), 1)
    end
end

local a, b = buffer.create(4 * 65536), buffer.create(3 * 65536 + 4)
buffer.fill(a, 0, 9)
buffer.fill(b, 0, 5)
local beforeA, beforeB = copy(a), copy(b)
local ok, err = gpu.run(partly, { a, b }, 4)
assert(not ok and string.find(err, "failed on the GPU", 1, true), err)
assert(same(a, beforeA) and same(b, beforeB))

-- and the next run starts again from the Luau contents
assert(gpu.run(partly, { a }, 4))
assert(buffer.readu32(a, 0) == 1 and buffer.readu32(a, 4) == 0x09090909 and buffer.readu32(a, 3 * 65536) == 1)
)") == "");
}

TEST_CASE("buffers_reusing_an_address_are_uploaded_again")
{
    std::unique_ptr<VulkanBackend> backend = createBackend();
    if (!backend)
        return;

    // freed buffers of the same size usually get the same address back; the run must still see the new contents
    CHECK(run(*backend, R"(
@gpu
local function incWords(b: buffer, i: number)
    buffer.writeu32(b, i * 4, buffer.readu32(b, i * 4) + 1)
end

for k = 1, 20 do
    local b = buffer.create(1024)
    buffer.fill(b, 0, k)
    assert(gpu.run(incWords, { b }, 256))
    assert(buffer.readu8(b, 1000) == k + 1 and buffer.readu8(b, 1001) == k)
    b = nil
    -- garbage to make the collector run
    for j = 1, 2000 do
        local t = table.create(64, j)
    end
end
)") == "");

    CHECK(backend->stats().succeeded == 20);
}

TEST_CASE("atomics_are_atomic_on_the_gpu_and_plain_in_luau")
{
    std::unique_ptr<VulkanBackend> backend = createBackend();
    if (!backend)
        return;

    // invocations sharing words of the buffer must use atomics
    CHECK(run(*backend, R"(
@gpu
local function histogram(b: buffer, i: number)
    gpu.atomicadd(b, (i % 16) * 4, 1)
    gpu.atomicmin(b, 64, (i * 7919) % 10007 + 5)
end

local b = buffer.create(68)
buffer.writeu32(b, 64, 0xffffffff)
assert(gpu.run(histogram, { b }, 16000))
for k = 0, 15 do
    assert(buffer.readu32(b, k * 4) == 1000)
end
assert(buffer.readu32(b, 64) == 5)

-- from Luau, they return the previous value
local c = buffer.create(8)
assert(gpu.atomicadd(c, 0, 3) == 0 and gpu.atomicadd(c, 0, -1) == 3 and buffer.readu32(c, 0) == 2)
assert(gpu.atomicmin(c, 4, 7) == 0 and buffer.readu32(c, 4) == 0)
assert(not pcall(gpu.atomicadd, c, 8, 1))
)") == "");
}

TEST_CASE("coroutines_can_run")
{
    std::unique_ptr<VulkanBackend> backend = createBackend();
    if (!backend)
        return;

    CHECK(run(*backend, std::string(kSquare) + R"(
local b = buffer.create(16)
local co = coroutine.create(function()
    assert(gpu.run(square, { b }, 4))
    coroutine.yield()
    assert(gpu.run(square, { b }, 4))
end)
assert(coroutine.resume(co))
assert(coroutine.resume(co))
assert(buffer.readu32(b, 12) == 9)
)") == "");

    CHECK(backend->stats().succeeded == 2);
}

TEST_CASE("buffer_copies_stay_within_the_cache_budget")
{
    VulkanOptions options;
    options.cacheBytes = 1 << 20;
    std::unique_ptr<VulkanBackend> backend = createBackend(options);
    if (!backend)
        return;

    // 64 live buffers of 256 KiB, each run once: 16 MiB if every copy were kept
    CHECK(run(*backend, std::string(kSquare) + R"(
local keep = {}
for k = 1, 64 do
    local b = buffer.create(256 * 1024)
    keep[k] = b
    assert(gpu.run(square, { b }, 64))
    assert(buffer.readu32(b, 63 * 4) == 63 * 63)
end
)") == "");

    RunStats stats = backend->stats();
    CHECK(stats.succeeded == 64);
    CHECK(stats.cachedBytes <= options.cacheBytes);

    // a single run may need more than the budget
    CHECK(run(*backend, std::string(kSquare) + R"(
local a, b, c = buffer.create(512 * 1024), buffer.create(512 * 1024), buffer.create(512 * 1024)
assert(gpu.run(square, { a, b, c }, 4))
assert(buffer.readu32(c, 12) == 9)
)") == "");
    CHECK(backend->stats().succeeded == 65);
}

TEST_CASE("the_backend_can_be_used_from_several_threads")
{
    std::unique_ptr<VulkanBackend> backend = createBackend();
    if (!backend)
        return;

    std::string script = std::string(kSquare) + R"(
for k = 1, 50 do
    local b = buffer.create(4096)
    assert(gpu.run(square, { b }, 1024))
    assert(buffer.readu32(b, 4092) == 1023 * 1023)
end
)";

    // set once here, since threads setting and restoring it would race
    ScopedFastFlag flag{FFlag::LuauGpuKernels, true};

    std::string errors[4];
    std::vector<std::thread> threads;
    for (std::string& error : errors)
        threads.emplace_back(
            [&]
            {
                error = run(*backend, script);
            }
        );
    for (std::thread& t : threads)
        t.join();

    for (const std::string& error : errors)
        CHECK(error == "");
    CHECK(backend->stats().succeeded == 200);
}

TEST_CASE("registers_dead_at_a_join_dont_need_exact_conversions")
{
    std::unique_ptr<VulkanBackend> backend = createBackend();
    if (!backend)
        return;

    // 'big' doesn't fit a float exactly, and its register is reused for floats after the branch, where it's dead
    CHECK(run(*backend, R"(
@gpu
local function k(b: buffer, i: number)
    if i % 2 == 0 then
        local big = buffer.readu32(b, 0)
        buffer.writeu32(b, 4 + i * 4, big)
    end
    local acc = i * 0.5
    for j = 1, 3 do
        acc = acc * 1.5 + j
    end
    buffer.writef32(b, 64 + i * 4, acc)
end

local b = buffer.create(128)
buffer.writeu32(b, 0, 0xfffffff1)
assert(gpu.run(k, { b }, 8))
assert(buffer.readu32(b, 4) == 0xfffffff1)
)") == "");
}

TEST_CASE("devices_that_break_the_kernel_contract_are_refused")
{
#if !defined(_WIN32)
    // llvmpipe, Mesa's CPU implementation, reports SignedZeroInfNanPreserve but folds inf - inf to 0; the loader picks it when
    // it's the only driver listed
    const char* const kDirs[] = {"/run/opengl-driver/share/vulkan/icd.d", "/usr/share/vulkan/icd.d", "/etc/vulkan/icd.d"};
    std::string icd;
    for (const char* dir : kDirs)
    {
        std::error_code ec;
        for (const std::filesystem::directory_entry& e : std::filesystem::directory_iterator(dir, ec))
        {
            if (e.path().filename().string().rfind("lvp_icd", 0) == 0)
                icd = e.path().string();
        }
    }

    if (icd.empty())
    {
        MESSAGE("skipped, llvmpipe isn't installed");
        return;
    }

    const char* previous = getenv("VK_DRIVER_FILES");
    std::string saved = previous ? previous : "";
    setenv("VK_DRIVER_FILES", icd.c_str(), 1);

    std::string error;
    std::unique_ptr<VulkanBackend> backend = VulkanBackend::create(error);

    if (previous)
        setenv("VK_DRIVER_FILES", saved.c_str(), 1);
    else
        unsetenv("VK_DRIVER_FILES");

    CHECK(!backend);
    bool refused = error.find("is refused:") != std::string::npos && error.find("NaN, infinities, signed zeros and denormal") != std::string::npos;
    CHECK_MESSAGE(refused, error);
#endif

    // a device that keeps its contract is accepted, and runs kernels
    std::unique_ptr<VulkanBackend> accepted = createBackend();
    if (!accepted)
        return;

    CHECK(run(*accepted, std::string(kSquare) + R"(
local b = buffer.create(16)
assert(gpu.run(square, { b }, 4))
assert(buffer.readu32(b, 12) == 9)
)") == "");
}

TEST_CASE("kernels_keep_nan_infinities_and_signed_zeros")
{
    std::unique_ptr<VulkanBackend> backend = createBackend();
    if (!backend)
        return;

    // without SignedZeroInfNanPreserve, drivers fold 0/0 to 0, x/x to 1, inf-inf to 0 and -0+0 to -0; inputs are in the first
    // 16 bytes, results after them
    CHECK(run(*backend, R"(
@gpu
local function k(b: buffer, i: number)
    local z = buffer.readf32(b, 0)
    local nz = buffer.readf32(b, 4)
    local n = buffer.readi32(b, 8)
    local h = buffer.readf32(b, 12)
    buffer.writef32(b, 16, 0 / n)
    buffer.writef32(b, 20, z / z)
    buffer.writef32(b, 24, 1 / z)
    buffer.writef32(b, 28, 1 / nz)
    buffer.writef32(b, 32, h - h)
    buffer.writef32(b, 36, h * 0)
    buffer.writef32(b, 40, nz + 0)
    buffer.writef32(b, 44, -z)
    buffer.writeu32(b, 48, if z / z ~= z / z then 1 else 0)
    buffer.writef32(b, 52, math.huge - math.huge)
end

local onGpu = buffer.create(56)
buffer.writef32(onGpu, 4, -0.0)
buffer.writef32(onGpu, 12, math.huge)
local inLuau = buffer.create(56)
buffer.copy(inLuau, 0, onGpu)

assert(gpu.run(k, { onGpu }, 1))
k(inLuau, 0)

for o = 16, 52, 4 do
    local x, y = buffer.readf32(onGpu, o), buffer.readf32(inLuau, o)
    -- NaN payloads are platform specific, in the VM too
    local same = if x ~= x then y ~= y else buffer.readu32(onGpu, o) == buffer.readu32(inLuau, o)
    assert(same, `offset {o}: gpu {x} vm {y}`)
end
)") == "");
}

TEST_CASE("vector_math_matches_luau_bit_for_bit")
{
    std::unique_ptr<VulkanBackend> backend = createBackend();
    if (!backend)
        return;

    // the VM computes vectors in f32, rounding each operation correctly: the GPU must give the same bits, for every exponent,
    // denormal numbers, infinities and NaN included. Drivers may reorder operations ((s - t) + t was folded to s), flush
    // denormal numbers, and round divisions and square roots within a few ulps; none of that may show
    CHECK(run(*backend, R"(
local N = 131072

@gpu
local function k(b: buffer, i: number)
    local o = i * 64
    local x = vector.create(buffer.readf32(b, o), buffer.readf32(b, o + 4), buffer.readf32(b, o + 8))
    local y = vector.create(buffer.readf32(b, o + 12), 1.5, buffer.readf32(b, o + 4))
    local q = x / y
    local n = vector.normalize(x)
    local c = vector.cross(x, y)
    local r = vector.create(0.25, x.x - y.x, 2) + vector.create(x.x, y.x, 1)
    buffer.writef32(b, o + 16, q.x)
    buffer.writef32(b, o + 20, q.z)
    buffer.writef32(b, o + 24, vector.magnitude(x))
    buffer.writef32(b, o + 28, n.y)
    buffer.writef32(b, o + 32, vector.dot(x, y))
    buffer.writef32(b, o + 36, c.z)
    buffer.writef32(b, o + 40, (x * y).x)
    buffer.writef32(b, o + 44, r.y)
    buffer.writef32(b, o + 48, (vector.create(x.x, 0, 0) * vector.create(0.5, 0, 0)).x)
    buffer.writef32(b, o + 52, vector.floor(x / y).y)
end

local seed = 7
local function rnd()
    seed = (seed * 1103515245 + 12345) % 2147483648
    return seed
end

local onGpu = buffer.create(N * 64)
for i = 0, N - 1 do
    for w = 0, 3 do
        -- random bit patterns: every exponent, half of them near 1, where ties and exact results are common
        local bits = (rnd() * 2 + rnd() % 2) % 4294967296
        if rnd() % 2 == 0 then
            bits = bit32.bor(bit32.band(bits, 0x807fffff), bit32.lshift(117 + rnd() % 20, 23))
        end
        buffer.writeu32(onGpu, i * 64 + w * 4, bits)
    end
end
local inLuau = buffer.create(N * 64)
buffer.copy(inLuau, 0, onGpu)

assert(gpu.run(k, { onGpu }, N))
for i = 0, N - 1 do
    k(inLuau, i)
end

for i = 0, N - 1 do
    for o = i * 64 + 16, i * 64 + 52, 4 do
        local x, y = buffer.readf32(onGpu, o), buffer.readf32(inLuau, o)
        -- NaN payloads are platform specific, in the VM too
        local same = if x ~= x then y ~= y else buffer.readu32(onGpu, o) == buffer.readu32(inLuau, o)
        assert(same, `index {i} word {(o - i * 64) // 4}: gpu {x} luau {y}`)
    end
end
)") == "");
}

TEST_CASE("division_and_square_roots_stay_exact_whatever_the_hardware_gives")
{
    // the hardware's division and square root only seed the exact ones: shaders whose seeds are a few units off, or hundreds
    // (where a 32-bit remainder would wrap around), must still give the VM's bits
    for (const char* error : {"1.0000003", "1.00005", "0.9998", "2.0"})
    {
        INFO("hardware results scaled by " << std::string(error));

        std::unique_ptr<VulkanBackend> backend = createBackend();
        if (!backend)
            return;

        std::string scale = error;
        backend->setLowering(
            [scale](lua_State* L, int idx)
            {
                CodeGen::GpuKernel kernel = CodeGen::compileGpu(L, idx);
                for (std::string from : {"    float q = a / b;", "    float q = sqrt(a);"})
                {
                    size_t at = kernel.glsl.find(from);
                    REQUIRE(at != std::string::npos);
                    std::string to = from.substr(0, from.size() - 1) + " * " + scale + ";";
                    kernel.glsl.replace(at, from.size(), to);
                }
                return kernel;
            }
        );

        CHECK(run(*backend, R"(
local N = 65536

@gpu
local function k(b: buffer, i: number)
    local o = i * 16
    local x, y = buffer.readf32(b, o), buffer.readf32(b, o + 4)
    buffer.writef32(b, o + 8, (vector.create(x, 1, 1) / vector.create(y, 1, 1)).x)
    buffer.writef32(b, o + 12, vector.magnitude(vector.create(x, 0, 0)))
end

local seed = 11
local function rnd()
    seed = (seed * 1103515245 + 12345) % 2147483648
    return seed
end

local onGpu = buffer.create(N * 16)
for i = 0, N - 1 do
    for w = 0, 1 do
        local bits = (rnd() * 2 + rnd() % 2) % 4294967296
        if rnd() % 2 == 0 then
            bits = bit32.bor(bit32.band(bits, 0x807fffff), bit32.lshift(117 + rnd() % 20, 23))
        end
        buffer.writeu32(onGpu, i * 16 + w * 4, bits)
    end
end
local inLuau = buffer.create(N * 16)
buffer.copy(inLuau, 0, onGpu)

assert(gpu.run(k, { onGpu }, N))
for i = 0, N - 1 do
    k(inLuau, i)
end

for i = 0, N - 1 do
    for o = i * 16 + 8, i * 16 + 12, 4 do
        local x, y = buffer.readf32(onGpu, o), buffer.readf32(inLuau, o)
        local same = if x ~= x then y ~= y else buffer.readu32(onGpu, o) == buffer.readu32(inLuau, o)
        assert(same, `index {i} word {(o - i * 16) // 4}: gpu {x} luau {y}`)
    end
end
)") == "");
    }
}

TEST_CASE("integers_that_floats_cant_hold_fail_the_run")
{
    std::unique_ptr<VulkanBackend> backend = createBackend();
    if (!backend)
        return;

    // the VM computes these exactly in doubles; rounding them to f32 would give another answer
    CHECK(run(*backend, std::string(kSquare) + R"(
@gpu
local function converted(b: buffer, i: number)
    buffer.writef32(b, 4, math.floor(buffer.readu32(b, 0) % 2.5))
end

@gpu
local function literal(b: buffer, i: number)
    buffer.writef32(b, 4, 16777217 * buffer.readf32(b, 8))
end

for _, k in { converted, literal } do
    local b = buffer.create(12)
    buffer.writeu32(b, 0, 4294967294)
    buffer.writef32(b, 8, 0.5)
    local ok, err = gpu.run(k, { b }, 1)
    assert(not ok and string.find(err, "failed on the GPU", 1, true), err)
end

-- below 2^24 they're exact, and divisions are correctly rounded: 37 / 37 is 1, not 0.99999994
@gpu
local function quotient(b: buffer, i: number)
    buffer.writef32(b, 4, (buffer.readu32(b, 0) / 37) // 0.25)
end
local b = buffer.create(8)
buffer.writeu32(b, 0, 37)
assert(gpu.run(quotient, { b }, 1))
assert(buffer.readf32(b, 4) == 4, buffer.readf32(b, 4))
)") == "");
}

TEST_CASE("float_results_that_overflow_where_doubles_dont_fail_the_run")
{
    std::unique_ptr<VulkanBackend> backend = createBackend();
    if (!backend)
        return;

    // scalar math is f64 in the VM: -1e10 * 4.6e28 is finite there, times 0 it's -0; in f32 it's -inf, times 0 NaN
    CHECK(run(*backend, std::string(kSquare) + R"(
@gpu
local function product(b: buffer, i: number)
    local t, d, z = buffer.readf32(b, 0), buffer.readf32(b, 4), buffer.readf32(b, 8)
    buffer.writef32(b, 12, (t * d) * z)
end

@gpu
local function power(b: buffer, i: number)
    -- e^100 is finite in f64
    buffer.writef32(b, 12, math.exp(buffer.readf32(b, 8) + 100))
end

local b = buffer.create(16)
buffer.writef32(b, 0, -1e10)
buffer.writef32(b, 4, 4.6e28)
local before = copy(b)
for _, k in { product, power } do
    local ok, err = gpu.run(k, { b }, 1)
    assert(not ok and string.find(err, "float overflow", 1, true), err)
    assert(same(b, before))
end

-- overflows are noted and fail the invocation once it returns: one that loops on an infinity is still stopped, and what it
-- wrote, including the infinity, never leaves the GPU
@gpu
local function spin(b: buffer, i: number)
    local x = buffer.readf32(b, 4)
    while x == x do
        x = x * 1e10
        buffer.writef32(b, 12, x)
    end
end
for _, k in { spin } do
    local ok, err = gpu.run(k, { b }, 1)
    assert(not ok and string.find(err, "failed on the GPU", 1, true), err)
    assert(same(b, before))
end

-- infinities the VM gets too, and underflows, are results
@gpu
local function legit(b: buffer, i: number)
    local x = buffer.readf32(b, 0)
    buffer.writef32(b, 4, 1 / x)
    buffer.writef32(b, 8, (x * 1e-20 + 1e-30) ^ 2)
end
local c = buffer.create(12)
assert(gpu.run(legit, { c }, 1))
assert(buffer.readf32(c, 4) == math.huge and buffer.readf32(c, 8) == 0)
)") == "");
}

TEST_CASE("kernels_that_crashed_the_spirv_optimizer_run_on_the_gpu")
{
    std::unique_ptr<VulkanBackend> backend = createBackend();
    if (!backend)
        return;

    // folding nested float multiplications and divisions crashed the SPIR-V optimizer, which is no longer used
    CHECK(run(*backend, R"(
@gpu
local function k(b: buffer, i: number)
    local s = buffer.readf32(b, 0)
    local r = vector.create(s, s, s)
    local w0 = (r * -2) / (vector.create(1, -2, 0.5) * r)
    buffer.writef32(b, 4 + i * 4, w0.x + w0.y + w0.z)
end

local onGpu, inLuau = buffer.create(20), buffer.create(20)
buffer.writef32(onGpu, 0, 3)
buffer.writef32(inLuau, 0, 3)
assert(gpu.run(k, { onGpu }, 4))
for i = 0, 3 do k(inLuau, i) end
assert(buffer.readf32(onGpu, 16) == buffer.readf32(inLuau, 16), buffer.readf32(onGpu, 16))
)") == "");
}

TEST_CASE("collected_buffers_and_functions_release_what_the_backend_holds_for_them")
{
    std::unique_ptr<VulkanBackend> backend = createBackend();
    if (!backend)
        return;

    CHECK(run(*backend, R"(
local function makeKernel()
    return @gpu function(b: buffer, i: number)
        buffer.writeu32(b, i * 4, i)
    end
end

-- in a function, so that nothing on the stack keeps the buffer and the kernel alive once it returns
local function use()
    local big = buffer.create(1024 * 1024)
    assert(gpu.run(makeKernel(), { big }, 4))
    local cached, kernels = backendstats()
    assert(cached >= 1024 * 1024 and kernels == 1, `{cached} {kernels}`)
end
use()

-- the GC collects the buffer and the function, then the sentinels of the weak tables, whose values it still marks in the
-- cycle where their keys die; the end of the next run frees the copy and the kernel
collect()
collect()
assert(gpu.run(makeKernel(), {}, 0))
local cached, _, memory = backendstats()
assert(cached == 0 and memory == 0, `{cached} {memory}`)
)") == "");

    // a function without upvalues lives as long as its prototype, so kernels are released with the state; the next run
    // frees them, leaving only the kernel of the function it runs
    CHECK(run(*backend, "assert(gpu.run(@gpu function(b: buffer, i: number) end, {}, 0))") == "");
    CHECK(backend->stats().kernels == 1);
}

TEST_CASE("fresh_buffers_every_iteration_keep_a_flat_cost_and_bounded_memory")
{
    std::unique_ptr<VulkanBackend> backend = createBackend();
    if (!backend)
        return;

    CHECK(run(*backend, R"(
@gpu
local function fill(b: buffer, i: number)
    buffer.writeu32(b, i * 4, i)
end

local n = 3000
local times = table.create(n)
local maxCached, maxMemory = 0, 0
for k = 1, n do
    local b = buffer.create(64 * 1024 + (k % 7) * 1024)
    local t = os.clock()
    assert(gpu.run(fill, { b }, 1024))
    times[k] = os.clock() - t
    assert(buffer.readu32(b, 4092) == 1023)
    if k % 100 == 0 then
        collect()
    end
    local cached, _, memory = backendstats()
    maxCached = math.max(maxCached, cached)
    maxMemory = math.max(maxMemory, memory)
end

-- medians, since other processes using the GPU make single runs slow at random
local function median(a, b)
    local t = table.create(b - a + 1)
    for k = a, b do
        table.insert(t, times[k])
    end
    table.sort(t)
    return t[#t // 2 + 1]
end

-- what is alive between two collections, not everything that ever ran
assert(maxCached < 101 * 72 * 1024, `{maxCached}`)
-- buffers share memory blocks for the device copies and host mirrors, which grow with what is alive
assert(maxMemory <= 4 * 101 * 72 * 1024, `{maxMemory}`)
-- the cost of a run doesn't grow with the number of buffers that came before
local first, last = median(1, n // 10), median(n - n // 10 + 1, n)
assert(last < first * 1.5 + 0.0002, `{first} {last}`)
)") == "");
}

TEST_CASE("large_kernels_with_many_loops_run_on_the_gpu")
{
    std::unique_ptr<VulkanBackend> backend = createBackend();
    if (!backend)
        return;

    // about 600 lines: a chain of loops and branches, each depending on the previous result
    std::string kernel = "@gpu\nlocal function big(b: buffer, i: number)\n    local x = i\n    local y = 0.5\n";
    for (int k = 0; k < 100; k++)
    {
        kernel += "    for j = 1, " + std::to_string(k % 7 + 1) + " do\n";
        kernel += "        x = (x * " + std::to_string(k % 13 + 3) + " + j) % 65521\n";
        kernel += "        y = y * 0.75 + x / 65521\n";
        kernel += "    end\n";
        kernel += "    if x % 3 == " + std::to_string(k % 3) + " then x += " + std::to_string(k) + " end\n";
        kernel += "\n";
    }
    kernel += "    buffer.writeu32(b, i * 8, x)\n    buffer.writef32(b, i * 8 + 4, y)\nend\n";

    CHECK(run(*backend, kernel + R"(
local n = 1024
local onGpu, inLuau = buffer.create(n * 8), buffer.create(n * 8)
assert(gpu.run(big, { onGpu }, n))
for i = 0, n - 1 do big(inLuau, i) end
for i = 0, n - 1 do
    assert(buffer.readu32(onGpu, i * 8) == buffer.readu32(inLuau, i * 8))
    assert(math.abs(buffer.readf32(onGpu, i * 8 + 4) - buffer.readf32(inLuau, i * 8 + 4)) < 1e-4)
end
)") == "");

    CHECK(backend->stats().succeeded == 1);
}

TEST_SUITE_END();

#endif
