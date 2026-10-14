// This file is part of the Luau programming language and is licensed under MIT License; see LICENSE.txt for details

// only built with LUAU_BUILD_GPU, which provides the Vulkan backend and the shader compilers
#if defined(LUAU_GPU_BACKEND_TESTS)

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
LUAU_FASTFLAG(LuauGpuLibrary)

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

// runs the chunk with the backend installed, like a sandboxed script compiled at -O2, and returns the error it raised
std::string run(VulkanBackend& backend, const std::string& source, int interruptLimit = 0)
{
    ScopedFastFlag flags[] = {{FFlag::LuauGpuKernels, true}, {FFlag::LuauGpuLibrary, true}};

    lua_State* L = luaL_newstate();
    luaL_openlibs(L);

    // backendstats() returns the device memory and the kernels the backend keeps, and the memory it allocated for buffers;
    // collect() runs a full GC cycle
    lua_pushlightuserdata(L, &backend);
    lua_pushcclosure(
        L,
        [](lua_State* L)
        {
            DispatchStats stats = static_cast<VulkanBackend*>(lua_tolightuserdata(L, lua_upvalueindex(1)))->stats();
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
    backend.install(L);

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

const char* const kSquare = R"(
@gpu
local function square(i: number, out: buffer)
    buffer.writeu32(out, i * 4, i * i)
end
)";

} // namespace

TEST_SUITE_BEGIN("GpuBackend");

TEST_CASE("examples_compile_for_every_target")
{
    ScopedFastFlag flags[] = {{FFlag::LuauGpuKernels, true}, {FFlag::LuauGpuLibrary, true}};
    ShaderCompiler compiler;
    int compiled = 0;

    for (const auto& file : std::filesystem::directory_iterator(LUAU_GPU_EXAMPLES_DIR))
    {
        if (file.path().extension() != ".luau")
            continue;

        std::string source = readFile(file.path());

        lua_State* L = luaL_newstate();
        luaL_openlibs(L);
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

TEST_CASE("valid_kernels_run_on_the_gpu")
{
    ScopedFastFlag sff{FFlag::LuauGpuKernels, true};

    std::unique_ptr<VulkanBackend> backend = createBackend();
    if (!backend)
        return;

    CHECK(run(*backend, std::string(kSquare) + R"(
local b = buffer.create(40)
gpu.dispatch(square, 10, b)
assert(buffer.readu32(b, 36) == 81)
assert(gpu.status(square) == "gpu")
)") == "");

    CHECK(backend->stats().gpuDispatches == 1);
}

TEST_CASE("kernels_that_do_not_compile_run_in_the_vm")
{
    ScopedFastFlag sff{FFlag::LuauGpuKernels, true};

    std::unique_ptr<VulkanBackend> backend = createBackend();
    if (!backend)
        return;

    int reports = 0;
    std::string report;
    backend->setFallbackHandler(
        [&](const std::string& message)
        {
            reports++;
            report = message;
        }
    );

    backend->setLowering(
        [](lua_State* L, int idx)
        {
            CodeGen::GpuKernel kernel = CodeGen::compileGpu(L, idx);
            kernel.glsl = "#version 450\nvoid main() { this is not glsl; }\n";
            return kernel;
        }
    );

    CHECK(run(*backend, std::string(kSquare) + R"(
local b = buffer.create(40)
gpu.dispatch(square, 10, b)
gpu.dispatch(square, 10, b)
assert(buffer.readu32(b, 36) == 81)
local where, why = gpu.status(square)
assert(where == "vm" and string.find(why, "doesn't compile", 1, true), why)
)") == "");

    CHECK(backend->stats().gpuDispatches == 0);
    CHECK(reports == 1);
    CHECK(report.find("'square' runs in the VM") != std::string::npos);
}

TEST_CASE("kernels_over_the_device_limits_run_in_the_vm")
{
    ScopedFastFlag sff{FFlag::LuauGpuKernels, true};

    VulkanOptions options;
    options.minimumLimits = true;
    std::unique_ptr<VulkanBackend> backend = createBackend(options);
    if (!backend)
        return;

    const char* source = R"(
@gpu
local function four(i: number, a: buffer, b: buffer, c: buffer, d: buffer)
    buffer.writeu32(d, i * 4, buffer.readu32(a, i * 4) + buffer.readu32(b, i * 4) + buffer.readu32(c, i * 4))
end

@gpu
local function two(i: number, a: buffer, b: buffer)
    buffer.writeu32(b, i * 4, buffer.readu32(a, i * 4) + 1)
end
)";

    CHECK(run(*backend, std::string(source) + R"(
local a, b, c, d = buffer.create(16), buffer.create(16), buffer.create(16), buffer.create(16)
buffer.writeu32(a, 0, 1)
gpu.dispatch(four, 4, a, b, c, d)
assert(buffer.readu32(d, 0) == 1)
gpu.dispatch(two, 4, a, b)
assert(buffer.readu32(b, 0) == 2)

local where, why = gpu.status(four)
assert(where == "vm" and string.find(why, "storage buffers", 1, true), why)
assert(gpu.status(two) == "gpu")
)") == "");

    CHECK(backend->stats().gpuDispatches == 1);
}

TEST_CASE("dispatches_over_the_workgroup_count_limit_use_two_dimensions")
{
    ScopedFastFlag sff{FFlag::LuauGpuKernels, true};

    VulkanOptions options;
    options.minimumLimits = true;
    std::unique_ptr<VulkanBackend> backend = createBackend(options);
    if (!backend)
        return;


    // 5M invocations are 78125 workgroups, over the 65535 guaranteed along x
    CHECK(run(*backend, R"(
@gpu
local function index(i: number, out: buffer)
    buffer.writeu32(out, i * 4, i)
end
local n = 5000000
local b = buffer.create(n * 4)
gpu.dispatch(index, n, b)
assert(buffer.readu32(b, 0) == 0)
assert(buffer.readu32(b, 65535 * 64 * 4) == 65535 * 64)
assert(buffer.readu32(b, (n - 1) * 4) == n - 1)
)") == "");

    CHECK(backend->stats().gpuDispatches == 1);
}

TEST_CASE("buffers_over_the_storage_range_run_in_the_vm")
{
    ScopedFastFlag sff{FFlag::LuauGpuKernels, true};

    VulkanOptions options;
    options.minimumLimits = true;
    std::unique_ptr<VulkanBackend> backend = createBackend(options);
    if (!backend)
        return;


    // 128 MiB + 4 is over the minimum guaranteed storage buffer range
    CHECK(run(*backend, std::string(kSquare) + R"(
local b = buffer.create(128 * 1024 * 1024 + 4)
gpu.dispatch(square, 4, b)
assert(buffer.readu32(b, 12) == 9)
)") == "");

    CHECK(backend->stats().gpuDispatches == 0);
}

TEST_CASE("errors_on_the_gpu_rerun_the_dispatch_in_the_vm")
{
    ScopedFastFlag sff{FFlag::LuauGpuKernels, true};

    std::unique_ptr<VulkanBackend> backend = createBackend();
    if (!backend)
        return;

    const char* source = R"(
@gpu
local function outOfBounds(i: number, out: buffer)
    buffer.writeu32(out, i * 4 + 4, i)
end
)";

    std::string error = run(*backend, std::string(source) + R"(
local b = buffer.create(16)
gpu.dispatch(outOfBounds, 4, b)
)");

    CHECK(error.find("out of bounds") != std::string::npos);
    CHECK(backend->stats().fallbacks == 1);
    CHECK(backend->stats().gpuDispatches == 0);
}

TEST_CASE("infinite_loops_stop_on_the_gpu_and_are_left_to_the_host_in_the_vm")
{
    ScopedFastFlag sff{FFlag::LuauGpuKernels, true};

    std::unique_ptr<VulkanBackend> backend = createBackend();
    if (!backend)
        return;

    const char* source = R"(
@gpu
local function spin(i: number, out: buffer)
    local n = 0
    while true do
        n += 1
        buffer.writeu32(out, i * 4, n)
    end
end

gpu.dispatch(spin, 64, buffer.create(256))
)";

    // the GPU gives up once the loop budget is spent, then the VM reruns the dispatch and the host's interrupt stops it
    CHECK(run(*backend, source, 100000).find("script timed out") != std::string::npos);
    CHECK(backend->stats().fallbacks == 1);
    CHECK(backend->stats().gpuDispatches == 0);
}

TEST_CASE("kernels_come_from_the_function_itself")
{
    std::unique_ptr<VulkanBackend> backend = createBackend();
    if (!backend)
        return;

    // two kernels on one line, closures of the same function, and a function that isn't @gpu with a kernel's shape
    CHECK(run(*backend, R"(
local a, b = @gpu function(i: number, o: buffer) buffer.writeu32(o, i * 4, 1) end, @gpu function(i: number, o: buffer) buffer.writeu32(o, i * 4, 2) end
local function plain(i: number, o: buffer) buffer.writeu32(o, i * 4, 3) end

local out = buffer.create(16)
gpu.dispatch(a, 4, out)
assert(buffer.readu32(out, 12) == 1)
gpu.dispatch(b, 4, out)
assert(buffer.readu32(out, 12) == 2)
gpu.dispatch(plain, 4, out)
assert(buffer.readu32(out, 12) == 3)
assert(gpu.status(a) == "gpu" and gpu.status(b) == "gpu" and gpu.status(plain) == "vm")

for k = 1, 3 do
    local f = @gpu function(i: number, o: buffer) buffer.writeu32(o, i * 4, i + 10) end
    gpu.dispatch(f, 4, out)
    assert(buffer.readu32(out, 12) == 13)
end
)") == "");

    CHECK(backend->stats().gpuDispatches == 5);
}

TEST_CASE("functions_whose_environment_isnt_safe_run_in_the_vm")
{
    std::unique_ptr<VulkanBackend> backend = createBackend();
    if (!backend)
        return;

    // getfenv marks the environment unsafe, so the builtins a kernel uses can't be trusted anymore
    CHECK(run(*backend, std::string(kSquare) + R"(
local b = buffer.create(16)
gpu.dispatch(square, 4, b)
getfenv(1)
gpu.dispatch(square, 4, b)
assert(buffer.readu32(b, 12) == 9)
)") == "");

    CHECK(backend->stats().gpuDispatches == 1);
}

TEST_CASE("fallback_handlers_can_throw_or_use_the_backend")
{
    ScopedFastFlag sff{FFlag::LuauGpuKernels, true};

    std::unique_ptr<VulkanBackend> backend = createBackend();
    if (!backend)
        return;

    const char* source = R"(
@gpu
local function first(i: number, out: buffer)
    local t = {}
    buffer.writeu32(out, i * 4, 1)
end

@gpu
local function second(i: number, out: buffer)
    local t = {}
    buffer.writeu32(out, i * 4, 2)
end
)";

    int calls = 0;
    backend->setFallbackHandler(
        [&](const std::string& message)
        {
            calls++;
            // the handler runs outside of the backend's lock
            backend->stats();
            if (message.find("'first'") != std::string::npos)
                throw std::runtime_error("handler failure");
        }
    );

    CHECK(run(*backend, std::string(source) + R"(
local b = buffer.create(16)
gpu.dispatch(first, 4, b)
assert(buffer.readu32(b, 12) == 1)
gpu.dispatch(second, 4, b)
assert(buffer.readu32(b, 12) == 2)
)") == "");

    CHECK(calls == 2);
}

TEST_CASE("kernels_run_in_the_vm_can_dispatch_and_coroutines_can_dispatch")
{
    ScopedFastFlag sff{FFlag::LuauGpuKernels, true};

    std::unique_ptr<VulkanBackend> backend = createBackend();
    if (!backend)
        return;

    const char* source = R"(
@gpu
local function inner(i: number, b: buffer)
    buffer.writeu32(b, i * 4, i + 1)
end

-- calls gpu.dispatch, so it runs in the VM, and dispatches inner on the GPU from inside a dispatch
@gpu
local function outer(i: number, b: buffer)
    gpu.dispatch(inner, 4, b)
end
)";

    CHECK(run(*backend, std::string(source) + R"(
local b = buffer.create(16)
gpu.dispatch(outer, 3, b)
assert(buffer.readu32(b, 12) == 4)

local co = coroutine.create(function()
    gpu.dispatch(inner, 4, b)
    coroutine.yield()
    gpu.dispatch(inner, 4, b)
end)
assert(coroutine.resume(co))
assert(coroutine.resume(co))

-- kernels that yield in the VM raise an error instead of leaving the dispatch half done
local yielding = coroutine.create(function()
    gpu.dispatch(function() coroutine.yield() end, 4, b)
end)
local ok, err = coroutine.resume(yielding)
assert(not ok and string.find(err, "yield"), err)

-- an error in the middle of a dispatch in the VM leaves the backend working
assert(not pcall(gpu.dispatch, function(i) if i == 2 then error("boom") end end, 4, b))
buffer.fill(b, 0, 0)
gpu.dispatch(inner, 4, b)
assert(buffer.readu32(b, 12) == 4)
)") == "");

    CHECK(backend->stats().gpuDispatches == 6);
}

TEST_CASE("status_of_functions_without_kernels")
{
    ScopedFastFlag sff{FFlag::LuauGpuKernels, true};

    std::unique_ptr<VulkanBackend> backend = createBackend();
    if (!backend)
        return;

    CHECK(run(*backend, R"(
local where, why = gpu.status(print)
assert(where == "vm" and string.find(why, "not a Luau function"), why)
where, why = gpu.status(function() end)
assert(where == "vm" and string.find(why, "gpu attribute"), why)
)") == "");
}

TEST_CASE("buffer_copies_stay_within_the_cache_budget")
{
    ScopedFastFlag sff{FFlag::LuauGpuKernels, true};

    VulkanOptions options;
    options.cacheBytes = 1 << 20;
    std::unique_ptr<VulkanBackend> backend = createBackend(options);
    if (!backend)
        return;


    // 64 live buffers of 256 KiB, each dispatched once: 16 MiB if every copy were kept
    CHECK(run(*backend, std::string(kSquare) + R"(
local keep = {}
for k = 1, 64 do
    local b = buffer.create(256 * 1024)
    keep[k] = b
    gpu.dispatch(square, 64, b)
    assert(buffer.readu32(b, 63 * 4) == 63 * 63)
end
)") == "");

    DispatchStats stats = backend->stats();
    CHECK(stats.gpuDispatches == 64);
    CHECK(stats.cachedBytes <= options.cacheBytes);

    // a single dispatch may need more than the budget
    const char* three = R"(
@gpu
local function sum(i: number, a: buffer, b: buffer, c: buffer)
    buffer.writeu32(c, i * 4, buffer.readu32(a, i * 4) + buffer.readu32(b, i * 4))
end
)";
    CHECK(run(*backend, std::string(three) + R"(
local a, b, c = buffer.create(512 * 1024), buffer.create(512 * 1024), buffer.create(512 * 1024)
buffer.writeu32(a, 0, 2)
buffer.writeu32(b, 0, 3)
gpu.dispatch(sum, 4, a, b, c)
assert(buffer.readu32(c, 0) == 5)
)") == "");
    CHECK(backend->stats().gpuDispatches == 65);
}

TEST_CASE("buffers_reusing_an_address_are_uploaded_again")
{
    ScopedFastFlag sff{FFlag::LuauGpuKernels, true};

    std::unique_ptr<VulkanBackend> backend = createBackend();
    if (!backend)
        return;

    const char* source = R"(
@gpu
local function copy(i: number, src: buffer, dst: buffer)
    buffer.writeu32(dst, i * 4, buffer.readu32(src, i * 4))
end
)";

    // freed buffers of the same size usually get the same address back; the copy must still see the new contents
    CHECK(run(*backend, std::string(source) + R"(
local dst = buffer.create(1024)
for k = 1, 20 do
    local src = buffer.create(1024)
    buffer.fill(src, 0, k)
    gpu.dispatch(copy, 256, src, dst)
    assert(buffer.readu8(dst, 1000) == k)
    src = nil
    -- garbage to make the collector run
    for j = 1, 2000 do
        local t = table.create(64, j)
    end
end
)") == "");

    CHECK(backend->stats().gpuDispatches == 20);
}

TEST_CASE("a_dispatch_over_the_timeout_disables_the_backend_without_copying_results")
{
    ScopedFastFlag sff{FFlag::LuauGpuKernels, true};

    VulkanOptions options;
    options.timeoutMs = 1;
    options.loopBudget = 0;
    std::unique_ptr<VulkanBackend> backend = createBackend(options);
    if (!backend)
        return;

    // takes tens of milliseconds on a fast GPU, far over the timeout, and much longer in the VM, where the host's interrupt
    // stops it; short enough not to keep the GPU busy for the tests that follow
    const char* source = R"(
@gpu
local function heavy(i: number, out: buffer)
    local x = 0
    for k = 1, 20000 do
        x = (x * 1664525 + k) % 65536
    end
    buffer.writeu32(out, (i % 1024) * 4, x)
end
)";
    std::string error =
        run(*backend,
            std::string(source) + R"(
local out = buffer.create(4096)
gpu.dispatch(heavy, 65536, out)
)",
            100000);
    CHECK(error.find("script timed out") != std::string::npos);

    DispatchStats stats = backend->stats();
    CHECK(stats.gpuDispatches == 0);
    CHECK(stats.downloadedBytes == 0);

    // the backend stays disabled, including for functions never dispatched before, and nothing touches the busy device
    CHECK(run(*backend, std::string(kSquare) + R"(
local b = buffer.create(16)
gpu.dispatch(square, 4, b)
assert(buffer.readu32(b, 12) == 9)
local where, why = gpu.status(square)
assert(where == "vm" and string.find(why, "disabled"), why)
)") == "");
    CHECK(backend->stats().gpuDispatches == 0);
}

TEST_CASE("the_backend_can_be_used_from_several_threads")
{
    // set once here, since threads setting and restoring them would race
    ScopedFastFlag flags[] = {{FFlag::LuauGpuKernels, true}, {FFlag::LuauGpuLibrary, true}};

    std::unique_ptr<VulkanBackend> backend = createBackend();
    if (!backend)
        return;

    std::string fallback;
    backend->setFallbackHandler(
        [&](const std::string& message)
        {
            fallback = message;
        }
    );

    std::string script = std::string(kSquare) + R"(
for k = 1, 50 do
    local b = buffer.create(4096)
    gpu.dispatch(square, 1024, b)
    assert(buffer.readu32(b, 4092) == 1023 * 1023)
end
)";

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
    CHECK_MESSAGE(backend->stats().gpuDispatches == 200, fallback);
}

TEST_CASE("registers_dead_at_a_join_dont_need_exact_conversions")
{
    std::unique_ptr<VulkanBackend> backend = createBackend();
    if (!backend)
        return;

    // 'big' doesn't fit a float exactly, and its register is reused for floats after the branch, where it's dead
    CHECK(run(*backend, R"(
@gpu
local function k(i: number, b: buffer)
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
gpu.dispatch(k, 8, b)
assert(buffer.readu32(b, 4) == 0xfffffff1)
)") == "");

    DispatchStats stats = backend->stats();
    CHECK(stats.gpuDispatches == 1);
    CHECK(stats.fallbacks == 0);
}

TEST_CASE("a_runaway_dispatch_gives_the_gpu_back_quickly")
{
    // no loop budget: only the request to stop bounds the GPU time
    VulkanOptions options;
    options.loopBudget = 0;
    std::unique_ptr<VulkanBackend> backend = createBackend(options);
    if (!backend)
        return;

    const char* source = R"(
@gpu
local function spin(i: number, out: buffer)
    local n = 0
    while true do
        n += 1
        buffer.writeu32(out, (i % 64) * 4, n)
    end
end

gpu.dispatch(spin, 1000000, buffer.create(256))
)";

    // the GPU attempt stops shortly after abortMs, then the VM reruns the dispatch and the host's interrupt stops it
    auto start = std::chrono::steady_clock::now();
    CHECK(run(*backend, source, 1000).find("script timed out") != std::string::npos);
    double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

    MESSAGE("runaway dispatch and VM rerun took " << seconds * 1000 << " ms");
    CHECK(seconds < 0.25);
    DispatchStats stats = backend->stats();
    CHECK(stats.fallbacks == 1);

    // the backend is still usable
    CHECK(run(*backend, std::string(kSquare) + R"(
local b = buffer.create(16)
gpu.dispatch(square, 4, b)
assert(buffer.readu32(b, 12) == 9 and gpu.status(square) == "gpu")
)") == "");
}

TEST_CASE("kernels_keep_nan_infinities_and_signed_zeros")
{
    std::unique_ptr<VulkanBackend> backend = createBackend();
    if (!backend)
        return;

    // without SignedZeroInfNanPreserve, drivers fold 0/0 to 0, x/x to 1, inf-inf to 0 and -0+0 to -0
    CHECK(run(*backend, R"(
@gpu
local function k(i: number, src: buffer, out: buffer)
    local z = buffer.readf32(src, 0)
    local nz = buffer.readf32(src, 4)
    local b = buffer.readi32(src, 8)
    local h = buffer.readf32(src, 12)
    buffer.writef32(out, 0, 0 / b)
    buffer.writef32(out, 4, z / z)
    buffer.writef32(out, 8, 1 / z)
    buffer.writef32(out, 12, 1 / nz)
    buffer.writef32(out, 16, h - h)
    buffer.writef32(out, 20, h * 0)
    buffer.writef32(out, 24, nz + 0)
    buffer.writef32(out, 28, -z)
    buffer.writeu32(out, 32, if z / z ~= z / z then 1 else 0)
    buffer.writef32(out, 36, math.huge - math.huge)
end

local src = buffer.create(16)
buffer.writef32(src, 4, -0.0)
buffer.writef32(src, 12, math.huge)

local gpuOut, vmOut = buffer.create(40), buffer.create(40)
gpu.dispatch(k, 1, src, gpuOut)
assert(gpu.status(k) == "gpu")
k(0, src, vmOut)

for o = 0, 36, 4 do
    local x, y = buffer.readf32(gpuOut, o), buffer.readf32(vmOut, o)
    -- NaN payloads are platform specific, in the VM too
    local same = if x ~= x then y ~= y else buffer.readu32(gpuOut, o) == buffer.readu32(vmOut, o)
    assert(same, `offset {o}: gpu {x} vm {y}`)
end
)") == "");

    CHECK(backend->stats().gpuDispatches == 1);
}

TEST_CASE("kernels_that_crashed_the_spirv_optimizer_run_on_the_gpu")
{
    std::unique_ptr<VulkanBackend> backend = createBackend();
    if (!backend)
        return;

    // folding nested float multiplications and divisions crashed the SPIR-V optimizer, which is no longer used
    CHECK(run(*backend, R"(
@gpu
local function k(i: number, src: buffer, out: buffer)
    local s = buffer.readf32(src, 0)
    local r = vector.create(s, s, s)
    local w0 = (r * -2) / (vector.create(1, -2, 0.5) * r)
    buffer.writef32(out, i * 4, w0.x + w0.y + w0.z)
end

local src, gpuOut, vmOut = buffer.create(4), buffer.create(16), buffer.create(16)
buffer.writef32(src, 0, 3)
gpu.dispatch(k, 4, src, gpuOut)
assert(gpu.status(k) == "gpu")
for i = 0, 3 do k(i, src, vmOut) end
assert(buffer.readf32(gpuOut, 12) == buffer.readf32(vmOut, 12), buffer.readf32(gpuOut, 12))
)") == "");

    CHECK(backend->stats().gpuDispatches == 1);
}

TEST_CASE("collected_buffers_and_functions_release_what_the_backend_holds_for_them")
{
    std::unique_ptr<VulkanBackend> backend = createBackend();
    if (!backend)
        return;

    CHECK(run(*backend, R"(
local function makeKernel()
    return @gpu function(i: number, out: buffer)
        buffer.writeu32(out, i * 4, i)
    end
end

-- in a function, so that nothing on the stack keeps the buffer and the kernel alive once it returns
local function use()
    local big = buffer.create(1024 * 1024)
    gpu.dispatch(makeKernel(), 4, big)
    local cached, kernels = backendstats()
    assert(cached >= 1024 * 1024 and kernels == 1, `{cached} {kernels}`)
end
use()

-- the GC collects the buffer and the function, then the sentinels of the weak tables, whose values it still marks in the
-- cycle where their keys die; the next call into the backend frees the copy and the kernel
collect()
collect()
gpu.status(print)
local cached, _, memory = backendstats()
assert(cached == 0 and memory == 0, `{cached} {memory}`)
)") == "");

    // a function without upvalues lives as long as its prototype, so kernels are released with the state; the next call into
    // the backend frees them, leaving only the kernel of the function it is called with
    CHECK(run(*backend, "gpu.status(print)") == "");
    CHECK(backend->stats().kernels == 1);
}

TEST_CASE("fresh_buffers_every_iteration_keep_a_flat_cost_and_bounded_memory")
{
    std::unique_ptr<VulkanBackend> backend = createBackend();
    if (!backend)
        return;

    CHECK(run(*backend, R"(
@gpu
local function fill(i: number, out: buffer)
    buffer.writeu32(out, i * 4, i)
end

local n = 3000
local times = table.create(n)
local maxCached, maxMemory = 0, 0
for k = 1, n do
    local b = buffer.create(64 * 1024 + (k % 7) * 1024)
    local t = os.clock()
    gpu.dispatch(fill, 1024, b)
    times[k] = os.clock() - t
    assert(buffer.readu32(b, 4092) == 1023)
    if k % 100 == 0 then
        collect()
    end
    local cached, _, memory = backendstats()
    maxCached = math.max(maxCached, cached)
    maxMemory = math.max(maxMemory, memory)
end

-- medians, since other processes using the GPU make single dispatches slow at random
local function median(a, b)
    local t = table.create(b - a + 1)
    for k = a, b do
        table.insert(t, times[k])
    end
    table.sort(t)
    return t[#t // 2 + 1]
end

-- what is alive between two collections, not everything that was ever dispatched
assert(maxCached < 101 * 72 * 1024, `{maxCached}`)
-- buffers share memory blocks for the device copies and host mirrors, which grow with what is alive
assert(maxMemory <= 4 * 101 * 72 * 1024, `{maxMemory}`)
-- the cost of a dispatch doesn't grow with the number of buffers that came before
local first, last = median(1, n // 10), median(n - n // 10 + 1, n)
assert(last < first * 1.5 + 0.0002, `{first} {last}`)
)") == "");
}

TEST_CASE("a_batch_is_the_steps_in_order_and_downloads_only_its_outputs")
{
    std::unique_ptr<VulkanBackend> backend = createBackend();
    if (!backend)
        return;

    // three double buffered steps of a 1D cellular automaton; the intermediate generations stay on the GPU
    CHECK(run(*backend, R"(
@gpu
local function step(i: number, src: buffer, dst: buffer)
    local n = buffer.len(src) // 4
    local l = buffer.readu32(src, ((i + n - 1) % n) * 4)
    local r = buffer.readu32(src, ((i + 1) % n) * 4)
    buffer.writeu32(dst, i * 4, bit32.bxor(l, r))
end

local n = 4096
local a, b = buffer.create(n * 4), buffer.create(n * 4)
buffer.writeu32(a, (n // 2) * 4, 1)

local va, vb = buffer.create(n * 4), buffer.create(n * 4)
buffer.copy(va, 0, a)
for _ = 1, 3 do
    for i = 0, n - 1 do step(i, va, vb) end
    va, vb = vb, va
end

gpu.dispatch({ { step, n, a, b }, { step, n, b, a }, { step, n, a, b } }, { b })
for o = 0, n * 4 - 4, 4 do
    assert(buffer.readu32(b, o) == buffer.readu32(va, o))
end
)") == "");

    DispatchStats stats = backend->stats();
    CHECK(stats.gpuDispatches == 1);
    CHECK(stats.downloadedBytes == 4096 * 4);
}

TEST_CASE("an_empty_batch_can_be_the_first_call_of_a_backend")
{
    std::unique_ptr<VulkanBackend> backend = createBackend();
    if (!backend)
        return;

    // nothing has been dispatched yet, so no descriptor pool exists; scripts probe for batch support this way
    CHECK(run(*backend, std::string(kSquare) + R"(
assert(pcall(gpu.dispatch, {}, {}, {}))
local b = buffer.create(40)
gpu.dispatch({}, { b })
gpu.dispatch({ { square, 10, b } }, { b })
assert(buffer.readu32(b, 36) == 81)
)") == "");
}

TEST_CASE("buffers_stay_on_the_gpu_between_batches_until_luau_changes_them")
{
    std::unique_ptr<VulkanBackend> backend = createBackend();
    if (!backend)
        return;

    CHECK(run(*backend, R"(
@gpu
local function add(i: number, acc: buffer, inc: buffer)
    buffer.writeu32(acc, i * 4, buffer.readu32(acc, i * 4) + buffer.readu32(inc, i * 4))
end

local acc, inc = buffer.create(64), buffer.create(64)
for i = 0, 15 do buffer.writeu32(inc, i * 4, i) end

-- acc is accumulated on the GPU, and only read back when asked for
for _ = 1, 5 do
    gpu.dispatch({ { add, 16, acc, inc } })
end
gpu.dispatch({}, { acc })
assert(buffer.readu32(acc, 60) == 75)

-- Luau changes inc: listing it in 'changed' makes the next batch see it
for i = 0, 15 do buffer.writeu32(inc, i * 4, 100) end
gpu.dispatch({ { add, 16, acc, inc } }, { acc }, { inc })
assert(buffer.readu32(acc, 60) == 175)
)") == "");

    DispatchStats stats = backend->stats();
    CHECK(stats.gpuDispatches == 7);
    // acc and inc once each at first, then inc again after it changed
    CHECK(stats.uploadedBytes == 64 * 3);
}

TEST_CASE("a_batch_that_fails_on_the_gpu_reruns_in_the_vm_from_the_contents_left_by_earlier_batches")
{
    std::unique_ptr<VulkanBackend> backend = createBackend();
    if (!backend)
        return;

    CHECK(run(*backend, R"(
@gpu
local function inc(i: number, b: buffer)
    buffer.writeu32(b, i * 4, buffer.readu32(b, i * 4) + 1)
end

@gpu
local function outOfBounds(i: number, b: buffer)
    buffer.writeu32(b, i * 4 + 4, 0)
end

local b = buffer.create(16)
gpu.dispatch({ { inc, 4, b } }) -- b is 1 on the GPU only
-- inc runs on the GPU, then the second step fails: the VM reruns both from b = 1, and raises the error after writing
-- the first three words of the second step
local ok, err = pcall(gpu.dispatch, { { inc, 4, b }, { outOfBounds, 4, b } }, { b })
assert(not ok and string.find(err, "out of bounds"), err)
assert(buffer.readu32(b, 0) == 2 and buffer.readu32(b, 4) == 0 and buffer.readu32(b, 12) == 0)
)") == "");

    DispatchStats stats = backend->stats();
    CHECK(stats.gpuDispatches == 1);
    CHECK(stats.fallbacks == 1);
}

TEST_CASE("large_kernels_with_many_loops_run_on_the_gpu")
{
    std::unique_ptr<VulkanBackend> backend = createBackend();
    if (!backend)
        return;

    // about 600 lines: a chain of loops and branches, each depending on the previous result
    std::string kernel = "@gpu\nlocal function big(i: number, out: buffer)\n    local x = i\n    local y = 0.5\n";
    for (int k = 0; k < 100; k++)
    {
        kernel += "    for j = 1, " + std::to_string(k % 7 + 1) + " do\n";
        kernel += "        x = (x * " + std::to_string(k % 13 + 3) + " + j) % 65521\n";
        kernel += "        y = y * 0.75 + x / 65521\n";
        kernel += "    end\n";
        kernel += "    if x % 3 == " + std::to_string(k % 3) + " then x += " + std::to_string(k) + " end\n";
        kernel += "\n";
    }
    kernel += "    buffer.writeu32(out, i * 8, x)\n    buffer.writef32(out, i * 8 + 4, y)\nend\n";

    CHECK(run(*backend, kernel + R"(
local n = 1024
local gpuOut, vmOut = buffer.create(n * 8), buffer.create(n * 8)
gpu.dispatch(big, n, gpuOut)
assert(gpu.status(big) == "gpu", select(2, gpu.status(big)))
for i = 0, n - 1 do big(i, vmOut) end
for i = 0, n - 1 do
    assert(buffer.readu32(gpuOut, i * 8) == buffer.readu32(vmOut, i * 8))
    assert(math.abs(buffer.readf32(gpuOut, i * 8 + 4) - buffer.readf32(vmOut, i * 8 + 4)) < 1e-4)
end
)") == "");

    CHECK(backend->stats().gpuDispatches == 1);
}

TEST_SUITE_END();

#endif
