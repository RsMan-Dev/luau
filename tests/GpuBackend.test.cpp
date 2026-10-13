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

// runs the chunk with the backend installed and returns the error it raised, if any; the caller enables LuauGpuKernels
std::string run(VulkanBackend& backend, const std::string& source, int interruptLimit = 0)
{
    lua_State* L = luaL_newstate();
    luaL_openlibs(L);
    luaopen_gpu(L);
    lua_pop(L, 1);
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

    size_t bytecodeSize = 0;
    char* bytecode = luau_compile(source.data(), source.size(), nullptr, &bytecodeSize);
    int status = luau_load(L, kChunkName, bytecode, bytecodeSize, 0);
    free(bytecode);

    if (status == 0)
        status = lua_pcall(L, 0, 0, 0);

    std::string error = status == 0 ? "" : lua_tostring(L, -1);
    lua_close(L);
    return error;
}

std::vector<Kernel> load(VulkanBackend& backend, const std::string& source)
{
    return backend.load(kChunkName, source.data(), source.size());
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
    ScopedFastFlag sff{FFlag::LuauGpuKernels, true};
    ShaderCompiler compiler;
    int compiled = 0;

    for (const auto& file : std::filesystem::directory_iterator(LUAU_GPU_EXAMPLES_DIR))
    {
        if (file.path().extension() != ".luau")
            continue;

        std::string source = readFile(file.path());
        for (const Kernel& kernel : translate(source.data(), source.size()))
        {
            if (!kernel.reason.empty())
                continue;

            INFO(file.path().filename().string() << ": " << kernel.name);

            std::vector<uint32_t> spirv;
            std::string error;
            REQUIRE_MESSAGE(compiler.compileSpirv(kernel, spirv, error), error);

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

    std::vector<Kernel> kernels = load(*backend, kSquare);
    REQUIRE(kernels.size() == 1);
    CHECK(kernels[0].reason.empty());

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

    std::vector<Kernel> kernels = translate(kSquare, strlen(kSquare));
    REQUIRE(kernels.size() == 1);
    kernels[0].glsl = "#version 450\nvoid main() { this is not glsl; }\n";
    backend->loadKernels(kChunkName, kernels);
    CHECK(kernels[0].reason.find("doesn't compile") != std::string::npos);

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

    std::vector<Kernel> kernels = load(*backend, source);
    REQUIRE(kernels.size() == 2);

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

    load(*backend, kSquare);

    // 5M invocations are 78125 workgroups, over the 65535 guaranteed along x
    CHECK(run(*backend, std::string(kSquare) + R"(
local n = 5000000
local b = buffer.create(n * 4)
gpu.dispatch(square, n, b)
assert(buffer.readu32(b, 0) == 0)
assert(buffer.readu32(b, 65535 * 64 * 4) == (65535 * 64) ^ 2 % 2 ^ 32)
assert(buffer.readu32(b, (n - 1) * 4) == ((n - 1) * (n - 1)) % 2 ^ 32)
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

    load(*backend, kSquare);

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
    load(*backend, source);

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

    std::vector<Kernel> kernels = load(*backend, source);
    REQUIRE(kernels.size() == 1);
    REQUIRE(kernels[0].reason.empty());

    // the GPU gives up once the loop budget is spent, then the VM reruns the dispatch and the host's interrupt stops it
    CHECK(run(*backend, source, 100000).find("script timed out") != std::string::npos);
    CHECK(backend->stats().fallbacks == 1);
    CHECK(backend->stats().gpuDispatches == 0);
}

TEST_CASE("reloading_a_chunk_drops_kernels_that_are_gone")
{
    ScopedFastFlag sff{FFlag::LuauGpuKernels, true};

    std::unique_ptr<VulkanBackend> backend = createBackend();
    if (!backend)
        return;

    load(*backend, kSquare);

    // the same chunk, edited: the function now at the kernel's line has the same shape but isn't @gpu
    const char* edited = R"(
local function square(i: number, out: buffer)
    buffer.writeu32(out, i * 4, 7)
end
)";
    CHECK(load(*backend, edited).empty());

    CHECK(run(*backend, std::string(edited) + R"(
local b = buffer.create(16)
gpu.dispatch(square, 4, b)
assert(buffer.readu32(b, 12) == 7)
assert(gpu.status(square) == "vm")
)") == "");

    CHECK(backend->stats().gpuDispatches == 0);
}

TEST_CASE("functions_compiled_from_another_source_dont_run_registered_kernels")
{
    ScopedFastFlag sff{FFlag::LuauGpuKernels, true};

    std::unique_ptr<VulkanBackend> backend = createBackend();
    if (!backend)
        return;

    load(*backend, kSquare);

    // the host registered one source but runs another one under the same chunk name
    CHECK(run(*backend, R"(
@gpu
local function other(i: number, out: buffer, extra: buffer)
    buffer.writeu32(out, i * 4, 5)
end
local b = buffer.create(16)
gpu.dispatch(other, 4, b, buffer.create(4))
assert(buffer.readu32(b, 12) == 5)
assert(gpu.status(other) == "vm")
)") == "");

    CHECK(backend->stats().gpuDispatches == 0);
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
    load(*backend, source);

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
    load(*backend, source);

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
assert(where == "vm" and string.find(why, "not marked @gpu"), why)
where, why = gpu.status(function() end)
assert(where == "vm" and string.find(why, "not marked @gpu"), why)
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

    load(*backend, kSquare);

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
    load(*backend, three);
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
    load(*backend, source);

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

    // takes a few hundred milliseconds on a fast GPU and much longer in the VM, where the host's interrupt stops it
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
    std::vector<Kernel> kernels = load(*backend, source);
    REQUIRE(kernels.size() == 1);
    REQUIRE(kernels[0].reason.empty());

    std::string error =
        run(*backend,
            std::string(source) + R"(
local out = buffer.create(4096)
gpu.dispatch(heavy, 1048576, out)
)",
            100000);
    CHECK(error.find("script timed out") != std::string::npos);

    DispatchStats stats = backend->stats();
    CHECK(stats.gpuDispatches == 0);
    CHECK(stats.downloadedBytes == 0);

    // the backend stays disabled, including for chunks loaded afterwards, and nothing touches the busy device
    kernels = load(*backend, kSquare);
    REQUIRE(kernels.size() == 1);
    CHECK(kernels[0].reason.find("disabled") != std::string::npos);

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
    ScopedFastFlag sff{FFlag::LuauGpuKernels, true};

    std::unique_ptr<VulkanBackend> backend = createBackend();
    if (!backend)
        return;

    load(*backend, kSquare);

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
    CHECK(backend->stats().gpuDispatches == 200);
}

TEST_SUITE_END();

#endif
