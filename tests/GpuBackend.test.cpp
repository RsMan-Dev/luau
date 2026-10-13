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

// runs the chunk with the backend installed and returns the error it raised, if any; the caller enables LuauGpuKernels
std::string run(VulkanBackend& backend, const std::string& source)
{
    lua_State* L = luaL_newstate();
    luaL_openlibs(L);
    luaopen_gpu(L);
    lua_pop(L, 1);
    backend.install(L);

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
