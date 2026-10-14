// This file is part of the Luau programming language and is licensed under MIT License; see LICENSE.txt for details
#pragma once

#include "Luau/CodeGenGpu.h"
#include "Luau/GpuAbi.h"

#include <stdint.h>

#include <functional>
#include <memory>
#include <string>

struct lua_State;

namespace Luau
{
namespace Gpu
{

struct RunStats
{
    uint64_t runs = 0;      // gpu.run calls the backend saw
    uint64_t succeeded = 0; // calls that ran on the GPU and returned true
    uint64_t failures = 0;  // calls whose kernel hit an error on the GPU
    uint64_t uploadedBytes = 0;
    uint64_t downloadedBytes = 0;
    uint64_t cachedBytes = 0; // device memory currently holding copies of Luau buffers
    uint64_t memoryBytes = 0; // device and host memory currently allocated for those copies, which share it
    uint64_t kernels = 0;     // kernels kept for live functions, including the ones that can't run on the GPU
};

struct VulkanOptions
{
    // loop iterations allowed per invocation before the run fails, 0 for no limit
    uint32_t loopBudget = 1u << 24;

    // a run that takes longer is asked to stop: its loops give up, and it fails; counted from when it starts on the GPU, so
    // that waiting for other work on the device doesn't count
    uint32_t abortMs = 100;

    // a run that takes longer, even after being asked to stop, disables the backend, since the device can't be trusted
    uint32_t timeoutMs = 5000;

    // uses the minimum limits guaranteed by Vulkan instead of the device's, to behave like low-end mobile devices
    bool minimumLimits = false;

    // device memory kept for copies of Luau buffers (host memory of the same size mirrors it); the least recently used
    // copies are freed beyond it, and a single run can use more
    uint64_t cacheBytes = uint64_t(1) << 30;

    // pipelines are created on host threads, since drivers can take minutes on large kernels: until a kernel's is ready,
    // runs of it return kCompiling and leave the buffers as they were. Set to have runs wait for it instead
    bool waitForPipelines = false;

    // file that keeps the pipelines the driver compiled between processes, read when the backend is created and written
    // when it's destroyed; empty for none
    std::string pipelineCachePath;
};

// what runs return while their kernel's pipeline is being created
constexpr const char* kCompiling = "the kernel is being compiled for the GPU";

// runs gpu.run on a Vulkan device, with the kernels native code generation lowers from @gpu functions
//
// A run succeeds only when every step does: lowering, GLSL compilation and SPIR-V validation, pipeline creation, device
// limits, buffer allocation, execution within the timeout and an error-free status word. Otherwise it fails and leaves the
// Luau buffers as they were; nothing runs in the VM instead. Copies of the buffers stay on the device between runs, and are
// only uploaded again when their Luau contents change. The backend can be used from several threads, and must outlive the
// states that use it.
class VulkanBackend
{
public:
    // returns nullptr and sets error when no usable device is found, or when the device is refused because it doesn't run
    // kernels as the VM would
    static std::unique_ptr<VulkanBackend> create(std::string& error, const VulkanOptions& options = VulkanOptions());
    ~VulkanBackend();

    // runs the function at stack index fn on each buffer of the array at stack index list, for index 0 .. runs-1, and
    // downloads the buffers; returns an empty string, or why the buffers were left as they were
    std::string run(lua_State* L, int fn, int list, uint32_t runs);

    // replaces CodeGen::compileGpu, which turns a function into a kernel; for tests
    void setLowering(std::function<CodeGen::GpuKernel(lua_State* L, int idx)> lowering);

    const std::string& deviceName() const;
    RunStats stats() const;

    struct Impl;

private:
    VulkanBackend() = default;
    std::unique_ptr<Impl> impl;
};

} // namespace Gpu
} // namespace Luau
