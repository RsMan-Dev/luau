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

struct DispatchStats
{
    uint64_t dispatches = 0;    // gpu.dispatch calls the backend saw
    uint64_t gpuDispatches = 0; // calls that ran on the GPU
    uint64_t fallbacks = 0;     // GPU runs that hit an error and were rerun in the VM
    uint64_t uploadedBytes = 0;
    uint64_t downloadedBytes = 0;
    uint64_t cachedBytes = 0; // device memory currently holding copies of Luau buffers
};

struct VulkanOptions
{
    // loop iterations allowed per invocation before the dispatch is discarded and rerun in the VM, 0 for no limit
    uint32_t loopBudget = 1u << 24;

    // a dispatch that takes longer disables the backend, since the device can't be trusted afterwards
    uint32_t timeoutMs = 5000;

    // uses the minimum limits guaranteed by Vulkan instead of the device's, to behave like low-end mobile devices
    bool minimumLimits = false;

    // device memory kept for copies of Luau buffers (host memory of the same size mirrors it); the least recently used
    // copies are freed beyond it, except copies newer than their Luau buffer, and a single dispatch can use more
    uint64_t cacheBytes = uint64_t(1) << 30;
};

// runs gpu.dispatch on a Vulkan device, with the kernels native code generation lowers from @gpu functions
//
// A dispatch runs on the GPU only when every step succeeds: lowering, GLSL compilation and SPIR-V validation, pipeline
// creation, device limits, buffer allocation, execution within the timeout and an error-free status word. Otherwise the VM
// runs it and gpu.status reports why. The backend can be used from several threads, and must outlive the states it is
// installed in.
class VulkanBackend
{
public:
    // returns nullptr and sets error when no usable device is found
    static std::unique_ptr<VulkanBackend> create(std::string& error, const VulkanOptions& options = VulkanOptions());
    ~VulkanBackend();

    // makes gpu.dispatch use this backend in the given state
    void install(lua_State* L);

    // called once per kernel the first time it is dispatched but has to run in the VM; exceptions it throws are ignored
    void setFallbackHandler(std::function<void(const std::string& message)> handler);

    // replaces CodeGen::compileGpu, which turns a function into a kernel; for tests
    void setLowering(std::function<CodeGen::GpuKernel(lua_State* L, int idx)> lowering);

    const std::string& deviceName() const;
    DispatchStats stats() const;

    struct Impl;

private:
    VulkanBackend() = default;
    std::unique_ptr<Impl> impl;
};

} // namespace Gpu
} // namespace Luau
