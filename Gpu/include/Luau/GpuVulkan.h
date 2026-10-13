// This file is part of the Luau programming language and is licensed under MIT License; see LICENSE.txt for details
#pragma once

#include "Luau/GpuTranslate.h"

#include <stdint.h>

#include <functional>
#include <memory>
#include <string>
#include <vector>

struct lua_State;

namespace Luau
{
namespace Gpu
{

struct DispatchStats
{
    uint64_t dispatches = 0;
    uint64_t gpuDispatches = 0;
    uint64_t fallbacks = 0; // GPU runs that hit an error and were rerun in the VM
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
    // copies are freed beyond it, but a single dispatch can use more
    uint64_t cacheBytes = uint64_t(1) << 30;
};

// runs gpu.dispatch on a Vulkan device; kernels come from chunks registered with load()
//
// A dispatch runs on the GPU only when every step succeeds: translation, GLSL compilation and SPIR-V validation, pipeline
// creation, device limits, buffer allocation, execution within the timeout and an error-free status word. Otherwise the VM
// runs it and gpu.status reports why. The backend can be used from several threads.
class VulkanBackend
{
public:
    // returns nullptr and sets error when no usable device is found
    static std::unique_ptr<VulkanBackend> create(std::string& error, const VulkanOptions& options = VulkanOptions());
    ~VulkanBackend();

    // translates the @gpu functions of a chunk that is (or will be) loaded with this chunk name and builds their pipelines
    std::vector<Kernel> load(const std::string& chunkname, const char* source, size_t size);

    // registers already translated kernels, replacing every kernel previously registered for the chunk; their reason is set
    // when they can't run on this device
    void loadKernels(const std::string& chunkname, std::vector<Kernel>& kernels);

    // makes gpu.dispatch use this backend in the given state; the backend must outlive the state
    void install(lua_State* L);

    // called once per kernel the first time it is dispatched but has to run in the VM; exceptions it throws are ignored
    void setFallbackHandler(std::function<void(const std::string& message)> handler);

    const std::string& deviceName() const;
    DispatchStats stats() const;

    struct Impl;

private:
    VulkanBackend() = default;
    std::unique_ptr<Impl> impl;
};

} // namespace Gpu
} // namespace Luau
