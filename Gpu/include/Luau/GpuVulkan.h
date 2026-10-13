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
};

// runs gpu.dispatch on a Vulkan device; kernels come from chunks registered with load()
class VulkanBackend
{
public:
    // returns nullptr and sets error when no usable device is found
    static std::unique_ptr<VulkanBackend> create(std::string& error);
    ~VulkanBackend();

    // translates the @gpu functions of a chunk that is (or will be) loaded with this chunk name
    std::vector<Kernel> load(const std::string& chunkname, const char* source, size_t size);

    // makes gpu.dispatch use this backend in the given state; the backend must outlive the state
    void install(lua_State* L);

    // called once per kernel the first time it is dispatched but has to run in the VM
    void setFallbackHandler(std::function<void(const std::string& message)> handler);

    const std::string& deviceName() const;
    DispatchStats& stats();

    struct Impl;

private:
    VulkanBackend() = default;
    std::unique_ptr<Impl> impl;
};

} // namespace Gpu
} // namespace Luau
