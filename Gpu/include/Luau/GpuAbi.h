// This file is part of the Luau programming language and is licensed under MIT License; see LICENSE.txt for details
#pragma once

#include <stdint.h>

// Interface between translated kernels and the backends that run them. Any backend (Vulkan, Metal, OpenGL ES, ...) binds
// resources this way:
//   - bindings 0 .. buffers-1 (set 0): the kernel's buffers, as arrays of 32-bit words
//   - binding 'buffers': three status words, 'err', 'stop' and 'started'. Kernels set 'err' to a nonzero value when the results
//     must be discarded and the dispatch rerun in the VM. The host sets 'stop' to a nonzero value to make invocations that
//     start afterwards give up at once, and loops within 256 iterations, which bounds the GPU time a runaway kernel takes.
//     Invocation 0 sets 'started', so that the host measures that time from when the dispatch actually runs
//   - push constants: GpuPushConstants
//   - workgroups of kGpuWorkgroupSize invocations along x; invocation i = x + y * groupsX * kGpuWorkgroupSize
namespace Luau
{
namespace Gpu
{

constexpr int kMaxKernelBuffers = 15;
constexpr uint32_t kWorkgroupSize = 64;

struct PushConstants
{
    // number of invocations
    uint32_t count;
    // maximum number of loop iterations per invocation before the dispatch is discarded, 0 for no limit
    uint32_t loopBudget;
    // size of each buffer in bytes
    uint32_t lengths[kMaxKernelBuffers];
};

static_assert(sizeof(PushConstants) <= 128, "push constants must fit the minimum size guaranteed by Vulkan");

} // namespace Gpu
} // namespace Luau
