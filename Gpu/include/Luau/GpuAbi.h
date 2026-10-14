// This file is part of the Luau programming language and is licensed under MIT License; see LICENSE.txt for details
#pragma once

#include <stdint.h>

// Interface between translated kernels and the backends that run them. A kernel is a function f(buffer, index) that
// gpu.run calls for index = 0 .. runs-1 on each buffer it's given; any backend (Vulkan, Metal, OpenGL ES, ...) binds
// resources this way:
//   - binding 0 (set 0): the buffer, as an array of 32-bit words
//   - binding 1: three status words, 'err', 'stop' and 'started'. Kernels set 'err' to a nonzero value when the results
//     must be discarded: the run then fails, and leaves the Luau buffer as it was. The host sets 'stop' to a nonzero value
//     to make invocations that start afterwards give up at once, and loops within 256 iterations, which bounds the GPU time
//     a runaway kernel takes. Invocation 0 sets 'started', so that the host measures that time from when the run starts
//   - push constants: GpuPushConstants
//   - workgroups of kGpuWorkgroupSize invocations along x; invocation index = x + y * groupsX * kGpuWorkgroupSize
namespace Luau
{
namespace Gpu
{

constexpr int kMaxKernelBuffers = 1;
constexpr uint32_t kWorkgroupSize = 64;

struct PushConstants
{
    // number of invocations
    uint32_t count;
    // maximum number of loop iterations per invocation before the run fails, 0 for no limit
    uint32_t loopBudget;
    // size of the buffer in bytes
    uint32_t lengths[kMaxKernelBuffers];
};

static_assert(sizeof(PushConstants) <= 128, "push constants must fit the minimum size guaranteed by Vulkan");

} // namespace Gpu
} // namespace Luau
