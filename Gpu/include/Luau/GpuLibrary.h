// This file is part of the Luau programming language and is licensed under MIT License; see LICENSE.txt for details
#pragma once

#include <string>

struct lua_State;

namespace Luau
{
namespace Gpu
{

class VulkanBackend;

// Opens the gpu library in the state, before it's sandboxed so that its kernels see a safe environment:
//
//   gpu.run(fn: (buffer, index: number) -> buffer?, buffers: { buffer }, runs_per_buffer: number) -> (boolean, string?)
//     runs the @gpu function fn(b, index) on the GPU for index = 0 .. runs_per_buffer-1 on each buffer b of the array; the
//     invocations of a buffer run concurrently, and each mutates it. Returns true once every buffer holds its results, or
//     false and why while every buffer is left as it was; exactly "Function cannot run in GPU" when fn can't be lowered for
//     the GPU (it isn't @gpu, uses what kernels don't support, or its environment isn't safe). Nothing runs in the VM
//     instead. Arguments of the wrong type or out of range raise errors, as in other libraries. Invocations must only
//     write words of the buffer that no other invocation reads or writes, unless they use these atomics:
//   gpu.atomicadd(b: buffer, offset: number, value: number) -> number
//   gpu.atomicmin(b: buffer, offset: number, value: number) -> number
//     add value to, or take the minimum with value of, the unsigned 32-bit integer at offset, and return its previous value;
//     on the GPU they're atomic, and called from Luau they're plain reads and writes
//
// gpu.run uses the backend, which must outlive the state; without one, it returns false and 'unavailable'.
void openGpuLibrary(lua_State* L, VulkanBackend* backend, const std::string& unavailable = "no GPU backend");

} // namespace Gpu
} // namespace Luau
