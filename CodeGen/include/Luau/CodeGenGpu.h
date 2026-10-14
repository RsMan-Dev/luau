// This file is part of the Luau programming language and is licensed under MIT License; see LICENSE.txt for details
#pragma once

#include "Luau/CodeGen.h"

#include <stdint.h>

#include <string>
#include <vector>

struct lua_State;

namespace Luau
{
namespace CodeGen
{

// A function marked @gpu lowered from its IR to a GLSL compute shader, invocation 'index' running f(buffer, index)
// The shader binds resources as described in Gpu/include/Luau/GpuAbi.h
struct GpuKernel
{
    CodeGenCompilationResult result = CodeGenCompilationResult::Success;

    // debug name and first line of the function
    std::string name;
    int line = 0;

    // when the function can't run on the GPU: the IR command or construct responsible and its source line
    std::string reason;

    std::string glsl;

    // the variant of the kernel that computes the divisions and square roots whose hardware results it can't correct
    // cheaply, which the kernel fails with GpuFailure::NeedsExact on: larger and slower, but rarely needed
    std::string exactGlsl() const;

    // whether the kernel writes its buffer (one entry)
    std::vector<bool> bufferWritten;
};

// Why a kernel failed. The status word holds the first failure of a run: its kind in the top 8 bits, its source line in the
// others (0 when unknown). The kernel prelude writes these values.
enum class GpuFailure : uint8_t
{
    None = 0,
    Access = 1,     // out of bounds or unaligned buffer access
    LoopBudget = 2, // the run's loop iteration budget ran out
    Stopped = 3,    // the host stopped the run
    Fallback = 4,   // a value of a type the lowering doesn't handle there, which native code leaves to the VM
    NeedsExact = 5, // a division or square root only the kernel's exact variant computes, see GpuKernel::exactGlsl
};

// The reason gpu.run gives for a status word, like "the kernel failed on the GPU: loop limit reached at line 12"
std::string describeGpuFailure(uint32_t status);

// Lowers the Luau function at stack index idx, which must have the gpu attribute
GpuKernel compileGpu(lua_State* L, int idx);

// Whether the builtins the function at stack index idx sees are still the real ones, which kernels rely on since they can't
// check at run time
bool isGpuEnvironmentSafe(lua_State* L, int idx);

// Lowers every function with the gpu attribute defined in the module whose main function is at stack index idx
std::vector<GpuKernel> compileGpuModule(lua_State* L, int idx);

} // namespace CodeGen
} // namespace Luau
