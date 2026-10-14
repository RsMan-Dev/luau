// This file is part of the Luau programming language and is licensed under MIT License; see LICENSE.txt for details
#pragma once

#include "Luau/CodeGen.h"

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

    // whether the kernel writes its buffer (one entry)
    std::vector<bool> bufferWritten;
};

// Lowers the Luau function at stack index idx, which must have the gpu attribute
GpuKernel compileGpu(lua_State* L, int idx);

// Whether the builtins the function at stack index idx sees are still the real ones, which kernels rely on since they can't
// check at run time
bool isGpuEnvironmentSafe(lua_State* L, int idx);

// Lowers every function with the gpu attribute defined in the module whose main function is at stack index idx
std::vector<GpuKernel> compileGpuModule(lua_State* L, int idx);

} // namespace CodeGen
} // namespace Luau
