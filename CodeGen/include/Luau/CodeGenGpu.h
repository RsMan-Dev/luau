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

// A function marked @gpu lowered from its IR to a GLSL compute shader, invocation 'i' running f(i, ...buffers)
// The shader binds resources as described in Gpu/include/Luau/GpuAbi.h
struct GpuKernel
{
    CodeGenCompilationResult result = CodeGenCompilationResult::Success;

    // when the function can't run on the GPU: the IR command or construct responsible and its source line
    std::string reason;

    std::string glsl;

    // one entry per buffer parameter (the parameters after the invocation index)
    std::vector<bool> bufferWritten;
};

// Lowers the Luau function at stack index idx, which must have the gpu attribute
GpuKernel compileGpu(lua_State* L, int idx);

} // namespace CodeGen
} // namespace Luau
