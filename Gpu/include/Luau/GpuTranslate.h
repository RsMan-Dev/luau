// This file is part of the Luau programming language and is licensed under MIT License; see LICENSE.txt for details
#pragma once

#include <stddef.h>
#include <string>
#include <vector>

namespace Luau
{
namespace Gpu
{

constexpr int kMaxKernelBuffers = 15;

// a function marked @gpu, translated to a GLSL compute shader when it fits the supported subset
struct Kernel
{
    std::string name;
    // line of the 'function' keyword, 1-based; matches lua_Debug::linedefined of the compiled function
    int line = 0;

    // when empty, glsl holds the shader; otherwise the kernel can only run in the VM
    std::string reason;
    std::string glsl;

    // one entry per buffer parameter (the parameters after the invocation index)
    std::vector<bool> bufferWritten;
};

// parses the chunk and translates every @gpu function as a kernel taking (index, ...buffers)
std::vector<Kernel> translate(const char* source, size_t size);

} // namespace Gpu
} // namespace Luau
