// This file is part of the Luau programming language and is licensed under MIT License; see LICENSE.txt for details
#pragma once

#include "Luau/Location.h"

#include <stddef.h>
#include <string>
#include <vector>

namespace Luau
{

class AstStat;

namespace Gpu
{

constexpr int kMaxKernelBuffers = 15;

// a function marked @gpu, translated to a GLSL compute shader when it fits the supported subset
struct Kernel
{
    std::string name;
    Location location;

    // first line of the function including its attributes, 1-based; matches lua_Debug::linedefined of the compiled function
    int line = 0;

    // set when the function is called by other code, which makes it a helper rather than something to dispatch
    bool isCalled = false;

    // when empty, glsl holds the shader; otherwise the kernel can only run in the VM
    std::string reason;
    Location reasonLocation;
    std::string glsl;

    // one entry per buffer parameter (the parameters after the invocation index)
    std::vector<bool> bufferWritten;
};

// translates every @gpu function of a chunk as a kernel taking (index, ...buffers)
std::vector<Kernel> translate(AstStat* root);

// parses the chunk first, which needs FFlag::LuauGpuKernels for @gpu to be accepted; returns nothing if it doesn't parse
std::vector<Kernel> translate(const char* source, size_t size);

} // namespace Gpu
} // namespace Luau
