// This file is part of the Luau programming language and is licensed under MIT License; see LICENSE.txt for details
#pragma once

#include "Luau/GpuTranslate.h"

#include <stdint.h>

#include <memory>
#include <string>
#include <vector>

namespace Luau
{
namespace Gpu
{

// Shader languages a translated kernel can be compiled to. Every target keeps the resource interface of GpuAbi.h:
//   - SpirvVulkan: SPIR-V 1.0 for Vulkan 1.0 (desktop and Android); descriptor set 0, bindings as in GpuAbi.h
//   - MslIos, MslMacos: Metal Shading Language 2.2 (iOS 13, macOS 10.15); the entry point is 'main0', kernel buffer i is
//     [[buffer(i)]], the status word is [[buffer(buffers)]], push constants are a constant buffer at [[buffer(kMslPushConstantBuffer)]],
//     and threadgroups must have kWorkgroupSize threads along x
//   - GlslEs: GLSL ES 3.10 compute shader for OpenGL ES 3.1 (Android); SSBOs keep their bindings, and push constants become the
//     uniform struct 'pc' whose fields n, budget and len[] hold PushConstants::count, loopBudget and lengths (set with
//     glUniform1ui/glUniform1uiv); local_size_x is kWorkgroupSize
enum class ShaderTarget
{
    SpirvVulkan,
    MslIos,
    MslMacos,
    GlslEs,
};

constexpr uint32_t kMslPushConstantBuffer = kMaxKernelBuffers + 1;

// compiles the GLSL of translated kernels and cross-compiles it for other graphics APIs
class ShaderCompiler
{
public:
    ShaderCompiler();
    ~ShaderCompiler();

    ShaderCompiler(const ShaderCompiler&) = delete;
    ShaderCompiler& operator=(const ShaderCompiler&) = delete;

    // compiles the kernel's GLSL to SPIR-V and validates it for Vulkan 1.0
    bool compileSpirv(const Kernel& kernel, std::vector<uint32_t>& spirv, std::string& error);

    // translates validated SPIR-V to the source of another target (SPIR-V itself gives its disassembly). GLSL ES output is
    // validated by parsing it again; MSL can't be validated here since Apple's Metal compiler only runs on Apple platforms
    bool crossCompile(const std::vector<uint32_t>& spirv, ShaderTarget target, std::string& source, std::string& error);

    struct Impl;

private:
    std::unique_ptr<Impl> impl;
};

} // namespace Gpu
} // namespace Luau
