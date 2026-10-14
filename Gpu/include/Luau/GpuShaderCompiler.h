// This file is part of the Luau programming language and is licensed under MIT License; see LICENSE.txt for details
#pragma once

#include "Luau/GpuAbi.h"

#include <stdint.h>

#include <memory>
#include <string>
#include <vector>

namespace Luau
{
namespace Gpu
{

// Shader languages a kernel can be compiled to. Every target keeps the resource interface of GpuAbi.h:
//   - SpirvVulkan: SPIR-V 1.0 for Vulkan 1.0 (desktop and Android); descriptor set 0, bindings as in GpuAbi.h
//   - MslIos, MslMacos: Metal Shading Language 2.2 (iOS 13, macOS 10.15); the entry point is 'main0', the kernel's buffer is
//     [[buffer(0)]], the status word [[buffer(1)]], the written parts [[buffer(2)]], push constants are a constant buffer at
//     [[buffer(kMslPushConstantBuffer)]], and threadgroups must have kWorkgroupSize threads along x
//   - GlslEs: GLSL ES 3.10 compute shader for OpenGL ES 3.1 (Android); SSBOs keep their bindings, and push constants become the
//     uniform struct 'pc' whose fields n, budget, len[] and written hold PushConstants::count, loopBudget, lengths and written (set with
//     glUniform1ui/glUniform1uiv); local_size_x is kWorkgroupSize
enum class ShaderTarget
{
    SpirvVulkan,
    MslIos,
    MslMacos,
    GlslEs,
};

constexpr uint32_t kMslPushConstantBuffer = kBindings;

// compiles the GLSL of kernels and cross-compiles it for other graphics APIs
class ShaderCompiler
{
public:
    ShaderCompiler();
    ~ShaderCompiler();

    ShaderCompiler(const ShaderCompiler&) = delete;
    ShaderCompiler& operator=(const ShaderCompiler&) = delete;

    // compiles the GLSL of a kernel to SPIR-V and validates it for Vulkan 1.0; the name only appears in errors
    bool compileSpirv(const std::string& glsl, const std::string& name, std::vector<uint32_t>& spirv, std::string& error);

    // translates validated SPIR-V to the source of another target (SPIR-V itself gives its disassembly). GLSL ES output is
    // validated by parsing it again; MSL can't be validated here since Apple's Metal compiler only runs on Apple platforms
    bool crossCompile(const std::vector<uint32_t>& spirv, ShaderTarget target, std::string& source, std::string& error);

    struct Impl;

private:
    std::unique_ptr<Impl> impl;
};

} // namespace Gpu
} // namespace Luau
