// This file is part of the Luau programming language and is licensed under MIT License; see LICENSE.txt for details
#include "Luau/GpuShaderCompiler.h"

#include <glslang/Include/glslang_c_interface.h>
#include <glslang/Public/resource_limits_c.h>
#include <shaderc/shaderc.h>
#include <spirv-tools/libspirv.h>
#include <spirv_cross_c.h>

namespace Luau
{
namespace Gpu
{

struct ShaderCompiler::Impl
{
    shaderc_compiler_t compiler = nullptr;
    spv_context validator = nullptr;

    ~Impl()
    {
        if (validator)
            spvContextDestroy(validator);
        if (compiler)
            shaderc_compiler_release(compiler);
    }
};

ShaderCompiler::ShaderCompiler()
    : impl(std::make_unique<Impl>())
{
    impl->compiler = shaderc_compiler_initialize();
    impl->validator = spvContextCreate(SPV_ENV_VULKAN_1_0);
    glslang_initialize_process();
}

ShaderCompiler::~ShaderCompiler()
{
    glslang_finalize_process();
}

bool ShaderCompiler::compileSpirv(const Kernel& kernel, std::vector<uint32_t>& spirv, std::string& error)
{
    if (!impl->compiler || !impl->validator)
    {
        error = "shader compiler failed to initialize";
        return false;
    }

    shaderc_compile_options_t options = shaderc_compile_options_initialize();
    shaderc_compile_options_set_optimization_level(options, shaderc_optimization_level_performance);
    shaderc_compile_options_set_target_env(options, shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_0);
    shaderc_compile_options_set_target_spirv(options, shaderc_spirv_version_1_0);
    // keeps names, which the OpenGL ES output needs to find its uniforms
    shaderc_compile_options_set_generate_debug_info(options);

    shaderc_compilation_result_t result = shaderc_compile_into_spv(
        impl->compiler, kernel.glsl.data(), kernel.glsl.size(), shaderc_compute_shader, kernel.name.c_str(), "main", options
    );
    shaderc_compile_options_release(options);

    bool ok = shaderc_result_get_compilation_status(result) == shaderc_compilation_status_success;
    if (ok)
    {
        const uint32_t* words = reinterpret_cast<const uint32_t*>(shaderc_result_get_bytes(result));
        spirv.assign(words, words + shaderc_result_get_length(result) / sizeof(uint32_t));
    }
    else
    {
        error = std::string("generated GLSL doesn't compile: ") + shaderc_result_get_error_message(result);
    }
    shaderc_result_release(result);

    if (!ok)
        return false;

    spv_const_binary_t binary = {spirv.data(), spirv.size()};
    spv_diagnostic diagnostic = nullptr;
    if (spvValidate(impl->validator, &binary, &diagnostic) != SPV_SUCCESS)
    {
        error = std::string("generated SPIR-V is invalid: ") + (diagnostic ? diagnostic->error : "unknown error");
        spvDiagnosticDestroy(diagnostic);
        return false;
    }

    return true;
}

static bool disassemble(spv_context context, const std::vector<uint32_t>& spirv, std::string& text, std::string& error)
{
    spv_text result = nullptr;
    spv_diagnostic diagnostic = nullptr;
    if (spvBinaryToText(context, spirv.data(), spirv.size(), SPV_BINARY_TO_TEXT_OPTION_INDENT, &result, &diagnostic) != SPV_SUCCESS)
    {
        error = std::string("SPIR-V disassembly failed: ") + (diagnostic ? diagnostic->error : "unknown error");
        spvDiagnosticDestroy(diagnostic);
        return false;
    }

    text.assign(result->str, result->length);
    spvTextDestroy(result);
    return true;
}

// parses GLSL ES source to check that the cross-compiled shader is valid for OpenGL ES 3.1
static bool validateGlslEs(const std::string& source, std::string& error)
{
    glslang_input_t input = {};
    input.language = GLSLANG_SOURCE_GLSL;
    input.stage = GLSLANG_STAGE_COMPUTE;
    input.client = GLSLANG_CLIENT_NONE;
    input.target_language = GLSLANG_TARGET_NONE;
    input.code = source.c_str();
    input.default_version = 310;
    input.default_profile = GLSLANG_ES_PROFILE;
    input.messages = GLSLANG_MSG_DEFAULT_BIT;
    input.resource = glslang_default_resource();

    glslang_shader_t* shader = glslang_shader_create(&input);
    bool ok = glslang_shader_preprocess(shader, &input) && glslang_shader_parse(shader, &input);
    if (!ok)
        error = std::string("generated GLSL ES is invalid: ") + glslang_shader_get_info_log(shader);
    glslang_shader_delete(shader);
    return ok;
}

bool ShaderCompiler::crossCompile(const std::vector<uint32_t>& spirv, ShaderTarget target, std::string& source, std::string& error)
{
    if (target == ShaderTarget::SpirvVulkan)
        return disassemble(impl->validator, spirv, source, error);

    spvc_context context = nullptr;
    if (spvc_context_create(&context) != SPVC_SUCCESS)
    {
        error = "failed to create a SPIRV-Cross context";
        return false;
    }

    bool msl = target == ShaderTarget::MslIos || target == ShaderTarget::MslMacos;
    spvc_parsed_ir ir = nullptr;
    spvc_compiler compiler = nullptr;
    spvc_compiler_options options = nullptr;
    const char* result = nullptr;

    bool ok = spvc_context_parse_spirv(context, spirv.data(), spirv.size(), &ir) == SPVC_SUCCESS &&
              spvc_context_create_compiler(context, msl ? SPVC_BACKEND_MSL : SPVC_BACKEND_GLSL, ir, SPVC_CAPTURE_MODE_TAKE_OWNERSHIP, &compiler) ==
                  SPVC_SUCCESS &&
              spvc_compiler_create_compiler_options(compiler, &options) == SPVC_SUCCESS;

    if (ok && msl)
    {
        spvc_compiler_options_set_uint(options, SPVC_COMPILER_OPTION_MSL_VERSION, SPVC_MAKE_MSL_VERSION(2, 2, 0));
        spvc_compiler_options_set_uint(
            options, SPVC_COMPILER_OPTION_MSL_PLATFORM, target == ShaderTarget::MslIos ? SPVC_MSL_PLATFORM_IOS : SPVC_MSL_PLATFORM_MACOS
        );

        // binding i (kernel buffers, then the status word) becomes [[buffer(i)]], independently of which bindings the shader uses
        for (uint32_t i = 0; i <= kMaxKernelBuffers && ok; ++i)
        {
            spvc_msl_resource_binding_2 binding;
            spvc_msl_resource_binding_init_2(&binding);
            binding.stage = SpvExecutionModelGLCompute;
            binding.desc_set = 0;
            binding.binding = i;
            binding.count = 1;
            binding.msl_buffer = i;
            ok = spvc_compiler_msl_add_resource_binding_2(compiler, &binding) == SPVC_SUCCESS;
        }

        spvc_msl_resource_binding_2 push;
        spvc_msl_resource_binding_init_2(&push);
        push.stage = SpvExecutionModelGLCompute;
        push.desc_set = SPVC_MSL_PUSH_CONSTANT_DESC_SET;
        push.binding = SPVC_MSL_PUSH_CONSTANT_BINDING;
        push.count = 1;
        push.msl_buffer = kMslPushConstantBuffer;
        ok = ok && spvc_compiler_msl_add_resource_binding_2(compiler, &push) == SPVC_SUCCESS;
    }
    else if (ok)
    {
        spvc_compiler_options_set_uint(options, SPVC_COMPILER_OPTION_GLSL_VERSION, 310);
        spvc_compiler_options_set_bool(options, SPVC_COMPILER_OPTION_GLSL_ES, SPVC_TRUE);
    }

    ok = ok && spvc_compiler_install_compiler_options(compiler, options) == SPVC_SUCCESS && spvc_compiler_compile(compiler, &result) == SPVC_SUCCESS;

    if (ok)
        source = result;
    else
        error = std::string("SPIRV-Cross failed: ") + spvc_context_get_last_error_string(context);

    spvc_context_destroy(context);

    if (ok && target == ShaderTarget::GlslEs)
        ok = validateGlslEs(source, error);

    return ok;
}

} // namespace Gpu
} // namespace Luau
