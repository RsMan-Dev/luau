// This file is part of the Luau programming language and is licensed under MIT License; see LICENSE.txt for details
#include "Luau/GpuTranslate.h"

#include "Fixture.h"
#include "ScopedFlags.h"

#include "doctest.h"

#include <regex>
#include <set>
#include <string>

using namespace Luau;

LUAU_FASTFLAG(LuauGpuKernels)

namespace
{

std::vector<Gpu::Kernel> translate(const std::string& source)
{
    ScopedFastFlag sff{FFlag::LuauGpuKernels, true};
    return Gpu::translate(source.data(), source.size());
}

const Gpu::Kernel& kernelNamed(const std::vector<Gpu::Kernel>& kernels, const char* name)
{
    for (const Gpu::Kernel& k : kernels)
    {
        if (k.name == name)
            return k;
    }

    FAIL("no kernel named " << name);
    return kernels[0];
}

// translates a kernel whose body is 'body' and returns why it can't run on the GPU
std::string reasonFor(const std::string& body)
{
    std::vector<Gpu::Kernel> kernels = translate("@gpu\nlocal function k(i: number, b: buffer)\n" + body + "\nend\n");
    REQUIRE(kernels.size() == 1);
    return kernels[0].reason;
}

bool contains(const std::string& s, const std::string& part)
{
    return s.find(part) != std::string::npos;
}

bool matches(const std::string& s, const char* pattern)
{
    return std::regex_search(s, std::regex(pattern));
}

size_t count(const std::string& s, const std::string& part)
{
    size_t result = 0;
    for (size_t pos = s.find(part); pos != std::string::npos; pos = s.find(part, pos + 1))
        result++;
    return result;
}

std::set<std::string> identifiers(const std::string& glsl)
{
    std::set<std::string> result;
    std::regex word("[A-Za-z_][A-Za-z0-9_]*");
    for (auto it = std::sregex_iterator(glsl.begin(), glsl.end(), word); it != std::sregex_iterator(); ++it)
        result.insert(it->str());
    return result;
}

} // namespace

TEST_SUITE_BEGIN("GpuTranslate");

TEST_CASE("translates_a_kernel")
{
    std::vector<Gpu::Kernel> kernels = translate(R"(
local x = 1

@gpu
local function fill(i: number, out: buffer)
    buffer.writeu32(out, i * 4, i)
end
)");

    REQUIRE(kernels.size() == 1);
    const Gpu::Kernel& k = kernels[0];
    CHECK(k.name == "fill");
    CHECK(k.reason.empty());
    CHECK(k.line == 4); // the function starts at its attribute, like linedefined in bytecode
    CHECK(!k.isCalled);
    CHECK(contains(k.glsl, "void main()"));
    CHECK(contains(k.glsl, "layout(std430, set = 0, binding = 0) buffer B0"));
    REQUIRE(k.bufferWritten.size() == 1);
    CHECK(k.bufferWritten[0]);
}

TEST_CASE("buffers_that_are_only_read_are_not_written_back")
{
    std::vector<Gpu::Kernel> kernels = translate(R"(
@gpu
local function copy(i: number, src: buffer, dst: buffer, histogram: buffer)
    local v = buffer.readu32(src, i * 4)
    buffer.writeu32(dst, i * 4, v)
    gpu.atomicadd(histogram, (v % 4) * 4, 1)
end
)");

    REQUIRE(kernels.size() == 1);
    CHECK(kernels[0].reason.empty());
    CHECK(kernels[0].bufferWritten == std::vector<bool>{false, true, true});
}

TEST_CASE("number_types_are_inferred")
{
    std::vector<Gpu::Kernel> kernels = translate(R"(
@gpu
local function k(i: number, b: buffer)
    local offset = i * 4
    local half = offset / 2
    local bits = buffer.readu32(b, offset)
    local mixed = 1
    mixed += 0.5
    buffer.writef32(b, offset, half + bits + mixed)
end
)");

    REQUIRE(kernels.size() == 1);
    const std::string& glsl = kernels[0].glsl;
    CHECK(matches(glsl, "int l[0-9]+_offset "));
    CHECK(matches(glsl, "float l[0-9]+_half "));
    CHECK(matches(glsl, "uint l[0-9]+_bits "));
    CHECK(matches(glsl, "float l[0-9]+_mixed "));
}

TEST_CASE("helpers_are_specialized_per_argument_types")
{
    std::vector<Gpu::Kernel> kernels = translate(R"(
@gpu
local function twice(x: number)
    return x * 2
end

@gpu
local function k(i: number, b: buffer)
    buffer.writei32(b, i * 8, twice(i))
    buffer.writef32(b, i * 8 + 4, twice(i * 0.5))
end
)");

    REQUIRE(kernels.size() == 2);
    CHECK(kernelNamed(kernels, "twice").isCalled);

    const Gpu::Kernel& k = kernelNamed(kernels, "k");
    CHECK(k.reason.empty());
    CHECK(!k.isCalled);
    CHECK(contains(k.glsl, "int h0_twice(int p0)"));
    CHECK(contains(k.glsl, "float h1_twice(float p0)"));
}

TEST_CASE("numeric_constants_are_inlined")
{
    std::vector<Gpu::Kernel> kernels = translate(R"(
local WIDTH = 640
local scale = 2
scale = 3

@gpu
local function useConstant(i: number, b: buffer)
    buffer.writeu32(b, i * 4, i % WIDTH)
end

@gpu
local function useVariable(i: number, b: buffer)
    buffer.writeu32(b, i * 4, i * scale)
end
)");

    REQUIRE(kernels.size() == 2);
    CHECK(kernelNamed(kernels, "useConstant").reason.empty());
    CHECK(contains(kernelNamed(kernels, "useConstant").glsl, "640"));
    CHECK(kernelNamed(kernels, "useVariable").reason == "upvalue 'scale' is not a numeric constant");
}

TEST_CASE("unsupported_code_reports_the_reason")
{
    CHECK(reasonFor("local t = {}") == "tables are not supported on the GPU");
    CHECK(reasonFor("local s = 'x'") == "strings are not supported on the GPU");
    CHECK(reasonFor("local f = function() end") == "nested functions are not supported on the GPU");
    CHECK(reasonFor("if i then end") == "conditions must be boolean on the GPU (numbers are always truthy)");
    CHECK(reasonFor("local x = i and 1") == "'and'/'or' are only supported on booleans on the GPU; use an if-expression");
    CHECK(reasonFor("for _ in b do end") == "statement is not supported on the GPU");
    CHECK(reasonFor("print(i)") == "'print' is not a @gpu function");
    CHECK(
        reasonFor("buffer.writeu8(b, i, 1)") ==
        "buffer.writeu8 is not supported on the GPU (supported: readf32/readu32/readi32, writef32/writeu32/writei32, len)"
    );
    CHECK(reasonFor("local x = math.random()") == "math.random is not supported on the GPU");
    CHECK(reasonFor("local x = i\nx = true") == "variable is assigned values of incompatible types");
    CHECK(reasonFor("local x\nbuffer.writeu32(b, 0, 1)") == "local 'x' is never assigned a value");
}

TEST_CASE("reason_location_points_at_the_unsupported_code")
{
    std::vector<Gpu::Kernel> kernels = translate(R"(
@gpu
local function k(i: number, b: buffer)
    local ok = i + 1
    local t = { ok }
end
)");

    REQUIRE(kernels.size() == 1);
    CHECK(kernels[0].reasonLocation.begin == Position(4, 14));
}

TEST_CASE("kernels_must_take_an_index_and_buffers")
{
    std::vector<Gpu::Kernel> kernels = translate(R"(
@gpu
local function noIndex()
end

@gpu
local function useIndexAsBuffer(i: number, b: buffer)
    buffer.writeu32(i, 0, 1)
end

@gpu
local function returnsValue(i: number)
    return i
end
)");

    REQUIRE(kernels.size() == 3);
    CHECK(kernelNamed(kernels, "noIndex").reason == "kernels take the invocation index as their first parameter");
    CHECK(kernelNamed(kernels, "useIndexAsBuffer").reason == "expected a buffer parameter");
    CHECK(kernelNamed(kernels, "returnsValue").reason == "kernels can't return values");
}

TEST_CASE("recursion_is_rejected")
{
    std::vector<Gpu::Kernel> kernels = translate(R"(
@gpu
local function f(x: number)
    if x > 0 then
        return f(x - 1)
    end
    return 0
end

@gpu
local function k(i: number, b: buffer)
    buffer.writeu32(b, i * 4, f(i))
end
)");

    CHECK(kernelNamed(kernels, "k").reason == "recursive calls are not supported on the GPU");
}

TEST_CASE("kernels_on_the_same_line_are_ambiguous")
{
    std::vector<Gpu::Kernel> kernels = translate(R"(
local a, b = @gpu function(i: number) end, @gpu function(i: number) end
)");

    REQUIRE(kernels.size() == 2);
    CHECK(contains(kernels[0].reason, "same line"));
    CHECK(contains(kernels[1].reason, "same line"));
}

TEST_CASE("type_inference_that_does_not_settle_is_rejected")
{
    // each pass of inference widens one more variable to float, against the order of the statements
    auto chain = [](int length)
    {
        std::string source = "@gpu\nlocal function k(i: number, b: buffer)\n";
        for (int v = 1; v <= length; ++v)
            source += "    local x" + std::to_string(v) + " = 0\n";
        source += "    while i > 0 do\n";
        for (int v = length; v > 1; --v)
            source += "        x" + std::to_string(v) + " = x" + std::to_string(v - 1) + "\n";
        source += "        x1 = 0.5\n        i -= 1\n    end\n";
        source += "    buffer.writef32(b, 0, x" + std::to_string(length) + ")\nend\n";
        return translate(source);
    };

    std::vector<Gpu::Kernel> settles = chain(8);
    REQUIRE(settles.size() == 1);
    CHECK(settles[0].reason.empty());
    CHECK(matches(settles[0].glsl, "float l[0-9]+_x8 "));

    std::vector<Gpu::Kernel> unsettled = chain(40);
    REQUIRE(unsettled.size() == 1);
    CHECK(unsettled[0].reason == "types of the function's values don't settle after 32 passes");
    CHECK(unsettled[0].glsl.empty());
}

TEST_CASE("every_loop_spends_the_loop_budget")
{
    std::vector<Gpu::Kernel> kernels = translate(R"(
@gpu
local function sum(n: number)
    local s = 0
    for j = 1, n do
        s += j
    end
    return s
end

@gpu
local function k(i: number, b: buffer)
    local x = i
    while x > 0 do
        x -= 1
    end
    repeat
        x += 1
    until x > 10
    for j = 1, i, x do
        x += sum(j)
    end
    buffer.writeu32(b, 0, x)
end
)");

    const Gpu::Kernel& k = kernelNamed(kernels, "k");
    REQUIRE(k.reason.empty());

    // while, repeat and for in the kernel, and the for loop in the helper
    CHECK(count(k.glsl, "if (l_spent()) break;") == 4);
    CHECK(contains(k.glsl, "pc.budget"));

    // a step that isn't a constant can be zero, which must not loop forever
    CHECK(matches(k.glsl, "if \\(fs[0-9]+ == int\\(0\\)\\) l_fail\\(\\);"));
}

TEST_CASE("operations_undefined_in_glsl_are_guarded")
{
    std::vector<Gpu::Kernel> kernels = translate(R"(
@gpu
local function k(i: number, b: buffer)
    local n = buffer.readi32(b, 0)
    buffer.writei32(b, 0, n // i + n % i + math.abs(n) + math.clamp(n, i, 10))
    buffer.writeu32(b, 4, bit32.extract(n, i, 3))
    buffer.writef32(b, 8, math.sqrt(n) + math.log(n) + math.asin(n) + math.acos(n) + math.atan2(n, i) + math.clamp(0.5, n, i))
    buffer.writeu32(b, 12, -0.5)
end
)");

    REQUIRE(kernels.size() == 1);
    const std::string& glsl = kernels[0].glsl;
    REQUIRE(kernels[0].reason.empty());

    // INT_MIN // -1 overflows
    CHECK(contains(glsl, "b == -1 && a == int(0x80000000u)"));
    CHECK(contains(glsl, "l_idiv("));
    CHECK(contains(glsl, "l_imod("));
    CHECK(contains(glsl, "l_iabs("));
    CHECK(count(glsl, "l_clamp(") >= 3 + 2);
    CHECK(contains(glsl, "l_extract("));
    CHECK(count(glsl, "bitfieldExtract(") == 1); // only inside l_extract
    for (const char* fn : {"l_sqrt(", "l_log(", "l_asin(", "l_acos(", "l_atan2("})
        CHECK(contains(glsl, fn));

    // float to uint goes through int, since negative floats can't be converted to uint
    CHECK(contains(glsl, "uint(int("));
    CHECK(!matches(glsl, "uint\\(float"));
}

TEST_CASE("generated_identifiers_are_safe_in_every_shading_language")
{
    std::vector<Gpu::Kernel> kernels = translate(R"(
@gpu
local function half__(device: number)
    local kernel = device * 2
    return kernel
end

@gpu
local function k(i: number, b: buffer)
    local a__b = i
    local gl_x = 1
    local _ = 2
    local constant, sample, out, threadgroup = 1, 2, 3, 4
    buffer.writeu32(b, 0, half__(a__b + gl_x + _ + constant + sample + out + threadgroup))
end
)");

    const Gpu::Kernel& k = kernelNamed(kernels, "k");
    REQUIRE(k.reason.empty());

    std::set<std::string> ids = identifiers(k.glsl);
    for (const char* name : {"half__", "device", "kernel", "a__b", "gl_x", "_", "constant", "sample", "out", "threadgroup"})
        CHECK_MESSAGE(ids.count(name) == 0, name);

    for (const std::string& id : ids)
    {
        CHECK_MESSAGE(!contains(id, "__"), id);
        CHECK_MESSAGE((id.rfind("gl_", 0) != 0 || id == "gl_GlobalInvocationID" || id == "gl_NumWorkGroups"), id);
    }

    // names stay readable
    CHECK(matches(k.glsl, "h[0-9]+_half\\("));
    CHECK(matches(k.glsl, "int l[0-9]+_a_b "));
}

TEST_CASE("source_without_the_flag_does_not_parse")
{
    ScopedFastFlag sff{FFlag::LuauGpuKernels, false};
    std::string source = "@gpu\nlocal function k(i: number) end\n";
    CHECK(Gpu::translate(source.data(), source.size()).empty());
}

TEST_SUITE_END();
