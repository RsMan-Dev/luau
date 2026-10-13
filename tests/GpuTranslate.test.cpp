// This file is part of the Luau programming language and is licensed under MIT License; see LICENSE.txt for details
#include "Luau/GpuTranslate.h"

#include "Fixture.h"
#include "ScopedFlags.h"

#include "doctest.h"

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
    CHECK(contains(glsl, "int l_offset_"));
    CHECK(contains(glsl, "float l_half_"));
    CHECK(contains(glsl, "uint l_bits_"));
    CHECK(contains(glsl, "float l_mixed_"));
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

TEST_CASE("source_without_the_flag_does_not_parse")
{
    ScopedFastFlag sff{FFlag::LuauGpuKernels, false};
    std::string source = "@gpu\nlocal function k(i: number) end\n";
    CHECK(Gpu::translate(source.data(), source.size()).empty());
}

TEST_SUITE_END();
