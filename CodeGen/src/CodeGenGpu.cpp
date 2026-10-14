// This file is part of the Luau programming language and is licensed under MIT License; see LICENSE.txt for details
#include "Luau/CodeGenGpu.h"

#include "CodeGenLower.h"

#include "lapi.h"
#include "ldebug.h"

#include <algorithm>

#include <math.h>
#include <stdio.h>

// Lowers the IR of a @gpu function to a GLSL compute shader, after the same IR translation, limits and optimizations as
// native code. Every guard and fallback path of the IR fails the run instead, which leaves its buffers as they were.

namespace Luau
{
namespace CodeGen
{

// mirrors Gpu/include/Luau/GpuAbi.h
constexpr int kGpuMaxBuffers = 1;
constexpr int kGpuWrittenChunkShift = 16;
constexpr unsigned kGpuWorkgroupSize = 64;

namespace
{

const char* const kPrelude = R"(
uint l_steps = 0u;
// the first failure of a run gives its kind and source line, see GpuFailure. An invocation notes its first one and reports it
// once it returns, which compiles much faster than an atomic at each check: drivers take a time growing faster than the code
// with each side effect in a branch.
// Kernels have two variants. The one compiled with L_EXACT computes every division and square root exactly. The other, which
// runs first, only notes that one needed the exact computation, and the host then runs the first instead: the exact
// computations cost kernels time even when they don't run, in the registers they keep
uint l_at = 0u;
uint l_err = 0u;
void l_fail(uint kind) { if (l_err == 0u) l_err = (kind << 24) | l_at; }
#ifndef L_EXACT
bool l_flag = false;
#endif
// the host's stop word is read with an atomic, which goes to memory on GPUs whose plain loads of a coherent buffer can keep
// seeing a cached copy
bool l_stopped() { return atomicAdd(st.stop, 0) != 0; }
bool l_bad(int off, int b) { return off < 0 || (off & 3) != 0 || uint(off) + 4u > pc.len[b]; }
// Numbers are f32. What needs an integer converts the number as the VM does: an int as luaL_checkinteger, truncated, and the
// smallest int when it doesn't fit (or is NaN), as x86-64 converts it
int l_toi(float x) { return x >= -2147483648.0 && x < 2147483648.0 ? int(x) : int(0x80000000u); }
// a uint as luaL_checkunsigned: truncated to a 64-bit integer, whose low 32 bits it keeps, 0 when it doesn't fit (or is NaN)
uint l_tou(float x)
{
    uint u = floatBitsToUint(x), e = (u >> 23) & 0xffu;
    if (e < 127u || e >= 190u)
        return 0u;
    uint m = (u & 0x7fffffu) | 0x800000u, t = e >= 150u ? (e - 150u < 32u ? m << (e - 150u) : 0u) : m >> (150u - e);
    return (u & 0x80000000u) != 0u ? 0u - t : t;
}
// integers rounded to the nearest float, ties to even, as the VM rounds the double holding one
float l_u2f(uint a)
{
    if (a < 0x1000000u)
        return float(a);
    int e = findMSB(a) - 23;
    uint m = a >> e, rest = a - (m << e), mid = 1u << (e - 1);
    if (rest > mid || (rest == mid && (m & 1u) != 0u))
        m++;
    return float(m) * uintBitsToFloat(uint(127 + e) << 23);
}
float l_i2f(int a) { float r = l_u2f(a < 0 ? uint(-a) : uint(a)); return a < 0 ? -r : r; }
// round, trunc and the difference are exact: only halves round away from zero
float l_round(float a) { float t = trunc(a); return abs(a - t) >= 0.5 ? t + sign(a) : t; }
float l_log(float a) { return a < 0.0 ? uintBitsToFloat(0x7fc00000u) : a == 0.0 ? uintBitsToFloat(0xff800000u) : log(a); }
float l_log2(float a) { return a < 0.0 ? uintBitsToFloat(0x7fc00000u) : a == 0.0 ? uintBitsToFloat(0xff800000u) : log2(a); }
float l_asin(float a) { return abs(a) > 1.0 ? uintBitsToFloat(0x7fc00000u) : asin(a); }
float l_acos(float a) { return abs(a) > 1.0 ? uintBitsToFloat(0x7fc00000u) : acos(a); }
bool l_odd(float b) { return abs(b) < 16777216.0 && mod(b, 2.0) == 1.0; }
// the GPU's exp and pow flush results below the normal range to 0: those are computed 2^64 times larger, then scaled down,
// which rounds them once
float l_exp(float a) { return a < -87.0 ? exp(a + 44.36141955583650) * uintBitsToFloat(0x1f800000u) : exp(a); }
float l_powe(float a, float b) { float l = b * log2(a); return l < -125.0 ? exp2(l + 64.0) * uintBitsToFloat(0x1f800000u) : pow(a, b); }
float l_fdiv(float a, float b);
// the GPU's pow is exp2(b * log2(a)), a few ulps off even where the result is exact (3 ^ 3): integer exponents up to 32
// multiply instead, which is exact whenever each product is, unless a product overflows or underflows
float l_powp(float a, float b)
{
    if (b != floor(b) || abs(b) > 32.0)
        return l_powe(a, b);
    float r = 1.0, x = a;
    for (uint n = uint(abs(b)); n != 0u; n >>= 1)
    {
        if ((n & 1u) != 0u)
            r *= x;
        if (n > 1u)
            x *= x;
    }
    if (isinf(r) || r < 1.17549435e-38 || isinf(x) || x < 1.17549435e-38)
        return l_powe(a, b);
    return b < 0.0 ? l_fdiv(1.0, r) : r;
}
float l_pow(float a, float b)
{
    if (b == 0.0 || a == 1.0 || (a == -1.0 && isinf(b))) return 1.0;
    if (isnan(a) || isnan(b)) return uintBitsToFloat(0x7fc00000u);
    if (a == 0.0) return b > 0.0 ? (l_odd(b) ? a : 0.0) : (l_odd(b) && floatBitsToUint(a) != 0u ? uintBitsToFloat(0xff800000u) : uintBitsToFloat(0x7f800000u));
    if (isinf(a) && a < 0.0) return b > 0.0 ? (l_odd(b) ? a : -a) : (l_odd(b) ? uintBitsToFloat(0x80000000u) : 0.0);
    if (a < 0.0) { if (b != floor(b)) return uintBitsToFloat(0x7fc00000u); float r = l_powp(-a, b); return l_odd(b) ? -r : r; }
    return l_powp(a, b);
}
vec3 l_minv(vec3 a, vec3 b) { return mix(b, a, lessThan(a, b)); }
vec3 l_maxv(vec3 a, vec3 b) { return mix(b, a, greaterThan(a, b)); }
// Vulkan only bounds the error of division and square roots, which the VM rounds correctly for vectors (f32): these start
// from the hardware's result and correct it with exact integer arithmetic. A finite nonzero float is m * 2^(e - 150) with
// m in [2^23, 2^24) once denormals are normalized
// the exact computations, which cost kernels time even when they don't run, are only in the variant of the kernel compiled
// with L_EXACT: the other notes that it needs them, and the host runs the variant instead
#ifdef L_EXACT
uint l_fmant(uint u, out int e)
{
    e = int((u >> 23) & 0xffu);
    uint m = u & 0x7fffffu;
    if (e != 0)
        return m | 0x800000u;
    int shift = 23 - findMSB(m);
    e = 1 - shift;
    return m << shift;
}
// rounds m * 2^(e - 150) with m in [2^24, 2^25) (a guard bit below the 24 bits kept), plus 'sticky' nonzero bits below it
float l_fround(uint sign, int e, uint m, bool sticky)
{
    if (e >= 255)
        return uintBitsToFloat(sign | 0x7f800000u);
    if (e <= 0)
    {
        int shift = min(1 - e, 26);
        sticky = sticky || (m & ((1u << shift) - 1u)) != 0u;
        m >>= shift;
        e = 0;
    }
    uint r = m >> 1;
    if ((m & 1u) != 0u && (sticky || (r & 1u) != 0u))
        r += 1u;
    // a carry into the exponent is right: it only makes the significand 2^23, which encodes the next binade
    return uintBitsToFloat(sign | ((e == 0 ? 0u : uint(e - 1) << 23) + r));
}
float l_fdivexact(float a, float b)
{
    uint ua = floatBitsToUint(a), ub = floatBitsToUint(b);
    uint sign = (ua ^ ub) & 0x80000000u;
    uint xa = ua & 0x7fffffffu, xb = ub & 0x7fffffffu;
    if (xa > 0x7f800000u || xb > 0x7f800000u || (xa == xb && (xa == 0u || xa == 0x7f800000u)))
        return uintBitsToFloat(0x7fc00000u);
    if (xa == 0x7f800000u || xb == 0u)
        return uintBitsToFloat(sign | 0x7f800000u);
    if (xa == 0u || xb == 0x7f800000u)
        return uintBitsToFloat(sign);
    int ea, eb;
    uint ma = l_fmant(xa, ea), mb = l_fmant(xb, eb);
    int e = ea - eb + 127;
    if (ma < mb)
    {
        ma <<= 1;
        e -= 1;
    }
    // q = floor(ma * 2^24 / mb), 25 bits, by long division with integers, 8 bits at a time: the remainder stays below mb < 2^24
    uint q = 1u, rem = ma - mb;
    [[dont_unroll]] for (int k = 0; k < 3; k++)
    {
        rem <<= 8;
        uint digit = rem / mb;
        rem -= digit * mb;
        q = (q << 8) | digit;
    }
    return l_fround(sign, e, q, rem != 0u);
}
float l_fsqrtexact(float a)
{
    uint u = floatBitsToUint(a);
    if ((u & 0x7fffffffu) == 0u || u == 0x7f800000u)
        return a;
    if (u > 0x7f800000u)
        return uintBitsToFloat(0x7fc00000u);
    int e;
    uint m = l_fmant(u, e);
    // a = m * 2^(e - 150); with an even exponent, sqrt(a) = sqrt(m * 2^25) * 2^((e - 127) / 2 - 24)
    if (((e - 127) & 1) != 0)
    {
        m <<= 1;
        e -= 1;
    }
    // root = floor(sqrt(m * 2^25)), 25 bits, with integers, two bits of m * 2^25 at a time: the remainder stays below 2^27
    uint root = 0u, rem = 0u;
    [[dont_unroll]] for (int k = 24; k >= 0; k--)
    {
        int at = 2 * k - 25;
        rem = (rem << 2) | ((at >= 0 ? m >> at : m << -at) & 3u);
        uint trial = (root << 2) | 1u;
        root <<= 1;
        if (rem >= trial)
        {
            rem -= trial;
            root |= 1u;
        }
    }
    return l_fround(0u, (e - 127) / 2 + 127, root, rem != 0u);
}
#endif
// the hardware's results are usually within a few units of the correctly rounded ones, which one integer remainder tells: with
// normal operands and results, the remainder gives how many units to move the result by, and checks that the exact quotient
// or root is then within half a unit of it; anything else, rare, goes to the exact computation, which only uses integers. The
// remainder is computed in 32 bits, so the result must first be close enough for it to fit: an estimate of the remainder in
// floats, off by at most 2^24, must be below 2^29. A result that far off goes to the exact computation too
bool l_fdivfast(float a, float b, out float result)
{
    float q = a / b;
    uint ua = floatBitsToUint(a), ub = floatBitsToUint(b), uq = floatBitsToUint(q);
    // a power of two is moved to the float below it, the side its binade's rounding is finer on: the exact quotient is then
    // reached upward, where crossing to the next binade is handled
    if ((uq & 0x7fffffu) == 0u && ((uq >> 23) & 0xffu) > 1u && ((uq >> 23) & 0xffu) < 255u)
        uq -= 1u;
    uint ea = (ua >> 23) & 0xffu, eb = (ub >> 23) & 0xffu, eq = (uq >> 23) & 0xffu;
    // a / b = q * 2^(s - 23) * (ma * 2^s) / (qm * mb), so the exact quotient minus q, in units of q's last bit, is r / mb
    int s = int(ea) - int(eb) + 150 - int(eq);
    uint ma = (ua & 0x7fffffu) | 0x800000u, mb = (ub & 0x7fffffu) | 0x800000u, qm = (uq & 0x7fffffu) | 0x800000u;
    int r = int((ma << (s & 31)) - qm * mb);
    int k = int(floor(float(r) * (1.0 / float(mb)) + 0.5));
    r -= k * int(mb);
    // the estimate of k can be off by one near a tie
    if (2 * r >= int(mb))
    {
        k++;
        r -= int(mb);
    }
    else if (2 * r <= -int(mb))
    {
        k--;
        r += int(mb);
    }
    bool close = abs(float(ma) * float(1u << uint(s & 31)) - float(qm) * float(mb)) < 536870912.0;
    bool normal = ea - 1u < 254u && eb - 1u < 254u && eq - 1u < 254u && uint(s - 23) <= 2u && close;
    // the significand stays in its binade, where the remainder's scale holds, or reaches the next power of two P, as when
    // the hardware gives 0.99999994 for x / x: past P units are twice as large, so the result is P or the float above it
    uint m = qm + uint(k);
    bool fits = false;
    if (m - 0x800000u < 0x800000u)
        fits = uint(abs(r)) * 2u < mb;
    else if (m - 0x1000000u < 4u && eq < 254u)
    {
        int above = int(m - 0x1000000u), rp = r + above * int(mb), rc = rp - 2 * int(mb);
        if (rp >= 0 ? rp < int(mb) : -2 * rp < int(mb))
        {
            k -= above;
            fits = true;
        }
        else if (abs(rc) < int(mb))
        {
            k -= above - 1;
            fits = true;
        }
    }
    result = uintBitsToFloat(uq + uint(k));
    if (normal && fits)
        return true;
    // zeros, infinities and NaN, which IEEE defines whatever the other operand, as when a vector has a zero component; only
    // denormal numbers and results the hardware got too far off are left
    uint sign = (ua ^ ub) & 0x80000000u, xa = ua & 0x7fffffffu, xb = ub & 0x7fffffffu;
    result = uintBitsToFloat(xa > 0x7f800000u || xb > 0x7f800000u || (xa == xb && (xa == 0u || xa == 0x7f800000u)) ? 0x7fc00000u
                             : xa == 0x7f800000u || xb == 0u                                                       ? sign | 0x7f800000u
                                                                                                                   : sign);
    return xa == 0u || xb == 0u || xa >= 0x7f800000u || xb >= 0x7f800000u;
}
float l_fdiv(float a, float b)
{
    float q;
#ifdef L_EXACT
    return l_fdivfast(a, b, q) ? q : l_fdivexact(a, b);
#else
    if (!l_fdivfast(a, b, q))
        l_flag = true;
    return q;
#endif
}
// the exact computation is in a loop the compiler keeps, so that its code isn't repeated for each component
vec3 l_fdivv(vec3 a, vec3 b)
{
    vec3 q;
    bvec3 fast = bvec3(l_fdivfast(a.x, b.x, q.x), l_fdivfast(a.y, b.y, q.y), l_fdivfast(a.z, b.z, q.z));
#ifdef L_EXACT
    [[dont_unroll]] for (int c = 0; c < 3; c++)
    {
        if (!fast[c])
            q[c] = l_fdivexact(a[c], b[c]);
    }
#else
    if (!all(fast))
        l_flag = true;
#endif
    return q;
}
float l_fsqrt(float a)
{
    float q = sqrt(a);
    uint ua = floatBitsToUint(a), uq = floatBitsToUint(q);
    // as for quotients, a power of two is moved to the float below it
    if ((uq & 0x7fffffu) == 0u && ((uq >> 23) & 0xffu) > 1u && ((uq >> 23) & 0xffu) < 255u)
        uq -= 1u;
    uint ea = (ua >> 23) & 0xffu, eq = (uq >> 23) & 0xffu;
    // a = q^2 * 2^(t - 23 * 2) * (m * 2^t) / qm^2: the root is correctly rounded when m * 2^t - qm^2 is in (-qm, qm]
    int t = int(ea) + 150 - 2 * int(eq);
    uint m = (ua & 0x7fffffu) | 0x800000u, qm = (uq & 0x7fffffu) | 0x800000u;
    uint x = m << (t & 31);
    int k = int(floor(float(int(x - qm * qm)) * (0.5 / float(qm)) + 0.5));
    uint root = qm + uint(k);
    int r = int(x - root * root);
    // the estimate of k can be off by one near a tie
    if (r <= -int(root))
        root -= 1u;
    else if (r > int(root))
        root += 1u;
    r = int(x - root * root);
    // t is 25 or 26 when the root is in the binade below the exact one, as a power of two moved below is
    bool close = abs(float(m) * float(1u << uint(t & 31)) - float(qm) * float(qm)) < 536870912.0;
    bool normal = ua - 0x00800000u < 0x7f000000u && uint(t - 23) <= 3u && close;
    // as for quotients, past the next power of two P units are twice as large: the root is P or the float above it, each
    // correctly rounded when it's within half of the units around it
    if (normal && root - 0x800000u < 0x800000u && r > -int(root) && r <= int(root))
        return uintBitsToFloat(uq + (root - qm));
    uint p = 0x1000000u, c = 0x1000002u;
    int rp = int(x - p * p), rc = int(x - c * c);
    if (normal && root - p < 4u && rp > -int(p) && rp <= 2 * int(p))
        return uintBitsToFloat(uq + (p - qm));
    if (normal && root - p < 4u && rc > 1 - 2 * int(c) && rc <= 2 * int(c))
        return uintBitsToFloat(uq + (p - qm) + 1u);
    // zeros and infinities are their roots, and negative numbers and NaN have none
    if ((ua & 0x7fffffffu) == 0u || ua == 0x7f800000u)
        return a;
    if (ua > 0x7f800000u)
        return uintBitsToFloat(0x7fc00000u);
#ifdef L_EXACT
    return l_fsqrtexact(a);
#else
    l_flag = true;
    return q;
#endif
}
// fmod is exact, as in the VM: the remainder of the significands, shifted one bit at a time
float l_fmod(float a, float b)
{
    uint ua = floatBitsToUint(a), xa = ua & 0x7fffffffu, xb = floatBitsToUint(b) & 0x7fffffffu;
    if (xa >= 0x7f800000u || xb > 0x7f800000u || xb == 0u)
        return uintBitsToFloat(0x7fc00000u);
    if (xa < xb)
        return a;
    int ea = int(xa >> 23), eb = int(xb >> 23);
    uint ma = ea == 0 ? xa : (xa & 0x7fffffu) | 0x800000u, mb = eb == 0 ? xb : (xb & 0x7fffffu) | 0x800000u;
    ea = max(ea, 1);
    eb = max(eb, 1);
    uint r = ma % mb;
    [[dont_unroll]] for (int d = ea - eb; d > 0; d--)
        r = (r << 1) % mb;
    return uintBitsToFloat(floatBitsToUint(ldexp(float(r), eb - 150)) | (ua & 0x80000000u));
}
)";

// a number as a float literal: the VM's double rounded to the nearest float, as f32 numbers are
std::string literal(double v)
{
    float f = float(v);
    if (isnan(f))
        return "uintBitsToFloat(0x7fc00000u)";
    if (isinf(f))
        return f > 0 ? "uintBitsToFloat(0x7f800000u)" : "uintBitsToFloat(0xff800000u)";

    char buf[64];
    snprintf(buf, sizeof(buf), "%.9g", double(f));
    std::string s = buf;
    if (s.find_first_of(".e") == std::string::npos)
        s += ".0";
    return "float(" + s + ")";
}

const char* conditionOperator(IrCondition cond, bool& negate, bool& isUnsigned)
{
    negate = false;
    isUnsigned = false;

    switch (cond)
    {
    case IrCondition::Equal:
        return "==";
    case IrCondition::NotEqual:
        return "!=";
    case IrCondition::Less:
        return "<";
    case IrCondition::NotLess:
        negate = true;
        return "<";
    case IrCondition::LessEqual:
        return "<=";
    case IrCondition::NotLessEqual:
        negate = true;
        return "<=";
    case IrCondition::Greater:
        return ">";
    case IrCondition::NotGreater:
        negate = true;
        return ">";
    case IrCondition::GreaterEqual:
        return ">=";
    case IrCondition::NotGreaterEqual:
        negate = true;
        return ">=";
    case IrCondition::UnsignedLess:
        isUnsigned = true;
        return "<";
    case IrCondition::UnsignedLessEqual:
        isUnsigned = true;
        return "<=";
    case IrCondition::UnsignedGreater:
        isUnsigned = true;
        return ">";
    case IrCondition::UnsignedGreaterEqual:
        isUnsigned = true;
        return ">=";
    default:
        CODEGEN_ASSERT(!"unexpected condition");
        return "==";
    }
}

std::string compare(const std::string& a, const std::string& b, IrCondition cond)
{
    bool negate, isUnsigned;
    const char* op = conditionOperator(cond, negate, isUnsigned);

    std::string res = isUnsigned ? "(uint(" + a + ") " + op + " uint(" + b + "))" : "(" + a + " " + op + " " + b + ")";
    return negate ? "(!" + res + ")" : res;
}

enum class GpuCall : int8_t
{
    None,
    Unknown,
    AtomicAdd,
    AtomicMin,
    BufferLen,
};

struct GlslBuilder
{
    std::string reason;
    std::string glsl;
    std::vector<bool> bufferWritten;
};

// what a VM register holds at a point of the function
struct RegState
{
    int buffer = -1;              // binding of its buffer, -1 for none, -2 when it isn't known statically
    GpuCall call = GpuCall::None; // import it holds
    int tag = -1;                 // its tag, -1 when it isn't known statically
};

struct GlslLowering
{
    IrFunction& function;
    Proto* proto;
    GlslBuilder& build;

    std::vector<bool> lowered;     // per block: lowered, or a fallback/exit that fails the run
    std::vector<int> instBuffer;   // per instruction producing a buffer pointer or TValue
    std::vector<GpuCall> instCall; // per instruction loading a TValue: the import it holds
    std::vector<int> instTag;      // per instruction loading a tag or TValue: the tag, -1 when it isn't known statically

    // registers are reused for values of different types, so what they hold is tracked along the control flow
    std::vector<std::vector<RegState>> entry; // per block, when it starts
    std::vector<bool> reached;
    std::vector<RegState> cur; // at the instruction being analyzed or lowered
    bool changed = false;

    // while lowering an instruction: the line set in l_at last in the block, -1 when unknown, and the kind of failure of a
    // check that falls back
    int lastLine = -1;
    GpuFailure failKind = GpuFailure::Fallback;
    std::vector<uint32_t> alignedBases; // values whose alignment an access of the block checked

    // structured control flow, as in "Beyond Relooper" (Ramsey, 2022): the IR of Luau code is reducible, so each block is
    // emitted once, in its immediate dominator, inside a loop if it's a loop header, after the blocks that end where the
    // block's dominator tree children with several predecessors start. Jumps across constructs go through l_jmp. Regions
    // that only end where a merge starts are switches rather than loops left at once: drivers compile loops in a time that
    // grows with the square of their number (NVIDIA: 0.6 s for 800, 0.14 s as switches)
    struct Construct
    {
        uint32_t block; // loop header, or block that follows
        bool loop;
        uint32_t exit = ~0u;    // for a loop: the block that follows it
        bool exited = false;    // some jump from a nested construct ends after this one
        bool restarted = false; // some jump from a nested construct restarts this loop
        bool passed = false;    // some jump from a nested construct ends further out
    };
    std::vector<Construct> constructs;
    std::vector<int> order; // reverse postorder index of reached blocks
    std::vector<int> forwardPreds;
    std::vector<std::vector<uint32_t>> children; // in the dominator tree of the reached blocks
    uint32_t current = 0;                        // block being lowered
    uint32_t fallsTo = ~0u;                      // block that runs next when the code being emitted completes without jumping, if known

    // blocks emitted where the block being lowered jumps to them, once it's lowered; this keeps the recursion that nests
    // blocks out of the lowering of instructions, whose frames are large
    struct Inlined
    {
        uint32_t block;
        uint32_t fallsTo;
    };
    std::vector<Inlined>* inlined = nullptr;

    std::vector<bool> readBuffers;
    std::vector<uint8_t> atomics; // per buffer, a bit per atomic operation used
    std::vector<uint8_t> narrow;  // per buffer: 1, 2 for 8 and 16-bit reads, 4, 8 for 8 and 16-bit writes
    std::vector<int> pcOfInst;

    std::string out;
    bool failed = false;

    GlslLowering(IrFunction& function, Proto* proto, GlslBuilder& build)
        : function(function)
        , proto(proto)
        , build(build)
    {
    }

    bool reject(uint32_t index, const std::string& what)
    {
        if (!failed)
        {
            int pc = index < pcOfInst.size() ? pcOfInst[index] : -1;
            int line = pc >= 0 ? luaG_getline(proto, pc) : -1;

            build.reason = what + " is not supported on the GPU";
            if (line > 0)
                build.reason += " (line " + std::to_string(line) + ")";
        }

        failed = true;
        return false;
    }

    // a / b correctly rounded, so that quotients the VM gets exactly, like 37 / 37, are exact here too; dividing by a power of
    // two is multiplying by its inverse, which is one too: both round the same quotient
    std::string quotient(IrOp a, IrOp b)
    {
        if (double k; constNumber(b, k) && powerOfTwo(k) && fabs(k) >= 0x1p-126 && fabs(k) <= 0x1p126)
            return num(a) + " * " + literal(1 / k);
        return "l_fdiv(" + num(a) + ", " + num(b) + ")";
    }

    static bool powerOfTwo(double k)
    {
        int e;
        return k != 0 && std::isfinite(k) && fabs(frexp(k, &e)) == 0.5;
    }

    // the value of a double constant operand
    bool constNumber(IrOp op, double& v)
    {
        if (op.kind == IrOpKind::Constant && function.constOp(op).kind == IrConstKind::Double)
            v = function.doubleOp(op);
        else if (op.kind == IrOpKind::VmConst && ttisnumber(&proto->k[vmConstOp(op)]))
            v = nvalue(&proto->k[vmConstOp(op)]);
        else
            return false;
        return true;
    }

    // the number of a double operand: a float
    std::string num(IrOp op)
    {
        switch (op.kind)
        {
        case IrOpKind::Inst:
            return "v" + std::to_string(op.index) + (isTvalue(op) ? "n" : "");
        case IrOpKind::Constant:
        {
            IrConst& k = function.constOp(op);
            return literal(k.kind == IrConstKind::Double ? k.valueDouble : double(k.valueInt));
        }
        case IrOpKind::VmReg:
            return "r" + std::to_string(vmRegOp(op)) + "n";
        case IrOpKind::VmConst:
            return literal(nvalue(&proto->k[vmConstOp(op)]));
        default:
            return "0.0";
        }
    }

    std::string intv(IrOp op)
    {
        if (op.kind == IrOpKind::Constant)
        {
            IrConst& c = function.constOp(op);
            return c.kind == IrConstKind::Uint ? "int(" + std::to_string(c.valueUint) + "u)" : "(" + std::to_string(c.valueInt) + ")";
        }
        if (op.kind == IrOpKind::VmReg)
            return "r" + std::to_string(vmRegOp(op)) + "i";
        return "v" + std::to_string(op.index) + (isTvalue(op) ? "i" : "");
    }

    std::string vec(IrOp op)
    {
        if (op.kind == IrOpKind::VmReg)
            return "r" + std::to_string(vmRegOp(op)) + "v";
        if (op.kind == IrOpKind::VmConst)
        {
            const float* v = vvalue(&proto->k[vmConstOp(op)]);
            return "vec3(" + literal(v[0]) + ", " + literal(v[1]) + ", " + literal(v[2]) + ")";
        }
        return "v" + std::to_string(op.index) + "v";
    }

    // instructions producing TValues hold each part in its own variable
    bool isTvalue(IrOp op)
    {
        return op.kind == IrOpKind::Inst && getCmdValueKind(function.instOp(op).cmd) == IrValueKind::Tvalue;
    }

    // tags known statically spare the checks that loop iterations would otherwise repeat
    int knownTag(IrOp op)
    {
        if (op.kind == IrOpKind::Constant)
            return function.tagOp(op);
        if (op.kind == IrOpKind::VmReg)
            return cur[vmRegOp(op)].tag;
        if (op.kind == IrOpKind::VmConst)
            return ttype(&proto->k[vmConstOp(op)]);
        if (op.kind == IrOpKind::Inst)
            return instTag[op.index];
        return -1;
    }

    // the target a tag check or comparison always takes, if known statically
    IrOp knownTarget(IrInst& inst)
    {
        int a = knownTag(OP_A(inst)), b = knownTag(OP_B(inst));
        if (a < 0 || b < 0)
            return {};
        if (inst.cmd == IrCmd::JUMP_EQ_TAG)
            return a == b ? OP_C(inst) : OP_D(inst);
        if (inst.cmd == IrCmd::CHECK_TAG && a != b)
            return OP_C(inst);
        return {};
    }

    std::string tag(IrOp op)
    {
        if (int k = knownTag(op); k >= 0)
            return std::to_string(k) + "u";
        if (op.kind == IrOpKind::Constant)
            return std::to_string(function.tagOp(op)) + "u";
        if (op.kind == IrOpKind::VmReg)
            return "r" + std::to_string(vmRegOp(op)) + "t";
        if (op.kind == IrOpKind::VmConst)
            return std::to_string(ttype(&proto->k[vmConstOp(op)])) + "u";
        return "v" + std::to_string(op.index) + (isTvalue(op) ? "t" : "");
    }

    int buffer(IrOp op)
    {
        if (op.kind == IrOpKind::Inst)
            return instBuffer[op.index];
        if (op.kind == IrOpKind::VmReg)
            return cur[vmRegOp(op)].buffer;
        return -1;
    }

    GpuCall importCall(uint32_t id)
    {
        if ((id >> 30) != 2)
            return GpuCall::Unknown;

        TValue* lib = &proto->k[(id >> 20) & 1023];
        TValue* name = &proto->k[(id >> 10) & 1023];
        if (!ttisstring(lib) || !ttisstring(name))
            return GpuCall::Unknown;

        std::string l = svalue(lib), n = svalue(name);
        if (l == "gpu" && n == "atomicadd")
            return GpuCall::AtomicAdd;
        if (l == "gpu" && n == "atomicmin")
            return GpuCall::AtomicMin;
        if (l == "buffer" && n == "len")
            return GpuCall::BufferLen;
        return GpuCall::Unknown;
    }

    // facts about the value an instruction produces, from what registers hold before it
    void facts(IrInst& inst, uint32_t index)
    {
        if (inst.cmd == IrCmd::LOAD_POINTER || inst.cmd == IrCmd::LOAD_TVALUE)
            instBuffer[index] = buffer(OP_A(inst));

        if (inst.cmd == IrCmd::LOAD_TAG || inst.cmd == IrCmd::LOAD_TVALUE)
            instTag[index] = knownTag(OP_A(inst));

        if (inst.cmd == IrCmd::LOAD_TVALUE)
            instCall[index] = OP_A(inst).kind == IrOpKind::VmReg ? cur[vmRegOp(OP_A(inst))].call : GpuCall::Unknown;
    }

    // what registers hold after an instruction
    void effects(IrInst& inst, uint32_t index)
    {
        // past a tag check, the register it loaded the tag of holds that tag
        if (inst.cmd == IrCmd::CHECK_TAG && OP_A(inst).kind == IrOpKind::Inst && OP_B(inst).kind == IrOpKind::Constant && index > 0)
        {
            IrInst& load = function.instOp(OP_A(inst));
            if (load.cmd == IrCmd::LOAD_TAG && OP_A(load).kind == IrOpKind::VmReg && OP_A(inst).index == index - 1)
                cur[vmRegOp(OP_A(load))].tag = function.tagOp(OP_B(inst));
        }

        if (!writesRegister(inst) && inst.cmd != IrCmd::GET_CACHED_IMPORT)
            return;

        int r = vmRegOp(OP_A(inst));
        RegState s = {-1, GpuCall::Unknown};

        switch (inst.cmd)
        {
        case IrCmd::STORE_TAG:
            // the value is stored separately; only a register that may hold a buffer keeps it
            s = cur[r];
            s.call = GpuCall::Unknown;
            if (OP_B(inst).kind != IrOpKind::Constant || function.tagOp(OP_B(inst)) != LUA_TBUFFER)
                s.buffer = -1;
            s.tag = knownTag(OP_B(inst));
            break;
        case IrCmd::STORE_DOUBLE:
        case IrCmd::STORE_INT:
            s.tag = cur[r].tag;
            break;
        case IrCmd::STORE_VECTOR:
            s.tag = OP_E(inst).kind != IrOpKind::None ? knownTag(OP_E(inst)) : cur[r].tag;
            break;
        case IrCmd::STORE_TVALUE:
        {
            IrOp src = OP_B(inst);
            if (src.kind == IrOpKind::Inst && function.instOp(src).cmd == IrCmd::LOAD_TVALUE)
                s = {instBuffer[src.index] >= 0 ? instBuffer[src.index] : -1, instCall[src.index], instTag[src.index]};
            break;
        }
        case IrCmd::STORE_SPLIT_TVALUE:
            s.tag = knownTag(OP_B(inst));
            if (function.tagOp(OP_B(inst)) == LUA_TBUFFER)
                s.buffer = buffer(OP_C(inst));
            break;
        case IrCmd::STORE_POINTER:
            s.buffer = buffer(OP_B(inst));
            s.tag = cur[r].tag;
            break;
        case IrCmd::GET_CACHED_IMPORT:
            s.call = importCall(function.importOp(OP_C(inst)));
            break;
        case IrCmd::CALL:
        {
            // the calls that can be lowered return a number, and the arguments don't survive the call
            int nargs = function.intOp(OP_B(inst));
            for (int i = 1; i <= nargs && r + i < int(cur.size()); i++)
                cur[r + i] = {-1, GpuCall::Unknown};
            s.tag = LUA_TNUMBER;
            break;
        }
        default:
            break;
        }

        cur[r] = s;
    }

    // merges what registers hold at a jump into what they hold when the target block starts
    void reach(IrOp target)
    {
        if (target.kind != IrOpKind::Block || !lowered[target.index])
            return;

        std::vector<RegState>& to = entry[target.index];
        bool first = !reached[target.index];
        reached[target.index] = true;
        changed |= first;

        // only registers the block reads before writing matter; the others hold values that are never used
        for (size_t r = 0; r < to.size(); r++)
        {
            if (!live(target.index, r))
                continue;

            if (first)
            {
                to[r] = cur[r];
                continue;
            }

            RegState m = {
                to[r].buffer == cur[r].buffer ? to[r].buffer : -2,
                to[r].call == cur[r].call ? to[r].call : GpuCall::Unknown,
                to[r].tag == cur[r].tag ? to[r].tag : -1
            };
            changed |= m.buffer != to[r].buffer || m.call != to[r].call || m.tag != to[r].tag;
            to[r] = m;
        }
    }

    bool live(uint32_t block, size_t reg)
    {
        const RegisterSet& in = function.cfg.in[block];
        return in.regs.test(reg) || (in.varargSeq && reg >= in.varargStart);
    }

    void reachTargets(IrInst& inst)
    {
        // a tag check that always passes doesn't reach its fallback, nor a tag comparison known statically its other target
        if (inst.cmd == IrCmd::CHECK_TAG || inst.cmd == IrCmd::JUMP_EQ_TAG)
        {
            int a = knownTag(OP_A(inst)), b = knownTag(OP_B(inst));
            if (a >= 0 && b >= 0)
            {
                reach(knownTarget(inst));
                return;
            }
        }

        for (IrOp op : inst.ops)
            reach(op);
    }

    // finds what registers hold when each block starts; unknowns only grow, so this terminates
    bool analyze(const std::vector<uint32_t>& sortedBlocks)
    {
        size_t size = function.instructions.size();
        instBuffer.assign(size, -1);
        instCall.assign(size, GpuCall::None);
        instTag.assign(size, -1);

        // the parameters are the buffer and the index
        std::vector<RegState> start(proto->maxstacksize);
        start[0].buffer = 0;
        start[0].tag = LUA_TBUFFER;
        if (proto->numparams > 1)
            start[1].tag = LUA_TNUMBER;

        entry.assign(function.blocks.size(), std::vector<RegState>(proto->maxstacksize));
        reached.assign(function.blocks.size(), false);
        entry[function.entryBlock] = start;
        reached[function.entryBlock] = true;

        for (changed = true; changed;)
        {
            changed = false;

            for (uint32_t blockIndex : sortedBlocks)
            {
                if (!lowered[blockIndex] || !reached[blockIndex])
                    continue;

                IrBlock& block = function.blocks[blockIndex];
                cur = entry[blockIndex];

                for (uint32_t index = block.start; index <= block.finish; index++)
                {
                    IrInst& inst = function.instructions[index];

                    if (isPseudo(inst.cmd))
                        continue;

                    if ((inst.cmd == IrCmd::STORE_DOUBLE || inst.cmd == IrCmd::STORE_SPLIT_TVALUE || inst.cmd == IrCmd::STORE_TVALUE ||
                         inst.cmd == IrCmd::STORE_POINTER) &&
                        OP_A(inst).kind != IrOpKind::VmReg)
                        return reject(index, "a store outside of the stack");

                    // a failing tag check leaves with what registers held before it
                    facts(inst, index);
                    if (inst.cmd == IrCmd::CHECK_TAG)
                        reachTargets(inst);
                    effects(inst, index);
                    if (inst.cmd != IrCmd::CHECK_TAG)
                        reachTargets(inst);
                }
            }
        }

        return true;
    }

    void line(const std::string& s)
    {
        out += "        " + s + "\n";
    }

    std::string jump(IrOp target)
    {
        if (target.kind == IrOpKind::Block && lowered[target.index])
        {
            return "{ " + branch(target.index) + " }";
        }

        // fallbacks, exits to the VM and aborts fail the run
        return "{ l_fail(" + std::to_string(int(failKind)) + "u); return; }";
    }

    bool backward(uint32_t from, uint32_t to)
    {
        return order[to] <= order[from];
    }

    bool merge(uint32_t block)
    {
        return forwardPreds[block] > 1;
    }

    static std::string breakCode(uint32_t block)
    {
        return std::to_string(block * 2 + 2);
    }

    static std::string continueCode(uint32_t block)
    {
        return std::to_string(block * 2 + 3);
    }

    // leaves the constructs up to the one that restarts 'block' (a loop) or that 'block' follows
    std::string leave(uint32_t block, bool loop)
    {
        for (size_t i = constructs.size(); i > 0; i--)
        {
            Construct& c = constructs[i - 1];
            if (loop ? !c.loop || c.block != block : (c.loop ? c.exit : c.block) != block)
                continue;

            // regions are switches, which a continue goes through
            bool innermost = true;
            for (size_t j = i; j < constructs.size(); j++)
                innermost &= loop && !constructs[j].loop;
            if (i == constructs.size() || innermost)
                return loop ? "continue;" : "break;";

            (loop ? c.restarted : c.exited) = true;
            for (size_t j = i; j < constructs.size(); j++)
                constructs[j].passed = true;
            return "l_jmp = " + (loop ? continueCode(block) : breakCode(block)) + "; break;";
        }

        CODEGEN_ASSERT(!"jump to a construct that doesn't enclose it");
        return "l_fail(" + std::to_string(int(GpuFailure::Fallback)) + "u); return;";
    }

    std::string branch(uint32_t target)
    {
        if (backward(current, target))
            return leave(target, true);
        if (merge(target))
            return leave(target, false);

        // the only way into the target: it's emitted right here
        inlined->push_back({target, fallsTo});
        return "\x01" + std::to_string(inlined->size() - 1) + "\x01";
    }

    // after a construct closes: a jump that ends here stops, one to a construct further out keeps leaving
    std::string landing(const Construct& closed)
    {
        std::string code;
        if (closed.exited)
            code += "if (l_jmp == " + breakCode(closed.loop ? closed.exit : closed.block) + ") l_jmp = 0;\n";

        if (closed.passed)
        {
            const Construct& parent = constructs.back();
            code += "if (l_jmp != 0) { ";
            if (parent.loop && parent.restarted)
                code += "if (l_jmp == " + continueCode(parent.block) + ") { l_jmp = 0; continue; } ";
            code += "break; }\n";
        }

        return code;
    }

    // the blocks of the loop that 'header' starts: those that reach it again without going through it
    std::vector<bool> loopBody(uint32_t header)
    {
        std::vector<bool> body(function.blocks.size(), false);
        body[header] = true;

        std::vector<uint32_t> stack;
        for (uint32_t pred : predecessors(function.cfg, header))
        {
            if (lowered[pred] && reached[pred] && backward(pred, header) && !body[pred])
            {
                body[pred] = true;
                stack.push_back(pred);
            }
        }

        while (!stack.empty())
        {
            uint32_t block = stack.back();
            stack.pop_back();

            for (uint32_t pred : predecessors(function.cfg, block))
            {
                if (lowered[pred] && reached[pred] && !body[pred])
                {
                    body[pred] = true;
                    stack.push_back(pred);
                }
            }
        }

        return body;
    }

    std::string tree(uint32_t block)
    {
        std::vector<uint32_t> merges;
        for (uint32_t child : children[block])
        {
            if (merge(child))
                merges.push_back(child);
        }

        // the merge reached last ends outermost
        std::sort(
            merges.begin(),
            merges.end(),
            [&](uint32_t a, uint32_t b)
            {
                return order[a] > order[b];
            }
        );

        bool header = false;
        for (uint32_t pred : predecessors(function.cfg, block))
            header |= lowered[pred] && reached[pred] && backward(pred, block);

        if (!header)
            return within(block, merges, 0);

        // the merges outside of the loop follow it, so that leaving the loop for the nearest one is a plain break
        std::vector<bool> body = loopBody(block);
        std::vector<uint32_t> inner, exits;
        for (uint32_t m : merges)
            (body[m] ? inner : exits).push_back(m);

        return around(block, exits, 0, inner);
    }

    std::string around(uint32_t header, const std::vector<uint32_t>& exits, size_t i, const std::vector<uint32_t>& inner)
    {
        if (i + 1 < exits.size())
        {
            constructs.push_back({exits[i], false});
            uint32_t after = fallsTo;
            fallsTo = exits[i];
            std::string code = "switch (0)\n{\ndefault:\n" + around(header, exits, i + 1, inner) + "break;\n}\n";
            fallsTo = after;
            Construct region = constructs.back();
            constructs.pop_back();
            return code + landing(region) + tree(exits[i]);
        }

        // without a merge of its own after it, the loop can still be left with a plain break for the block that runs once the
        // code around it completes
        constructs.push_back({header, true, i < exits.size() ? exits[i] : fallsTo});
        uint32_t after = fallsTo;
        fallsTo = ~0u;
        std::string code = "for (;;)\n{\n" + within(header, inner, 0) + "}\n";
        fallsTo = after;
        Construct loop = constructs.back();
        constructs.pop_back();
        code += landing(loop);
        return i < exits.size() ? code + tree(exits[i]) : code;
    }

    std::string within(uint32_t block, const std::vector<uint32_t>& merges, size_t i)
    {
        if (i == merges.size())
            return blockCode(block);

        constructs.push_back({merges[i], false});
        uint32_t after = fallsTo;
        fallsTo = merges[i];
        std::string code = "switch (0)\n{\ndefault:\n" + within(block, merges, i + 1) + "break;\n}\n";
        fallsTo = after;
        Construct region = constructs.back();
        constructs.pop_back();
        return code + landing(region) + tree(merges[i]);
    }

    std::string blockCode(uint32_t blockIndex)
    {
        current = blockIndex;
        cur = entry[blockIndex];

        std::string saved;
        saved.swap(out);

        std::vector<Inlined> targets;
        std::vector<Inlined>* outer = inlined;
        inlined = &targets;

        int outerLine = lastLine;
        lastLine = -1;
        std::vector<uint32_t> outerBases;
        outerBases.swap(alignedBases);

        IrBlock& block = function.blocks[blockIndex];
        for (uint32_t index = block.start; index <= block.finish && !failed; index++)
        {
            IrInst& inst = function.instructions[index];
            if (isPseudo(inst.cmd))
                continue;

            // failures report the line of the instruction that failed, which the compiler tracks for free since it's
            // constant between jumps
            int pc = pcOfInst[index];
            int at = pc >= 0 ? std::min(luaG_getline(proto, pc), 0xffffff) : 0;
            if (at != lastLine)
                line("l_at = " + std::to_string(std::max(at, 0)) + "u;");
            lastLine = at;
            failKind = inst.cmd == IrCmd::CHECK_BUFFER_LEN ? GpuFailure::Access : GpuFailure::Fallback;

            facts(inst, index);
            if (!lowerInst(inst, index))
                break;
            effects(inst, index);

            // code jumped to may set another line
            if (isBlockTerminator(inst.cmd) || isNonTerminatingJump(inst.cmd))
                lastLine = -1;
        }
        lastLine = outerLine;
        alignedBases.swap(outerBases);

        inlined = outer;
        saved.swap(out);

        if (targets.empty())
            return saved;

        std::string code;
        uint32_t after = fallsTo;
        for (size_t pos = 0;;)
        {
            size_t start = saved.find('\x01', pos);
            if (start == std::string::npos)
            {
                code.append(saved, pos, std::string::npos);
                break;
            }

            size_t end = saved.find('\x01', start + 1);
            const Inlined& target = targets[std::stoul(saved.substr(start + 1, end - start - 1))];
            code.append(saved, pos, start - pos);
            fallsTo = target.fallsTo;
            code += tree(target.block);
            pos = end + 1;
        }
        fallsTo = after;
        return code;
    }

    // reverse postorder of the reached blocks, and their predecessors along forward edges
    void orderBlocks()
    {
        order.assign(function.blocks.size(), 0);
        forwardPreds.assign(function.blocks.size(), 0);

        std::vector<uint32_t> postorder;
        std::vector<uint8_t> visited(function.blocks.size(), 0);
        std::vector<std::pair<uint32_t, bool>> stack = {{function.entryBlock, false}};
        while (!stack.empty())
        {
            auto [block, done] = stack.back();
            stack.pop_back();

            if (done)
            {
                postorder.push_back(block);
                continue;
            }
            if (visited[block])
                continue;

            visited[block] = 1;
            stack.push_back({block, true});
            for (uint32_t succ : successors(function.cfg, block))
            {
                if (lowered[succ] && reached[succ] && !visited[succ])
                    stack.push_back({succ, false});
            }
        }

        for (size_t i = 0; i < postorder.size(); i++)
            order[postorder[i]] = int(postorder.size() - 1 - i);

        for (uint32_t block : postorder)
        {
            for (uint32_t pred : predecessors(function.cfg, block))
            {
                if (lowered[pred] && reached[pred] && !backward(pred, block))
                    forwardPreds[block]++;
            }
        }

        // dominators of the blocks that are actually lowered, which jumps to fallbacks don't change ("A Simple, Fast Dominance
        // Algorithm", Cooper, Harvey and Kennedy)
        std::vector<uint32_t> rpo(postorder.rbegin(), postorder.rend());
        std::vector<int> idom(function.blocks.size(), -1);
        idom[function.entryBlock] = int(function.entryBlock);

        for (bool progress = true; progress;)
        {
            progress = false;
            for (size_t i = 1; i < rpo.size(); i++)
            {
                int dom = -1;
                for (uint32_t pred : predecessors(function.cfg, rpo[i]))
                {
                    if (!lowered[pred] || !reached[pred] || idom[pred] < 0)
                        continue;

                    if (dom < 0)
                    {
                        dom = int(pred);
                        continue;
                    }

                    int a = int(pred), b = dom;
                    while (a != b)
                    {
                        while (order[a] > order[b])
                            a = idom[a];
                        while (order[b] > order[a])
                            b = idom[b];
                    }
                    dom = a;
                }

                if (dom != idom[rpo[i]])
                {
                    idom[rpo[i]] = dom;
                    progress = true;
                }
            }
        }

        children.assign(function.blocks.size(), {});
        for (size_t i = 1; i < rpo.size(); i++)
            children[idom[rpo[i]]].push_back(rpo[i]);
    }

    void guard(const std::string& ok, IrOp target)
    {
        // the rest of the block follows the code jumped to
        uint32_t after = fallsTo;
        fallsTo = ~0u;
        line("if (!" + ok + ") " + jump(target));
        fallsTo = after;
    }

    std::string truthy(IrOp reg)
    {
        std::string r = "r" + std::to_string(vmRegOp(reg));
        return "(" + r + "t != 0u && !(" + r + "t == 1u && " + r + "i == 0))";
    }

    std::string compareNum(IrOp a, IrOp b, IrCondition cond)
    {
        return compare(num(a), num(b), cond);
    }

    // an offset as a value plus a constant, the value being ~0u when the offset is a constant
    void offsetParts(IrOp op, uint32_t& base, int64_t& k)
    {
        k = 0;
        while (op.kind == IrOpKind::Inst && (function.instOp(op).cmd == IrCmd::ADD_INT || function.instOp(op).cmd == IrCmd::SUB_INT) &&
               OP_B(function.instOp(op)).kind == IrOpKind::Constant)
        {
            IrInst& add = function.instOp(op);
            k += add.cmd == IrCmd::ADD_INT ? function.intOp(OP_B(add)) : -int64_t(function.intOp(OP_B(add)));
            op = OP_A(add);
        }

        base = ~0u;
        if (op.kind == IrOpKind::Constant)
            k += function.intOp(op);
        else if (op.kind == IrOpKind::Inst)
            base = op.index;
    }

    std::string access(int binding, IrOp offset, const char* op, const std::string& value = "")
    {
        if (op[0] == 'r')
            readBuffers[binding] = true;
        else
            build.bufferWritten[binding] = true;

        // an access at a multiple of 4 from a value an earlier access of the block checked is aligned too: the check that
        // offset passed, or the run fails anyway
        uint32_t base;
        int64_t k;
        offsetParts(offset, base, k);
        bool aligned = (k & 3) == 0 && (base == ~0u || std::find(alignedBases.begin(), alignedBases.end(), base) != alignedBases.end());
        if ((k & 3) == 0 && base != ~0u && !aligned)
            alignedBases.push_back(base);

        std::string b = std::to_string(binding);
        std::string name = std::string(op[0] == 'r' ? "l_rd" : "l_wr") + (aligned ? "a" : "") + b;
        return value.empty() ? name + "(" + intv(offset) + ")" : name + "(" + intv(offset) + ", " + value + ")";
    }

    bool binding(uint32_t index, IrOp op, int& b)
    {
        b = buffer(op);
        return b >= 0 ? true : reject(index, "a buffer other than the kernel's parameter");
    }

    static bool writesRegister(IrInst& inst)
    {
        switch (inst.cmd)
        {
        case IrCmd::STORE_TAG:
        case IrCmd::STORE_DOUBLE:
        case IrCmd::STORE_INT:
        case IrCmd::STORE_VECTOR:
        case IrCmd::STORE_TVALUE:
        case IrCmd::STORE_SPLIT_TVALUE:
        case IrCmd::STORE_POINTER:
        case IrCmd::CALL:
            return OP_A(inst).kind == IrOpKind::VmReg;
        default:
            return false;
        }
    }

    bool lowerInst(IrInst& inst, uint32_t index)
    {
        std::string v = "v" + std::to_string(index);

        switch (inst.cmd)
        {
        case IrCmd::NOP:
        case IrCmd::CHECK_SAFE_ENV: // the host only translates functions whose environment is safe
        case IrCmd::CHECK_GC:
        case IrCmd::SET_SAVEDPC:
        case IrCmd::STORE_POINTER: // buffer bindings are static, see analyze
            break;
        case IrCmd::LOAD_TAG:
            line(v + " = " + tag(OP_A(inst)) + ";");
            break;
        case IrCmd::LOAD_POINTER:
            break;
        case IrCmd::LOAD_DOUBLE:
            line(v + " = " + num(OP_A(inst)) + ";");
            break;
        case IrCmd::LOAD_INT:
            line(v + " = " + intv(OP_A(inst)) + ";");
            break;
        case IrCmd::LOAD_TVALUE:
        {
            IrOp a = OP_A(inst);
            if (a.kind == IrOpKind::VmReg)
            {
                std::string r = "r" + std::to_string(vmRegOp(a));
                line(v + "t = " + r + "t; " + v + "n = " + r + "n; " + v + "i = " + r + "i; " + v + "v = " + r + "v;");
            }
            else if (a.kind == IrOpKind::VmConst)
            {
                TValue* k = &proto->k[vmConstOp(a)];
                if (ttisvector(k))
                    line(v + "t = " + std::to_string(LUA_TVECTOR) + "u; " + v + "v = " + vec(a) + ";");
                else if (ttisnumber(k))
                    line(v + "t = " + std::to_string(LUA_TNUMBER) + "u; " + v + "n = " + literal(nvalue(k)) + ";");
                else if (ttisboolean(k))
                    line(v + "t = " + std::to_string(LUA_TBOOLEAN) + "u; " + v + "i = " + std::to_string(bvalue(k)) + ";");
                else if (ttisnil(k))
                    line(v + "t = 0u;");
                else
                    return reject(index, "a constant of this type");
            }
            else
                return reject(index, "a value load from memory");
            break;
        }
        case IrCmd::STORE_TAG:
            line("r" + std::to_string(vmRegOp(OP_A(inst))) + "t = " + tag(OP_B(inst)) + ";");
            break;
        case IrCmd::STORE_DOUBLE:
            line("r" + std::to_string(vmRegOp(OP_A(inst))) + "n = " + num(OP_B(inst)) + ";");
            break;
        case IrCmd::STORE_INT:
            line("r" + std::to_string(vmRegOp(OP_A(inst))) + "i = " + intv(OP_B(inst)) + ";");
            break;
        case IrCmd::STORE_VECTOR:
        {
            std::string r = "r" + std::to_string(vmRegOp(OP_A(inst)));
            line(r + "v = vec3(" + floatv(OP_B(inst)) + ", " + floatv(OP_C(inst)) + ", " + floatv(OP_D(inst)) + ");");
            if (OP_E(inst).kind != IrOpKind::None)
                line(r + "t = " + tag(OP_E(inst)) + ";");
            break;
        }
        case IrCmd::LOAD_FLOAT:
            line(v + " = " + vec(OP_A(inst)) + "[" + std::to_string(function.intOp(OP_B(inst)) / 4) + "];");
            break;
        case IrCmd::ADD_VEC:
        case IrCmd::SUB_VEC:
        case IrCmd::MUL_VEC:
        case IrCmd::DIV_VEC:
        {
            if (inst.cmd == IrCmd::DIV_VEC)
            {
                std::string a = vec(OP_A(inst)), b = vec(OP_B(inst));
                line(v + "v = l_fdivv(" + a + ", " + b + ");");
                break;
            }
            const char* op = inst.cmd == IrCmd::ADD_VEC ? " + " : inst.cmd == IrCmd::SUB_VEC ? " - " : " * ";
            line(v + "v = " + vec(OP_A(inst)) + op + vec(OP_B(inst)) + ";");
            break;
        }
        case IrCmd::IDIV_VEC:
        {
            std::string a = vec(OP_A(inst)), b = vec(OP_B(inst));
            line(v + "v = floor(l_fdivv(" + a + ", " + b + "));");
            break;
        }
        case IrCmd::MULADD_VEC:
            line(v + "v = " + vec(OP_A(inst)) + " * " + vec(OP_B(inst)) + " + " + vec(OP_C(inst)) + ";");
            break;
        case IrCmd::UNM_VEC:
            line(v + "v = -" + vec(OP_A(inst)) + ";");
            break;
        case IrCmd::MIN_VEC:
        case IrCmd::MAX_VEC:
            line(v + "v = " + (inst.cmd == IrCmd::MIN_VEC ? "l_minv(" : "l_maxv(") + vec(OP_A(inst)) + ", " + vec(OP_B(inst)) + ");");
            break;
        case IrCmd::FLOOR_VEC:
        case IrCmd::CEIL_VEC:
        case IrCmd::ABS_VEC:
        {
            const char* fn = inst.cmd == IrCmd::FLOOR_VEC ? "floor(" : inst.cmd == IrCmd::CEIL_VEC ? "ceil(" : "abs(";
            line(v + "v = " + fn + vec(OP_A(inst)) + ");");
            break;
        }
        case IrCmd::SELECT_VEC:
            line(v + "v = mix(" + vec(OP_A(inst)) + ", " + vec(OP_B(inst)) + ", equal(" + vec(OP_C(inst)) + ", " + vec(OP_D(inst)) + "));");
            break;
        case IrCmd::DOT_VEC:
        {
            // as the VM computes it, in this order; native code adds the third product to a fourth lane of +0 first, which turns
            // a -0 into +0
            std::string a = vec(OP_A(inst)), b = vec(OP_B(inst));
            line(v + " = (" + a + ".x * " + b + ".x + " + a + ".y * " + b + ".y) + " + a + ".z * " + b + ".z;");
            break;
        }
        case IrCmd::EXTRACT_VEC:
            line(v + " = " + vec(OP_A(inst)) + "[" + std::to_string(function.intOp(OP_B(inst))) + "];");
            break;
        case IrCmd::FLOAT_TO_VEC:
            line(v + "v = vec3(" + floatv(OP_A(inst)) + ");");
            break;
        case IrCmd::TAG_VECTOR:
            line(v + "v = " + vec(OP_A(inst)) + ";");
            break;
        case IrCmd::ADD_FLOAT:
        case IrCmd::SUB_FLOAT:
        case IrCmd::MUL_FLOAT:
        case IrCmd::DIV_FLOAT:
        {
            if (inst.cmd == IrCmd::DIV_FLOAT)
            {
                line(v + " = l_fdiv(" + floatv(OP_A(inst)) + ", " + floatv(OP_B(inst)) + ");");
                break;
            }
            const char* op = inst.cmd == IrCmd::ADD_FLOAT ? " + " : inst.cmd == IrCmd::SUB_FLOAT ? " - " : " * ";
            line(v + " = " + floatv(OP_A(inst)) + op + floatv(OP_B(inst)) + ";");
            break;
        }
        case IrCmd::MIN_FLOAT:
        case IrCmd::MAX_FLOAT:
        {
            std::string a = floatv(OP_A(inst)), b = floatv(OP_B(inst));
            line(v + " = " + a + (inst.cmd == IrCmd::MIN_FLOAT ? " < " : " > ") + b + " ? " + a + " : " + b + ";");
            break;
        }
        case IrCmd::UNM_FLOAT:
            line(v + " = -" + floatv(OP_A(inst)) + ";");
            break;
        case IrCmd::FLOOR_FLOAT:
        case IrCmd::CEIL_FLOAT:
        case IrCmd::ABS_FLOAT:
        {
            const char* fn = inst.cmd == IrCmd::FLOOR_FLOAT ? "floor(" : inst.cmd == IrCmd::CEIL_FLOAT ? "ceil(" : "abs(";
            line(v + " = " + fn + floatv(OP_A(inst)) + ");");
            break;
        }
        case IrCmd::SQRT_FLOAT:
            line(v + " = l_fsqrt(" + floatv(OP_A(inst)) + ");");
            break;
        case IrCmd::SIGN_FLOAT:
        {
            std::string a = floatv(OP_A(inst));
            line(v + " = " + a + " > 0.0 ? 1.0 : " + a + " < 0.0 ? -1.0 : 0.0;");
            break;
        }
        case IrCmd::UINT_TO_FLOAT:
            line(v + " = l_u2f(uint(" + intv(OP_A(inst)) + "));");
            break;
        case IrCmd::INVOKE_LIBM:
            if (!lowerLibm(inst, v))
                return reject(index, "this math function");
            break;
        case IrCmd::GET_CACHED_IMPORT:
            line("r" + std::to_string(vmRegOp(OP_A(inst))) + "t = " + std::to_string(LUA_TFUNCTION) + "u;");
            break;
        case IrCmd::CALL:
            if (!lowerCall(inst, index))
                return false;
            break;
        case IrCmd::STORE_TVALUE:
        {
            int r = vmRegOp(OP_A(inst));
            IrOp src = OP_B(inst);
            if (src.kind != IrOpKind::Inst || getCmdValueKind(function.instOp(src).cmd) != IrValueKind::Tvalue)
                return reject(index, "a value copy from memory");

            std::string s = "v" + std::to_string(src.index);
            std::string rs = "r" + std::to_string(r);

            // vector operations produce TValues holding a vector
            if (function.instOp(src).cmd != IrCmd::LOAD_TVALUE)
            {
                line(rs + "t = " + std::to_string(LUA_TVECTOR) + "u; " + rs + "v = " + s + "v;");
                break;
            }

            line(rs + "t = " + s + "t; " + rs + "n = " + s + "n; " + rs + "i = " + s + "i; " + rs + "v = " + s + "v;");
            break;
        }
        case IrCmd::STORE_SPLIT_TVALUE:
        {
            int r = vmRegOp(OP_A(inst));
            uint8_t t = function.tagOp(OP_B(inst));
            std::string rs = "r" + std::to_string(r);
            line(rs + "t = " + std::to_string(t) + "u;");

            if (t == LUA_TNUMBER)
                line(rs + "n = " + num(OP_C(inst)) + ";");
            else if (t == LUA_TBOOLEAN)
                line(rs + "i = " + intv(OP_C(inst)) + ";");
            else if (t != LUA_TBUFFER)
                return reject(index, "a value of this type");
            break;
        }
        case IrCmd::ADD_INT:
            line(v + " = " + intv(OP_A(inst)) + " + " + intv(OP_B(inst)) + ";");
            break;
        case IrCmd::SUB_INT:
            line(v + " = " + intv(OP_A(inst)) + " - " + intv(OP_B(inst)) + ";");
            break;
        case IrCmd::ADD_NUM:
        case IrCmd::SUB_NUM:
        case IrCmd::MUL_NUM:
        {
            const char* op = inst.cmd == IrCmd::ADD_NUM ? " + " : inst.cmd == IrCmd::SUB_NUM ? " - " : " * ";
            line(v + " = " + num(OP_A(inst)) + op + num(OP_B(inst)) + ";");
            break;
        }
        case IrCmd::MULADD_NUM:
            line(v + " = " + num(OP_A(inst)) + " * " + num(OP_B(inst)) + " + " + num(OP_C(inst)) + ";");
            break;
        case IrCmd::DIV_NUM:
            line(v + " = " + quotient(OP_A(inst), OP_B(inst)) + ";");
            break;
        case IrCmd::IDIV_NUM:
            // floor(a / b) and a - floor(a / b) * b, as the VM computes them
            line(v + " = floor(" + quotient(OP_A(inst), OP_B(inst)) + ");");
            break;
        case IrCmd::MOD_NUM:
        {
            std::string b = num(OP_B(inst));
            line(v + " = " + num(OP_A(inst)) + " - floor(" + quotient(OP_A(inst), OP_B(inst)) + ") * " + b + ";");
            break;
        }
        case IrCmd::MIN_NUM:
        case IrCmd::MAX_NUM:
        {
            // when one value is NaN, the second one is the result
            std::string a = num(OP_A(inst)), b = num(OP_B(inst));
            line(v + " = " + a + (inst.cmd == IrCmd::MIN_NUM ? " < " : " > ") + b + " ? " + a + " : " + b + ";");
            break;
        }
        case IrCmd::UNM_NUM:
            line(v + " = -" + num(OP_A(inst)) + ";");
            break;
        case IrCmd::FLOOR_NUM:
        case IrCmd::CEIL_NUM:
        case IrCmd::ROUND_NUM:
        {
            const char* fn = inst.cmd == IrCmd::FLOOR_NUM ? "floor(" : inst.cmd == IrCmd::CEIL_NUM ? "ceil(" : "l_round(";
            line(v + " = " + fn + num(OP_A(inst)) + ");");
            break;
        }
        case IrCmd::SQRT_NUM:
            line(v + " = l_fsqrt(" + num(OP_A(inst)) + ");");
            break;
        case IrCmd::ABS_NUM:
            line(v + " = abs(" + num(OP_A(inst)) + ");");
            break;
        case IrCmd::SIGN_NUM:
        {
            std::string a = num(OP_A(inst));
            line(v + " = " + a + " > 0.0 ? 1.0 : " + a + " < 0.0 ? -1.0 : 0.0;");
            break;
        }
        case IrCmd::SELECT_NUM:
            line(v + " = " + compareNum(OP_C(inst), OP_D(inst), IrCondition::Equal) + " ? " + num(OP_B(inst)) + " : " + num(OP_A(inst)) + ";");
            break;
        case IrCmd::NOT_ANY:
            line(v + " = int(" + tag(OP_A(inst)) + " == 0u || (" + tag(OP_A(inst)) + " == 1u && " + intv(OP_B(inst)) + " == 0));");
            break;
        case IrCmd::CMP_ANY:
        {
            // values of other types, or an order between values that aren't numbers, are left to the VM
            std::string ta = tag(OP_A(inst)), tb = tag(OP_B(inst));
            IrCondition cond = conditionOp(OP_C(inst));
            std::string number = std::to_string(LUA_TNUMBER) + "u";
            std::string fail = "(l_fail(" + std::to_string(int(GpuFailure::Fallback)) + "u), 0)";

            if (cond == IrCondition::Equal)
                line(
                    v + " = " + ta + " != " + tb + " ? 0 : " + ta + " == " + std::to_string(LUA_TNIL) + "u ? 1 : " + ta +
                    " == " + std::to_string(LUA_TBOOLEAN) + "u ? int(" + intv(OP_A(inst)) + " == " + intv(OP_B(inst)) + ") : " + ta +
                    " == " + number + " ? int(" + compareNum(OP_A(inst), OP_B(inst), cond) + ") : " + fail + ";"
                );
            else
                line(
                    v + " = " + ta + " == " + number + " && " + tb + " == " + number + " ? int(" + compareNum(OP_A(inst), OP_B(inst), cond) +
                    ") : " + fail + ";"
                );
            break;
        }
        case IrCmd::CMP_INT:
            line(v + " = int(" + compare(intv(OP_A(inst)), intv(OP_B(inst)), conditionOp(OP_C(inst))) + ");");
            break;
        case IrCmd::CMP_SPLIT_TVALUE:
        {
            uint8_t t = function.tagOp(OP_B(inst));
            std::string eq;
            if (t == LUA_TBOOLEAN)
                eq = "(" + intv(OP_C(inst)) + " == " + intv(OP_D(inst)) + ")";
            else if (t == LUA_TNUMBER)
                eq = compareNum(OP_C(inst), OP_D(inst), IrCondition::Equal);
            else
                return reject(index, "a comparison of this type");

            eq = "(" + tag(OP_A(inst)) + " == " + std::to_string(t) + "u && " + eq + ")";
            line(v + " = int(" + (conditionOp(OP_E(inst)) == IrCondition::Equal ? eq : "!" + eq) + ");");
            break;
        }
        case IrCmd::JUMP:
            line(jump(OP_A(inst)));
            break;
        case IrCmd::JUMP_IF_TRUTHY:
        case IrCmd::JUMP_IF_FALSY:
            line(
                "if (" + std::string(inst.cmd == IrCmd::JUMP_IF_FALSY ? "!" : "") + truthy(OP_A(inst)) + ") " + jump(OP_B(inst)) + " else " +
                jump(OP_C(inst))
            );
            break;
        case IrCmd::JUMP_EQ_TAG:
            if (IrOp known = knownTarget(inst); known.kind != IrOpKind::None)
            {
                line(jump(known));
                break;
            }
            line("if (" + tag(OP_A(inst)) + " == " + tag(OP_B(inst)) + ") " + jump(OP_C(inst)) + " else " + jump(OP_D(inst)));
            break;
        case IrCmd::JUMP_CMP_INT:
            line("if " + compare(intv(OP_A(inst)), intv(OP_B(inst)), conditionOp(OP_C(inst))) + " " + jump(OP_D(inst)) + " else " + jump(OP_E(inst)));
            break;
        case IrCmd::JUMP_CMP_NUM:
            line("if " + compareNum(OP_A(inst), OP_B(inst), conditionOp(OP_C(inst))) + " " + jump(OP_D(inst)) + " else " + jump(OP_E(inst)));
            break;
        case IrCmd::JUMP_FORN_LOOP_COND:
        {
            std::string idx = num(OP_A(inst)), limit = num(OP_B(inst)), step = num(OP_C(inst));
            line(
                "if (" + step + " > 0.0 ? " + idx + " <= " + limit + " : " + limit + " <= " + idx + ") " + jump(OP_D(inst)) + " else " +
                jump(OP_E(inst))
            );
            break;
        }
        case IrCmd::INT_TO_NUM:
            line(v + " = l_i2f(" + intv(OP_A(inst)) + ");");
            break;
        case IrCmd::UINT_TO_NUM:
            line(v + " = l_u2f(uint(" + intv(OP_A(inst)) + "));");
            break;
        case IrCmd::NUM_TO_INT:
            line(v + " = l_toi(" + num(OP_A(inst)) + ");");
            break;
        case IrCmd::NUM_TO_UINT:
            line(v + " = int(l_tou(" + num(OP_A(inst)) + "));");
            break;
        case IrCmd::FLOAT_TO_NUM:
        case IrCmd::NUM_TO_FLOAT:
            line(v + " = " + floatv(OP_A(inst)) + ";");
            break;
        case IrCmd::TRUNCATE_UINT:
            line(v + " = " + intv(OP_A(inst)) + ";");
            break;
        case IrCmd::CHECK_TAG:
            if (knownTag(OP_A(inst)) >= 0 && knownTag(OP_A(inst)) == knownTag(OP_B(inst)))
                break;
            if (IrOp known = knownTarget(inst); known.kind != IrOpKind::None)
            {
                guard("false", known);
                break;
            }
            guard("(" + tag(OP_A(inst)) + " == " + tag(OP_B(inst)) + ")", OP_C(inst));
            break;
        case IrCmd::CHECK_CMP_NUM:
            guard(compareNum(OP_A(inst), OP_B(inst), conditionOp(OP_C(inst))), OP_D(inst));
            break;
        case IrCmd::CHECK_CMP_INT:
            guard(compare(intv(OP_A(inst)), intv(OP_B(inst)), conditionOp(OP_C(inst))), OP_D(inst));
            break;
        case IrCmd::CHECK_BUFFER_LEN:
        {
            int b;
            if (!binding(index, OP_A(inst), b))
                return false;

            // the offset is the number truncated, as the VM converts it: native code only checks that the number is an integer to
            // leave the others to the VM
            std::string base = intv(OP_B(inst));
            std::string len = "int(pc.len[" + std::to_string(b) + "])";
            guard("(" + base + " >= -(" + intv(OP_C(inst)) + ") && " + base + " <= " + len + " - " + intv(OP_D(inst)) + ")", OP_F(inst));
            break;
        }
        case IrCmd::INTERRUPT:
            // native code checks for interrupts here, at every loop iteration and before returning
            // and every 256 iterations, whether the host asked a run that takes too long to stop
            line(
                "if ((++l_steps > pc.budget && pc.budget != 0u) || ((l_steps & 255u) == 0u && l_stopped())) { l_fail(l_steps > pc.budget && "
                "pc.budget != 0u ? " +
                std::to_string(int(GpuFailure::LoopBudget)) + "u : " + std::to_string(int(GpuFailure::Stopped)) + "u); return; }"
            );
            break;
        case IrCmd::RETURN:
            line("return;");
            break;
        case IrCmd::BUFFER_READI32:
        case IrCmd::BUFFER_READF32:
        {
            int b;
            if (!binding(index, OP_A(inst), b))
                return false;

            std::string raw = access(b, OP_B(inst), "r");
            line(v + " = " + (inst.cmd == IrCmd::BUFFER_READF32 ? "uintBitsToFloat(" + raw + ")" : "int(" + raw + ")") + ";");
            break;
        }
        case IrCmd::BUFFER_READI8:
        case IrCmd::BUFFER_READU8:
        case IrCmd::BUFFER_READI16:
        case IrCmd::BUFFER_READU16:
        {
            int b;
            if (!binding(index, OP_A(inst), b))
                return false;

            bool half = inst.cmd == IrCmd::BUFFER_READI16 || inst.cmd == IrCmd::BUFFER_READU16;
            narrow[b] |= half ? 2 : 1;
            std::string raw = std::string(half ? "l_rdh" : "l_rdb") + std::to_string(b) + "(" + intv(OP_B(inst)) + ")";
            std::string shift = half ? "16" : "24";
            if (inst.cmd == IrCmd::BUFFER_READI8 || inst.cmd == IrCmd::BUFFER_READI16)
                line(v + " = int(" + raw + " << " + shift + "u) >> " + shift + ";");
            else
                line(v + " = int(" + raw + ");");
            break;
        }
        case IrCmd::BUFFER_WRITEI8:
        case IrCmd::BUFFER_WRITEI16:
        {
            int b;
            if (!binding(index, OP_A(inst), b))
                return false;

            bool half = inst.cmd == IrCmd::BUFFER_WRITEI16;
            narrow[b] |= half ? 8 : 4;
            build.bufferWritten[b] = true;
            line(std::string(half ? "l_wrh" : "l_wrb") + std::to_string(b) + "(" + intv(OP_B(inst)) + ", uint(" + intv(OP_C(inst)) + "));");
            break;
        }
        case IrCmd::BUFFER_WRITEI32:
        case IrCmd::BUFFER_WRITEF32:
        {
            int b;
            if (!binding(index, OP_A(inst), b))
                return false;

            std::string value = inst.cmd == IrCmd::BUFFER_WRITEF32 ? "floatBitsToUint(" + floatv(OP_C(inst)) + ")" : "uint(" + intv(OP_C(inst)) + ")";
            line(access(b, OP_B(inst), "w", value) + ";");
            break;
        }
        case IrCmd::BITAND_UINT:
        case IrCmd::BITXOR_UINT:
        case IrCmd::BITOR_UINT:
        {
            const char* op = inst.cmd == IrCmd::BITAND_UINT ? " & " : inst.cmd == IrCmd::BITXOR_UINT ? " ^ " : " | ";
            line(v + " = " + intv(OP_A(inst)) + op + intv(OP_B(inst)) + ";");
            break;
        }
        case IrCmd::BITNOT_UINT:
            line(v + " = ~" + intv(OP_A(inst)) + ";");
            break;
        case IrCmd::BITLSHIFT_UINT:
        case IrCmd::BITRSHIFT_UINT:
            // like native code, the shift amount is taken modulo 32
            line(
                v + " = int(uint(" + intv(OP_A(inst)) + ")" + (inst.cmd == IrCmd::BITLSHIFT_UINT ? " << " : " >> ") + "(uint(" + intv(OP_B(inst)) +
                ") & 31u));"
            );
            break;
        case IrCmd::BITARSHIFT_UINT:
            line(v + " = " + intv(OP_A(inst)) + " >> (" + intv(OP_B(inst)) + " & 31);");
            break;
        case IrCmd::BITLROTATE_UINT:
        case IrCmd::BITRROTATE_UINT:
        {
            std::string a = "uint(" + intv(OP_A(inst)) + ")", n = "(uint(" + intv(OP_B(inst)) + ") & 31u)";
            bool left = inst.cmd == IrCmd::BITLROTATE_UINT;
            line(v + " = int((" + a + (left ? " << " : " >> ") + n + ") | (" + a + (left ? " >> " : " << ") + "((32u - " + n + ") & 31u)));");
            break;
        }
        case IrCmd::BITCOUNTLZ_UINT:
            line(v + " = 31 - findMSB(uint(" + intv(OP_A(inst)) + "));");
            break;
        case IrCmd::BITCOUNTRZ_UINT:
            line(v + " = " + intv(OP_A(inst)) + " == 0 ? 32 : findLSB(" + intv(OP_A(inst)) + ");");
            break;
        default:
            return reject(index, inst.cmd == IrCmd::CALL ? "a call that wasn't inlined" : getCmdName(inst.cmd));
        }

        return true;
    }

    bool lowerLibm(IrInst& inst, const std::string& v)
    {
        std::string a = num(OP_B(inst));
        std::string b = OP_C(inst).kind != IrOpKind::None ? num(OP_C(inst)) : "";

        switch (function.uintOp(OP_A(inst)))
        {
        case LBF_MATH_EXP:
            line(v + " = l_exp(" + a + ");");
            return true;
        case LBF_MATH_LOG:
            line(v + " = l_log(" + a + ");");
            return true;
        case LBF_IR_MATH_LOG2:
            line(v + " = l_log2(" + a + ");");
            return true;
        case LBF_MATH_LOG10:
            line(v + " = l_log2(" + a + ") * 0.30102999566;");
            return true;
        case LBF_MATH_SIN:
        case LBF_MATH_COS:
        case LBF_MATH_TAN:
        case LBF_MATH_ATAN:
        case LBF_MATH_SINH:
        case LBF_MATH_COSH:
        case LBF_MATH_TANH:
        {
            uint32_t id = function.uintOp(OP_A(inst));
            const char* fn = id == LBF_MATH_SIN    ? "sin("
                             : id == LBF_MATH_COS  ? "cos("
                             : id == LBF_MATH_TAN  ? "tan("
                             : id == LBF_MATH_ATAN ? "atan("
                             : id == LBF_MATH_SINH ? "sinh("
                             : id == LBF_MATH_COSH ? "cosh("
                                                   : "tanh(";
            line(v + " = " + fn + a + ");");
            return true;
        }
        case LBF_MATH_ASIN:
            line(v + " = l_asin(" + a + ");");
            return true;
        case LBF_MATH_ACOS:
            line(v + " = l_acos(" + a + ");");
            return true;
        case LBF_MATH_ATAN2:
            line(v + " = atan(" + a + ", " + b + ");");
            return true;
        case LBF_MATH_POW:
            line(v + " = l_pow(" + a + ", " + b + ");");
            return true;
        case LBF_MATH_FMOD:
            line(v + " = l_fmod(" + a + ", " + b + ");");
            return true;
        default:
            return false;
        }
    }

    bool lowerCall(IrInst& inst, uint32_t index)
    {
        if (OP_A(inst).kind != IrOpKind::VmReg)
            return reject(index, "a call that wasn't inlined");

        int ra = vmRegOp(OP_A(inst));
        int nargs = function.intOp(OP_B(inst));
        int nresults = function.intOp(OP_C(inst));
        GpuCall call = cur[ra].call;

        if (call == GpuCall::None || call == GpuCall::Unknown)
            return reject(index, "a call that wasn't inlined");
        if (nresults < 0 || nresults > 1)
            return reject(index, "a call with several results");

        std::string r = "r" + std::to_string(ra);
        int b = ra + 1 < int(cur.size()) ? cur[ra + 1].buffer : -1;
        if (b < 0)
            return reject(index, "a buffer other than the kernel's parameter");

        std::string value;
        if (call == GpuCall::BufferLen)
        {
            if (nargs != 1)
                return reject(index, "buffer.len with these arguments");
            value = "l_u2f(pc.len[" + std::to_string(b) + "])";
        }
        else
        {
            if (nargs != 3)
                return reject(index, "an atomic with these arguments");

            // offsets and values are converted as the gpu library does in the VM
            std::string offset = "l_toi(" + num(IrOp{IrOpKind::VmReg, uint32_t(ra + 2)}) + ")";
            std::string operand = "l_tou(" + num(IrOp{IrOpKind::VmReg, uint32_t(ra + 3)}) + ")";

            int op = call == GpuCall::AtomicAdd ? 0 : 1;
            atomics[b] |= uint8_t(1 << op);
            build.bufferWritten[b] = true;

            static const char* const kNames[] = {"add", "min"};
            value = "l_u2f(l_atomic" + std::string(kNames[op]) + std::to_string(b) + "(" + offset + ", " + operand + "))";
        }

        line(r + "t = " + std::to_string(LUA_TNUMBER) + "u;");
        if (nresults != 0)
            line(r + "n = " + value + ";");
        else
            line(value + ";");
        return true;
    }

    std::string floatv(IrOp op)
    {
        if (op.kind == IrOpKind::Inst && getCmdValueKind(function.instOp(op).cmd) == IrValueKind::Float)
            return "v" + std::to_string(op.index);
        return num(op);
    }

    std::string declarations()
    {
        std::string decl;

        for (int r = 0; r < proto->maxstacksize; r++)
        {
            std::string rs = "r" + std::to_string(r);
            decl += "    uint " + rs + "t = 0u; int " + rs + "i = 0; precise vec3 " + rs + "v = vec3(0.0); precise float " + rs + "n = 0.0;\n";
        }

        for (size_t i = 0; i < function.instructions.size(); i++)
        {
            IrInst& inst = function.instructions[i];
            if (isPseudo(inst.cmd))
                continue;

            std::string v = "v" + std::to_string(i);

            switch (getCmdValueKind(inst.cmd))
            {
            case IrValueKind::Tag:
                decl += "    uint " + v + " = 0u;\n";
                break;
            case IrValueKind::Int:
                decl += "    int " + v + " = 0;\n";
                break;
            case IrValueKind::Float:
                decl += "    precise float " + v + " = 0.0;\n";
                break;
            case IrValueKind::Double:
                decl += "    precise float " + v + " = 0.0;\n";
                break;
            case IrValueKind::Tvalue:
                decl += "    uint " + v + "t = 0u; precise float " + v + "n = 0.0; int " + v + "i = 0; precise vec3 " + v + "v = vec3(0.0);\n";
                break;
            default:
                break;
            }
        }

        return decl;
    }

    bool lower(const std::vector<uint32_t>& sortedBlocks)
    {
        lowered.assign(function.blocks.size(), false);

        for (uint32_t blockIndex : sortedBlocks)
        {
            IrBlockKind kind = function.blocks[blockIndex].kind;
            lowered[blockIndex] = kind != IrBlockKind::Dead && kind != IrBlockKind::Fallback && kind != IrBlockKind::ExitSync;
        }

        pcOfInst.assign(function.instructions.size(), -1);
        for (size_t pc = 0; pc < function.bcMapping.size(); pc++)
        {
            // instructions that produce no IR map to where the next one would start, possibly the end
            if (function.bcMapping[pc].irLocation < pcOfInst.size())
                pcOfInst[function.bcMapping[pc].irLocation] = int(pc);
        }
        for (size_t i = 1; i < pcOfInst.size(); i++)
        {
            if (pcOfInst[i] < 0)
                pcOfInst[i] = pcOfInst[i - 1];
        }

        if (proto->is_vararg)
            return reject(0, "a variadic function");
        if (proto->numparams == 0)
            return reject(0, "a function without the buffer parameter");
        if (proto->numparams > 2)
            return reject(0, "a function with parameters other than the buffer and the index");

        int buffers = kGpuMaxBuffers;
        build.bufferWritten.assign(buffers, false);
        readBuffers.assign(buffers, false);
        atomics.assign(buffers, 0);
        narrow.assign(buffers, 0);

        // which registers each block reads, on the IR after the optimizations
        computeCfgInfo(function);

        if (!analyze(sortedBlocks))
            return false;

        orderBlocks();
        std::string body = tree(function.entryBlock);
        if (failed)
            return false;

        std::string& glsl = build.glsl;
        // like the VM, shaders keep NaN, infinities and signed zeros (SignedZeroInfNanPreserve for 32-bit floats), and denormal
        // numbers (DenormPreserve)
        glsl = "#version 450\n#extension GL_EXT_spirv_intrinsics : require\n#extension GL_EXT_control_flow_attributes : require\n"
               "spirv_execution_mode(extensions = [\"SPV_KHR_float_controls\"], capabilities = [4466], 4461, 32);\n"
               "spirv_execution_mode(extensions = [\"SPV_KHR_float_controls\"], capabilities = [4464], 4459, 32);\n"
               "layout(local_size_x = " +
               std::to_string(kGpuWorkgroupSize) + ") in;\n\n";

        for (int i = 0; i < buffers; i++)
            glsl += "layout(std430, set = 0, binding = " + std::to_string(i) + ") buffer B" + std::to_string(i) + " { uint d[]; } b" +
                    std::to_string(i) + ";\n";
        // 'stop' is an int so that the block doesn't have the structure of the result of umulExtended, which SPIR-V would make
        // the same type and SPIRV-Cross would then declare with the block's name
        glsl += "layout(std430, set = 0, binding = " + std::to_string(buffers) +
                ") coherent volatile buffer BS { uint err; int stop; uint started; } st;\n";
        glsl += "layout(std430, set = 0, binding = " + std::to_string(buffers + 1) + ") buffer BW { uint d[]; } wr;\n";
        glsl += "layout(push_constant) uniform PC { uint n; uint budget; uint len[" + std::to_string(kGpuMaxBuffers) + "]; uint written; } pc;\n";
        glsl += kPrelude;
        // the host only downloads the parts of the buffer that kernels write; reading the word first is faster than storing
        // it again from every invocation
        glsl += "void l_mark(int off) { uint w = pc.written + (uint(off) >> " + std::to_string(kGpuWrittenChunkShift) +
                "); if (wr.d[w] == 0u) wr.d[w] = 1u; }\n";

        for (int i = 0; i < buffers; i++)
        {
            std::string b = std::to_string(i);
            // reads and writes follow a CHECK_BUFFER_LEN that covers their bounds, as in native code; words must be aligned
            if (readBuffers[i])
                glsl += "uint l_rd" + b + "(int off) { if ((off & 3) != 0) { l_fail(1u); return 0u; } return b" + b + ".d[off >> 2]; }\n" +
                        "uint l_rda" + b + "(int off) { return b" + b + ".d[off >> 2]; }\n";
            if (build.bufferWritten[i])
                glsl += "void l_wr" + b + "(int off, uint v) { if ((off & 3) != 0) { l_fail(1u); return; } b" + b +
                        ".d[off >> 2] = v; l_mark(off); }\n" + "void l_wra" + b + "(int off, uint v) { b" + b + ".d[off >> 2] = v; l_mark(off); }\n";

            // 8 and 16-bit accesses are parts of a word: a 16-bit one at a byte offset of 3 is also in the next word, which the
            // device copy has, since it's a whole number of words. Writes only change their bits, with atomics: the other
            // bytes of the word can be written concurrently by other invocations
            std::string d = "b" + b + ".d";
            if (narrow[i] & 1)
                glsl += "uint l_rdb" + b + "(int off) { return (" + d + "[off >> 2] >> (uint(off & 3) * 8u)) & 0xffu; }\n";
            if (narrow[i] & 2)
                glsl += "uint l_rdh" + b + "(int off) { uint s = uint(off & 3) * 8u, w = " + d + "[off >> 2] >> s; if (s == 24u) w |= " + d +
                        "[(off >> 2) + 1] << 8u; return w & 0xffffu; }\n";
            if (narrow[i] & 12)
                glsl += "void l_wrm" + b + "(int off, uint v, uint m) { uint s = uint(off & 3) * 8u; atomicAnd(" + d +
                        "[off >> 2], ~(m << s)); atomicOr(" + d + "[off >> 2], (v & m) << s); l_mark(off); }\n";
            if (narrow[i] & 4)
                glsl += "void l_wrb" + b + "(int off, uint v) { l_wrm" + b + "(off, v, 0xffu); }\n";
            if (narrow[i] & 8)
                glsl += "void l_wrh" + b + "(int off, uint v) { if ((off & 3) == 3) { l_wrm" + b + "(off, v, 0xffu); l_wrm" + b +
                        "(off + 1, v >> 8u, 0xffu); } else l_wrm" + b + "(off, v, 0xffffu); }\n";

            static const char* const kAtomics[][2] = {{"add", "atomicAdd"}, {"min", "atomicMin"}};
            for (int op = 0; op < 2; op++)
            {
                if (atomics[i] & (1 << op))
                    glsl += std::string("uint l_atomic") + kAtomics[op][0] + b + "(int off, uint v) { if (l_bad(off, " + b +
                            ")) { l_fail(1u); return 0u; } l_mark(off); return " + kAtomics[op][1] + "(b" + b + ".d[off >> 2], v); }\n";
            }
        }

        glsl += "\nvoid l_kernel(int index)\n{\n" + declarations();
        glsl += "    r0t = " + std::to_string(LUA_TBUFFER) + "u;\n";
        if (proto->numparams > 1)
            glsl += "    r1t = " + std::to_string(LUA_TNUMBER) + "u; r1n = float(index);\n";

        glsl += "    int l_jmp = 0;\n";
        glsl += body;
        glsl += "}\n\n";
        glsl += "void main()\n{\n";
        glsl +=
            "    uint gid = gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * gl_NumWorkGroups.x * " + std::to_string(kGpuWorkgroupSize) + "u;\n";
        // invocations that start after the host asked the run to stop give up at once
        glsl += "    if (gid >= pc.n) return;\n    if (gid == 0u) st.started = 1u;\n    if (l_stopped())\n        l_fail(" +
                std::to_string(int(GpuFailure::Stopped)) +
                "u);\n    else\n        l_kernel(int(gid));\n"
                "#ifndef L_EXACT\n    if (l_flag)\n        l_err = " +
                std::to_string(int(GpuFailure::NeedsExact)) +
                "u << 24;\n#endif\n"
                "    if (l_err != 0u)\n        atomicCompSwap(st.err, 0u, l_err);\n}\n";

        return true;
    }
};

} // namespace

inline bool lowerIr(
    LogBuilder* logger,
    GlslBuilder& build,
    IrBuilder& ir,
    const std::vector<uint32_t>& sortedBlocks,
    ModuleHelpers& helpers,
    Proto* proto,
    AssemblyOptions options,
    LoweringStats* stats
)
{
    GlslLowering lowering(ir.function, proto, build);
    return lowering.lower(sortedBlocks);
}

static GpuKernel compileProto(Proto* proto, LuaTable* env)
{
    GpuKernel kernel;
    kernel.result = CodeGenCompilationResult::CodeGenLoweringFailure;
    kernel.name = proto->debugname ? getstr(proto->debugname) : "anonymous";
    kernel.line = proto->linedefined;

    if ((proto->flags & LPF_GPU_FUNCTION) == 0)
    {
        kernel.reason = "the function doesn't have the gpu attribute";
        return kernel;
    }

    // the shader can't check the environment at run time, so builtins must be the real ones
    if (!env->safeenv)
    {
        kernel.reason = "the function's environment isn't safe (sandbox the state and don't modify builtin libraries)";
        return kernel;
    }

    // the builder keeps references to the hooks and the environment
    CompilationOptions options;
    VmEnvironmentInfo environment;
    IrBuilder ir(options.hooks, environment);
    ir.float32 = true;
    ir.buildFunctionIr(proto);

    if (ir.function.instructions.size() >= unsigned(FInt::CodegenHeuristicsInstructionLimit.value))
    {
        kernel.result = CodeGenCompilationResult::CodeGenOverflowInstructionLimit;
        kernel.reason = "the function is too large";
        return kernel;
    }

    GlslBuilder build;
    ModuleHelpers helpers;
    kernel.result = CodeGenCompilationResult::Success;

    if (!lowerFunction(ir, nullptr, build, helpers, proto, AssemblyOptions(), nullptr, kernel.result))
    {
        kernel.reason = build.reason.empty() ? "the function exceeds native code generation limits" : build.reason;
        return kernel;
    }

    kernel.glsl = std::move(build.glsl);
    kernel.bufferWritten = std::move(build.bufferWritten);
    return kernel;
}

std::string GpuKernel::exactGlsl() const
{
    size_t line = glsl.find('\n');
    if (line == std::string::npos)
        return glsl;
    return glsl.substr(0, line + 1) + "#define L_EXACT\n" + glsl.substr(line + 1);
}

std::string describeGpuFailure(uint32_t status)
{
    static const char* const kKinds[] = {
        "an unknown failure",
        "out of bounds or unaligned buffer access",
        "loop limit reached",
        "stopped by the host",
        "case native code leaves to the VM (a value of another type, a bit32 argument out of range)",
        "division or square root that needs the exact computation",
    };

    uint32_t kind = status >> 24, line = status & 0xffffff;
    std::string reason = std::string("the kernel failed on the GPU: ") + kKinds[kind < sizeof(kKinds) / sizeof(kKinds[0]) ? kind : 0];
    if (line != 0)
        reason += " at line " + std::to_string(line);
    return reason;
}

bool isGpuEnvironmentSafe(lua_State* L, int idx)
{
    return lua_isLfunction(L, idx) && clvalue(luaA_toobject(L, idx))->env->safeenv;
}

GpuKernel compileGpu(lua_State* L, int idx)
{
    if (!lua_isLfunction(L, idx))
    {
        GpuKernel kernel;
        kernel.result = CodeGenCompilationResult::CodeGenLoweringFailure;
        kernel.reason = "not a Luau function";
        return kernel;
    }

    Closure* cl = clvalue(luaA_toobject(L, idx));
    return compileProto(cl->l.p, cl->env);
}

static void gatherGpuProtos(Proto* proto, LuaTable* env, std::vector<GpuKernel>& kernels)
{
    if (proto->flags & LPF_GPU_FUNCTION)
        kernels.push_back(compileProto(proto, env));

    for (int i = 0; i < proto->sizep; i++)
        gatherGpuProtos(proto->p[i], env, kernels);
}

std::vector<GpuKernel> compileGpuModule(lua_State* L, int idx)
{
    std::vector<GpuKernel> kernels;

    if (lua_isLfunction(L, idx))
    {
        Closure* cl = clvalue(luaA_toobject(L, idx));
        gatherGpuProtos(cl->l.p, cl->env, kernels);
    }

    return kernels;
}

} // namespace CodeGen
} // namespace Luau
