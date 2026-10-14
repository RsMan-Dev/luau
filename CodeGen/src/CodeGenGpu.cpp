// This file is part of the Luau programming language and is licensed under MIT License; see LICENSE.txt for details
#include "Luau/CodeGenGpu.h"

#include "CodeGenLower.h"

#include "lapi.h"
#include "ldebug.h"

#include <algorithm>

#include <math.h>
#include <stdio.h>

// Lowers the IR of a @gpu function to a GLSL compute shader, after the same IR translation, limits and optimizations as
// native code. Every guard and fallback path of the IR fails the dispatch instead: the host then reruns it in the VM.

namespace Luau
{
namespace CodeGen
{

// mirrors Gpu/include/Luau/GpuAbi.h
constexpr int kGpuMaxBuffers = 15;
constexpr unsigned kGpuWorkgroupSize = 64;

namespace
{

// Numbers are lowered to 32-bit types. Integers stay exact: an operation whose result doesn't fit fails the dispatch.
// Ordered so that joining two classes is taking the larger one.
enum class NumClass : uint8_t
{
    None,
    NonNeg, // integer in [0, 2^31), valid as both int and uint
    Uint,
    Int,
    Float,
};

const char* const kPrelude = R"(
uint l_steps = 0u;
void l_fail() { atomicOr(st.err, 1u); }
bool l_bad(int off, int b) { return off < 0 || (off & 3) != 0 || uint(off) + 4u > pc.len[b]; }
int l_chk(bool bad, int r) { if (bad) l_fail(); return r; }
uint l_chk(bool bad, uint r) { if (bad) l_fail(); return r; }
int l_idivk(int a, int k) { int q = a / k; if (q * k != a && a < 0) q -= 1; return q; }
int l_addi(int a, int b) { int r = a + b; if (((a ^ r) & (b ^ r)) < 0) l_fail(); return r; }
int l_subi(int a, int b) { int r = a - b; if (((a ^ b) & (a ^ r)) < 0) l_fail(); return r; }
int l_muli(int a, int b) { int hi, lo; imulExtended(a, b, hi, lo); if (hi != (lo >> 31)) l_fail(); return lo; }
int l_negi(int a) { if (a == int(0x80000000u)) l_fail(); return -a; }
int l_idivi(int a, int b) { if (b == 0 || (b == -1 && a == int(0x80000000u))) { l_fail(); return 0; } int q = a / b; if (q * b != a && ((a < 0) != (b < 0))) q -= 1; return q; }
int l_modi(int a, int b) { if (b == 0) { l_fail(); return 0; } if (b == -1) return 0; return a - l_idivi(a, b) * b; }
uint l_addu(uint a, uint b) { uint c; uint r = uaddCarry(a, b, c); if (c != 0u) l_fail(); return r; }
uint l_mulu(uint a, uint b) { uint hi, lo; umulExtended(a, b, hi, lo); if (hi != 0u) l_fail(); return lo; }
uint l_idivu(uint a, uint b) { if (b == 0u) { l_fail(); return 0u; } return a / b; }
uint l_modu(uint a, uint b) { if (b == 0u) { l_fail(); return 0u; } return a % b; }
int l_u2i(uint a) { if (a > 0x7fffffffu) l_fail(); return int(a); }
uint l_i2u(int a) { if (a < 0) l_fail(); return uint(a); }
float l_i2f(int a) { if (a < -16777216 || a > 16777216) l_fail(); return float(a); }
float l_u2f(uint a) { if (a > 16777216u) l_fail(); return float(a); }
int l_f2i(float a) { if (!(a >= -2147483648.0 && a < 2147483648.0)) { l_fail(); return 0; } return int(a); }
uint l_f2u(float a) { if (!(a > -2147483649.0 && a < 4294967296.0)) { l_fail(); return 0u; } return a >= 0.0 ? uint(a) : uint(int(a)); }
float l_sqrt(float a) { return a < 0.0 ? uintBitsToFloat(0x7fc00000u) : sqrt(a); }
float l_round(float a) { return sign(a) * floor(abs(a) + 0.5); }
float l_log(float a) { return a < 0.0 ? uintBitsToFloat(0x7fc00000u) : a == 0.0 ? uintBitsToFloat(0xff800000u) : log(a); }
float l_log2(float a) { return a < 0.0 ? uintBitsToFloat(0x7fc00000u) : a == 0.0 ? uintBitsToFloat(0xff800000u) : log2(a); }
float l_asin(float a) { return abs(a) > 1.0 ? uintBitsToFloat(0x7fc00000u) : asin(a); }
float l_acos(float a) { return abs(a) > 1.0 ? uintBitsToFloat(0x7fc00000u) : acos(a); }
float l_fmod(float a, float b) { return b == 0.0 ? uintBitsToFloat(0x7fc00000u) : a - b * trunc(a / b); }
bool l_odd(float b) { return abs(b) < 16777216.0 && mod(b, 2.0) == 1.0; }
float l_pow(float a, float b)
{
    if (b == 0.0 || a == 1.0) return 1.0;
    if (isnan(a) || isnan(b)) return uintBitsToFloat(0x7fc00000u);
    if (a == 0.0) return b > 0.0 ? (l_odd(b) ? a : 0.0) : (l_odd(b) && floatBitsToUint(a) != 0u ? uintBitsToFloat(0xff800000u) : uintBitsToFloat(0x7f800000u));
    if (a < 0.0) { if (b != floor(b)) return uintBitsToFloat(0x7fc00000u); float r = pow(-a, b); return l_odd(b) ? -r : r; }
    return pow(a, b);
}
vec3 l_minv(vec3 a, vec3 b) { return mix(b, a, lessThan(a, b)); }
vec3 l_maxv(vec3 a, vec3 b) { return mix(b, a, greaterThan(a, b)); }
)";

const char* glslType(NumClass c)
{
    return c == NumClass::Float ? "float" : c == NumClass::Uint ? "uint" : "int";
}

NumClass constClass(double v)
{
    if (v != floor(v) || (v == 0 && signbit(v)))
        return NumClass::Float;
    if (v >= 0 && v <= 2147483647.0)
        return NumClass::NonNeg;
    if (v >= -2147483648.0 && v < 0)
        return NumClass::Int;
    if (v > 0 && v <= 4294967295.0)
        return NumClass::Uint;
    return NumClass::Float;
}

std::string literal(double v, NumClass c)
{
    char buf[64];

    if (c == NumClass::Float)
    {
        if (isnan(v))
            return "uintBitsToFloat(0x7fc00000u)";
        if (isinf(v))
            return v > 0 ? "uintBitsToFloat(0x7f800000u)" : "uintBitsToFloat(0xff800000u)";

        snprintf(buf, sizeof(buf), "%.9g", v);
        std::string s = buf;
        if (s.find_first_of(".e") == std::string::npos)
            s += ".0";
        return "float(" + s + ")";
    }

    if (v == -2147483648.0)
        return "int(0x80000000u)";

    snprintf(buf, sizeof(buf), c == NumClass::Uint ? "%.0fu" : "(%.0f)", v);
    return buf;
}

// converts an expression between number classes; only widens, with a failure check where the value may not fit
// a register that holds both integers and floats keeps the integers exact, so they can still be used as offsets
std::string convert(const std::string& code, NumClass from, NumClass to, bool exact = false)
{
    if (exact && to == NumClass::Float && from != NumClass::Float && from != NumClass::None)
        return (from == NumClass::Uint ? "l_u2f(" : "l_i2f(") + code + ")";

    if (from == to || from == NumClass::None)
        return code;
    if (to == NumClass::Float)
        return "float(" + code + ")";
    if (to == NumClass::Uint)
        return from == NumClass::Int ? "l_i2u(" + code + ")" : "uint(" + code + ")";
    if (to == NumClass::Int || to == NumClass::NonNeg)
        return from == NumClass::Uint ? "l_u2i(" + code + ")" : code;

    CODEGEN_ASSERT(!"unexpected conversion");
    return code;
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
    AtomicMax,
    BufferLen,
};

struct GlslBuilder
{
    std::string reason;
    std::string glsl;
    std::vector<bool> bufferWritten;
};

struct Num
{
    std::string code;
    NumClass cls;
};

// what a VM register holds at a point of the function
struct RegState
{
    NumClass cls = NumClass::None; // class of its number, which lives in the variable of that class
    int buffer = -1;               // binding of its buffer, -1 for none, -2 when it isn't known statically
    GpuCall call = GpuCall::None;  // import it holds
    int tag = -1;                  // its tag, -1 when it isn't known statically
};

struct GlslLowering
{
    IrFunction& function;
    Proto* proto;
    GlslBuilder& build;

    std::vector<bool> lowered;       // per block: lowered, or a fallback/exit that fails the dispatch
    std::vector<NumClass> instClass; // per instruction producing a double, or the number of a loaded TValue
    std::vector<int> instBuffer;     // per instruction producing a buffer pointer or TValue
    std::vector<GpuCall> instCall;   // per instruction loading a TValue: the import it holds
    std::vector<int> instTag;        // per instruction loading a tag or TValue: the tag, -1 when it isn't known statically

    // bounds of the integer an instruction produces, which spare overflow checks when they fit its class
    struct Range
    {
        double lo = -INFINITY;
        double hi = INFINITY;
    };
    std::vector<Range> instRange;

    // registers are reused for values of different types, so what they hold is tracked along the control flow
    std::vector<std::vector<RegState>> entry; // per block, when it starts
    std::vector<bool> reached;
    std::vector<RegState> cur; // at the instruction being analyzed or lowered
    bool changed = false;

    // structured control flow, as in "Beyond Relooper" (Ramsey, 2022): the IR of Luau code is reducible, so each block is
    // emitted once, in its immediate dominator, inside a loop if it's a loop header, after the blocks that end where the
    // block's dominator tree children with several predecessors start. Jumps across constructs go through l_jmp.
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

    std::vector<bool> readBuffers;
    std::vector<uint8_t> atomics; // per buffer, a bit per atomic operation used
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

    NumClass cls(IrOp op)
    {
        switch (op.kind)
        {
        case IrOpKind::Inst:
            if (isTvalue(op) && function.instOp(op).cmd != IrCmd::LOAD_TVALUE)
                return NumClass::Float;
            return instClass[op.index];
        case IrOpKind::Constant:
        {
            IrConst& c = function.constOp(op);
            if (c.kind == IrConstKind::Double)
                return constClass(c.valueDouble);
            if (c.kind == IrConstKind::Int)
                return constClass(double(c.valueInt));
            return NumClass::Float;
        }
        case IrOpKind::VmReg:
            return cur[vmRegOp(op)].cls;
        case IrOpKind::VmConst:
        {
            TValue* k = &proto->k[vmConstOp(op)];
            return ttisnumber(k) ? constClass(nvalue(k)) : NumClass::None;
        }
        default:
            return NumClass::Float;
        }
    }

    // value of a double operand, in its own class
    Num num(IrOp op)
    {
        NumClass c = cls(op);

        switch (op.kind)
        {
        case IrOpKind::Inst:
            return {"v" + std::to_string(op.index) + (isTvalue(op) ? "n" : ""), c};
        case IrOpKind::Constant:
        {
            IrConst& k = function.constOp(op);
            return {literal(k.kind == IrConstKind::Double ? k.valueDouble : double(k.valueInt), c), c};
        }
        case IrOpKind::VmReg:
            return {numReg(vmRegOp(op), c), c};
        case IrOpKind::VmConst:
            return {literal(nvalue(&proto->k[vmConstOp(op)]), c), c};
        default:
            return {"0", NumClass::NonNeg};
        }
    }

    // the value of a constant operand that is an integer of the class
    bool constInt(IrOp op, NumClass c, int64_t& k)
    {
        double v;
        if (op.kind == IrOpKind::Constant && function.constOp(op).kind == IrConstKind::Double)
            v = function.doubleOp(op);
        else if (op.kind == IrOpKind::VmConst && ttisnumber(&proto->k[vmConstOp(op)]))
            v = nvalue(&proto->k[vmConstOp(op)]);
        else
            return false;

        if (v != floor(v) || v < (c == NumClass::Uint ? 0.0 : -2147483648.0) || v > (c == NumClass::Uint ? 4294967295.0 : 2147483647.0))
            return false;

        k = int64_t(v);
        return true;
    }

    // integer arithmetic with a constant operand needs simpler checks, and none for divisions by positive constants; the
    // classes of the operands bound them further: a NonNeg value is at least 0
    std::string intArith(uint32_t index, IrCmd cmd, IrOp opa, IrOp opb, NumClass c)
    {
        std::string a = num(opa, c), b = num(opb, c);

        // results that fit the class can't overflow
        Range r = instRange[index], domain = classRange(c);
        if (r.lo >= domain.lo && r.hi <= domain.hi && (cmd == IrCmd::ADD_NUM || cmd == IrCmd::SUB_NUM || cmd == IrCmd::MUL_NUM))
            return "(" + a + (cmd == IrCmd::ADD_NUM ? " + " : cmd == IrCmd::SUB_NUM ? " - " : " * ") + b + ")";

        bool uns = c == NumClass::Uint;
        bool nonneg = c == NumClass::NonNeg;
        std::string u = uns ? "u" : "";
        int64_t max = uns ? 4294967295ll : 2147483647ll, min = uns ? 0 : -2147483648ll;

        int64_t k;
        bool constB = constInt(opb, c, k);
        if (!constB && (cmd == IrCmd::ADD_NUM || cmd == IrCmd::MUL_NUM) && constInt(opa, c, k))
        {
            std::swap(a, b);
            constB = true;
        }

        auto lit = [&](int64_t v)
        {
            return v == -2147483648ll ? std::string("int(0x80000000u)") : "(" + std::to_string(v) + u + ")";
        };

        switch (cmd)
        {
        case IrCmd::ADD_NUM:
        case IrCmd::SUB_NUM:
        {
            bool add = cmd == IrCmd::ADD_NUM;
            if (constB && (add || k != min))
            {
                int64_t d = add ? k : -k;
                if (d == 0)
                    return a;
                std::string r = a + (add ? " + " : " - ") + b;
                if (d > 0)
                    return "l_chk(" + a + " > " + lit(max - d) + ", " + r + ")";
                if (uns)
                    break;
                return "l_chk(" + a + " < " + lit(min - d) + ", " + r + ")";
            }
            if (add && nonneg)
                return "l_chk((" + a + " + " + b + ") < 0, " + a + " + " + b + ")";
            break;
        }
        case IrCmd::MUL_NUM:
            if (constB && k > 0)
            {
                if (k == 1)
                    return a;
                std::string r = a + " * " + b;
                if (uns || nonneg)
                    return "l_chk(" + a + " > " + lit(max / k) + ", " + r + ")";
                return "l_chk(" + a + " > " + lit(max / k) + " || " + a + " < " + lit(-(2147483648ll / k)) + ", " + r + ")";
            }
            break;
        case IrCmd::IDIV_NUM:
        case IrCmd::MOD_NUM:
            if (constB && k > 0)
            {
                bool idiv = cmd == IrCmd::IDIV_NUM;
                bool pow2 = (k & (k - 1)) == 0;
                int shift = 0;
                while ((int64_t(1) << shift) < k)
                    shift++;

                // shifting a two's complement value right rounds down, like Luau's floor division
                if (pow2 && !uns)
                    return idiv ? "(" + a + " >> " + std::to_string(shift) + ")" : "(" + a + " & " + lit(k - 1) + ")";
                if (uns || nonneg || range(opa).lo >= 0)
                    return "(" + a + (idiv ? " / " : " % ") + b + ")";
                return idiv ? "l_idivk(" + a + ", " + b + ")" : "(" + a + " - l_idivk(" + a + ", " + b + ") * " + b + ")";
            }
            break;
        default:
            break;
        }

        a = num(opa, c);
        b = num(opb, c);
        switch (cmd)
        {
        case IrCmd::ADD_NUM:
            return (uns ? "l_addu(" : "l_addi(") + a + ", " + b + ")";
        case IrCmd::SUB_NUM:
            return "l_subi(" + a + ", " + b + ")";
        case IrCmd::MUL_NUM:
            return (uns ? "l_mulu(" : "l_muli(") + a + ", " + b + ")";
        case IrCmd::IDIV_NUM:
            return (uns ? "l_idivu(" : "l_idivi(") + a + ", " + b + ")";
        case IrCmd::MOD_NUM:
            return (uns ? "l_modu(" : "l_modi(") + a + ", " + b + ")";
        default:
            CODEGEN_ASSERT(!"not an integer operation");
            return "0";
        }
    }

    std::string num(IrOp op, NumClass to, bool exact = false)
    {
        Num n = num(op);
        return convert(n.code, n.cls, to, exact);
    }

    // the variable holding the number of a register when it's of that class
    static std::string numReg(int reg, NumClass c)
    {
        return "r" + std::to_string(reg) + (c == NumClass::Float ? "f" : c == NumClass::Uint ? "u" : "a");
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
            return "vec3(" + literal(v[0], NumClass::Float) + ", " + literal(v[1], NumClass::Float) + ", " + literal(v[2], NumClass::Float) + ")";
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

    static NumClass join(NumClass a, NumClass b)
    {
        return a > b ? a : b;
    }

    static NumClass arith(NumClass a, NumClass b)
    {
        return join(join(a, b), NumClass::NonNeg);
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
        if (l == "gpu" && n == "atomicmax")
            return GpuCall::AtomicMax;
        if (l == "buffer" && n == "len")
            return GpuCall::BufferLen;
        return GpuCall::Unknown;
    }

    // class of the double produced by an instruction
    NumClass resultClass(IrInst& inst)
    {
        switch (inst.cmd)
        {
        case IrCmd::LOAD_DOUBLE:
            return join(cls(OP_A(inst)), NumClass::NonNeg);
        case IrCmd::ADD_NUM:
        case IrCmd::MUL_NUM:
        case IrCmd::IDIV_NUM:
        case IrCmd::MOD_NUM:
        case IrCmd::MIN_NUM:
        case IrCmd::MAX_NUM:
            return arith(cls(OP_A(inst)), cls(OP_B(inst)));
        case IrCmd::MULADD_NUM:
            return arith(arith(cls(OP_A(inst)), cls(OP_B(inst))), cls(OP_C(inst)));
        case IrCmd::SUB_NUM:
            return join(arith(cls(OP_A(inst)), cls(OP_B(inst))), NumClass::Int);
        case IrCmd::UNM_NUM:
        case IrCmd::SIGN_NUM:
            return join(cls(OP_A(inst)), NumClass::Int);
        case IrCmd::FLOOR_NUM:
        case IrCmd::CEIL_NUM:
        case IrCmd::ROUND_NUM:
        case IrCmd::ABS_NUM:
            return join(cls(OP_A(inst)), NumClass::NonNeg);
        case IrCmd::SELECT_NUM:
            return arith(cls(OP_A(inst)), cls(OP_B(inst)));
        case IrCmd::INT_TO_NUM:
            return NumClass::Int;
        case IrCmd::UINT_TO_NUM:
            return NumClass::Uint;
        default:
            return NumClass::Float;
        }
    }

    static Range classRange(NumClass c)
    {
        switch (c)
        {
        case NumClass::NonNeg:
            return {0, 2147483647.0};
        case NumClass::Uint:
            return {0, 4294967295.0};
        case NumClass::Int:
            return {-2147483648.0, 2147483647.0};
        default:
            return {};
        }
    }

    Range range(IrOp op)
    {
        if (op.kind == IrOpKind::Inst)
            return instRange[op.index];
        if (op.kind == IrOpKind::Constant)
        {
            IrConst& k = function.constOp(op);
            double v = k.kind == IrConstKind::Double ? k.valueDouble : k.kind == IrConstKind::Uint ? double(k.valueUint) : double(k.valueInt);
            return v == floor(v) ? Range{v, v} : Range{};
        }
        if (op.kind == IrOpKind::VmConst && ttisnumber(&proto->k[vmConstOp(op)]))
        {
            double v = nvalue(&proto->k[vmConstOp(op)]);
            return v == floor(v) ? Range{v, v} : Range{};
        }
        return classRange(cls(op));
    }

    Range resultRange(IrInst& inst)
    {
        Range a = range(OP_A(inst)), b = inst.ops.size() > 1 ? range(OP_B(inst)) : Range();
        Range r;

        switch (inst.cmd)
        {
        case IrCmd::LOAD_DOUBLE:
            return a;
        case IrCmd::ADD_NUM:
        case IrCmd::ADD_INT:
            r = {a.lo + b.lo, a.hi + b.hi};
            break;
        case IrCmd::SUB_NUM:
        case IrCmd::SUB_INT:
            r = {a.lo - b.hi, a.hi - b.lo};
            break;
        case IrCmd::MUL_NUM:
        {
            double p[] = {a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi};
            r = {std::min(std::min(p[0], p[1]), std::min(p[2], p[3])), std::max(std::max(p[0], p[1]), std::max(p[2], p[3]))};
            break;
        }
        case IrCmd::IDIV_NUM:
            if (b.lo == b.hi && b.lo > 0)
                r = {floor(a.lo / b.lo), floor(a.hi / b.lo)};
            break;
        case IrCmd::MOD_NUM:
            if (b.lo == b.hi && b.lo > 0)
                r = a.lo >= 0 && a.hi < b.lo ? a : Range{0, b.lo - 1};
            break;
        case IrCmd::MIN_NUM:
            r = {std::min(a.lo, b.lo), std::min(a.hi, b.hi)};
            break;
        case IrCmd::MAX_NUM:
            r = {std::max(a.lo, b.lo), std::max(a.hi, b.hi)};
            break;
        case IrCmd::UNM_NUM:
            r = {-a.hi, -a.lo};
            break;
        case IrCmd::FLOOR_NUM:
        case IrCmd::CEIL_NUM:
        case IrCmd::ROUND_NUM:
            r = {floor(a.lo), ceil(a.hi)};
            break;
        case IrCmd::INT_TO_NUM:
        case IrCmd::NUM_TO_INT:
            r = a;
            break;
        case IrCmd::UINT_TO_NUM:
            r = a.lo >= 0 ? a : Range{0, 4294967295.0};
            break;
        case IrCmd::BITAND_UINT:
            if (a.lo >= 0 || b.lo >= 0)
                r = {0, std::min(a.lo >= 0 ? a.hi : INFINITY, b.lo >= 0 ? b.hi : INFINITY)};
            break;
        default:
            break;
        }

        // NaN from infinite bounds
        if (!(r.lo <= r.hi))
            return {};
        return r;
    }

    // facts about the value an instruction produces, from what registers hold before it
    void facts(IrInst& inst, uint32_t index)
    {
        if (getCmdValueKind(inst.cmd) == IrValueKind::Double)
            instClass[index] = resultClass(inst);

        // only integers computed exactly have bounds; f32 rounding would move those of floats
        if ((getCmdValueKind(inst.cmd) == IrValueKind::Double && instClass[index] != NumClass::Float) ||
            getCmdValueKind(inst.cmd) == IrValueKind::Int)
            instRange[index] = resultRange(inst);

        if (inst.cmd == IrCmd::LOAD_POINTER || inst.cmd == IrCmd::LOAD_TVALUE)
            instBuffer[index] = buffer(OP_A(inst));

        if (inst.cmd == IrCmd::LOAD_TAG || inst.cmd == IrCmd::LOAD_TVALUE)
            instTag[index] = knownTag(OP_A(inst));

        if (inst.cmd == IrCmd::LOAD_TVALUE)
        {
            instClass[index] = cls(OP_A(inst));
            instCall[index] = OP_A(inst).kind == IrOpKind::VmReg ? cur[vmRegOp(OP_A(inst))].call : GpuCall::Unknown;
        }
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
        RegState s = {NumClass::None, -1, GpuCall::Unknown};

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
            s.cls = cls(OP_B(inst));
            s.tag = cur[r].tag;
            break;
        case IrCmd::STORE_INT:
            s.cls = cur[r].cls;
            s.tag = cur[r].tag;
            break;
        case IrCmd::STORE_VECTOR:
            s.tag = OP_E(inst).kind != IrOpKind::None ? knownTag(OP_E(inst)) : cur[r].tag;
            break;
        case IrCmd::STORE_TVALUE:
        {
            IrOp src = OP_B(inst);
            if (src.kind == IrOpKind::Inst && function.instOp(src).cmd == IrCmd::LOAD_TVALUE)
                s = {instClass[src.index], instBuffer[src.index] >= 0 ? instBuffer[src.index] : -1, instCall[src.index], instTag[src.index]};
            break;
        }
        case IrCmd::STORE_SPLIT_TVALUE:
            s.tag = knownTag(OP_B(inst));
            if (function.tagOp(OP_B(inst)) == LUA_TNUMBER)
                s.cls = cls(OP_C(inst));
            else if (function.tagOp(OP_B(inst)) == LUA_TBUFFER)
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
            // the calls that can be lowered return unsigned 32-bit integers, and the arguments don't survive the call
            int nargs = function.intOp(OP_B(inst));
            for (int i = 1; i <= nargs && r + i < int(cur.size()); i++)
                cur[r + i] = {NumClass::None, -1, GpuCall::Unknown};
            s.cls = NumClass::Uint;
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
                join(to[r].cls, cur[r].cls),
                to[r].buffer == cur[r].buffer ? to[r].buffer : -2,
                to[r].call == cur[r].call ? to[r].call : GpuCall::Unknown,
                to[r].tag == cur[r].tag ? to[r].tag : -1
            };
            changed |= m.cls != to[r].cls || m.buffer != to[r].buffer || m.call != to[r].call || m.tag != to[r].tag;
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

    // finds what registers hold when each block starts; classes and unknowns only grow, so this terminates
    bool analyze(const std::vector<uint32_t>& sortedBlocks)
    {
        size_t size = function.instructions.size();
        instClass.assign(size, NumClass::None);
        instBuffer.assign(size, -1);
        instCall.assign(size, GpuCall::None);
        instTag.assign(size, -1);
        instRange.assign(size, Range());

        std::vector<RegState> start(proto->maxstacksize);
        start[0].cls = NumClass::NonNeg;
        start[0].tag = LUA_TNUMBER;
        for (int i = 1; i < proto->numparams; i++)
        {
            start[i].buffer = i - 1;
            start[i].tag = LUA_TBUFFER;
        }

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
            // numbers move to the variable of the class the target block expects, widening exactly
            std::string moves;
            const std::vector<RegState>& to = entry[target.index];
            for (size_t r = 0; r < to.size(); r++)
            {
                if (!live(target.index, r))
                    continue;

                NumClass from = cur[r].cls;
                if (from == NumClass::None || from == to[r].cls)
                    continue;

                std::string src = numReg(int(r), from);
                std::string value = convert(src, from, to[r].cls, true);
                std::string dst = numReg(int(r), to[r].cls);
                if (value != src || dst != src)
                    moves += dst + " = " + value + "; ";
            }

            return "{ " + moves + branch(target.index) + " }";
        }

        // fallbacks, exits to the VM and aborts: the host reruns the dispatch in the VM
        return "{ l_fail(); return; }";
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

            if (i == constructs.size())
                return loop ? "continue;" : "break;";

            (loop ? c.restarted : c.exited) = true;
            for (size_t j = i; j < constructs.size(); j++)
                constructs[j].passed = true;
            return "l_jmp = " + (loop ? continueCode(block) : breakCode(block)) + "; break;";
        }

        CODEGEN_ASSERT(!"jump to a construct that doesn't enclose it");
        return "l_fail(); return;";
    }

    std::string branch(uint32_t target)
    {
        if (backward(current, target))
            return leave(target, true);
        if (merge(target))
            return leave(target, false);

        // the only way into the target: it's emitted right here
        uint32_t from = current;
        std::vector<RegState> state = cur;
        std::string code = tree(target);
        current = from;
        cur = state;
        return code;
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
            std::string code = "for (;;)\n{\n" + around(header, exits, i + 1, inner) + "break;\n}\n";
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
        std::string code = "for (;;)\n{\n" + within(block, merges, i + 1) + "break;\n}\n";
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

        IrBlock& block = function.blocks[blockIndex];
        for (uint32_t index = block.start; index <= block.finish && !failed; index++)
        {
            IrInst& inst = function.instructions[index];
            if (isPseudo(inst.cmd))
                continue;

            facts(inst, index);
            if (!lowerInst(inst, index))
                break;
            effects(inst, index);
        }

        saved.swap(out);
        return saved;
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

    // compares two numbers in a class both fit in
    std::string compareNum(IrOp a, IrOp b, IrCondition cond)
    {
        NumClass c = arith(cls(a), cls(b));
        return compare(num(a, c), num(b, c), cond);
    }

    std::string access(int binding, const std::string& offset, const char* op, const std::string& value = "")
    {
        if (op[0] == 'r')
            readBuffers[binding] = true;
        else
            build.bufferWritten[binding] = true;

        std::string b = std::to_string(binding);
        return value.empty() ? "l_rd" + b + "(" + offset + ")" : "l_wr" + b + "(" + offset + ", " + value + ")";
    }

    bool binding(uint32_t index, IrOp op, int& b)
    {
        b = buffer(op);
        return b >= 0 ? true : reject(index, "a buffer that isn't a parameter");
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
        NumClass c = instClass[index];

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
            line(v + " = " + num(OP_A(inst), c) + ";");
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
                line(v + "t = " + r + "t; " + v + "n = " + numReg(vmRegOp(a), cls(a)) + "; " + v + "i = " + r + "i; " + v + "v = " + r + "v;");
            }
            else if (a.kind == IrOpKind::VmConst)
            {
                TValue* k = &proto->k[vmConstOp(a)];
                if (ttisvector(k))
                    line(v + "t = " + std::to_string(LUA_TVECTOR) + "u; " + v + "v = " + vec(a) + ";");
                else if (ttisnumber(k))
                    line(v + "t = " + std::to_string(LUA_TNUMBER) + "u; " + v + "n = " + literal(nvalue(k), cls(a)) + ";");
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
        {
            Num n = num(OP_B(inst));
            line(numReg(vmRegOp(OP_A(inst)), n.cls) + " = " + n.code + ";");
            break;
        }
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
            const char* op = inst.cmd == IrCmd::ADD_VEC ? " + " : inst.cmd == IrCmd::SUB_VEC ? " - " : inst.cmd == IrCmd::MUL_VEC ? " * " : " / ";
            line(v + "v = " + vec(OP_A(inst)) + op + vec(OP_B(inst)) + ";");
            break;
        }
        case IrCmd::IDIV_VEC:
            line(v + "v = floor(" + vec(OP_A(inst)) + " / " + vec(OP_B(inst)) + ");");
            break;
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
            line(v + " = dot(" + vec(OP_A(inst)) + ", " + vec(OP_B(inst)) + ");");
            break;
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
            const char* op = inst.cmd == IrCmd::ADD_FLOAT   ? " + "
                             : inst.cmd == IrCmd::SUB_FLOAT ? " - "
                             : inst.cmd == IrCmd::MUL_FLOAT ? " * "
                                                            : " / ";
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
            line(v + " = l_sqrt(" + floatv(OP_A(inst)) + ");");
            break;
        case IrCmd::SIGN_FLOAT:
        {
            std::string a = floatv(OP_A(inst));
            line(v + " = " + a + " > 0.0 ? 1.0 : " + a + " < 0.0 ? -1.0 : 0.0;");
            break;
        }
        case IrCmd::UINT_TO_FLOAT:
            line(v + " = float(uint(" + intv(OP_A(inst)) + "));");
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

            line(rs + "t = " + s + "t; " + numReg(r, instClass[src.index]) + " = " + s + "n; " + rs + "i = " + s + "i; " + rs + "v = " + s + "v;");
            break;
        }
        case IrCmd::STORE_SPLIT_TVALUE:
        {
            int r = vmRegOp(OP_A(inst));
            uint8_t t = function.tagOp(OP_B(inst));
            std::string rs = "r" + std::to_string(r);
            line(rs + "t = " + std::to_string(t) + "u;");

            if (t == LUA_TNUMBER)
            {
                Num n = num(OP_C(inst));
                line(numReg(r, n.cls) + " = " + n.code + ";");
            }
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
        case IrCmd::MUL_NUM:
        {
            std::string a = num(OP_A(inst), c), b = num(OP_B(inst), c);
            bool add = inst.cmd == IrCmd::ADD_NUM;
            if (c == NumClass::Float)
                line(v + " = " + a + (add ? " + " : " * ") + b + ";");
            else
                line(v + " = " + intArith(index, inst.cmd, OP_A(inst), OP_B(inst), c) + ";");
            break;
        }
        case IrCmd::SUB_NUM:
            if (c == NumClass::Float)
                line(v + " = " + num(OP_A(inst), c) + " - " + num(OP_B(inst), c) + ";");
            else
                line(v + " = " + intArith(index, inst.cmd, OP_A(inst), OP_B(inst), c) + ";");
            break;
        case IrCmd::MULADD_NUM:
            if (c == NumClass::Float)
                line(v + " = " + num(OP_A(inst), c) + " * " + num(OP_B(inst), c) + " + " + num(OP_C(inst), c) + ";");
            else if (c == NumClass::Uint)
                line(v + " = l_addu(l_mulu(" + num(OP_A(inst), c) + ", " + num(OP_B(inst), c) + "), " + num(OP_C(inst), c) + ");");
            else
                line(v + " = l_addi(l_muli(" + num(OP_A(inst), c) + ", " + num(OP_B(inst), c) + "), " + num(OP_C(inst), c) + ");");
            break;
        case IrCmd::DIV_NUM:
            line(v + " = " + num(OP_A(inst), c) + " / " + num(OP_B(inst), c) + ";");
            break;
        case IrCmd::IDIV_NUM:
        case IrCmd::MOD_NUM:
        {
            std::string a = num(OP_A(inst), c), b = num(OP_B(inst), c);
            bool idiv = inst.cmd == IrCmd::IDIV_NUM;
            if (c == NumClass::Float)
                line(v + " = " + (idiv ? "floor(" + a + " / " + b + ")" : a + " - floor(" + a + " / " + b + ") * " + b) + ";");
            else
                line(v + " = " + intArith(index, inst.cmd, OP_A(inst), OP_B(inst), c) + ";");
            break;
        }
        case IrCmd::MIN_NUM:
        case IrCmd::MAX_NUM:
        {
            // when one value is NaN, the second one is the result
            std::string a = num(OP_A(inst), c), b = num(OP_B(inst), c);
            line(v + " = " + a + (inst.cmd == IrCmd::MIN_NUM ? " < " : " > ") + b + " ? " + a + " : " + b + ";");
            break;
        }
        case IrCmd::UNM_NUM:
            line(v + " = " + (c == NumClass::Float ? "-" + num(OP_A(inst), c) : "l_negi(" + num(OP_A(inst), c) + ")") + ";");
            break;
        case IrCmd::FLOOR_NUM:
        case IrCmd::CEIL_NUM:
        case IrCmd::ROUND_NUM:
        {
            std::string a = num(OP_A(inst), c);
            const char* fn = inst.cmd == IrCmd::FLOOR_NUM ? "floor(" : inst.cmd == IrCmd::CEIL_NUM ? "ceil(" : "l_round(";
            line(v + " = " + (c == NumClass::Float ? fn + a + ")" : a) + ";");
            break;
        }
        case IrCmd::SQRT_NUM:
            line(v + " = l_sqrt(" + num(OP_A(inst), c) + ");");
            break;
        case IrCmd::ABS_NUM:
        {
            std::string a = num(OP_A(inst), c);
            line(v + " = " + (c == NumClass::Float ? "abs(" + a + ")" : c == NumClass::Int ? a + " < 0 ? l_negi(" + a + ") : " + a : a) + ";");
            break;
        }
        case IrCmd::SIGN_NUM:
        {
            std::string a = num(OP_A(inst), c);
            line(v + " = " + (c == NumClass::Float ? a + " > 0.0 ? 1.0 : " + a + " < 0.0 ? -1.0 : 0.0" : "sign(" + a + ")") + ";");
            break;
        }
        case IrCmd::SELECT_NUM:
            line(v + " = " + compareNum(OP_C(inst), OP_D(inst), IrCondition::Equal) + " ? " + num(OP_B(inst), c) + " : " + num(OP_A(inst), c) + ";");
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
            std::string fail = "(l_fail(), 0)";

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
            NumClass k = arith(arith(cls(OP_A(inst)), cls(OP_B(inst))), cls(OP_C(inst)));
            std::string idx = num(OP_A(inst), k), limit = num(OP_B(inst), k), step = num(OP_C(inst), k);
            std::string zero = k == NumClass::Float ? "0.0" : k == NumClass::Uint ? "0u" : "0";
            line(
                "if (" + step + " > " + zero + " ? " + idx + " <= " + limit + " : " + limit + " <= " + idx + ") " + jump(OP_D(inst)) + " else " +
                jump(OP_E(inst))
            );
            break;
        }
        case IrCmd::INT_TO_NUM:
            line(v + " = " + intv(OP_A(inst)) + ";");
            break;
        case IrCmd::UINT_TO_NUM:
            line(v + " = uint(" + intv(OP_A(inst)) + ");");
            break;
        case IrCmd::NUM_TO_INT:
        {
            Num a = num(OP_A(inst));
            line(v + " = " + (a.cls == NumClass::Float ? "l_f2i(" + a.code + ")" : convert(a.code, a.cls, NumClass::Int)) + ";");
            break;
        }
        case IrCmd::NUM_TO_UINT:
        {
            // out of range values are architecture specific in native code; here they fail the dispatch
            Num a = num(OP_A(inst));
            line(v + " = int(" + (a.cls == NumClass::Float ? "l_f2u(" + a.code + ")" : a.code) + ");");
            break;
        }
        case IrCmd::FLOAT_TO_NUM:
            line(v + " = " + floatv(OP_A(inst)) + ";");
            break;
        case IrCmd::TRUNCATE_UINT:
            line(v + " = " + intv(OP_A(inst)) + ";");
            break;
        case IrCmd::NUM_TO_FLOAT:
            line(v + " = " + num(OP_A(inst), NumClass::Float) + ";");
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

            std::string base = intv(OP_B(inst));
            std::string len = "int(pc.len[" + std::to_string(b) + "])";
            guard("(" + base + " >= -(" + intv(OP_C(inst)) + ") && " + base + " <= " + len + " - " + intv(OP_D(inst)) + ")", OP_F(inst));

            // the offset must also be the exact integer the source number holds
            if (OP_E(inst).kind != IrOpKind::Undef)
            {
                Num src = num(OP_E(inst));
                if (src.cls == NumClass::Float)
                    guard("(float(" + base + ") == " + src.code + ")", OP_F(inst));
            }
            break;
        }
        case IrCmd::INTERRUPT:
            // native code checks for interrupts here, at every loop iteration and before returning
            // and every 256 iterations, whether the host asked a dispatch that runs for too long to stop
            line("if ((++l_steps > pc.budget && pc.budget != 0u) || ((l_steps & 255u) == 0u && st.stop != 0)) { l_fail(); return; }");
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

            std::string raw = access(b, intv(OP_B(inst)), "r");
            line(v + " = " + (inst.cmd == IrCmd::BUFFER_READF32 ? "uintBitsToFloat(" + raw + ")" : "int(" + raw + ")") + ";");
            break;
        }
        case IrCmd::BUFFER_WRITEI32:
        case IrCmd::BUFFER_WRITEF32:
        {
            int b;
            if (!binding(index, OP_A(inst), b))
                return false;

            std::string value = inst.cmd == IrCmd::BUFFER_WRITEF32 ? "floatBitsToUint(" + floatv(OP_C(inst)) + ")" : "uint(" + intv(OP_C(inst)) + ")";
            line(access(b, intv(OP_B(inst)), "w", value) + ";");
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
        std::string a = num(OP_B(inst), NumClass::Float);
        std::string b = OP_C(inst).kind != IrOpKind::None ? num(OP_C(inst), NumClass::Float) : "";

        switch (function.uintOp(OP_A(inst)))
        {
        case LBF_MATH_EXP:
            line(v + " = exp(" + a + ");");
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
            return reject(index, "a buffer that isn't a parameter");

        std::string value;
        if (call == GpuCall::BufferLen)
        {
            if (nargs != 1)
                return reject(index, "buffer.len with these arguments");
            value = "pc.len[" + std::to_string(b) + "]";
        }
        else
        {
            if (nargs != 3)
                return reject(index, "an atomic with these arguments");

            // offsets are rounded toward zero and values wrap from the ranges the VM accepts, as in the gpu library
            Num off = num(IrOp{IrOpKind::VmReg, uint32_t(ra + 2)});
            Num val = num(IrOp{IrOpKind::VmReg, uint32_t(ra + 3)});
            std::string offset = off.cls == NumClass::Float ? "l_f2i(" + off.code + ")" : convert(off.code, off.cls, NumClass::Int);
            std::string operand = val.cls == NumClass::Float ? "l_f2u(" + val.code + ")"
                                  : val.cls == NumClass::Int ? "uint(" + val.code + ")"
                                                             : convert(val.code, val.cls, NumClass::Uint);

            int op = call == GpuCall::AtomicAdd ? 0 : call == GpuCall::AtomicMin ? 1 : 2;
            atomics[b] |= uint8_t(1 << op);
            build.bufferWritten[b] = true;

            static const char* const kNames[] = {"add", "min", "max"};
            value = "l_atomic" + std::string(kNames[op]) + std::to_string(b) + "(" + offset + ", " + operand + ")";
        }

        line(r + "t = " + std::to_string(LUA_TNUMBER) + "u;");
        if (nresults != 0)
            line(numReg(ra, NumClass::Uint) + " = " + value + ";");
        else
            line(value + ";");
        return true;
    }

    std::string floatv(IrOp op)
    {
        if (op.kind == IrOpKind::Inst && getCmdValueKind(function.instOp(op).cmd) == IrValueKind::Float)
            return "v" + std::to_string(op.index);
        return num(op, NumClass::Float);
    }

    std::string declarations()
    {
        std::string decl;

        for (int r = 0; r < proto->maxstacksize; r++)
        {
            std::string rs = "r" + std::to_string(r);
            decl += "    uint " + rs + "t = 0u; int " + rs + "i = 0; vec3 " + rs + "v = vec3(0.0); int " + rs + "a = 0; uint " + rs +
                    "u = 0u; float " + rs + "f = 0.0;\n";
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
                decl += "    float " + v + " = 0.0;\n";
                break;
            case IrValueKind::Double:
                decl += std::string("    ") + glslType(instClass[i]) + " " + v + " = " + glslType(instClass[i]) + "(0);\n";
                break;
            case IrValueKind::Tvalue:
            {
                NumClass c = inst.cmd == IrCmd::LOAD_TVALUE ? instClass[i] : NumClass::Float;
                decl += "    uint " + v + "t = 0u; " + glslType(c) + " " + v + "n = " + glslType(c) + "(0); int " + v + "i = 0; vec3 " + v +
                        "v = vec3(0.0);\n";
                break;
            }
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
            return reject(0, "a function without the invocation index parameter");
        if (proto->numparams - 1 > kGpuMaxBuffers)
            return reject(0, "a function with more than " + std::to_string(kGpuMaxBuffers) + " buffer parameters");

        int buffers = proto->numparams - 1;
        build.bufferWritten.assign(buffers, false);
        readBuffers.assign(buffers, false);
        atomics.assign(buffers, 0);

        // which registers each block reads, on the IR after the optimizations
        computeCfgInfo(function);

        if (!analyze(sortedBlocks))
            return false;

        orderBlocks();
        std::string body = tree(function.entryBlock);
        if (failed)
            return false;

        std::string& glsl = build.glsl;
        // like the VM, shaders keep NaN, infinities and signed zeros (SignedZeroInfNanPreserve for 32-bit floats)
        glsl = "#version 450\n#extension GL_EXT_spirv_intrinsics : require\n"
               "spirv_execution_mode(extensions = [\"SPV_KHR_float_controls\"], capabilities = [4466], 4461, 32);\n"
               "layout(local_size_x = " +
               std::to_string(kGpuWorkgroupSize) + ") in;\n\n";

        for (int i = 0; i < buffers; i++)
            glsl += "layout(std430, set = 0, binding = " + std::to_string(i) + ") buffer B" + std::to_string(i) + " { uint d[]; } b" +
                    std::to_string(i) + ";\n";
        // 'stop' is an int so that the block doesn't have the structure of the result of umulExtended, which SPIR-V would make
        // the same type and SPIRV-Cross would then declare with the block's name
        glsl += "layout(std430, set = 0, binding = " + std::to_string(buffers) +
                ") coherent volatile buffer BS { uint err; int stop; uint started; } st;\n";
        glsl += "layout(push_constant) uniform PC { uint n; uint budget; uint len[" + std::to_string(kGpuMaxBuffers) + "]; } pc;\n";
        glsl += kPrelude;

        for (int i = 0; i < buffers; i++)
        {
            std::string b = std::to_string(i);
            // reads and writes follow a CHECK_BUFFER_LEN that covers their bounds, as in native code; words must be aligned
            if (readBuffers[i])
                glsl += "uint l_rd" + b + "(int off) { if ((off & 3) != 0) { l_fail(); return 0u; } return b" + b + ".d[off >> 2]; }\n";
            if (build.bufferWritten[i])
                glsl += "void l_wr" + b + "(int off, uint v) { if ((off & 3) != 0) { l_fail(); return; } b" + b + ".d[off >> 2] = v; }\n";

            static const char* const kAtomics[][2] = {{"add", "atomicAdd"}, {"min", "atomicMin"}, {"max", "atomicMax"}};
            for (int op = 0; op < 3; op++)
            {
                if (atomics[i] & (1 << op))
                    glsl += std::string("uint l_atomic") + kAtomics[op][0] + b + "(int off, uint v) { if (l_bad(off, " + b +
                            ")) { l_fail(); return 0u; } return " + kAtomics[op][1] + "(b" + b + ".d[off >> 2], v); }\n";
            }
        }

        glsl += "\nvoid l_kernel(int index)\n{\n" + declarations();
        glsl += "    r0t = " + std::to_string(LUA_TNUMBER) + "u; r0a = index;\n";
        for (int i = 1; i <= buffers; i++)
            glsl += "    r" + std::to_string(i) + "t = " + std::to_string(LUA_TBUFFER) + "u;\n";

        glsl += "    int l_jmp = 0;\n";
        glsl += body;
        glsl += "}\n\n";
        glsl += "void main()\n{\n";
        glsl +=
            "    uint gid = gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * gl_NumWorkGroups.x * " + std::to_string(kGpuWorkgroupSize) + "u;\n";
        // invocations that start after the host asked the dispatch to stop give up at once
        glsl += "    if (gid >= pc.n) return;\n    if (gid == 0u) st.started = 1u;\n    if (st.stop != 0) { l_fail(); return; }\n"
                "    l_kernel(int(gid));\n}\n";

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
