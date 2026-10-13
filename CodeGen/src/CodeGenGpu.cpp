// This file is part of the Luau programming language and is licensed under MIT License; see LICENSE.txt for details
#include "Luau/CodeGenGpu.h"

#include "CodeGenLower.h"

#include "lapi.h"
#include "ldebug.h"

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

struct GlslLowering
{
    IrFunction& function;
    Proto* proto;
    GlslBuilder& build;

    std::vector<bool> lowered;       // per block: lowered, or a fallback/exit that fails the dispatch
    std::vector<NumClass> instClass; // per instruction producing a double
    std::vector<int> instBuffer;     // per instruction producing a buffer pointer or TValue
    std::vector<NumClass> regClass;  // number slot of each VM register
    std::vector<int> regBuffer;      // buffer binding held by each VM register, -1 for none, -2 when it varies
    std::vector<bool> readBuffers;
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
            return regClass[vmRegOp(op)];
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
            return {"v" + std::to_string(op.index), c};
        case IrOpKind::Constant:
        {
            IrConst& k = function.constOp(op);
            return {literal(k.kind == IrConstKind::Double ? k.valueDouble : double(k.valueInt), c), c};
        }
        case IrOpKind::VmReg:
            return {"r" + std::to_string(vmRegOp(op)) + "n", c};
        case IrOpKind::VmConst:
            return {literal(nvalue(&proto->k[vmConstOp(op)]), c), c};
        default:
            return {"0", NumClass::NonNeg};
        }
    }

    std::string num(IrOp op, NumClass to, bool exact = false)
    {
        Num n = num(op);
        return convert(n.code, n.cls, to, exact);
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
        return "v" + std::to_string(op.index);
    }

    std::string tag(IrOp op)
    {
        if (op.kind == IrOpKind::Constant)
            return std::to_string(function.tagOp(op)) + "u";
        if (op.kind == IrOpKind::VmReg)
            return "r" + std::to_string(vmRegOp(op)) + "t";
        return "v" + std::to_string(op.index);
    }

    int buffer(IrOp op)
    {
        if (op.kind == IrOpKind::Inst)
            return instBuffer[op.index];
        if (op.kind == IrOpKind::VmReg)
            return regBuffer[vmRegOp(op)];
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

    void storeNum(int reg, NumClass c, bool& changed)
    {
        NumClass j = join(regClass[reg], c);
        changed |= j != regClass[reg];
        regClass[reg] = j;
    }

    void storeBuffer(int reg, int binding, bool& changed)
    {
        int b = regBuffer[reg] == -1 || regBuffer[reg] == binding ? binding : -2;
        changed |= b != regBuffer[reg];
        regBuffer[reg] = b;
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

    // infers the class of every number until it settles; classes only grow, so this terminates
    bool analyze(const std::vector<uint32_t>& sortedBlocks)
    {
        instClass.assign(function.instructions.size(), NumClass::None);
        instBuffer.assign(function.instructions.size(), -1);
        regClass.assign(proto->maxstacksize, NumClass::None);
        regBuffer.assign(proto->maxstacksize, -1);

        regClass[0] = NumClass::NonNeg;
        for (int i = 1; i < proto->numparams; i++)
            regBuffer[i] = i - 1;

        for (bool changed = true; changed;)
        {
            changed = false;

            for (uint32_t blockIndex : sortedBlocks)
            {
                if (!lowered[blockIndex])
                    continue;

                IrBlock& block = function.blocks[blockIndex];

                for (uint32_t index = block.start; index <= block.finish; index++)
                {
                    IrInst& inst = function.instructions[index];

                    if (isPseudo(inst.cmd))
                        continue;

                    if (getCmdValueKind(inst.cmd) == IrValueKind::Double)
                    {
                        NumClass c = join(instClass[index], resultClass(inst));
                        changed |= c != instClass[index];
                        instClass[index] = c;
                    }
                    else if (inst.cmd == IrCmd::LOAD_POINTER || inst.cmd == IrCmd::LOAD_TVALUE)
                    {
                        instBuffer[index] = buffer(OP_A(inst));
                    }

                    if ((inst.cmd == IrCmd::STORE_DOUBLE || inst.cmd == IrCmd::STORE_SPLIT_TVALUE || inst.cmd == IrCmd::STORE_TVALUE ||
                         inst.cmd == IrCmd::STORE_POINTER) &&
                        OP_A(inst).kind != IrOpKind::VmReg)
                        return reject(index, "a store outside of the stack");

                    if (inst.cmd == IrCmd::STORE_DOUBLE)
                        storeNum(vmRegOp(OP_A(inst)), cls(OP_B(inst)), changed);
                    else if (inst.cmd == IrCmd::STORE_SPLIT_TVALUE && function.tagOp(OP_B(inst)) == LUA_TNUMBER)
                        storeNum(vmRegOp(OP_A(inst)), cls(OP_C(inst)), changed);
                    else if (inst.cmd == IrCmd::STORE_SPLIT_TVALUE && function.tagOp(OP_B(inst)) == LUA_TBUFFER)
                        storeBuffer(vmRegOp(OP_A(inst)), buffer(OP_C(inst)), changed);
                    else if (inst.cmd == IrCmd::STORE_POINTER)
                        storeBuffer(vmRegOp(OP_A(inst)), buffer(OP_B(inst)), changed);
                    else if (inst.cmd == IrCmd::STORE_TVALUE)
                    {
                        IrOp src = OP_B(inst);
                        if (src.kind != IrOpKind::Inst || function.instOp(src).cmd != IrCmd::LOAD_TVALUE)
                            continue; // rejected when lowering the instruction producing the value, or the store

                        IrOp from = OP_A(function.instOp(src));
                        storeNum(vmRegOp(OP_A(inst)), cls(from), changed);
                        if (buffer(from) != -1)
                            storeBuffer(vmRegOp(OP_A(inst)), buffer(from), changed);
                    }
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
            return "{ l_block = " + std::to_string(target.index) + "; continue; }";

        // fallbacks, exits to the VM and aborts: the host reruns the dispatch in the VM
        return "{ l_fail(); return; }";
    }

    void guard(const std::string& ok, IrOp target)
    {
        line("if (!" + ok + ") " + jump(target));
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
                line(v + "t = " + r + "t; " + v + "n = " + r + "n; " + v + "i = " + r + "i;");
            }
            else if (a.kind == IrOpKind::VmConst)
            {
                TValue* k = &proto->k[vmConstOp(a)];
                if (ttisnumber(k))
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
            int r = vmRegOp(OP_A(inst));
            line("r" + std::to_string(r) + "n = " + num(OP_B(inst), regClass[r], true) + ";");
            break;
        }
        case IrCmd::STORE_INT:
            line("r" + std::to_string(vmRegOp(OP_A(inst))) + "i = " + intv(OP_B(inst)) + ";");
            break;
        case IrCmd::STORE_TVALUE:
        {
            int r = vmRegOp(OP_A(inst));
            IrOp src = OP_B(inst);
            if (src.kind != IrOpKind::Inst || function.instOp(src).cmd != IrCmd::LOAD_TVALUE)
                return reject(index, "a value copy from memory");

            std::string s = "v" + std::to_string(src.index);
            NumClass from = cls(OP_A(function.instOp(src)));
            std::string rs = "r" + std::to_string(r);
            line(rs + "t = " + s + "t; " + rs + "n = " + convert(s + "n", from, regClass[r], true) + "; " + rs + "i = " + s + "i;");
            break;
        }
        case IrCmd::STORE_SPLIT_TVALUE:
        {
            int r = vmRegOp(OP_A(inst));
            uint8_t t = function.tagOp(OP_B(inst));
            std::string rs = "r" + std::to_string(r);
            line(rs + "t = " + std::to_string(t) + "u;");

            if (t == LUA_TNUMBER)
                line(rs + "n = " + num(OP_C(inst), regClass[r], true) + ";");
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
                line(
                    v + " = " + (add ? (c == NumClass::Uint ? "l_addu(" : "l_addi(") : (c == NumClass::Uint ? "l_mulu(" : "l_muli(")) + a + ", " + b +
                    ");"
                );
            break;
        }
        case IrCmd::SUB_NUM:
            if (c == NumClass::Float)
                line(v + " = " + num(OP_A(inst), c) + " - " + num(OP_B(inst), c) + ";");
            else
                line(v + " = l_subi(" + num(OP_A(inst), c) + ", " + num(OP_B(inst), c) + ");");
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
                line(
                    v + " = " + (idiv ? (c == NumClass::Uint ? "l_idivu(" : "l_idivi(") : (c == NumClass::Uint ? "l_modu(" : "l_modi(")) + a + ", " +
                    b + ");"
                );
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
            line("if (pc.budget != 0u && ++l_steps > pc.budget) { l_fail(); return; }");
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
            decl += "    uint " + rs + "t = 0u; " + glslType(regClass[r]) + " " + rs + "n = " + glslType(regClass[r]) + "(0); int " + rs + "i = 0;\n";
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
                if (inst.cmd == IrCmd::LOAD_TVALUE)
                {
                    NumClass c = cls(OP_A(inst));
                    decl += "    uint " + v + "t = 0u; " + glslType(c) + " " + v + "n = " + glslType(c) + "(0); int " + v + "i = 0;\n";
                }
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
            if (function.bcMapping[pc].irLocation != ~0u)
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

        if (!analyze(sortedBlocks))
            return false;

        for (uint32_t blockIndex : sortedBlocks)
        {
            if (!lowered[blockIndex])
                continue;

            IrBlock& block = function.blocks[blockIndex];
            out += "    case " + std::to_string(blockIndex) + ":\n    {\n";

            for (uint32_t index = block.start; index <= block.finish; index++)
            {
                IrInst& inst = function.instructions[index];
                if (!isPseudo(inst.cmd) && !lowerInst(inst, index))
                    return false;
            }

            out += "    }\n";
        }

        std::string& glsl = build.glsl;
        glsl = "#version 450\nlayout(local_size_x = " + std::to_string(kGpuWorkgroupSize) + ") in;\n\n";

        for (int i = 0; i < buffers; i++)
            glsl += "layout(std430, set = 0, binding = " + std::to_string(i) + ") buffer B" + std::to_string(i) + " { uint d[]; } b" +
                    std::to_string(i) + ";\n";
        glsl += "layout(std430, set = 0, binding = " + std::to_string(buffers) + ") buffer BS { uint err; } st;\n";
        glsl += "layout(push_constant) uniform PC { uint n; uint budget; uint len[" + std::to_string(kGpuMaxBuffers) + "]; } pc;\n";
        glsl += kPrelude;

        for (int i = 0; i < buffers; i++)
        {
            std::string b = std::to_string(i);
            if (readBuffers[i])
                glsl += "uint l_rd" + b + "(int off) { if (l_bad(off, " + b + ")) { l_fail(); return 0u; } return b" + b + ".d[off >> 2]; }\n";
            if (build.bufferWritten[i])
                glsl += "void l_wr" + b + "(int off, uint v) { if (l_bad(off, " + b + ")) { l_fail(); return; } b" + b + ".d[off >> 2] = v; }\n";
        }

        glsl += "\nvoid l_kernel(int index)\n{\n" + declarations();
        glsl += "    r0t = " + std::to_string(LUA_TNUMBER) + "u; r0n = " + convert("index", NumClass::NonNeg, regClass[0], true) + ";\n";
        for (int i = 1; i <= buffers; i++)
            glsl += "    r" + std::to_string(i) + "t = " + std::to_string(LUA_TBUFFER) + "u;\n";

        glsl += "    int l_block = " + std::to_string(function.entryBlock) + ";\n    for (;;)\n    {\n    switch (l_block)\n    {\n";
        glsl += out;
        glsl += "    }\n    return;\n    }\n}\n\n";
        glsl += "void main()\n{\n";
        glsl +=
            "    uint gid = gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * gl_NumWorkGroups.x * " + std::to_string(kGpuWorkgroupSize) + "u;\n";
        glsl += "    if (gid >= pc.n) return;\n    l_kernel(int(gid));\n}\n";

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

GpuKernel compileGpu(lua_State* L, int idx)
{
    GpuKernel kernel;
    kernel.result = CodeGenCompilationResult::CodeGenLoweringFailure;

    if (!lua_isLfunction(L, idx))
    {
        kernel.reason = "not a Luau function";
        return kernel;
    }

    Closure* cl = clvalue(luaA_toobject(L, idx));
    Proto* proto = cl->l.p;

    if ((proto->flags & LPF_GPU_FUNCTION) == 0)
    {
        kernel.reason = "the function doesn't have the gpu attribute";
        return kernel;
    }

    // the shader can't check the environment at run time, so builtins must be the real ones
    if (!cl->env->safeenv)
    {
        kernel.reason = "the function's environment isn't safe (sandbox the state and don't modify builtin libraries)";
        return kernel;
    }

    IrBuilder ir(CompilationOptions().hooks, VmEnvironmentInfo{});
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

} // namespace CodeGen
} // namespace Luau
