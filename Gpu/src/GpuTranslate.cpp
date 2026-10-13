// This file is part of the Luau programming language and is licensed under MIT License; see LICENSE.txt for details
#include "Luau/GpuTranslate.h"

#include "Luau/Ast.h"
#include "Luau/Parser.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#include <map>
#include <set>
#include <unordered_map>

namespace Luau
{
namespace Gpu
{

namespace
{

// Luau numbers are doubles; on the GPU each value gets one of these types, inferred from how it's produced
enum class Ty
{
    Unknown,
    Bool,
    Int,
    UInt,
    Float,
    Vec,
    Buffer,
    Void,
};

struct Type
{
    Ty ty = Ty::Unknown;
    int binding = -1; // for Ty::Buffer

    bool operator==(const Type& other) const
    {
        return ty == other.ty && binding == other.binding;
    }
    bool operator!=(const Type& other) const
    {
        return !(*this == other);
    }
    bool operator<(const Type& other) const
    {
        return ty != other.ty ? ty < other.ty : binding < other.binding;
    }
};

Type type(Ty ty)
{
    return Type{ty, -1};
}

bool isNumeric(Ty ty)
{
    return ty == Ty::Int || ty == Ty::UInt || ty == Ty::Float;
}

bool isInteger(Ty ty)
{
    return ty == Ty::Int || ty == Ty::UInt;
}

struct Val
{
    std::string code;
    Type type;
    // integer literals that fit both int and uint stay unsigned when combined with uint values
    bool nonNegLiteral = false;
};

struct TranslateError
{
    Location location;
    std::string message;
};

[[noreturn]] void fail(const Location& location, const std::string& message)
{
    throw TranslateError{location, message};
}

std::string glslType(const Location& location, Type t)
{
    switch (t.ty)
    {
    case Ty::Bool:
        return "bool";
    case Ty::Int:
        return "int";
    case Ty::UInt:
        return "uint";
    case Ty::Float:
        return "float";
    case Ty::Vec:
        return "vec3";
    case Ty::Void:
        return "void";
    default:
        fail(location, "value has no GPU type");
    }
}

std::string floatLiteral(double v)
{
    if (isinf(v))
        return v > 0 ? "uintBitsToFloat(0x7F800000u)" : "uintBitsToFloat(0xFF800000u)";

    char buf[64];
    snprintf(buf, sizeof(buf), "%.9g", v);
    std::string s = buf;
    if (s.find_first_of(".en") == std::string::npos)
        s += ".0";
    return "float(" + s + ")";
}

Val convert(const Location& location, const Val& v, Ty to)
{
    if (v.type.ty == to)
        return v;

    if (!isNumeric(v.type.ty) || !isNumeric(to))
        fail(location, "can't convert between these GPU types");

    const char* ctor = to == Ty::Int ? "int" : to == Ty::UInt ? "uint" : "float";
    return Val{std::string(ctor) + "(" + v.code + ")", type(to)};
}

// common type of two numbers for arithmetic and comparisons
Ty commonNumeric(const Val& a, const Val& b)
{
    if (a.type.ty == Ty::Float || b.type.ty == Ty::Float)
        return Ty::Float;
    if (a.type.ty == Ty::UInt && b.type.ty == Ty::UInt)
        return Ty::UInt;
    if (a.type.ty == Ty::UInt && b.nonNegLiteral)
        return Ty::UInt;
    if (b.type.ty == Ty::UInt && a.nonNegLiteral)
        return Ty::UInt;
    return Ty::Int;
}

// type of a variable that is assigned values of both types
Type join(const Location& location, Type a, Type b)
{
    if (a.ty == Ty::Unknown)
        return b;
    if (b.ty == Ty::Unknown || a == b)
        return a;
    if (isNumeric(a.ty) && isNumeric(b.ty))
        return type(a.ty == Ty::Float || b.ty == Ty::Float ? Ty::Float : Ty::Int);

    fail(location, "variable is assigned values of incompatible types");
}

bool isGpuFunction(AstExprFunction* func)
{
    return func->hasAttribute(AstAttr::Type::Gpu);
}

// chunk-wide facts: which locals are constants and which functions can be called from kernels
struct ChunkInfo : AstVisitor
{
    std::set<AstLocal*> assigned;
    std::vector<std::pair<AstLocal*, AstExpr*>> topLevelLocals;
    std::unordered_map<AstLocal*, AstExprFunction*> localFunctions;
    std::unordered_map<std::string, AstExprFunction*> globalFunctions;
    std::vector<AstExprFunction*> gpuFunctions;

    std::unordered_map<AstLocal*, double> constants;

    void markAssigned(AstExpr* expr)
    {
        if (AstExprLocal* l = expr->as<AstExprLocal>())
            assigned.insert(l->local);
    }

    bool visit(AstStatAssign* node) override
    {
        for (AstExpr* var : node->vars)
            markAssigned(var);
        return true;
    }

    bool visit(AstStatCompoundAssign* node) override
    {
        markAssigned(node->var);
        return true;
    }

    bool visit(AstStatLocalFunction* node) override
    {
        if (isGpuFunction(node->func))
            localFunctions[node->name] = node->func;
        return true;
    }

    bool visit(AstStatFunction* node) override
    {
        if (AstExprGlobal* g = node->name->as<AstExprGlobal>(); g && isGpuFunction(node->func))
            globalFunctions[g->name.value] = node->func;
        return true;
    }

    bool visit(AstExprFunction* node) override
    {
        if (isGpuFunction(node))
            gpuFunctions.push_back(node);
        return true;
    }

    static bool constantValue(AstExpr* expr, double& value)
    {
        if (AstExprConstantNumber* n = expr->as<AstExprConstantNumber>())
        {
            value = n->value;
            return true;
        }
        if (AstExprUnary* u = expr->as<AstExprUnary>(); u && u->op == AstExprUnary::Op::Minus)
        {
            if (constantValue(u->expr, value))
            {
                value = -value;
                return true;
            }
        }
        if (AstExprGroup* g = expr->as<AstExprGroup>())
            return constantValue(g->expr, value);
        return false;
    }

    void analyze(AstStatBlock* root)
    {
        for (AstStat* stat : root->body)
        {
            if (AstStatLocal* local = stat->as<AstStatLocal>())
            {
                for (size_t i = 0; i < local->vars.size && i < local->values.size; ++i)
                    topLevelLocals.emplace_back(local->vars.data[i], local->values.data[i]);
            }
        }

        root->visit(this);

        for (auto& [local, expr] : topLevelLocals)
        {
            double value;
            if (!assigned.count(local) && constantValue(expr, value))
                constants[local] = value;
        }

        for (auto it = localFunctions.begin(); it != localFunctions.end();)
        {
            if (assigned.count(it->first))
                it = localFunctions.erase(it);
            else
                ++it;
        }
    }
};

struct Translator;

// translates one function body for a given set of argument types
struct FunctionEmitter
{
    Translator& tr;
    AstExprFunction* func;
    bool isKernel;

    std::unordered_map<AstLocal*, Type> localTypes;
    std::unordered_map<AstLocal*, std::string> localNames;
    Type returnType = type(Ty::Unknown);
    bool changed = false;
    int tempCounter = 0;

    std::string out;
    int indent = 1;

    // loop nesting; true for 'repeat' loops, where 'continue' would skip the condition
    std::vector<bool> loops;

    // locals declared without a value whose type isn't known yet
    std::vector<AstLocal*> unresolvedLocals;

    FunctionEmitter(Translator& tr, AstExprFunction* func, bool isKernel)
        : tr(tr)
        , func(func)
        , isKernel(isKernel)
    {
    }

    void line(const std::string& s)
    {
        out.append(indent * 4, ' ');
        out += s;
        out += '\n';
    }

    std::string nameOf(AstLocal* local);

    void declareType(const Location& location, AstLocal* local, Type t)
    {
        Type& current = localTypes[local];
        Type joined = join(location, current, t);
        if (joined != current)
        {
            current = joined;
            changed = true;
        }
    }

    Val expr(AstExpr* node);
    Val call(AstExprCall* node);
    Val builtin(AstExprCall* node, const std::string& lib, const std::string& name);
    Val binary(AstExprBinary* node);
    std::vector<Val> args(AstExprCall* node, size_t count);

    Val condition(AstExpr* node)
    {
        Val v = expr(node);
        if (v.type.ty != Ty::Bool)
            fail(node->location, "conditions must be boolean on the GPU (numbers are always truthy)");
        return v;
    }

    void block(AstStatBlock* node);
    void stat(AstStat* node);
    void assign(const Location& location, AstExpr* var, const Val& value);
};

struct HelperSpec
{
    std::string name;
    Type returnType;
    bool done = false;
};

struct Translator
{
    ChunkInfo& info;

    std::set<int> readBindings;
    std::set<int> writeBindings;
    std::set<std::pair<std::string, int>> atomics;
    std::set<int> writtenBindings;

    std::map<std::pair<AstExprFunction*, std::vector<Type>>, HelperSpec> helpers;
    std::string helperCode;
    int helperCounter = 0;

    explicit Translator(ChunkInfo& info)
        : info(info)
    {
    }

    AstExprFunction* calledFunction(AstExpr* callee)
    {
        if (AstExprLocal* l = callee->as<AstExprLocal>())
        {
            auto it = info.localFunctions.find(l->local);
            return it == info.localFunctions.end() ? nullptr : it->second;
        }
        if (AstExprGlobal* g = callee->as<AstExprGlobal>())
        {
            auto it = info.globalFunctions.find(g->name.value);
            return it == info.globalFunctions.end() ? nullptr : it->second;
        }
        return nullptr;
    }

    const HelperSpec& helper(const Location& location, AstExprFunction* func, const std::vector<Type>& argTypes)
    {
        auto key = std::make_pair(func, argTypes);
        auto it = helpers.find(key);
        if (it != helpers.end())
        {
            if (!it->second.done)
                fail(location, "recursive calls are not supported on the GPU");
            return it->second;
        }

        HelperSpec& spec = helpers[key];
        spec.name = "h" + std::to_string(helperCounter++) + "_" + (func->debugname.value ? func->debugname.value : "anon");

        if (func->vararg)
            fail(func->location, "variadic functions are not supported on the GPU");
        if (func->self)
            fail(func->location, "methods are not supported on the GPU");

        FunctionEmitter em(*this, func, false);

        std::vector<std::string> params;
        std::vector<std::string> copies;

        for (int iter = 0; iter < 32; ++iter)
        {
            em.changed = false;
            em.out.clear();
            em.unresolvedLocals.clear();
            params.clear();
            copies.clear();

            for (size_t i = 0; i < func->args.size; ++i)
            {
                AstLocal* arg = func->args.data[i];
                Type t = i < argTypes.size() ? argTypes[i] : type(Ty::Unknown);
                if (t.ty == Ty::Unknown)
                    fail(func->location, "helper function called with too few arguments");

                em.declareType(func->location, arg, t);

                if (t.ty == Ty::Buffer)
                    continue;

                std::string pname = "p" + std::to_string(i);
                params.push_back(glslType(func->location, t) + " " + pname);
                Val pv{pname, t};
                copies.push_back(
                    glslType(func->location, em.localTypes[arg]) + " " + em.nameOf(arg) + " = " +
                    convert(func->location, pv, em.localTypes[arg].ty).code + ";"
                );
            }

            em.block(func->body);

            if (!em.changed)
                break;
        }

        if (!em.unresolvedLocals.empty())
            fail(func->location, "local '" + std::string(em.unresolvedLocals[0]->name.value) + "' is never assigned a value");

        Type ret = em.returnType.ty == Ty::Unknown ? type(Ty::Void) : em.returnType;

        std::string code = glslType(func->location, ret) + " " + spec.name + "(";
        for (size_t i = 0; i < params.size(); ++i)
            code += (i ? ", " : "") + params[i];
        code += ")\n{\n";
        for (const std::string& c : copies)
            code += "    " + c + "\n";
        code += em.out;
        if (ret.ty != Ty::Void)
            code += "    return " + glslType(func->location, ret) + "(0);\n";
        code += "}\n\n";

        helperCode += code;

        // the map may have rehashed, look the entry up again
        HelperSpec& done = helpers[key];
        done.returnType = ret;
        done.done = true;
        return done;
    }
};

std::string FunctionEmitter::nameOf(AstLocal* local)
{
    auto it = localNames.find(local);
    if (it != localNames.end())
        return it->second;

    std::string name = "l_" + std::string(local->name.value) + "_" + std::to_string(localNames.size());
    localNames[local] = name;
    return name;
}

std::vector<Val> FunctionEmitter::args(AstExprCall* node, size_t count)
{
    if (node->args.size != count)
        fail(node->location, "expected " + std::to_string(count) + " arguments");

    std::vector<Val> result;
    for (AstExpr* a : node->args)
        result.push_back(expr(a));
    return result;
}

Val FunctionEmitter::expr(AstExpr* node)
{
    if (AstExprGroup* g = node->as<AstExprGroup>())
    {
        Val v = expr(g->expr);
        v.code = "(" + v.code + ")";
        return v;
    }

    if (AstExprTypeAssertion* ta = node->as<AstExprTypeAssertion>())
        return expr(ta->expr);

    if (AstExprConstantNumber* n = node->as<AstExprConstantNumber>())
    {
        double v = n->value;
        if (isnan(v))
            fail(node->location, "NaN constants are not supported");

        if (v == floor(v) && v >= -2147483648.0 && v <= 2147483647.0)
            return Val{std::to_string((long long)v), type(Ty::Int), v >= 0};
        if (v == floor(v) && v > 0 && v <= 4294967295.0)
            return Val{std::to_string((long long)v) + "u", type(Ty::UInt)};

        return Val{floatLiteral(v), type(Ty::Float)};
    }

    if (AstExprConstantBool* b = node->as<AstExprConstantBool>())
        return Val{b->value ? "true" : "false", type(Ty::Bool)};

    if (AstExprLocal* l = node->as<AstExprLocal>())
    {
        auto c = tr.info.constants.find(l->local);
        if (l->upvalue || c != tr.info.constants.end())
        {
            if (c == tr.info.constants.end())
                fail(node->location, "upvalue '" + std::string(l->local->name.value) + "' is not a numeric constant");

            AstExprConstantNumber constant(node->location, c->second);
            return expr(&constant);
        }

        auto it = localTypes.find(l->local);
        if (it == localTypes.end())
            fail(node->location, "unknown local '" + std::string(l->local->name.value) + "'");

        if (it->second.ty == Ty::Unknown)
            fail(node->location, "local '" + std::string(l->local->name.value) + "' is used before it has a value");

        return Val{it->second.ty == Ty::Buffer ? "" : nameOf(l->local), it->second};
    }

    if (AstExprIndexName* in = node->as<AstExprIndexName>())
    {
        if (AstExprGlobal* g = in->expr->as<AstExprGlobal>(); g && strcmp(g->name.value, "math") == 0)
        {
            if (strcmp(in->index.value, "huge") == 0)
                return Val{floatLiteral(HUGE_VAL), type(Ty::Float)};
            if (strcmp(in->index.value, "pi") == 0)
                return Val{floatLiteral(3.14159265358979323846), type(Ty::Float)};
        }

        if (in->op != '.')
            fail(node->location, "method syntax is not supported on the GPU");

        Val base = expr(in->expr);
        if (base.type.ty == Ty::Vec)
        {
            const char* f = in->index.value;
            if (strcmp(f, "x") == 0 || strcmp(f, "X") == 0)
                return Val{base.code + ".x", type(Ty::Float)};
            if (strcmp(f, "y") == 0 || strcmp(f, "Y") == 0)
                return Val{base.code + ".y", type(Ty::Float)};
            if (strcmp(f, "z") == 0 || strcmp(f, "Z") == 0)
                return Val{base.code + ".z", type(Ty::Float)};
        }

        fail(node->location, "field access is only supported on vectors");
    }

    if (AstExprCall* c = node->as<AstExprCall>())
        return call(c);

    if (AstExprUnary* u = node->as<AstExprUnary>())
    {
        Val v = expr(u->expr);
        switch (u->op)
        {
        case AstExprUnary::Op::Not:
            if (v.type.ty != Ty::Bool)
                fail(node->location, "'not' is only supported on booleans on the GPU");
            return Val{"(!" + v.code + ")", v.type};
        case AstExprUnary::Op::Minus:
            if (v.type.ty == Ty::UInt)
                v = convert(node->location, v, Ty::Int);
            if (!isNumeric(v.type.ty) && v.type.ty != Ty::Vec)
                fail(node->location, "unary minus needs a number or a vector");
            return Val{"(-" + v.code + ")", v.type};
        case AstExprUnary::Op::Len:
            fail(node->location, "'#' is not supported on the GPU");
        }
    }

    if (AstExprBinary* b = node->as<AstExprBinary>())
        return binary(b);

    if (AstExprIfElse* ie = node->as<AstExprIfElse>())
    {
        if (ie->conditionLocal)
            fail(node->location, "conditional bindings are not supported on the GPU");

        Val c = condition(ie->condition);
        Val t = expr(ie->trueExpr);
        Val f = expr(ie->falseExpr);

        Type rt = t.type;
        if (t.type != f.type)
        {
            if (!isNumeric(t.type.ty) || !isNumeric(f.type.ty))
                fail(node->location, "both branches of an if-expression must have the same type");
            rt = type(commonNumeric(t, f));
            t = convert(node->location, t, rt.ty);
            f = convert(node->location, f, rt.ty);
        }
        if (rt.ty == Ty::Buffer)
            fail(node->location, "if-expressions can't select buffers on the GPU");

        return Val{"(" + c.code + " ? " + t.code + " : " + f.code + ")", rt};
    }

    if (node->is<AstExprFunction>())
        fail(node->location, "nested functions are not supported on the GPU");
    if (node->is<AstExprGlobal>())
        fail(node->location, "global '" + std::string(node->as<AstExprGlobal>()->name.value) + "' is not supported on the GPU");
    if (node->is<AstExprConstantString>() || node->is<AstExprInterpString>())
        fail(node->location, "strings are not supported on the GPU");
    if (node->is<AstExprConstantNil>())
        fail(node->location, "nil is not supported on the GPU");
    if (node->is<AstExprTable>())
        fail(node->location, "tables are not supported on the GPU");
    if (node->is<AstExprIndexExpr>())
        fail(node->location, "indexing is not supported on the GPU");
    if (node->is<AstExprVarargs>())
        fail(node->location, "varargs are not supported on the GPU");

    fail(node->location, "expression is not supported on the GPU");
}

Val FunctionEmitter::binary(AstExprBinary* node)
{
    Val a = expr(node->left);
    Val b = expr(node->right);
    const Location& loc = node->location;

    switch (node->op)
    {
    case AstExprBinary::And:
    case AstExprBinary::Or:
        if (a.type.ty != Ty::Bool || b.type.ty != Ty::Bool)
            fail(loc, "'and'/'or' are only supported on booleans on the GPU; use an if-expression");
        return Val{"(" + a.code + (node->op == AstExprBinary::And ? " && " : " || ") + b.code + ")", type(Ty::Bool)};

    case AstExprBinary::Concat:
        fail(loc, "strings are not supported on the GPU");

    case AstExprBinary::CompareEq:
    case AstExprBinary::CompareNe:
    case AstExprBinary::CompareLt:
    case AstExprBinary::CompareLe:
    case AstExprBinary::CompareGt:
    case AstExprBinary::CompareGe:
    {
        static const char* ops[] = {"", "", "", "", "", "", "", "", "!=", "==", "<", "<=", ">", ">="};
        const char* op = ops[node->op];

        if (a.type.ty == Ty::Bool && b.type.ty == Ty::Bool && (node->op == AstExprBinary::CompareEq || node->op == AstExprBinary::CompareNe))
            return Val{"(" + a.code + " " + op + " " + b.code + ")", type(Ty::Bool)};

        if (a.type.ty == Ty::Vec && b.type.ty == Ty::Vec && (node->op == AstExprBinary::CompareEq || node->op == AstExprBinary::CompareNe))
            return Val{
                std::string(node->op == AstExprBinary::CompareEq ? "all" : "any") + "(" +
                    (node->op == AstExprBinary::CompareEq ? "equal" : "notEqual") + "(" + a.code + ", " + b.code + "))",
                type(Ty::Bool)
            };

        if (!isNumeric(a.type.ty) || !isNumeric(b.type.ty))
            fail(loc, "comparison is only supported between numbers on the GPU");

        Ty t = commonNumeric(a, b);
        a = convert(loc, a, t);
        b = convert(loc, b, t);
        return Val{"(" + a.code + " " + op + " " + b.code + ")", type(Ty::Bool)};
    }

    default:
        break;
    }

    // vector arithmetic follows Luau: vector op vector and vector op number (number op vector for multiplication and division)
    if (a.type.ty == Ty::Vec || b.type.ty == Ty::Vec)
    {
        if (a.type.ty != Ty::Vec)
            a = convert(loc, a, Ty::Float);
        if (b.type.ty != Ty::Vec)
            b = convert(loc, b, Ty::Float);

        switch (node->op)
        {
        case AstExprBinary::Add:
            return Val{"(" + a.code + " + " + b.code + ")", type(Ty::Vec)};
        case AstExprBinary::Sub:
            return Val{"(" + a.code + " - " + b.code + ")", type(Ty::Vec)};
        case AstExprBinary::Mul:
            return Val{"(" + a.code + " * " + b.code + ")", type(Ty::Vec)};
        case AstExprBinary::Div:
            return Val{"(" + a.code + " / " + b.code + ")", type(Ty::Vec)};
        case AstExprBinary::FloorDiv:
            return Val{"floor(" + a.code + " / " + b.code + ")", type(Ty::Vec)};
        default:
            fail(loc, "operator is not supported on vectors");
        }
    }

    if (!isNumeric(a.type.ty) || !isNumeric(b.type.ty))
        fail(loc, "arithmetic is only supported on numbers and vectors on the GPU");

    Ty t = node->op == AstExprBinary::Div || node->op == AstExprBinary::Pow ? Ty::Float : commonNumeric(a, b);
    a = convert(loc, a, t);
    b = convert(loc, b, t);

    switch (node->op)
    {
    case AstExprBinary::Add:
        return Val{"(" + a.code + " + " + b.code + ")", type(t)};
    case AstExprBinary::Sub:
        return Val{"(" + a.code + " - " + b.code + ")", type(t)};
    case AstExprBinary::Mul:
        return Val{"(" + a.code + " * " + b.code + ")", type(t)};
    case AstExprBinary::Div:
        return Val{"(" + a.code + " / " + b.code + ")", type(t)};
    case AstExprBinary::Pow:
        return Val{"l_pow(" + a.code + ", " + b.code + ")", type(t)};
    case AstExprBinary::FloorDiv:
        if (t == Ty::Float)
            return Val{"floor(" + a.code + " / " + b.code + ")", type(t)};
        return Val{std::string(t == Ty::Int ? "l_idiv(" : "l_udiv(") + a.code + ", " + b.code + ")", type(t)};
    case AstExprBinary::Mod:
        if (t == Ty::Float)
            return Val{"l_fmod(" + a.code + ", " + b.code + ")", type(t)};
        return Val{std::string(t == Ty::Int ? "l_imod(" : "l_umod(") + a.code + ", " + b.code + ")", type(t)};
    default:
        fail(loc, "operator is not supported on the GPU");
    }
}

Val FunctionEmitter::call(AstExprCall* node)
{
    if (node->self)
        fail(node->location, "method calls are not supported on the GPU");

    if (AstExprIndexName* in = node->func->as<AstExprIndexName>())
    {
        if (AstExprGlobal* g = in->expr->as<AstExprGlobal>())
            return builtin(node, g->name.value, in->index.value);
    }

    if (AstExprFunction* target = tr.calledFunction(node->func))
    {
        std::vector<Type> types;
        std::vector<Val> values;
        for (AstExpr* a : node->args)
        {
            Val v = expr(a);
            if (v.type.ty == Ty::Void)
                fail(a->location, "argument has no value");
            types.push_back(v.type);
            values.push_back(v);
        }
        if (types.size() > target->args.size)
            fail(node->location, "too many arguments");

        const HelperSpec& spec = tr.helper(node->location, target, types);

        std::string code = spec.name + "(";
        bool first = true;
        for (const Val& v : values)
        {
            if (v.type.ty == Ty::Buffer)
                continue;
            code += (first ? "" : ", ") + v.code;
            first = false;
        }
        code += ")";
        return Val{code, spec.returnType};
    }

    if (AstExprLocal* l = node->func->as<AstExprLocal>())
        fail(node->location, "'" + std::string(l->local->name.value) + "' is not a local @gpu function");
    if (AstExprGlobal* g = node->func->as<AstExprGlobal>())
        fail(node->location, "'" + std::string(g->name.value) + "' is not a @gpu function");

    fail(node->location, "call is not supported on the GPU");
}

Val FunctionEmitter::builtin(AstExprCall* node, const std::string& lib, const std::string& name)
{
    const Location& loc = node->location;

    auto bufferArg = [&](const Val& v) -> int
    {
        if (v.type.ty != Ty::Buffer)
            fail(loc, "expected a buffer parameter");
        return v.type.binding;
    };
    auto offsetArg = [&](const Val& v) -> std::string
    {
        if (!isNumeric(v.type.ty))
            fail(loc, "buffer offset must be a number");
        return convert(loc, v, Ty::Int).code;
    };
    auto num = [&](const Val& v, Ty t) -> std::string
    {
        return convert(loc, v, t).code;
    };
    auto unaryFloat = [&](const char* fn) -> Val
    {
        auto a = args(node, 1);
        return Val{std::string(fn) + "(" + num(a[0], Ty::Float) + ")", type(Ty::Float)};
    };

    if (lib == "buffer")
    {
        if (name == "readf32" || name == "readu32" || name == "readi32")
        {
            auto a = args(node, 2);
            int b = bufferArg(a[0]);
            tr.readBindings.insert(b);
            std::string raw = "l_rd" + std::to_string(b) + "(" + offsetArg(a[1]) + ")";
            if (name == "readf32")
                return Val{"uintBitsToFloat(" + raw + ")", type(Ty::Float)};
            if (name == "readi32")
                return Val{"int(" + raw + ")", type(Ty::Int)};
            return Val{raw, type(Ty::UInt)};
        }
        if (name == "writef32" || name == "writeu32" || name == "writei32")
        {
            auto a = args(node, 3);
            int b = bufferArg(a[0]);
            tr.writeBindings.insert(b);
            tr.writtenBindings.insert(b);

            std::string value;
            if (name == "writef32")
                value = "floatBitsToUint(" + num(a[2], Ty::Float) + ")";
            else if (name == "writei32")
                value = "uint(" + num(a[2], Ty::Int) + ")";
            else
                value = num(a[2], a[2].type.ty == Ty::Float ? Ty::Float : Ty::UInt);
            if (name == "writeu32" && a[2].type.ty == Ty::Float)
                value = "uint(int(" + value + "))";

            return Val{"l_wr" + std::to_string(b) + "(" + offsetArg(a[1]) + ", " + value + ")", type(Ty::Void)};
        }
        if (name == "len")
        {
            auto a = args(node, 1);
            return Val{"int(pc.len[" + std::to_string(bufferArg(a[0])) + "])", type(Ty::Int)};
        }

        fail(loc, "buffer." + name + " is not supported on the GPU (supported: readf32/readu32/readi32, writef32/writeu32/writei32, len)");
    }

    if (lib == "gpu")
    {
        if (name == "atomicadd" || name == "atomicmin" || name == "atomicmax")
        {
            auto a = args(node, 3);
            int b = bufferArg(a[0]);
            tr.atomics.insert({name, b});
            tr.writtenBindings.insert(b);
            std::string value = a[2].type.ty == Ty::Float ? "uint(int(" + num(a[2], Ty::Float) + "))" : num(a[2], Ty::UInt);
            return Val{"l_" + name + std::to_string(b) + "(" + offsetArg(a[1]) + ", " + value + ")", type(Ty::UInt)};
        }

        fail(loc, "gpu." + name + " can't be called from a kernel");
    }

    if (lib == "math")
    {
        if (name == "floor" || name == "ceil")
        {
            auto a = args(node, 1);
            if (isInteger(a[0].type.ty))
                return a[0];
            return Val{"int(" + name + "(" + num(a[0], Ty::Float) + "))", type(Ty::Int)};
        }
        if (name == "round")
        {
            auto a = args(node, 1);
            if (isInteger(a[0].type.ty))
                return a[0];
            return Val{"l_round(" + num(a[0], Ty::Float) + ")", type(Ty::Int)};
        }
        if (name == "abs")
        {
            auto a = args(node, 1);
            if (a[0].type.ty == Ty::UInt)
                return a[0];
            return Val{"abs(" + num(a[0], a[0].type.ty) + ")", a[0].type};
        }
        if (name == "sign")
        {
            auto a = args(node, 1);
            Ty t = a[0].type.ty == Ty::Float ? Ty::Float : Ty::Int;
            return Val{"sign(" + num(a[0], t) + ")", type(t)};
        }
        if (name == "min" || name == "max")
        {
            if (node->args.size < 1)
                fail(loc, "math." + name + " needs arguments");
            Val acc = expr(node->args.data[0]);
            for (size_t i = 1; i < node->args.size; ++i)
            {
                Val v = expr(node->args.data[i]);
                Ty t = commonNumeric(acc, v);
                acc = Val{name + "(" + num(acc, t) + ", " + num(v, t) + ")", type(t)};
            }
            if (!isNumeric(acc.type.ty))
                fail(loc, "math." + name + " needs numbers");
            return acc;
        }
        if (name == "clamp")
        {
            auto a = args(node, 3);
            Ty t = commonNumeric(a[0], a[1]);
            t = commonNumeric(Val{"", type(t)}, a[2]);
            return Val{"clamp(" + num(a[0], t) + ", " + num(a[1], t) + ", " + num(a[2], t) + ")", type(t)};
        }
        if (name == "sqrt" || name == "sin" || name == "cos" || name == "tan" || name == "asin" || name == "acos" || name == "exp")
            return unaryFloat(name.c_str());
        if (name == "atan")
            return unaryFloat("atan");
        if (name == "atan2")
        {
            auto a = args(node, 2);
            return Val{"atan(" + num(a[0], Ty::Float) + ", " + num(a[1], Ty::Float) + ")", type(Ty::Float)};
        }
        if (name == "log")
        {
            if (node->args.size == 2)
            {
                auto a = args(node, 2);
                return Val{"(log(" + num(a[0], Ty::Float) + ") / log(" + num(a[1], Ty::Float) + "))", type(Ty::Float)};
            }
            return unaryFloat("log");
        }
        if (name == "pow")
        {
            auto a = args(node, 2);
            return Val{"l_pow(" + num(a[0], Ty::Float) + ", " + num(a[1], Ty::Float) + ")", type(Ty::Float)};
        }
        if (name == "fmod")
        {
            auto a = args(node, 2);
            std::string x = num(a[0], Ty::Float), y = num(a[1], Ty::Float);
            return Val{"(" + x + " - " + y + " * trunc(" + x + " / " + y + "))", type(Ty::Float)};
        }
        if (name == "lerp")
        {
            auto a = args(node, 3);
            return Val{"mix(" + num(a[0], Ty::Float) + ", " + num(a[1], Ty::Float) + ", " + num(a[2], Ty::Float) + ")", type(Ty::Float)};
        }

        fail(loc, "math." + name + " is not supported on the GPU");
    }

    if (lib == "bit32")
    {
        auto u = [&](const Val& v)
        {
            if (v.type.ty == Ty::Float)
                return "uint(int(" + v.code + "))";
            return num(v, Ty::UInt);
        };

        if (name == "band" || name == "bor" || name == "bxor")
        {
            const char* op = name == "band" ? " & " : name == "bor" ? " | " : " ^ ";
            if (node->args.size == 0)
                return Val{name == "band" ? "0xFFFFFFFFu" : "0u", type(Ty::UInt)};
            std::string code = "(";
            for (size_t i = 0; i < node->args.size; ++i)
                code += (i ? op : "") + u(expr(node->args.data[i]));
            return Val{code + ")", type(Ty::UInt)};
        }
        if (name == "bnot")
        {
            auto a = args(node, 1);
            return Val{"(~" + u(a[0]) + ")", type(Ty::UInt)};
        }
        if (name == "lshift" || name == "rshift" || name == "arshift")
        {
            auto a = args(node, 2);
            const char* fn = name == "lshift" ? "l_shl(" : name == "rshift" ? "l_shr(" : "l_sar(";
            return Val{std::string(fn) + u(a[0]) + ", " + u(a[1]) + ")", type(Ty::UInt)};
        }
        if (name == "btest")
        {
            auto a = args(node, 2);
            return Val{"((" + u(a[0]) + " & " + u(a[1]) + ") != 0u)", type(Ty::Bool)};
        }
        if (name == "extract")
        {
            auto a = args(node, 3);
            return Val{"bitfieldExtract(" + u(a[0]) + ", int(" + u(a[1]) + "), int(" + u(a[2]) + "))", type(Ty::UInt)};
        }

        fail(loc, "bit32." + name + " is not supported on the GPU");
    }

    if (lib == "vector")
    {
        auto vec = [&](const Val& v)
        {
            if (v.type.ty != Ty::Vec)
                fail(loc, "expected a vector");
            return v.code;
        };

        if (name == "create")
        {
            if (node->args.size != 3)
                fail(loc, "vector.create needs 3 components on the GPU");
            auto a = args(node, 3);
            return Val{"vec3(" + num(a[0], Ty::Float) + ", " + num(a[1], Ty::Float) + ", " + num(a[2], Ty::Float) + ")", type(Ty::Vec)};
        }
        if (name == "dot")
        {
            auto a = args(node, 2);
            return Val{"dot(" + vec(a[0]) + ", " + vec(a[1]) + ")", type(Ty::Float)};
        }
        if (name == "cross")
        {
            auto a = args(node, 2);
            return Val{"cross(" + vec(a[0]) + ", " + vec(a[1]) + ")", type(Ty::Vec)};
        }
        if (name == "magnitude")
        {
            auto a = args(node, 1);
            return Val{"length(" + vec(a[0]) + ")", type(Ty::Float)};
        }
        if (name == "normalize")
        {
            auto a = args(node, 1);
            return Val{"normalize(" + vec(a[0]) + ")", type(Ty::Vec)};
        }
        if (name == "floor" || name == "ceil" || name == "abs")
        {
            auto a = args(node, 1);
            return Val{name + "(" + vec(a[0]) + ")", type(Ty::Vec)};
        }
        if (name == "min" || name == "max")
        {
            auto a = args(node, 2);
            return Val{name + "(" + vec(a[0]) + ", " + vec(a[1]) + ")", type(Ty::Vec)};
        }

        fail(loc, "vector." + name + " is not supported on the GPU");
    }

    fail(loc, lib + "." + name + " is not supported on the GPU");
}

void FunctionEmitter::assign(const Location& location, AstExpr* var, const Val& value)
{
    AstExprLocal* l = var->as<AstExprLocal>();
    if (!l)
        fail(location, "only locals can be assigned on the GPU");
    if (l->upvalue)
        fail(location, "assigning upvalues is not supported on the GPU");
    if (value.type.ty == Ty::Buffer || value.type.ty == Ty::Void)
        fail(location, "value can't be stored in a local on the GPU");

    declareType(location, l->local, value.type);
    Type t = localTypes[l->local];
    line(nameOf(l->local) + " = " + convert(location, value, t.ty).code + ";");
}

void FunctionEmitter::block(AstStatBlock* node)
{
    for (AstStat* s : node->body)
        stat(s);
}

void FunctionEmitter::stat(AstStat* node)
{
    if (AstStatBlock* b = node->as<AstStatBlock>())
    {
        line("{");
        indent++;
        block(b);
        indent--;
        line("}");
        return;
    }

    if (AstStatLocal* local = node->as<AstStatLocal>())
    {
        if (local->values.size > local->vars.size)
            fail(node->location, "extra values in local declaration");

        std::vector<Val> values;
        for (AstExpr* v : local->values)
        {
            if (v->is<AstExprCall>() && local->values.size < local->vars.size)
                fail(node->location, "multiple return values are not supported on the GPU");
            values.push_back(expr(v));
        }

        // evaluate everything before declaring, as Luau does
        std::vector<std::string> temps;
        for (size_t i = 0; i < values.size(); ++i)
        {
            if (values[i].type.ty == Ty::Buffer)
            {
                declareType(node->location, local->vars.data[i], values[i].type);
                temps.push_back("");
                continue;
            }
            if (values[i].type.ty == Ty::Void)
                fail(node->location, "value has no result");

            if (values.size() == 1)
            {
                temps.push_back(values[i].code);
                continue;
            }
            std::string t = "t" + std::to_string(tempCounter++);
            line(glslType(node->location, values[i].type) + " " + t + " = " + values[i].code + ";");
            temps.push_back(t);
        }

        for (size_t i = 0; i < local->vars.size; ++i)
        {
            AstLocal* var = local->vars.data[i];
            if (i < values.size())
            {
                if (values[i].type.ty == Ty::Buffer)
                    continue;
                declareType(node->location, var, values[i].type);
                Type t = localTypes[var];
                Val v{temps[i], values[i].type, values[i].nonNegLiteral};
                line(glslType(node->location, t) + " " + nameOf(var) + " = " + convert(node->location, v, t.ty).code + ";");
            }
            else
            {
                // a local without a value starts as nil in Luau; the GPU zero-initializes it and requires an assignment before use
                Type& t = localTypes[var];
                if (t.ty == Ty::Unknown)
                {
                    // type is learned from a later assignment; the fixpoint iteration declares it on the next pass
                    line("/* " + std::string(var->name.value) + " */");
                    unresolvedLocals.push_back(var);
                    continue;
                }
                line(glslType(node->location, t) + " " + nameOf(var) + " = " + glslType(node->location, t) + "(0);");
            }
        }
        return;
    }

    if (AstStatAssign* a = node->as<AstStatAssign>())
    {
        if (a->vars.size != a->values.size)
            fail(node->location, "assignments must have as many values as targets on the GPU");

        if (a->vars.size == 1)
        {
            assign(node->location, a->vars.data[0], expr(a->values.data[0]));
            return;
        }

        std::vector<Val> temps;
        for (AstExpr* v : a->values)
        {
            Val value = expr(v);
            if (value.type.ty == Ty::Buffer || value.type.ty == Ty::Void)
                fail(node->location, "value can't be stored on the GPU");
            std::string t = "t" + std::to_string(tempCounter++);
            line(glslType(node->location, value.type) + " " + t + " = " + value.code + ";");
            temps.push_back(Val{t, value.type});
        }
        for (size_t i = 0; i < a->vars.size; ++i)
            assign(node->location, a->vars.data[i], temps[i]);
        return;
    }

    if (AstStatCompoundAssign* ca = node->as<AstStatCompoundAssign>())
    {
        AstExprBinary bin(node->location, ca->op, ca->var, ca->value);
        assign(node->location, ca->var, binary(&bin));
        return;
    }

    if (AstStatExpr* e = node->as<AstStatExpr>())
    {
        if (!e->expr->is<AstExprCall>())
            fail(node->location, "expression statement must be a call");
        Val v = expr(e->expr);
        line(v.code + ";");
        return;
    }

    if (AstStatIf* i = node->as<AstStatIf>())
    {
        if (i->conditionLocal)
            fail(node->location, "conditional bindings are not supported on the GPU");

        line("if (" + condition(i->condition).code + ")");
        line("{");
        indent++;
        block(i->thenbody);
        indent--;
        line("}");
        if (i->elsebody)
        {
            line("else");
            line("{");
            indent++;
            if (AstStatBlock* eb = i->elsebody->as<AstStatBlock>())
                block(eb);
            else
                stat(i->elsebody);
            indent--;
            line("}");
        }
        return;
    }

    if (AstStatWhile* w = node->as<AstStatWhile>())
    {
        line("while (" + condition(w->condition).code + ")");
        line("{");
        indent++;
        loops.push_back(false);
        block(w->body);
        loops.pop_back();
        indent--;
        line("}");
        return;
    }

    if (AstStatRepeat* r = node->as<AstStatRepeat>())
    {
        line("for (;;)");
        line("{");
        indent++;
        loops.push_back(true);
        block(r->body);
        loops.pop_back();
        line("if (" + condition(r->condition).code + ") break;");
        indent--;
        line("}");
        return;
    }

    if (AstStatFor* f = node->as<AstStatFor>())
    {
        Val from = expr(f->from);
        Val to = expr(f->to);
        Val step = f->step ? expr(f->step) : Val{"1", type(Ty::Int), true};

        if (!isNumeric(from.type.ty) || !isNumeric(to.type.ty) || !isNumeric(step.type.ty))
            fail(node->location, "for loop bounds must be numbers");

        Ty t = from.type.ty == Ty::Float || to.type.ty == Ty::Float || step.type.ty == Ty::Float ? Ty::Float : Ty::Int;

        double stepValue = 0;
        bool constantStep = !f->step || ChunkInfo::constantValue(f->step, stepValue);
        if (!f->step)
            stepValue = 1;
        if (constantStep && stepValue == 0)
            fail(node->location, "for loop step is zero");

        std::string ty = t == Ty::Int ? "int" : "float";
        std::string id = std::to_string(tempCounter++);
        std::string vi = "f" + id, ve = "fe" + id, vs = "fs" + id;

        line("{");
        indent++;
        line(ty + " " + ve + " = " + convert(f->to->location, to, t).code + ";");
        line(ty + " " + vs + " = " + convert(f->step ? f->step->location : node->location, step, t).code + ";");

        std::string cond = constantStep ? (stepValue > 0 ? vi + " <= " + ve : vi + " >= " + ve)
                                        : "(" + vs + " > " + ty + "(0)) ? (" + vi + " <= " + ve + ") : (" + vi + " >= " + ve + ")";

        line("for (" + ty + " " + vi + " = " + convert(f->from->location, from, t).code + "; " + cond + "; " + vi + " += " + vs + ")");
        line("{");
        indent++;

        declareType(node->location, f->var, type(t));
        Type vt = localTypes[f->var];
        line(glslType(node->location, vt) + " " + nameOf(f->var) + " = " + convert(node->location, Val{vi, type(t)}, vt.ty).code + ";");

        loops.push_back(false);
        block(f->body);
        loops.pop_back();

        indent--;
        line("}");
        indent--;
        line("}");
        return;
    }

    if (node->is<AstStatBreak>())
    {
        if (loops.empty())
            fail(node->location, "break outside of a loop");
        line("break;");
        return;
    }

    if (node->is<AstStatContinue>())
    {
        if (loops.empty())
            fail(node->location, "continue outside of a loop");
        if (loops.back())
            fail(node->location, "continue inside repeat-until is not supported on the GPU");
        line("continue;");
        return;
    }

    if (AstStatReturn* r = node->as<AstStatReturn>())
    {
        if (r->list.size == 0)
        {
            line(returnType.ty == Ty::Unknown || returnType.ty == Ty::Void ? "return;" : "return " + glslType(node->location, returnType) + "(0);");
            return;
        }
        if (isKernel)
            fail(node->location, "kernels can't return values");
        if (r->list.size > 1)
            fail(node->location, "multiple return values are not supported on the GPU");

        Val v = expr(r->list.data[0]);
        if (v.type.ty == Ty::Buffer || v.type.ty == Ty::Void)
            fail(node->location, "returned value is not supported on the GPU");

        Type joined = join(node->location, returnType, v.type);
        if (joined != returnType)
        {
            returnType = joined;
            changed = true;
        }
        line("return " + convert(node->location, v, returnType.ty).code + ";");
        return;
    }

    if (node->is<AstStatLocalFunction>() || node->is<AstStatFunction>())
        fail(node->location, "nested functions are not supported on the GPU");

    fail(node->location, "statement is not supported on the GPU");
}

const char* kPrelude = R"(
void l_fail() { atomicOr(st.err, 1u); }
int l_idiv(int a, int b) { if (b == 0) { l_fail(); return 0; } int q = a / b; if (q * b != a && ((a < 0) != (b < 0))) q -= 1; return q; }
int l_imod(int a, int b) { if (b == 0) { l_fail(); return 0; } return a - l_idiv(a, b) * b; }
uint l_udiv(uint a, uint b) { if (b == 0u) { l_fail(); return 0u; } return a / b; }
uint l_umod(uint a, uint b) { if (b == 0u) { l_fail(); return 0u; } return a % b; }
float l_fmod(float a, float b) { return a - floor(a / b) * b; }
float l_pow(float a, float b) { if (a < 0.0 && b == floor(b)) { float r = pow(-a, b); return mod(b, 2.0) == 1.0 ? -r : r; } return pow(a, b); }
uint l_shl(uint x, uint n) { return n >= 32u ? 0u : x << n; }
uint l_shr(uint x, uint n) { return n >= 32u ? 0u : x >> n; }
uint l_sar(uint x, uint n) { return uint(int(x) >> min(n, 31u)); }
int l_round(float x) { return int(sign(x) * floor(abs(x) + 0.5)); }
bool l_bad(int off, int b) { return off < 0 || (off & 3) != 0 || uint(off) + 4u > pc.len[b]; }
)";

std::string accessors(const Translator& tr)
{
    std::string code;
    for (int b : tr.readBindings)
    {
        std::string s = std::to_string(b);
        code += "uint l_rd" + s + "(int off) { if (l_bad(off, " + s + ")) { l_fail(); return 0u; } return b" + s + ".d[off >> 2]; }\n";
    }
    for (int b : tr.writeBindings)
    {
        std::string s = std::to_string(b);
        code += "void l_wr" + s + "(int off, uint v) { if (l_bad(off, " + s + ")) { l_fail(); return; } b" + s + ".d[off >> 2] = v; }\n";
    }
    for (const auto& [op, b] : tr.atomics)
    {
        std::string s = std::to_string(b);
        std::string fn = op == "atomicadd" ? "atomicAdd" : op == "atomicmin" ? "atomicMin" : "atomicMax";
        code += "uint l_" + op + s + "(int off, uint v) { if (l_bad(off, " + s + ")) { l_fail(); return 0u; } return " + fn + "(b" + s +
                ".d[off >> 2], v); }\n";
    }
    return code;
}

Kernel translateKernel(ChunkInfo& info, AstExprFunction* func)
{
    Kernel kernel;
    kernel.name = func->debugname.value ? func->debugname.value : "anonymous";
    kernel.line = func->location.begin.line + 1;

    try
    {
        if (func->vararg)
            fail(func->location, "kernels can't be variadic");
        if (func->self)
            fail(func->location, "kernels can't be methods");
        if (func->args.size == 0)
            fail(func->location, "kernels take the invocation index as their first parameter");
        if (func->args.size - 1 > size_t(kMaxKernelBuffers))
            fail(func->location, "kernels take at most " + std::to_string(kMaxKernelBuffers) + " buffers");

        Translator tr(info);
        FunctionEmitter em(tr, func, true);

        size_t buffers = func->args.size - 1;

        for (int iter = 0; iter < 32; ++iter)
        {
            em.changed = false;
            em.out.clear();
            em.unresolvedLocals.clear();

            em.declareType(func->location, func->args.data[0], type(Ty::Int));
            for (size_t i = 0; i < buffers; ++i)
                em.declareType(func->location, func->args.data[i + 1], Type{Ty::Buffer, int(i)});

            em.block(func->body);

            if (!em.changed)
                break;
        }

        if (!em.unresolvedLocals.empty())
            fail(func->location, "local '" + std::string(em.unresolvedLocals[0]->name.value) + "' is never assigned a value");

        AstLocal* index = func->args.data[0];
        Type indexType = em.localTypes[index];

        std::string glsl = "#version 450\nlayout(local_size_x = 64) in;\n\n";
        for (size_t i = 0; i < buffers; ++i)
            glsl += "layout(std430, set = 0, binding = " + std::to_string(i) + ") buffer B" + std::to_string(i) + " { uint d[]; } b" +
                    std::to_string(i) + ";\n";
        glsl += "layout(std430, set = 0, binding = " + std::to_string(buffers) + ") buffer BS { uint err; } st;\n";
        glsl += "layout(push_constant) uniform PC { uint n; uint len[" + std::to_string(kMaxKernelBuffers) + "]; } pc;\n";
        glsl += kPrelude;
        glsl += accessors(tr);
        glsl += "\n";
        glsl += tr.helperCode;
        glsl += "void l_kernel(int index)\n{\n";
        glsl += "    " + glslType(func->location, indexType) + " " + em.nameOf(index) + " = " +
                convert(func->location, Val{"index", type(Ty::Int)}, indexType.ty).code + ";\n";
        glsl += em.out;
        glsl += "}\n\n";
        glsl += "void main()\n{\n";
        glsl += "    uint gid = gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * gl_NumWorkGroups.x * 64u;\n";
        glsl += "    if (gid >= pc.n) return;\n";
        glsl += "    l_kernel(int(gid));\n";
        glsl += "}\n";

        kernel.glsl = glsl;
        for (size_t i = 0; i < buffers; ++i)
            kernel.bufferWritten.push_back(tr.writtenBindings.count(int(i)) != 0);
    }
    catch (TranslateError& e)
    {
        kernel.reason = "line " + std::to_string(e.location.begin.line + 1) + ": " + e.message;
        kernel.glsl.clear();
    }

    return kernel;
}

} // namespace

std::vector<Kernel> translate(const char* source, size_t size)
{
    Allocator allocator;
    AstNameTable names(allocator);
    ParseResult result = Parser::parse(source, size, names, allocator);

    std::vector<Kernel> kernels;
    if (!result.errors.empty())
        return kernels;

    ChunkInfo info;
    info.analyze(result.root);

    for (AstExprFunction* func : info.gpuFunctions)
        kernels.push_back(translateKernel(info, func));

    return kernels;
}

} // namespace Gpu
} // namespace Luau
