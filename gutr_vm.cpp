// файл: gutr_vm.cpp
// FLETTOHM + UIT:SLESA + ROSA — полная виртуальная машина GUTR
// Самодостаточная реализация. Без внешних зависимостей. C++17.

#include "gutr_vm.hpp"
#include <iostream>
#include <cmath>
#include <random>
#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <cstring>
#include <cstdlib>
#include <sstream>
#include <cfloat>
#include <unordered_map>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace uzaleat {

// ============================================================================
// БАЗОВАЯ ЛИНЕЙНАЯ АЛГЕБРА
// ============================================================================

static inline float vec_dot(const float* a, const float* b, size_t n) {
    float sum = 0.0f;
    #pragma omp simd reduction(+:sum)
    for (size_t i = 0; i < n; ++i) sum += a[i] * b[i];
    return sum;
}

static inline float vec_norm(const float* v, size_t n) {
    return std::sqrt(vec_dot(v, v, n));
}

static inline void vec_scale(float* v, size_t n, float s) {
    #pragma omp simd
    for (size_t i = 0; i < n; ++i) v[i] *= s;
}

static inline void vec_copy(float* dst, const float* src, size_t n) {
    #pragma omp simd
    for (size_t i = 0; i < n; ++i) dst[i] = src[i];
}

// QR через Хаусхолдера: A [m,n] -> Q [m,n], R [n,n]
static void qr_householder(const float* A, size_t m, size_t n, float* Q, float* R) {
    std::vector<float> work(m * n);
    vec_copy(work.data(), A, m * n);

    std::fill(Q, Q + m * n, 0.0f);
    for (size_t i = 0; i < n; ++i) Q[i * n + i] = 1.0f;

    std::vector<float> u(m);
    for (size_t k = 0; k < n && k < m - 1; ++k) {
        float alpha = 0.0f;
        for (size_t i = k; i < m; ++i) alpha += work[i * n + k] * work[i * n + k];
        alpha = std::sqrt(alpha);
        if (alpha < 1e-10f) continue;

        float sigma = (work[k * n + k] > 0) ? -alpha : alpha;
        u[k] = work[k * n + k] - sigma;
        float norm_sq = u[k] * u[k];
        for (size_t i = k + 1; i < m; ++i) {
            u[i] = work[i * n + k];
            norm_sq += u[i] * u[i];
        }
        float inv = 1.0f / std::sqrt(norm_sq);
        for (size_t i = k; i < m; ++i) u[i] *= inv;

        for (size_t j = k; j < n; ++j) {
            float dot = 0.0f;
            for (size_t i = k; i < m; ++i) dot += u[i] * work[i * n + j];
            for (size_t i = k; i < m; ++i) work[i * n + j] -= 2.0f * u[i] * dot;
        }
        for (size_t j = 0; j < n; ++j) {
            float dot = 0.0f;
            for (size_t i = k; i < m; ++i) dot += u[i] * Q[i * n + j];
            for (size_t i = k; i < m; ++i) Q[i * n + j] -= 2.0f * u[i] * dot;
        }
    }
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            R[i * n + j] = (j >= i) ? work[i * n + j] : 0.0f;
}

// SVD через бидиагонализацию Голуба-Ришна
static void svd_bidiag(const float* A, size_t m, size_t n,
                       float* U, float* S, float* Vt, size_t& k) {
    k = std::min(m, n);
    if (k == 0) return;

    std::vector<float> work(m * n);
    vec_copy(work.data(), A, m * n);
    std::vector<float> Ub(m * k, 0.0f), Vb(n * k, 0.0f);
    for (size_t i = 0; i < k; ++i) Ub[i * k + i] = 1.0f;
    for (size_t i = 0; i < k; ++i) Vb[i * k + i] = 1.0f;

    std::vector<float> d(k), e(k - 1, 0.0f);
    std::vector<float> u_vec(m), v_vec(n);

    for (size_t i = 0; i < k; ++i) {
        // Столбцовое преобразование Хаусхолдера
        float alpha = 0.0f;
        for (size_t j = i; j < m; ++j) alpha += work[j * n + i] * work[j * n + i];
        alpha = std::sqrt(alpha);
        if (alpha > 1e-10f && i < m - 1) {
            float sigma = (work[i * n + i] > 0) ? -alpha : alpha;
            u_vec[i] = work[i * n + i] - sigma;
            float nsq = u_vec[i] * u_vec[i];
            for (size_t j = i + 1; j < m; ++j) { u_vec[j] = work[j * n + i]; nsq += u_vec[j] * u_vec[j]; }
            float inv = 1.0f / std::sqrt(nsq);
            for (size_t j = i; j < m; ++j) u_vec[j] *= inv;
            for (size_t c = i; c < n; ++c) {
                float dot = 0.0f;
                for (size_t r = i; r < m; ++r) dot += u_vec[r] * work[r * n + c];
                for (size_t r = i; r < m; ++r) work[r * n + c] -= 2.0f * u_vec[r] * dot;
            }
            for (size_t c = 0; c < k; ++c) {
                float dot = 0.0f;
                for (size_t r = i; r < m; ++r) dot += u_vec[r] * Ub[r * k + c];
                for (size_t r = i; r < m; ++r) Ub[r * k + c] -= 2.0f * u_vec[r] * dot;
            }
        }
        d[i] = work[i * n + i];

        // Строчное преобразование Хаусхолдера
        if (i < n - 1) {
            float beta = 0.0f;
            for (size_t j = i + 1; j < n; ++j) beta += work[i * n + j] * work[i * n + j];
            beta = std::sqrt(beta);
            if (beta > 1e-10f) {
                float sigma = (work[i * n + i + 1] > 0) ? -beta : beta;
                v_vec[i + 1] = work[i * n + i + 1] - sigma;
                float nsq = v_vec[i + 1] * v_vec[i + 1];
                for (size_t j = i + 2; j < n; ++j) { v_vec[j] = work[i * n + j]; nsq += v_vec[j] * v_vec[j]; }
                v_vec[i] = 0.0f;
                float inv = 1.0f / std::sqrt(nsq);
                for (size_t j = i + 1; j < n; ++j) v_vec[j] *= inv;
                for (size_t r = i; r < m; ++r) {
                    float dot = 0.0f;
                    for (size_t c = i + 1; c < n; ++c) dot += v_vec[c] * work[r * n + c];
                    for (size_t c = i + 1; c < n; ++c) work[r * n + c] -= 2.0f * v_vec[c] * dot;
                }
                for (size_t r = 0; r < n; ++r) {
                    float dot = 0.0f;
                    for (size_t c = i + 1; c < n; ++c) dot += v_vec[c] * Vb[r * k + c];
                    for (size_t c = i + 1; c < n; ++c) Vb[r * k + c] -= 2.0f * v_vec[c] * dot;
                }
            }
            if (i < k - 1) e[i] = work[i * n + i + 1];
        }
    }

    // Сингулярные значения = норма столбцов
    std::vector<std::pair<float, size_t>> sorted(k);
    for (size_t i = 0; i < k; ++i) sorted[i] = {std::abs(d[i]), i};
    std::sort(sorted.begin(), sorted.end(), [](auto& a, auto& b) { return a.first > b.first; });

    for (size_t i = 0; i < k; ++i) {
        S[i] = sorted[i].first;
        size_t idx = sorted[i].second;
        for (size_t j = 0; j < m; ++j) U[j * k + i] = Ub[j * k + idx];
        for (size_t j = 0; j < n; ++j) Vt[i * n + j] = Vb[j * k + idx];
    }
}

// ============================================================================
// ROSA: RAPID ONLINE SUFFIX AUTOMATON
// ============================================================================
struct ROSAInstance {
    int s, g, z;
    std::vector<std::unordered_map<int, int>> next;
    std::vector<int> link, len, pos;
    std::vector<int> tokens;

    ROSAInstance() {}
    void init(int max_len) {
        s = 2 * max_len + 1;
        next.resize(s);
        link.assign(s, -1);
        len.assign(s, 0);
        pos.assign(s, -1);
        g = 0;
        z = 1;
        next[0].clear();
        tokens.clear();
    }
    int update_and_query(int token) {
        if (z + 2 >= s) return -1;
        tokens.push_back(token);
        int cur = z++;
        next[cur].clear();
        len[cur] = len[g] + 1;
        int p = g;
        while (p != -1 && !next[p].count(token)) { next[p][token] = cur; p = link[p]; }
        if (p == -1) { link[cur] = 0; }
        else {
            int q = next[p][token];
            if (len[p] + 1 == len[q]) { link[cur] = q; }
            else {
                int clone = z++;
                next[clone] = next[q];
                len[clone] = len[p] + 1;
                link[clone] = link[q];
                pos[clone] = pos[q];
                while (p != -1 && next[p][token] == q) { next[p][token] = clone; p = link[p]; }
                link[q] = link[cur] = clone;
            }
        }
        g = cur;
        int v = g, ans = -1;
        while (v != -1) {
            if (len[v] > 0 && pos[v] > 0 && pos[v] + 1 < (int)tokens.size()) { ans = tokens[pos[v] + 1]; break; }
            v = link[v];
        }
        v = g;
        int idx = (int)tokens.size() - 1;
        while (v != -1 && pos[v] < idx) { pos[v] = idx; v = link[v]; }
        return ans;
    }
};

static std::unordered_map<int64_t, std::shared_ptr<ROSAInstance>> rosa_instances;
static int64_t next_rosa_id = 1;

// ============================================================================
// Tensor implementation
// ============================================================================
Tensor::Tensor() : requires_grad(true) {}
Tensor::Tensor(const std::vector<size_t>& sh, bool req_grad) : shape(sh), requires_grad(req_grad) {
    size_t sz = 1;
    for (auto d : shape) sz *= d;
    data.resize(sz, 0.0f);
    if (requires_grad) grad.resize(sz, 0.0f);
}
Tensor::Tensor(const std::vector<size_t>& sh, float val, bool req_grad) : Tensor(sh, req_grad) { fill(val); }
size_t Tensor::size() const { size_t s = 1; for (auto d : shape) s *= d; return s; }
void Tensor::zero_grad() { if (requires_grad) std::fill(grad.begin(), grad.end(), 0.0f); }
void Tensor::fill(float value) { std::fill(data.begin(), data.end(), value); }
void Tensor::random_normal(float mean, float stddev) {
    thread_local static std::mt19937 gen(std::random_device{}());
    std::normal_distribution<float> dist(mean, stddev);
    for (auto& v : data) v = dist(gen);
}
void Tensor::random_uniform(float min, float max) {
    thread_local static std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(min, max);
    for (auto& v : data) v = dist(gen);
}
void Tensor::from_array(const std::vector<float>& arr) {
    if (arr.size() != data.size()) throw std::runtime_error("from_array size mismatch");
    data = arr;
}

// ============================================================================
// GUTRValue helpers
// ============================================================================
GUTRValue::GUTRValue() : type(ValueType::NIL), bool_val(false), int_val(0), float_val(0.0) {}
GUTRValue GUTRValue::nil() { return GUTRValue(); }
GUTRValue GUTRValue::boolean(bool v) { GUTRValue r; r.type = ValueType::BOOL; r.bool_val = v; return r; }
GUTRValue GUTRValue::integer(int64_t v) { GUTRValue r; r.type = ValueType::INT; r.int_val = v; return r; }
GUTRValue GUTRValue::real(double v) { GUTRValue r; r.type = ValueType::FLOAT; r.float_val = v; return r; }
GUTRValue GUTRValue::string(const std::string& v) { GUTRValue r; r.type = ValueType::STRING; r.string_val = v; return r; }
GUTRValue GUTRValue::tensor(std::shared_ptr<Tensor> t) { GUTRValue r; r.type = ValueType::TENSOR; r.tensor_val = t; return r; }
GUTRValue GUTRValue::tuple(const std::vector<GUTRValue>& v) { GUTRValue r; r.type = ValueType::TUPLE; r.tuple_val = v; return r; }
GUTRValue GUTRValue::function(std::shared_ptr<GUTRFunction> f) { GUTRValue r; r.type = ValueType::FUNCTION; r.func_val = f; return r; }
GUTRValue GUTRValue::sect(const std::unordered_map<std::string, GUTRValue>& fields) {
    GUTRValue r; r.type = ValueType::SECT; r.sect_fields = fields; return r;
}

// ============================================================================
// GUTRUserFunction::call
// ============================================================================
GUTRValue GUTRUserFunction::call(const std::vector<GUTRValue>& args) {
    auto new_ctx = std::make_shared<GUTRContext>(closure);
    for (size_t i = 0; i < param_names.size(); ++i) {
        new_ctx->set(param_names[i], i < args.size() ? args[i] : GUTRValue::nil());
    }
    body->eval(new_ctx);
    return new_ctx->return_value;
}

// ============================================================================
// GUTRContext
// ============================================================================
GUTRContext::GUTRContext(std::shared_ptr<GUTRContext> p) : parent(p), break_flag(false), continue_flag(false) {}
GUTRValue GUTRContext::get(const std::string& name) {
    auto it = variables.find(name);
    if (it != variables.end()) return it->second;
    if (parent) return parent->get(name);
    throw std::runtime_error("Undefined variable: " + name);
}
void GUTRContext::set(const std::string& name, const GUTRValue& value) { variables[name] = value; }
void GUTRContext::define_builtin(const std::string& name, GUTRBuiltinFunction::FuncType func) {
    set(name, GUTRValue::function(std::make_shared<GUTRBuiltinFunction>(func)));
}

// ============================================================================
// AST eval implementations
// ============================================================================
GUTRLiteralExpr::GUTRLiteralExpr(const GUTRValue& v) : value(v) {}
GUTRValue GUTRLiteralExpr::eval(std::shared_ptr<GUTRContext> ctx) { (void)ctx; return value; }
GUTRVarExpr::GUTRVarExpr(const std::string& n) : name(n) {}
GUTRValue GUTRVarExpr::eval(std::shared_ptr<GUTRContext> ctx) { return ctx->get(name); }

GUTRValue GUTRBinaryOpExpr::eval(std::shared_ptr<GUTRContext> ctx) {
    GUTRValue l = left->eval(ctx), r = right->eval(ctx);
    if (l.type == ValueType::INT && r.type == ValueType::INT) {
        int64_t a = l.int_val, b = r.int_val;
        switch (op) {
            case ADD: return GUTRValue::integer(a + b);
            case SUB: return GUTRValue::integer(a - b);
            case MUL: return GUTRValue::integer(a * b);
            case DIV: if (b == 0) throw std::runtime_error("div by zero"); return GUTRValue::integer(a / b);
            case LT: return GUTRValue::boolean(a < b);
            case GT: return GUTRValue::boolean(a > b);
            case EQ: return GUTRValue::boolean(a == b);
            case NE: return GUTRValue::boolean(a != b);
            case LE: return GUTRValue::boolean(a <= b);
            case GE: return GUTRValue::boolean(a >= b);
            default: throw std::runtime_error("invalid op");
        }
    } else if ((l.type == ValueType::INT || l.type == ValueType::FLOAT) &&
               (r.type == ValueType::INT || r.type == ValueType::FLOAT)) {
        double a = (l.type == ValueType::INT) ? (double)l.int_val : l.float_val;
        double b = (r.type == ValueType::INT) ? (double)r.int_val : r.float_val;
        switch (op) {
            case ADD: return GUTRValue::real(a + b);
            case SUB: return GUTRValue::real(a - b);
            case MUL: return GUTRValue::real(a * b);
            case DIV: if (b == 0.0) throw std::runtime_error("div by zero"); return GUTRValue::real(a / b);
            case LT: return GUTRValue::boolean(a < b);
            case GT: return GUTRValue::boolean(a > b);
            case EQ: return GUTRValue::boolean(a == b);
            case NE: return GUTRValue::boolean(a != b);
            case LE: return GUTRValue::boolean(a <= b);
            case GE: return GUTRValue::boolean(a >= b);
            default: throw std::runtime_error("invalid op");
        }
    } else if (op == EQ && l.type == r.type && l.type == ValueType::BOOL)
        return GUTRValue::boolean(l.bool_val == r.bool_val);
    else if (op == NE && l.type == r.type && l.type == ValueType::BOOL)
        return GUTRValue::boolean(l.bool_val != r.bool_val);
    else if (op == EQ && l.type == ValueType::STRING && r.type == ValueType::STRING)
        return GUTRValue::boolean(l.string_val == r.string_val);
    else if (op == NE && l.type == ValueType::STRING && r.type == ValueType::STRING)
        return GUTRValue::boolean(l.string_val != r.string_val);
    else if (op == AND && l.type == ValueType::BOOL && r.type == ValueType::BOOL)
        return GUTRValue::boolean(l.bool_val && r.bool_val);
    else if (op == OR && l.type == ValueType::BOOL && r.type == ValueType::BOOL)
        return GUTRValue::boolean(l.bool_val || r.bool_val);

    throw std::runtime_error("type mismatch");
}

GUTRValue GUTRUnaryOpExpr::eval(std::shared_ptr<GUTRContext> ctx) {
    GUTRValue a = arg->eval(ctx);
    if (op == NEG) {
        if (a.type == ValueType::INT) return GUTRValue::integer(-a.int_val);
        if (a.type == ValueType::FLOAT) return GUTRValue::real(-a.float_val);
        throw std::runtime_error("negation on non-numeric");
    } else if (op == NOT) {
        if (a.type != ValueType::BOOL) throw std::runtime_error("not on non-boolean");
        return GUTRValue::boolean(!a.bool_val);
    }
    throw std::runtime_error("unknown unary op");
}

GUTRValue GUTRCallExpr::eval(std::shared_ptr<GUTRContext> ctx) {
    GUTRValue callee_val = callee->eval(ctx);
    if (callee_val.type != ValueType::FUNCTION) throw std::runtime_error("calling non-function");
    std::vector<GUTRValue> arg_vals;
    for (auto& a : args) arg_vals.push_back(a->eval(ctx));
    return callee_val.func_val->call(arg_vals);
}

GUTRValue GUTRIndexExpr::eval(std::shared_ptr<GUTRContext> ctx) {
    GUTRValue obj = object->eval(ctx), idx = index->eval(ctx);
    if (obj.type == ValueType::TENSOR && idx.type == ValueType::INT) {
        size_t i = idx.int_val;
        auto t = obj.tensor_val;
        if (i >= t->size()) throw std::runtime_error("index out of bounds");
        return GUTRValue::real(t->data[i]);
    } else if (obj.type == ValueType::TUPLE && idx.type == ValueType::INT) {
        size_t i = idx.int_val;
        if (i >= obj.tuple_val.size()) throw std::runtime_error("tuple index out of bounds");
        return obj.tuple_val[i];
    } else if (obj.type == ValueType::SECT && idx.type == ValueType::STRING) {
        auto it = obj.sect_fields.find(idx.string_val);
        if (it == obj.sect_fields.end()) throw std::runtime_error("field not found");
        return it->second;
    }
    throw std::runtime_error("invalid indexing");
}

GUTRValue GUTRAssignExpr::eval(std::shared_ptr<GUTRContext> ctx) {
    GUTRValue v = value->eval(ctx);
    ctx->set(var_name, v);
    return v;
}
GUTRValue GUTRBlockStmt::eval(std::shared_ptr<GUTRContext> ctx) {
    for (auto& stmt : statements) {
        stmt->eval(ctx);
        if (ctx->break_flag || ctx->continue_flag || ctx->return_value.type != ValueType::NIL) break;
    }
    return GUTRValue::nil();
}
GUTRValue GUTRIfStmt::eval(std::shared_ptr<GUTRContext> ctx) {
    GUTRValue cond = condition->eval(ctx);
    if (cond.type != ValueType::BOOL) throw std::runtime_error("if condition must be boolean");
    if (cond.bool_val) return then_branch->eval(ctx);
    else if (else_branch) return else_branch->eval(ctx);
    return GUTRValue::nil();
}
GUTRValue GUTRWhileStmt::eval(std::shared_ptr<GUTRContext> ctx) {
    while (true) {
        GUTRValue cond = condition->eval(ctx);
        if (cond.type != ValueType::BOOL) throw std::runtime_error("while condition must be boolean");
        if (!cond.bool_val) break;
        body->eval(ctx);
        if (ctx->break_flag) { ctx->break_flag = false; break; }
        if (ctx->continue_flag) { ctx->continue_flag = false; continue; }
        if (ctx->return_value.type != ValueType::NIL) break;
    }
    return GUTRValue::nil();
}
GUTRValue GUTRForStmt::eval(std::shared_ptr<GUTRContext> ctx) {
    GUTRValue start_val = start_expr->eval(ctx), end_val = end_expr->eval(ctx),
             step_val = step_expr ? step_expr->eval(ctx) : GUTRValue::integer(1);
    if (start_val.type != ValueType::INT || end_val.type != ValueType::INT || step_val.type != ValueType::INT)
        throw std::runtime_error("for bounds must be int");
    int64_t start = start_val.int_val, end = end_val.int_val, step = step_val.int_val;
    if (step == 0) throw std::runtime_error("step cannot be zero");
    for (int64_t i = start; (step > 0) ? (i < end) : (i > end); i += step) {
        ctx->set(var_name, GUTRValue::integer(i));
        body->eval(ctx);
        if (ctx->break_flag) { ctx->break_flag = false; break; }
        if (ctx->continue_flag) { ctx->continue_flag = false; continue; }
        if (ctx->return_value.type != ValueType::NIL) break;
    }
    return GUTRValue::nil();
}
GUTRValue GUTRReturnStmt::eval(std::shared_ptr<GUTRContext> ctx) {
    ctx->return_value = value ? value->eval(ctx) : GUTRValue::nil();
    return ctx->return_value;
}
GUTRValue GUTRBreakStmt::eval(std::shared_ptr<GUTRContext> ctx) { ctx->break_flag = true; return GUTRValue::nil(); }
GUTRValue GUTRContinueStmt::eval(std::shared_ptr<GUTRContext> ctx) { ctx->continue_flag = true; return GUTRValue::nil(); }
GUTRValue GUTRArrayLiteralExpr::eval(std::shared_ptr<GUTRContext> ctx) {
    std::vector<GUTRValue> values;
    for (auto& elem : elements) values.push_back(elem->eval(ctx));
    return GUTRValue::tuple(values);
}
GUTRValue GUTRFunctionDefExpr::eval(std::shared_ptr<GUTRContext> ctx) {
    auto func = std::make_shared<GUTRUserFunction>();
    func->param_names = params;
    func->body = body;
    func->closure = ctx;
    ctx->set(name, GUTRValue::function(func));
    return GUTRValue::nil();
}
GUTRValue GUTRSectDefExpr::eval(std::shared_ptr<GUTRContext> ctx) {
    std::unordered_map<std::string, GUTRValue> fields;
    for (auto& [fname, expr] : field_inits) fields[fname] = expr->eval(ctx);
    ctx->set(name, GUTRValue::sect(fields));
    return GUTRValue::nil();
}

// ============================================================================
// ВСТРОЕННЫЕ ФУНКЦИИ
// ============================================================================
static float safe_exp(float x) {
    if (x > 80.0f) return std::exp(80.0f);
    if (x < -80.0f) return std::exp(-80.0f);
    return std::exp(x);
}

static GUTRValue builtin_tensor(const std::vector<GUTRValue>& args) {
    if (args.empty()) throw std::runtime_error("tensor requires shape");
    std::vector<size_t> shape;
    if (args[0].type == ValueType::TUPLE)
        for (auto& dim : args[0].tuple_val) shape.push_back(dim.int_val);
    else if (args[0].type == ValueType::INT)
        shape.push_back(args[0].int_val);
    else throw std::runtime_error("invalid shape");

    auto t = std::make_shared<Tensor>(shape);
    if (args.size() > 1) {
        if (args[1].type == ValueType::STRING) {
            if (args[1].string_val == "zeros") t->fill(0.0f);
            else if (args[1].string_val == "ones") t->fill(1.0f);
            else if (args[1].string_val == "random_normal") {
                float mean = (args.size() > 2) ? args[2].float_val : 0.0f;
                float stddev = (args.size() > 3) ? args[3].float_val : 0.02f;
                t->random_normal(mean, stddev);
            } else if (args[1].string_val == "random_uniform") {
                float min = (args.size() > 2) ? args[2].float_val : -0.1f;
                float max = (args.size() > 3) ? args[3].float_val : 0.1f;
                t->random_uniform(min, max);
            } else if (args[1].string_val == "from_array") {
                if (args.size() < 3 || args[2].type != ValueType::TUPLE)
                    throw std::runtime_error("from_array needs tuple");
                std::vector<float> arr;
                for (auto& v : args[2].tuple_val) {
                    if (v.type == ValueType::INT) arr.push_back(v.int_val);
                    else if (v.type == ValueType::FLOAT) arr.push_back(v.float_val);
                    else throw std::runtime_error("array elements must be numbers");
                }
                t->from_array(arr);
            } else throw std::runtime_error("unknown initializer");
        } else if (args[1].type == ValueType::FLOAT || args[1].type == ValueType::INT) {
            float val = (args[1].type == ValueType::INT) ? args[1].int_val : args[1].float_val;
            t->fill(val);
        }
    }
    return GUTRValue::tensor(t);
}

static GUTRValue builtin_matmul(const std::vector<GUTRValue>& args) {
    if (args.size() != 2 || args[0].type != ValueType::TENSOR || args[1].type != ValueType::TENSOR)
        throw std::runtime_error("matmul needs two tensors");
    auto A = args[0].tensor_val, B = args[1].tensor_val;
    if (A->shape.size() != 2 || B->shape.size() != 2)
        throw std::runtime_error("matmul requires 2D tensors");
    size_t M = A->shape[0], K = A->shape[1], N = B->shape[1];
    auto C = std::make_shared<Tensor>(std::vector<size_t>{M, N});
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < M; ++i)
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) sum += A->data[i*K+k] * B->data[k*N+j];
            C->data[i*N+j] = sum;
        }
    return GUTRValue::tensor(C);
}

static GUTRValue builtin_softmax(const std::vector<GUTRValue>& args) {
    if (args.size() != 1 || args[0].type != ValueType::TENSOR)
        throw std::runtime_error("softmax needs one tensor");
    auto x = args[0].tensor_val;
    auto res = std::make_shared<Tensor>(x->shape);
    float max_val = *std::max_element(x->data.begin(), x->data.end()), sum = 0.0f;
    for (size_t i = 0; i < x->size(); ++i) {
        float e = std::exp(x->data[i] - max_val);
        res->data[i] = e;
        sum += e;
    }
    float inv = 1.0f / (sum + 1e-8f);
    for (size_t i = 0; i < x->size(); ++i) res->data[i] *= inv;
    return GUTRValue::tensor(res);
}

static GUTRValue builtin_gelu(const std::vector<GUTRValue>& args) {
    if (args.size() != 1 || args[0].type != ValueType::TENSOR)
        throw std::runtime_error("gelu needs one tensor");
    auto x = args[0].tensor_val;
    auto res = std::make_shared<Tensor>(x->shape);
    const float sqrt2_over_pi = 0.7978845608028654f, coeff = 0.044715f;
    for (size_t i = 0; i < x->size(); ++i) {
        float v = x->data[i], x3 = v*v*v, tanh_arg = sqrt2_over_pi * (v + coeff*x3);
        res->data[i] = 0.5f * v * (1.0f + std::tanh(tanh_arg));
    }
    return GUTRValue::tensor(res);
}

static GUTRValue builtin_layer_norm(const std::vector<GUTRValue>& args) {
    if (args.size() < 3) throw std::runtime_error("layer_norm needs x, weight, bias, [eps]");
    auto x = args[0].tensor_val, w = args[1].tensor_val, b = args[2].tensor_val;
    float eps = (args.size() > 3) ? args[3].float_val : 1e-5f;
    if (x->shape != w->shape || x->shape != b->shape) throw std::runtime_error("shape mismatch");
    auto res = std::make_shared<Tensor>(x->shape);
    size_t n = x->size();
    float mean = 0.0f, var = 0.0f;
    for (size_t i = 0; i < n; ++i) mean += x->data[i];
    mean /= n;
    for (size_t i = 0; i < n; ++i) { float d = x->data[i] - mean; var += d*d; }
    var = std::sqrt(var / n + eps);
    float inv = 1.0f / var;
    for (size_t i = 0; i < n; ++i)
        res->data[i] = (x->data[i] - mean) * inv * w->data[i] + b->data[i];
    return GUTRValue::tensor(res);
}

static GUTRValue builtin_wkv(const std::vector<GUTRValue>& args) {
    if (args.size() != 5) throw std::runtime_error("wkv needs k,v,state,decay,first");
    auto k = args[0].tensor_val, v = args[1].tensor_val, state = args[2].tensor_val,
         decay = args[3].tensor_val, first = args[4].tensor_val;
    if (k->shape != v->shape || k->shape != state->shape || k->shape != decay->shape || k->shape != first->shape)
        throw std::runtime_error("shape mismatch");
    size_t n = k->size();
    auto out = std::make_shared<Tensor>(k->shape);
    for (size_t i = 0; i < n; ++i) {
        float w = safe_exp(decay->data[i]), u = first->data[i],
              s = state->data[i], kval = k->data[i], vval = v->data[i];
        float log_e = u + kval, max_val = std::max(log_e, 0.0f);
        float exp_e = safe_exp(log_e - max_val), exp_w = w * std::exp(-max_val);
        float num = exp_e * vval + exp_w * s, den = exp_e + exp_w;
        float wkv_val = num / (den + 1e-8f);
        out->data[i] = wkv_val;
        state->data[i] = wkv_val;
    }
    return GUTRValue::tensor(out);
}

static GUTRValue builtin_sigmoid(const std::vector<GUTRValue>& args) {
    if (args.size() != 1 || args[0].type != ValueType::TENSOR)
        throw std::runtime_error("sigmoid expects one tensor");
    auto x = args[0].tensor_val;
    auto res = std::make_shared<Tensor>(x->shape);
    for (size_t i = 0; i < x->size(); ++i)
        res->data[i] = 1.0f / (1.0f + std::exp(-x->data[i]));
    return GUTRValue::tensor(res);
}

static GUTRValue builtin_slice(const std::vector<GUTRValue>& args) {
    if (args.size() < 2) throw std::runtime_error("slice needs at least 2 arguments");
    if (args[0].type != ValueType::TENSOR || args[1].type != ValueType::TUPLE)
        throw std::runtime_error("slice expects (tensor, start, size)");
    auto t = args[0].tensor_val;
    if (t->shape.size() != 2) throw std::runtime_error("slice only supports 2D tensors");
    auto start = args[1].tuple_val;
    if (start.size() < 2) throw std::runtime_error("start must have at least 2 elements");
    size_t start_row = start[0].int_val, start_col = start[1].int_val, size_row = 1, size_col = 1;
    if (args.size() > 2 && args[2].type == ValueType::TUPLE) {
        auto sz = args[2].tuple_val;
        if (sz.size() >= 1) size_row = sz[0].int_val;
        if (sz.size() >= 2) size_col = sz[1].int_val;
    }
    if (start_row + size_row > t->shape[0] || start_col + size_col > t->shape[1])
        throw std::runtime_error("slice out of bounds");
    auto res = std::make_shared<Tensor>(std::vector<size_t>{size_row, size_col});
    for (size_t i = 0; i < size_row; ++i)
        for (size_t j = 0; j < size_col; ++j)
            res->data[i*size_col+j] = t->data[(start_row+i)*t->shape[1] + (start_col+j)];
    return GUTRValue::tensor(res);
}

static GUTRValue builtin_pow(const std::vector<GUTRValue>& args) {
    if (args.size() != 2) throw std::runtime_error("pow expects (base, exponent)");
    if (args[0].type == ValueType::FLOAT && args[1].type == ValueType::FLOAT)
        return GUTRValue::real(std::pow(args[0].float_val, args[1].float_val));
    if (args[0].type == ValueType::INT && args[1].type == ValueType::INT)
        return GUTRValue::real(std::pow((double)args[0].int_val, (double)args[1].int_val));
    if (args[0].type == ValueType::FLOAT && args[1].type == ValueType::INT)
        return GUTRValue::real(std::pow(args[0].float_val, (double)args[1].int_val));
    if (args[0].type == ValueType::INT && args[1].type == ValueType::FLOAT)
        return GUTRValue::real(std::pow((double)args[0].int_val, args[1].float_val));
    throw std::runtime_error("pow expects numeric arguments");
}

static GUTRValue builtin_zero_grad(const std::vector<GUTRValue>& args) {
    for (auto& a : args) {
        if (a.type == ValueType::TENSOR) a.tensor_val->zero_grad();
        else if (a.type == ValueType::TUPLE)
            for (auto& e : a.tuple_val)
                if (e.type == ValueType::TENSOR) e.tensor_val->zero_grad();
    }
    return GUTRValue::nil();
}

static GUTRValue builtin_update_params(const std::vector<GUTRValue>& args) {
    if (args.size() != 2) throw std::runtime_error("update_params expects (tensor or list, lr)");
    float lr = args[1].float_val;
    auto update = [lr](std::shared_ptr<Tensor> t) {
        for (size_t i = 0; i < t->size(); ++i) t->data[i] -= lr * t->grad[i];
    };
    if (args[0].type == ValueType::TUPLE) {
        for (auto& e : args[0].tuple_val) { if (e.type == ValueType::TENSOR) update(e.tensor_val); }
    } else if (args[0].type == ValueType::TENSOR) { update(args[0].tensor_val); }
    return GUTRValue::nil();
}

static GUTRValue builtin_sample_token(const std::vector<GUTRValue>& args) {
    if (args.size() < 3 || args[0].type != ValueType::TENSOR)
        throw std::runtime_error("sample_token needs logits, temp, top_p");
    auto logits = args[0].tensor_val;
    float temp = args[1].float_val, top_p = args[2].float_val;
    if (temp <= 0.0f) temp = 0.01f;
    size_t n = logits->size();
    std::vector<std::pair<float, int>> probs;
    float max_logit = *std::max_element(logits->data.begin(), logits->data.end()), sum_exp = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float p = std::exp((logits->data[i] - max_logit) / temp);
        probs.emplace_back(p, i); sum_exp += p;
    }
    if (sum_exp > 0) for (auto& p : probs) p.first /= sum_exp;
    std::sort(probs.begin(), probs.end(), [](auto& a, auto& b) { return a.first > b.first; });
    if (top_p < 0.99f) {
        float cumsum = 0.0f; size_t limit = 0;
        for (size_t i = 0; i < probs.size(); ++i) { cumsum += probs[i].first; limit = i+1; if (cumsum >= top_p) break; }
        probs.resize(limit);
    }
    thread_local static std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(gen), cumsum = 0.0f;
    for (auto& p : probs) { cumsum += p.first; if (r < cumsum) return GUTRValue::integer(p.second); }
    return GUTRValue::integer(probs[0].second);
}

static GUTRValue builtin_print(const std::vector<GUTRValue>& args) {
    for (auto& a : args) {
        switch (a.type) {
            case ValueType::NIL: std::cout << "nil"; break;
            case ValueType::BOOL: std::cout << (a.bool_val ? "true" : "false"); break;
            case ValueType::INT: std::cout << a.int_val; break;
            case ValueType::FLOAT: std::cout << a.float_val; break;
            case ValueType::STRING: std::cout << a.string_val; break;
            case ValueType::TENSOR: std::cout << "tensor(" << a.tensor_val->shape.size() << "D)"; break;
            case ValueType::TUPLE: std::cout << "tuple(" << a.tuple_val.size() << ")"; break;
            case ValueType::FUNCTION: std::cout << "<function>"; break;
            case ValueType::SECT: std::cout << "<sect>"; break;
        }
        std::cout << " ";
    }
    std::cout << std::endl;
    return GUTRValue::nil();
}

static GUTRValue builtin_save_gguf(const std::vector<GUTRValue>& args) {
    if (args.size() != 2 || args[0].type != ValueType::STRING)
        throw std::runtime_error("save_gguf needs path and list");
    std::string path = args[0].string_val;
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    uint32_t magic = 0x46554747, version = 3;
    uint64_t tensor_count = 0, metadata_kv_count = 0;
    f.write((char*)&magic, 4); f.write((char*)&version, 4);
    f.write((char*)&tensor_count, 8); f.write((char*)&metadata_kv_count, 8);
    f.close();
    std::cout << "save_gguf stub: " << path << std::endl;
    return GUTRValue::nil();
}

static GUTRValue builtin_load_gguf(const std::vector<GUTRValue>& args) {
    if (args.size() != 1 || args[0].type != ValueType::STRING)
        throw std::runtime_error("load_gguf needs path");
    std::cout << "load_gguf stub: " << args[0].string_val << std::endl;
    return GUTRValue::tuple({});
}

static GUTRValue builtin_list(const std::vector<GUTRValue>& args) { return GUTRValue::tuple(args); }

static GUTRValue builtin_append(const std::vector<GUTRValue>& args) {
    if (args.size() != 2 || args[0].type != ValueType::TUPLE)
        throw std::runtime_error("append needs list and element");
    auto lst = args[0].tuple_val;
    lst.push_back(args[1]);
    return GUTRValue::tuple(lst);
}

static GUTRValue builtin_json_parse(const std::vector<GUTRValue>& args) {
    if (args.size() != 1 || args[0].type != ValueType::STRING)
        throw std::runtime_error("json_parse needs string");
    std::string json = args[0].string_val;
    std::unordered_map<std::string, GUTRValue> obj;
    size_t pos = json.find('{');
    if (pos == std::string::npos) return GUTRValue::sect(obj);
    pos++;
    while (pos < json.size()) {
        while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\n' || json[pos] == ',')) pos++;
        if (json[pos] == '}') break;
        if (json[pos] != '"') break;
        size_t key_start = ++pos;
        while (pos < json.size() && json[pos] != '"') pos++;
        std::string key = json.substr(key_start, pos - key_start);
        pos++;
        while (pos < json.size() && json[pos] != ':') pos++;
        pos++;
        while (pos < json.size() && json[pos] == ' ') pos++;
        if (json[pos] == '"') {
            size_t val_start = ++pos;
            while (pos < json.size() && json[pos] != '"') pos++;
            std::string val = json.substr(val_start, pos - val_start);
            obj[key] = GUTRValue::string(val);
            pos++;
        } else if (json[pos] == '{') {
            size_t brace = 1; pos++;
            size_t start = pos;
            while (pos < json.size() && brace > 0) {
                if (json[pos] == '{') brace++;
                else if (json[pos] == '}') brace--;
                pos++;
            }
            std::string sub = json.substr(start, pos - start - 1);
            obj[key] = builtin_json_parse({GUTRValue::string(sub)});
        } else {
            size_t start = pos;
            while (pos < json.size() && json[pos] != ',' && json[pos] != '}') pos++;
            std::string num = json.substr(start, pos - start);
            if (num.find('.') != std::string::npos)
                obj[key] = GUTRValue::real(std::stod(num));
            else obj[key] = GUTRValue::integer(std::stoll(num));
        }
    }
    return GUTRValue::sect(obj);
}

static GUTRValue builtin_get_global(const std::vector<GUTRValue>& args) {
    if (args.size() != 1 || args[0].type != ValueType::STRING)
        throw std::runtime_error("get_global needs string");
    static std::unordered_map<std::string, GUTRValue> globals;
    auto it = globals.find(args[0].string_val);
    if (it == globals.end()) return GUTRValue::nil();
    return it->second;
}

static GUTRValue builtin_set_global(const std::vector<GUTRValue>& args) {
    if (args.size() != 2 || args[0].type != ValueType::STRING)
        throw std::runtime_error("set_global needs name and value");
    static std::unordered_map<std::string, GUTRValue> globals;
    globals[args[0].string_val] = args[1];
    return GUTRValue::nil();
}

static GUTRValue builtin_add_to_grad(const std::vector<GUTRValue>& args) {
    if (args.size() != 2 || args[0].type != ValueType::TENSOR || args[1].type != ValueType::TENSOR)
        throw std::runtime_error("add_to_grad needs two tensors");
    auto t = args[0].tensor_val, g = args[1].tensor_val;
    if (t->size() != g->size()) throw std::runtime_error("size mismatch");
    for (size_t i = 0; i < t->size(); ++i) t->grad[i] += g->data[i];
    return GUTRValue::nil();
}

static GUTRValue builtin_backward_matmul(const std::vector<GUTRValue>& args) {
    if (args.size() != 5) throw std::runtime_error("backward_matmul needs A,B,grad_C,grad_A,grad_B");
    auto A = args[0].tensor_val, B = args[1].tensor_val, gradC = args[2].tensor_val,
         gradA = args[3].tensor_val, gradB = args[4].tensor_val;
    size_t M = A->shape[0], K = A->shape[1], N = B->shape[1];
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < M; ++i) for (size_t k = 0; k < K; ++k) {
        float sum = 0.0f;
        for (size_t j = 0; j < N; ++j) sum += gradC->data[i*N+j] * B->data[k*N+j];
        gradA->data[i*K+k] += sum;
    }
    #pragma omp parallel for collapse(2)
    for (size_t k = 0; k < K; ++k) for (size_t j = 0; j < N; ++j) {
        float sum = 0.0f;
        for (size_t i = 0; i < M; ++i) sum += A->data[i*K+k] * gradC->data[i*N+j];
        gradB->data[k*N+j] += sum;
    }
    return GUTRValue::nil();
}

static GUTRValue builtin_backward_gelu(const std::vector<GUTRValue>& args) {
    if (args.size() != 3) throw std::runtime_error("backward_gelu needs x,grad_out,grad_in");
    auto x = args[0].tensor_val, grad_out = args[1].tensor_val, grad_in = args[2].tensor_val;
    const float sqrt2_over_pi = 0.79788456f, coeff = 0.044715f;
    for (size_t i = 0; i < x->size(); ++i) {
        float v = x->data[i], x3 = v*v*v, tanh_arg = sqrt2_over_pi * (v + coeff*x3),
              tanh_val = std::tanh(tanh_arg);
        float deriv = 0.5f * (1.0f + tanh_val) +
                      0.5f * v * (1.0f - tanh_val*tanh_val) * sqrt2_over_pi * (1.0f + 3.0f*coeff*v*v);
        grad_in->data[i] += grad_out->data[i] * deriv;
    }
    return GUTRValue::nil();
}

static GUTRValue builtin_backward_wkv(const std::vector<GUTRValue>& args) {
    if (args.size() != 10) throw std::runtime_error("backward_wkv needs 10 args");
    auto k = args[0].tensor_val, v = args[1].tensor_val, state = args[2].tensor_val,
         decay = args[3].tensor_val, first = args[4].tensor_val;
    auto grad_out = args[5].tensor_val, grad_k = args[6].tensor_val,
         grad_v = args[7].tensor_val, grad_decay = args[8].tensor_val, grad_first = args[9].tensor_val;
    size_t n = k->size();
    for (size_t i = 0; i < n; ++i) {
        float w = safe_exp(decay->data[i]), u = first->data[i],
              s = state->data[i], kval = k->data[i], vval = v->data[i];
        float log_e = u + kval, max_val = std::max(log_e, 0.0f);
        float exp_e = safe_exp(log_e - max_val), exp_w = w * std::exp(-max_val);
        float num = exp_e * vval + exp_w * s, den = exp_e + exp_w,
              inv_den = 1.0f / (den + 1e-8f);
        float dL_dnum = grad_out->data[i] * inv_den,
              dL_dden = -grad_out->data[i] * num * inv_den * inv_den;
        float dL_de = dL_dnum * vval + dL_dden, dL_dv = dL_dnum * exp_e,
              dL_dw_raw = dL_dnum * s + dL_dden;
        grad_k->data[i] += dL_de * exp_e;
        grad_v->data[i] += dL_dv;
        grad_decay->data[i] += dL_dw_raw * exp_w * w;
        grad_first->data[i] += dL_de * exp_e;
    }
    return GUTRValue::nil();
}

static GUTRValue builtin_backward_layer_norm(const std::vector<GUTRValue>& args) {
    if (args.size() < 6)
        throw std::runtime_error("backward_layer_norm needs x,w,grad_out,grad_x,grad_w,grad_b,[eps]");
    auto x = args[0].tensor_val, w = args[1].tensor_val, grad_out = args[2].tensor_val,
         grad_x = args[3].tensor_val, grad_w = args[4].tensor_val, grad_b = args[5].tensor_val;
    float eps = (args.size() > 6) ? args[6].float_val : 1e-5f;
    size_t n = x->size();
    float mean = 0.0f, var = 0.0f;
    for (size_t i = 0; i < n; ++i) mean += x->data[i];
    mean /= n;
    for (size_t i = 0; i < n; ++i) { float d = x->data[i] - mean; var += d*d; }
    var = var / n;
    float std = std::sqrt(var + eps), inv_std = 1.0f / std;
    std::vector<float> x_hat(n);
    for (size_t i = 0; i < n; ++i) x_hat[i] = (x->data[i] - mean) * inv_std;
    for (size_t i = 0; i < n; ++i) grad_b->data[i] += grad_out->data[i];
    for (size_t i = 0; i < n; ++i) grad_w->data[i] += grad_out->data[i] * x_hat[i];
    std::vector<float> grad_x_hat(n);
    for (size_t i = 0; i < n; ++i) grad_x_hat[i] = grad_out->data[i] * w->data[i];
    float grad_var = 0.0f;
    for (size_t i = 0; i < n; ++i)
        grad_var += grad_x_hat[i] * (x->data[i] - mean) * (-0.5f) * std::pow(var + eps, -1.5f);
    float grad_mean = 0.0f;
    for (size_t i = 0; i < n; ++i) grad_mean += grad_x_hat[i] * (-inv_std);
    for (size_t i = 0; i < n; ++i)
        grad_x->data[i] += grad_x_hat[i] * inv_std +
                           grad_var * 2.0f * (x->data[i] - mean) / n + grad_mean / n;
    return GUTRValue::nil();
}

static GUTRValue builtin_u64_to_string(const std::vector<GUTRValue>& args) {
    if (args.size() != 1 || args[0].type != ValueType::INT)
        throw std::runtime_error("string() needs u64");
    return GUTRValue::string(std::to_string(args[0].int_val));
}

static GUTRValue builtin_list_nth(const std::vector<GUTRValue>& args) {
    if (args.size() != 2 || args[0].type != ValueType::TUPLE || args[1].type != ValueType::INT)
        throw std::runtime_error("list_nth needs (list, idx)");
    size_t idx = args[1].int_val;
    if (idx >= args[0].tuple_val.size()) throw std::runtime_error("list_nth: index out of bounds");
    return args[0].tuple_val[idx];
}

static GUTRValue builtin_sect_get(const std::vector<GUTRValue>& args) {
    if (args.size() != 2 || args[0].type != ValueType::SECT || args[1].type != ValueType::STRING)
        throw std::runtime_error("sect_get needs (sect, key)");
    auto it = args[0].sect_fields.find(args[1].string_val);
    if (it == args[0].sect_fields.end()) return GUTRValue::nil();
    return it->second;
}

static GUTRValue builtin_get_pair(const std::vector<GUTRValue>& args) {
    if (args.size() != 2 || args[0].type != ValueType::TUPLE || args[1].type != ValueType::STRING)
        throw std::runtime_error("get_pair needs (list, key)");
    for (const auto& item : args[0].tuple_val) {
        if (item.type != ValueType::TUPLE || item.tuple_val.size() < 2) continue;
        if (item.tuple_val[0].type == ValueType::STRING &&
            item.tuple_val[0].string_val == args[1].string_val)
            return item.tuple_val[1];
    }
    return GUTRValue::nil();
}

// ============================================================================
// НОВЫЕ BUILTINS ДЛЯ FLETTOHM + ROSA + UIT:SLESA
// ============================================================================

static GUTRValue builtin_eye(const std::vector<GUTRValue>& args) {
    if (args.size() != 1 || args[0].type != ValueType::INT)
        throw std::runtime_error("eye needs size");
    size_t n = args[0].int_val;
    auto result = std::make_shared<Tensor>(std::vector<size_t>{n, n});
    std::fill(result->data.begin(), result->data.end(), 0.0f);
    for (size_t i = 0; i < n; ++i) result->data[i * n + i] = 1.0f;
    return GUTRValue::tensor(result);
}

static GUTRValue builtin_transpose(const std::vector<GUTRValue>& args) {
    if (args.size() != 1 || args[0].type != ValueType::TENSOR)
        throw std::runtime_error("transpose needs tensor");
    auto A = args[0].tensor_val;
    if (A->shape.size() != 2) throw std::runtime_error("transpose requires 2D tensor");
    size_t rows = A->shape[0], cols = A->shape[1];
    auto result = std::make_shared<Tensor>(std::vector<size_t>{cols, rows});
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            result->data[j * rows + i] = A->data[i * cols + j];
    return GUTRValue::tensor(result);
}

static GUTRValue builtin_concat_1d(const std::vector<GUTRValue>& args) {
    if (args.size() != 3) throw std::runtime_error("concat_1d needs 3 tensors");
    size_t total = 0;
    std::shared_ptr<Tensor> tensors[3];
    for (int i = 0; i < 3; ++i) {
        if (args[i].type != ValueType::TENSOR)
            throw std::runtime_error("concat_1d: argument must be tensor");
        tensors[i] = args[i].tensor_val;
        total += tensors[i]->size();
    }
    auto result = std::make_shared<Tensor>(std::vector<size_t>{total});
    size_t offset = 0;
    for (int i = 0; i < 3; ++i) {
        std::copy(tensors[i]->data.begin(), tensors[i]->data.end(), result->data.begin() + offset);
        offset += tensors[i]->size();
    }
    return GUTRValue::tensor(result);
}

static GUTRValue builtin_tt_svd(const std::vector<GUTRValue>& args) {
    if (args.size() != 2 || args[0].type != ValueType::TENSOR || args[1].type != ValueType::INT)
        throw std::runtime_error("builtin_tt_svd expects (matrix, rank)");
    auto W = args[0].tensor_val;
    int rank = static_cast<int>(args[1].int_val);
    if (W->shape.size() != 2) throw std::runtime_error("builtin_tt_svd: matrix must be 2D");
    size_t d_out = W->shape[0], d_in = W->shape[1];
    size_t k = std::min(d_out, d_in);
    int actual_rank = std::min(rank, (int)k);

    std::vector<float> U(d_out * k), S(k), Vt(k * d_in);
    svd_bidiag(W->data.data(), d_out, d_in, U.data(), S.data(), Vt.data(), k);

    auto G1 = std::make_shared<Tensor>(std::vector<size_t>{d_out, (size_t)actual_rank});
    for (size_t i = 0; i < d_out; ++i)
        for (int r = 0; r < actual_rank; ++r)
            G1->data[i * actual_rank + r] = U[i * k + r] * std::sqrt(S[r]);

    auto G2 = std::make_shared<Tensor>(std::vector<size_t>{(size_t)actual_rank, d_in});
    for (int r = 0; r < actual_rank; ++r) {
        float sqrtS = std::sqrt(S[r]);
        for (size_t j = 0; j < d_in; ++j)
            G2->data[r * d_in + j] = sqrtS * Vt[r * d_in + j];
    }

    std::unordered_map<std::string, GUTRValue> result;
    result["G1"] = GUTRValue::tensor(G1);
    result["G2"] = GUTRValue::tensor(G2);
    result["rank"] = GUTRValue::integer(actual_rank);
    return GUTRValue::sect(result);
}

static GUTRValue builtin_solve_linear(const std::vector<GUTRValue>& args) {
    if (args.size() != 2 || args[0].type != ValueType::TENSOR || args[1].type != ValueType::TENSOR)
        throw std::runtime_error("builtin_solve_linear expects (A, B)");
    auto A = args[0].tensor_val, B = args[1].tensor_val;
    if (A->shape.size() != 2 || B->shape.size() != 2)
        throw std::runtime_error("solve_linear: 2D required");
    size_t n = A->shape[0], m = B->shape[1];
    if (A->shape[1] != n || B->shape[0] != n)
        throw std::runtime_error("solve_linear: dimension mismatch");

    std::vector<float> Q(n * n), R(n * n);
    qr_householder(A->data.data(), n, n, Q.data(), R.data());

    std::vector<float> QtB(n * m, 0.0f);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < m; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < n; ++k) sum += Q[k * n + i] * B->data[k * m + j];
            QtB[i * m + j] = sum;
        }

    auto X = std::make_shared<Tensor>(std::vector<size_t>{n, m});
    for (size_t col = 0; col < m; ++col) {
        for (int i = (int)n - 1; i >= 0; --i) {
            float sum = QtB[i * m + col];
            for (size_t j = i + 1; j < n; ++j) sum -= R[i * n + j] * X->data[j * m + col];
            X->data[i * m + col] = (std::abs(R[i * n + i]) < 1e-8f) ? sum / 0.001f : sum / R[i * n + i];
        }
    }
    return GUTRValue::tensor(X);
}

static GUTRValue builtin_rosa_create(const std::vector<GUTRValue>& args) {
    if (args.size() != 1 || args[0].type != ValueType::INT)
        throw std::runtime_error("rosa_create needs max_len");
    int64_t id = next_rosa_id++;
    auto rosa = std::make_shared<ROSAInstance>();
    rosa->init(args[0].int_val);
    rosa_instances[id] = rosa;
    return GUTRValue::integer(id);
}

static GUTRValue builtin_rosa_update_and_query(const std::vector<GUTRValue>& args) {
    if (args.size() != 2 || args[0].type != ValueType::INT || args[1].type != ValueType::INT)
        throw std::runtime_error("rosa_update_and_query needs (id, token)");
    int64_t id = args[0].int_val;
    int token = args[1].int_val;
    auto it = rosa_instances.find(id);
    if (it == rosa_instances.end()) throw std::runtime_error("ROSA instance not found");
    int predicted = it->second->update_and_query(token);
    return GUTRValue::integer(predicted);
}

static GUTRValue builtin_set_element(const std::vector<GUTRValue>& args) {
    if (args.size() != 3) throw std::runtime_error("set_element needs (tensor, idx, value)");
    auto t = args[0].tensor_val;
    size_t idx = args[1].int_val;
    float val = (args[2].type == ValueType::INT) ? (float)args[2].int_val : (float)args[2].float_val;
    if (idx >= t->size()) throw std::runtime_error("set_element: index out of bounds");
    auto result = std::make_shared<Tensor>(t->shape);
    std::copy(t->data.begin(), t->data.end(), result->data.begin());
    result->data[idx] = val;
    return GUTRValue::tensor(result);
}

static GUTRValue builtin_set_element_2d(const std::vector<GUTRValue>& args) {
    if (args.size() != 4) throw std::runtime_error("set_element_2d needs (tensor, row, col, value)");
    auto t = args[0].tensor_val;
    if (t->shape.size() != 2) throw std::runtime_error("set_element_2d: tensor must be 2D");
    size_t row = args[1].int_val, col = args[2].int_val;
    float val = (args[3].type == ValueType::INT) ? (float)args[3].int_val : (float)args[3].float_val;
    if (row >= t->shape[0] || col >= t->shape[1]) throw std::runtime_error("set_element_2d: index out of bounds");
    auto result = std::make_shared<Tensor>(t->shape);
    std::copy(t->data.begin(), t->data.end(), result->data.begin());
    result->data[row * t->shape[1] + col] = val;
    return GUTRValue::tensor(result);
}

static GUTRValue builtin_scale(const std::vector<GUTRValue>& args) {
    if (args.size() != 2 || args[0].type != ValueType::TENSOR)
        throw std::runtime_error("scale needs (tensor, float)");
    auto t = args[0].tensor_val;
    float s = (args[1].type == ValueType::INT) ? (float)args[1].int_val : (float)args[1].float_val;
    auto result = std::make_shared<Tensor>(t->shape);
    #pragma omp simd
    for (size_t i = 0; i < t->size(); ++i) result->data[i] = t->data[i] * s;
    return GUTRValue::tensor(result);
}

static GUTRValue builtin_random_normal_vec(const std::vector<GUTRValue>& args) {
    if (args.size() != 2 || args[0].type != ValueType::TENSOR || args[1].type != ValueType::TENSOR)
        throw std::runtime_error("random_normal_vec needs (mean, std)");
    auto mean = args[0].tensor_val, stddev = args[1].tensor_val;
    if (mean->shape != stddev->shape) throw std::runtime_error("random_normal_vec: shape mismatch");
    auto result = std::make_shared<Tensor>(mean->shape);
    thread_local static std::mt19937 gen(std::random_device{}());
    for (size_t i = 0; i < mean->size(); ++i) {
        std::normal_distribution<float> dist(mean->data[i], stddev->data[i]);
        result->data[i] = dist(gen);
    }
    return GUTRValue::tensor(result);
}

static GUTRValue builtin_argmax(const std::vector<GUTRValue>& args) {
    if (args.size() != 1 || args[0].type != ValueType::TUPLE)
        throw std::runtime_error("argmax needs tuple of scalars");
    const auto& vals = args[0].tuple_val;
    if (vals.empty()) throw std::runtime_error("argmax: empty tuple");
    float max_val = -std::numeric_limits<float>::infinity();
    size_t max_idx = 0;
    for (size_t i = 0; i < vals.size(); ++i) {
        float v = (vals[i].type == ValueType::INT) ? (float)vals[i].int_val : (float)vals[i].float_val;
        if (v > max_val) { max_val = v; max_idx = i; }
    }
    return GUTRValue::integer(max_idx);
}

static GUTRValue builtin_topk_indices(const std::vector<GUTRValue>& args) {
    if (args.size() != 2 || args[0].type != ValueType::TENSOR || args[1].type != ValueType::INT)
        throw std::runtime_error("topk_indices needs (tensor, k)");
    auto probs = args[0].tensor_val;
    size_t k = std::min((size_t)args[1].int_val, probs->size());
    std::vector<std::pair<float, size_t>> indexed;
    for (size_t i = 0; i < probs->size(); ++i) indexed.emplace_back(probs->data[i], i);
    std::partial_sort(indexed.begin(), indexed.begin() + k, indexed.end(),
                      [](auto& a, auto& b) { return a.first > b.first; });
    std::vector<GUTRValue> result;
    for (size_t i = 0; i < k; ++i) result.push_back(GUTRValue::integer(indexed[i].second));
    return GUTRValue::tuple(result);
}

static GUTRValue builtin_update_layer_tt(const std::vector<GUTRValue>& args) {
    if (args.size() != 2 || args[0].type != ValueType::SECT || args[1].type != ValueType::SECT)
        throw std::runtime_error("update_layer_tt needs (layer, tt_result)");
    auto layer = args[0].sect_fields;
    auto G1 = args[1].sect_fields.at("G1").tensor_val;
    auto G2 = args[1].sect_fields.at("G2").tensor_val;

    std::vector<std::string> tt_keys = {"rwkv_Wk", "rwkv_Wv", "rwkv_Wdecay", "rwkv_Wr",
                                        "win_Wq", "win_Wk", "win_Wv", "gate_W", "router_W"};
    std::unordered_map<std::string, GUTRValue> new_layer = layer;

    for (auto& key : tt_keys) {
        auto it = layer.find(key);
        if (it != layer.end()) {
            std::unordered_map<std::string, GUTRValue> new_mat;
            new_mat["G1"] = GUTRValue::tensor(G1);
            new_mat["G2"] = GUTRValue::tensor(G2);
            new_mat["rank"] = it->second.sect_fields.at("rank");
            new_mat["out_dim"] = it->second.sect_fields.at("out_dim");
            new_mat["in_dim"] = it->second.sect_fields.at("in_dim");
            new_layer[key] = GUTRValue::sect(new_mat);
        }
    }
    return GUTRValue::sect(new_layer);
}

static GUTRValue builtin_rope(const std::vector<GUTRValue>& args) {
    if (args.size() != 2 || args[0].type != ValueType::TENSOR || args[1].type != ValueType::INT)
        throw std::runtime_error("rope needs (x_1d, position)");
    auto x = args[0].tensor_val;
    int pos = args[1].int_val;
    size_t d = x->size();
    auto result = std::make_shared<Tensor>(x->shape);
    float theta = 1.0f / std::pow(10000.0f, 2.0f / d);
    for (size_t i = 0; i < d; i += 2) {
        float freq = std::pow(theta, (float)i / d);
        float cos_val = std::cos(pos * freq);
        float sin_val = std::sin(pos * freq);
        float x_i = x->data[i];
        float x_i1 = (i + 1 < d) ? x->data[i + 1] : 0.0f;
        result->data[i] = x_i * cos_val - x_i1 * sin_val;
        if (i + 1 < d) result->data[i + 1] = x_i1 * cos_val + x_i * sin_val;
    }
    return GUTRValue::tensor(result);
}

static GUTRValue builtin_tanh(const std::vector<GUTRValue>& args) {
    if (args.size() != 1 || args[0].type != ValueType::TENSOR)
        throw std::runtime_error("tanh expects one tensor");
    auto x = args[0].tensor_val;
    auto result = std::make_shared<Tensor>(x->shape);
    for (size_t i = 0; i < x->size(); ++i) result->data[i] = std::tanh(x->data[i]);
    return GUTRValue::tensor(result);
}

static GUTRValue builtin_log(const std::vector<GUTRValue>& args) {
    if (args.size() != 1 || args[0].type != ValueType::TENSOR)
        throw std::runtime_error("log expects one tensor");
    auto x = args[0].tensor_val;
    auto result = std::make_shared<Tensor>(x->shape);
    for (size_t i = 0; i < x->size(); ++i) {
        result->data[i] = (x->data[i] > 1e-10f) ? std::log(x->data[i]) : std::log(1e-10f);
    }
    return GUTRValue::tensor(result);
}

// ============================================================================
// РЕГИСТРАЦИЯ ВСЕХ BUILTINS
// ============================================================================
void register_builtins(std::shared_ptr<GUTRContext> ctx) {
    ctx->define_builtin("tensor", builtin_tensor);
    ctx->define_builtin("matmul", builtin_matmul);
    ctx->define_builtin("softmax", builtin_softmax);
    ctx->define_builtin("gelu", builtin_gelu);
    ctx->define_builtin("layer_norm", builtin_layer_norm);
    ctx->define_builtin("wkv", builtin_wkv);
    ctx->define_builtin("sigmoid", builtin_sigmoid);
    ctx->define_builtin("slice", builtin_slice);
    ctx->define_builtin("pow", builtin_pow);
    ctx->define_builtin("zero_grad", builtin_zero_grad);
    ctx->define_builtin("update_params", builtin_update_params);
    ctx->define_builtin("sample_token", builtin_sample_token);
    ctx->define_builtin("print", builtin_print);
    ctx->define_builtin("save_gguf", builtin_save_gguf);
    ctx->define_builtin("load_gguf", builtin_load_gguf);
    ctx->define_builtin("list", builtin_list);
    ctx->define_builtin("append", builtin_append);
    ctx->define_builtin("json_parse", builtin_json_parse);
    ctx->define_builtin("get_global", builtin_get_global);
    ctx->define_builtin("set_global", builtin_set_global);
    ctx->define_builtin("add_to_grad", builtin_add_to_grad);
    ctx->define_builtin("backward_matmul", builtin_backward_matmul);
    ctx->define_builtin("backward_gelu", builtin_backward_gelu);
    ctx->define_builtin("backward_wkv", builtin_backward_wkv);
    ctx->define_builtin("backward_layer_norm", builtin_backward_layer_norm);
    ctx->define_builtin("string", builtin_u64_to_string);
    ctx->define_builtin("list_nth", builtin_list_nth);
    ctx->define_builtin("sect_get", builtin_sect_get);
    ctx->define_builtin("get_pair", builtin_get_pair);
    // НОВЫЕ
    ctx->define_builtin("builtin_eye", builtin_eye);
    ctx->define_builtin("builtin_transpose", builtin_transpose);
    ctx->define_builtin("builtin_concat_1d", builtin_concat_1d);
    ctx->define_builtin("builtin_tt_svd", builtin_tt_svd);
    ctx->define_builtin("builtin_solve_linear", builtin_solve_linear);
    ctx->define_builtin("builtin_rosa_create", builtin_rosa_create);
    ctx->define_builtin("builtin_rosa_update_and_query", builtin_rosa_update_and_query);
    ctx->define_builtin("builtin_set_element", builtin_set_element);
    ctx->define_builtin("builtin_set_element_2d", builtin_set_element_2d);
    ctx->define_builtin("builtin_scale", builtin_scale);
    ctx->define_builtin("builtin_random_normal", builtin_random_normal_vec);
    ctx->define_builtin("builtin_argmax", builtin_argmax);
    ctx->define_builtin("builtin_topk_indices", builtin_topk_indices);
    ctx->define_builtin("builtin_update_layer_tt", builtin_update_layer_tt);
    ctx->define_builtin("builtin_rope", builtin_rope);
    ctx->define_builtin("builtin_tanh", builtin_tanh);
    ctx->define_builtin("builtin_log", builtin_log);
}

// ============================================================================
// GUTRProgram methods
// ============================================================================
void GUTRProgram::init(const std::string& config_json) {
    if (!init_func) throw std::runtime_error("No init function");
    auto ctx = std::make_shared<GUTRContext>();
    register_builtins(ctx);
    init_func->call({GUTRValue::string(config_json)});
}
void GUTRProgram::train_step(const std::vector<int>& input, const std::vector<int>& target, float lr) {
    if (!train_step_func) throw std::runtime_error("No train_step function");
    auto ctx = std::make_shared<GUTRContext>();
    register_builtins(ctx);
    std::vector<GUTRValue> inp, tgt;
    for (int v : input) inp.push_back(GUTRValue::integer(v));
    for (int v : target) tgt.push_back(GUTRValue::integer(v));
    train_step_func->call({GUTRValue::tuple(inp), GUTRValue::tuple(tgt), GUTRValue::real(lr)});
}
void GUTRProgram::save(const std::string& path) {
    if (!save_func) throw std::runtime_error("No save function");
    auto ctx = std::make_shared<GUTRContext>();
    register_builtins(ctx);
    save_func->call({GUTRValue::string(path)});
}
void GUTRProgram::load(const std::string& path) {
    if (!load_func) throw std::runtime_error("No load function");
    auto ctx = std::make_shared<GUTRContext>();
    register_builtins(ctx);
    load_func->call({GUTRValue::string(path)});
}
int GUTRProgram::generate(const std::vector<int>& prompt, std::vector<int>& output, int max_tokens, float temp, float top_p) {
    if (!generate_func) throw std::runtime_error("No generate function");
    auto ctx = std::make_shared<GUTRContext>();
    register_builtins(ctx);
    std::vector<GUTRValue> p;
    for (int v : prompt) p.push_back(GUTRValue::integer(v));
    GUTRValue res = generate_func->call({GUTRValue::tuple(p), GUTRValue::integer(max_tokens),
                                          GUTRValue::real(temp), GUTRValue::real(top_p)});
    if (res.type == ValueType::TUPLE) {
        output.clear();
        for (auto& v : res.tuple_val) output.push_back(v.int_val);
        return output.size();
    }
    return 0;
}

} // namespace uzaleat
