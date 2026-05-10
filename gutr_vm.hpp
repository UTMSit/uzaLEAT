#ifndef GUTR_VM_HPP
#define GUTR_VM_HPP

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <cstdint>
#include <functional>
#include <mutex>
#include <stdexcept>  // ДОБАВЛЕНО для std::runtime_error

namespace uzaleat {

// Forward declarations
struct GUTRContext;
struct GUTRFunction;
struct GUTRBlockStmt;

enum class ValueType { NIL, BOOL, INT, FLOAT, STRING, TENSOR, FUNCTION, TUPLE, SECT };

struct Tensor {
    std::vector<size_t> shape;
    std::vector<float> data;
    std::vector<float> grad;
    bool requires_grad;
    std::string name; // Для отладки

    Tensor();
    explicit Tensor(const std::vector<size_t>& shape, bool requires_grad = true);
    Tensor(const std::vector<size_t>& shape, float fill_value, bool requires_grad = true);

    ~Tensor() = default;
    Tensor(const Tensor&) = default;
    Tensor& operator=(const Tensor&) = default;
    Tensor(Tensor&&) noexcept = default;
    Tensor& operator=(Tensor&&) noexcept = default;

    size_t size() const;
    void zero_grad();
    void fill(float value);
    void random_normal(float mean = 0.0f, float stddev = 0.02f);
    void random_uniform(float min = -0.1f, float max = 0.1f);
    void from_array(const std::vector<float>& arr);

    // Вспомогательные методы безопасности
    float& at(size_t i) { return data.at(i); }
    const float& at(size_t i) const { return data.at(i); }
    float* data_ptr() { return data.data(); }
    const float* data_ptr() const { return data.data(); }
    float* grad_ptr() { return requires_grad && !grad.empty() ? grad.data() : nullptr; }
    bool is_valid() const { return !data.empty() && (!requires_grad || grad.size() == data.size()); }
};

struct GUTRValue {
    ValueType type = ValueType::NIL;
    bool bool_val = false;
    int64_t int_val = 0;
    double float_val = 0.0;
    std::string string_val;
    std::shared_ptr<Tensor> tensor_val;
    std::vector<GUTRValue> tuple_val;
    std::shared_ptr<GUTRFunction> func_val;
    std::unordered_map<std::string, GUTRValue> sect_fields;

    GUTRValue();

    ~GUTRValue() = default;
    GUTRValue(const GUTRValue&) = default;
    GUTRValue& operator=(const GUTRValue&) = default;
    GUTRValue(GUTRValue&&) noexcept = default;
    GUTRValue& operator=(GUTRValue&&) noexcept = default;

    static GUTRValue nil();
    static GUTRValue boolean(bool v);
    static GUTRValue integer(int64_t v);
    static GUTRValue real(double v);
    static GUTRValue string(const std::string& v);
    static GUTRValue tensor(std::shared_ptr<Tensor> t);
    static GUTRValue tuple(const std::vector<GUTRValue>& v);
    static GUTRValue function(std::shared_ptr<GUTRFunction> f);
    static GUTRValue sect(const std::unordered_map<std::string, GUTRValue>& fields);

    bool is_nil() const { return type == ValueType::NIL; }
    bool is_number() const { return type == ValueType::INT || type == ValueType::FLOAT; }
    double as_number() const {
        if (type == ValueType::INT) return static_cast<double>(int_val);
        if (type == ValueType::FLOAT) return float_val;
        throw std::runtime_error("Not a number");
    }
    std::string type_name() const {
        switch(type) {
            case ValueType::NIL: return "nil";
            case ValueType::BOOL: return "bool";
            case ValueType::INT: return "int";
            case ValueType::FLOAT: return "float";
            case ValueType::STRING: return "string";
            case ValueType::TENSOR: return "tensor";
            case ValueType::FUNCTION: return "function";
            case ValueType::TUPLE: return "tuple";
            case ValueType::SECT: return "sect";
            default: return "unknown";
        }
    }
};

struct GUTRFunction {
    std::string name;
    virtual ~GUTRFunction() = default;
    virtual GUTRValue call(const std::vector<GUTRValue>& args) = 0;
    virtual size_t min_args() const { return 0; }
    virtual size_t max_args() const { return SIZE_MAX; }
};

struct GUTRBuiltinFunction : public GUTRFunction {
    using FuncType = std::function<GUTRValue(const std::vector<GUTRValue>&)>;
    FuncType impl;
    size_t min_args_ = 0;
    size_t max_args_ = SIZE_MAX;

    GUTRBuiltinFunction(FuncType f, const std::string& func_name = "") : impl(f) {
        name = func_name;
    }

    GUTRBuiltinFunction& set_args(size_t min, size_t max = SIZE_MAX) {
        min_args_ = min;
        max_args_ = max;
        return *this;
    }

    GUTRValue call(const std::vector<GUTRValue>& args) override {
        if (args.size() < min_args_)
            throw std::runtime_error(name + ": expected at least " + std::to_string(min_args_) +
                                   " args, got " + std::to_string(args.size()));
        if (args.size() > max_args_)
            throw std::runtime_error(name + ": expected at most " + std::to_string(max_args_) +
                                   " args, got " + std::to_string(args.size()));
        return impl(args);
    }

    size_t min_args() const override { return min_args_; }
    size_t max_args() const override { return max_args_; }
};

struct GUTRUserFunction : public GUTRFunction {
    std::vector<std::string> param_names;
    std::shared_ptr<GUTRBlockStmt> body;
    std::shared_ptr<GUTRContext> closure;
    GUTRValue call(const std::vector<GUTRValue>& args) override;
    size_t min_args() const override { return param_names.size(); }
    size_t max_args() const override { return param_names.size(); }
};

struct GUTRExpr {
    int line = 0;
    int col = 0;
    virtual ~GUTRExpr() = default;
    virtual GUTRValue eval(std::shared_ptr<GUTRContext> ctx) = 0;
    virtual std::string to_string() const { return "<expr>"; }
};

struct GUTRLiteralExpr : public GUTRExpr {
    GUTRValue value;
    explicit GUTRLiteralExpr(const GUTRValue& v);
    GUTRValue eval(std::shared_ptr<GUTRContext> ctx) override;
    std::string to_string() const override;
};

struct GUTRVarExpr : public GUTRExpr {
    std::string name;
    explicit GUTRVarExpr(const std::string& n);
    GUTRValue eval(std::shared_ptr<GUTRContext> ctx) override;
    std::string to_string() const override { return name; }
};

struct GUTRBinaryOpExpr : public GUTRExpr {
    enum Op { ADD, SUB, MUL, DIV, LT, GT, EQ, NE, LE, GE, AND, OR };
    Op op;
    std::shared_ptr<GUTRExpr> left, right;
    GUTRValue eval(std::shared_ptr<GUTRContext> ctx) override;
    std::string to_string() const override;
};

struct GUTRUnaryOpExpr : public GUTRExpr {
    enum Op { NEG, NOT };
    Op op;
    std::shared_ptr<GUTRExpr> arg;
    GUTRValue eval(std::shared_ptr<GUTRContext> ctx) override;
};

struct GUTRCallExpr : public GUTRExpr {
    std::shared_ptr<GUTRExpr> callee;
    std::vector<std::shared_ptr<GUTRExpr>> args;
    GUTRValue eval(std::shared_ptr<GUTRContext> ctx) override;
};

struct GUTRIndexExpr : public GUTRExpr {
    std::shared_ptr<GUTRExpr> object;
    std::shared_ptr<GUTRExpr> index;
    GUTRValue eval(std::shared_ptr<GUTRContext> ctx) override;
};

struct GUTRAssignExpr : public GUTRExpr {
    std::string var_name;
    std::shared_ptr<GUTRExpr> value;
    bool is_global = false;
    GUTRValue eval(std::shared_ptr<GUTRContext> ctx) override;
};

struct GUTRBlockStmt : public GUTRExpr {
    std::vector<std::shared_ptr<GUTRExpr>> statements;
    GUTRValue eval(std::shared_ptr<GUTRContext> ctx) override;
    void append(std::shared_ptr<GUTRExpr> stmt) { statements.push_back(stmt); }
};

struct GUTRIfStmt : public GUTRExpr {
    std::shared_ptr<GUTRExpr> condition;
    std::shared_ptr<GUTRExpr> then_branch;
    std::shared_ptr<GUTRExpr> else_branch;
    GUTRValue eval(std::shared_ptr<GUTRContext> ctx) override;
};

struct GUTRWhileStmt : public GUTRExpr {
    std::shared_ptr<GUTRExpr> condition;
    std::shared_ptr<GUTRExpr> body;
    GUTRValue eval(std::shared_ptr<GUTRContext> ctx) override;
};

struct GUTRForStmt : public GUTRExpr {
    std::string var_name;
    std::shared_ptr<GUTRExpr> start_expr;
    std::shared_ptr<GUTRExpr> end_expr;
    std::shared_ptr<GUTRExpr> step_expr;
    std::shared_ptr<GUTRExpr> body;
    GUTRValue eval(std::shared_ptr<GUTRContext> ctx) override;
};

struct GUTRReturnStmt : public GUTRExpr {
    std::shared_ptr<GUTRExpr> value;
    GUTRValue eval(std::shared_ptr<GUTRContext> ctx) override;
};

struct GUTRBreakStmt : public GUTRExpr {
    GUTRValue eval(std::shared_ptr<GUTRContext> ctx) override;
};

struct GUTRContinueStmt : public GUTRExpr {
    GUTRValue eval(std::shared_ptr<GUTRContext> ctx) override;
};

struct GUTRArrayLiteralExpr : public GUTRExpr {
    std::vector<std::shared_ptr<GUTRExpr>> elements;
    GUTRValue eval(std::shared_ptr<GUTRContext> ctx) override;
};

struct GUTRFunctionDefExpr : public GUTRExpr {
    std::string name;
    std::vector<std::string> params;
    std::shared_ptr<GUTRBlockStmt> body;
    GUTRValue eval(std::shared_ptr<GUTRContext> ctx) override;
};

struct GUTRSectDefExpr : public GUTRExpr {
    std::string name;
    std::vector<std::pair<std::string, std::shared_ptr<GUTRExpr>>> field_inits;
    GUTRValue eval(std::shared_ptr<GUTRContext> ctx) override;
};

struct GUTRContext : public std::enable_shared_from_this<GUTRContext> {
    std::unordered_map<std::string, GUTRValue> variables;
    std::shared_ptr<GUTRContext> parent;
    bool break_flag = false;
    bool continue_flag = false;
    GUTRValue return_value;
    std::string name;
    int depth = 0;

    explicit GUTRContext(std::shared_ptr<GUTRContext> p = nullptr);
    ~GUTRContext() = default;

    GUTRValue get(const std::string& name);
    void set(const std::string& name, const GUTRValue& value);
    void define_builtin(const std::string& name, GUTRBuiltinFunction::FuncType func);

    bool has(const std::string& name) const {
        return variables.find(name) != variables.end() || (parent && parent->has(name));
    }

    void define_builtin_advanced(const std::string& name,
                                 GUTRBuiltinFunction::FuncType func,
                                 size_t min_args = 0,
                                 size_t max_args = SIZE_MAX) {
        auto builtin = std::make_shared<GUTRBuiltinFunction>(func, name);
        builtin->set_args(min_args, max_args);
        set(name, GUTRValue::function(builtin));
    }

    std::shared_ptr<GUTRContext> create_child(const std::string& child_name = "") {
        auto child = std::make_shared<GUTRContext>(shared_from_this());
        child->name = child_name;
        child->depth = depth + 1;
        return child;
    }

    GUTRValue* find(const std::string& name) {
        auto it = variables.find(name);
        if (it != variables.end()) return &it->second;
        if (parent) return parent->find(name);
        return nullptr;
    }
};

class GUTRProgram {
public:
    GUTRProgram() = default;

    void set_global(const std::string& key, const GUTRValue& val) {
        if (global_ctx) global_ctx->set(key, val);
    }

    GUTRValue get_global(const std::string& key) const {
        if (global_ctx) return global_ctx->get(key);
        return GUTRValue::nil();
    }

    void init(const std::string& config_json);
    void train_step(const std::vector<int>& input, const std::vector<int>& target, float lr);
    void save(const std::string& path);
    void load(const std::string& path);
    int generate(const std::vector<int>& prompt, std::vector<int>& output, int max_tokens, float temp, float top_p);

    bool is_ready() const {
        return global_ctx && init_func && train_step_func && save_func && load_func && generate_func;
    }

private:
    std::shared_ptr<GUTRContext> global_ctx;
    std::shared_ptr<GUTRFunction> init_func;
    std::shared_ptr<GUTRFunction> train_step_func;
    std::shared_ptr<GUTRFunction> save_func;
    std::shared_ptr<GUTRFunction> load_func;
    std::shared_ptr<GUTRFunction> generate_func;
    friend class GUTRParser;
};

void register_builtins(std::shared_ptr<GUTRContext> ctx);

} // namespace uzaleat

#endif
