#ifndef GUTR_VM_HPP
#define GUTR_VM_HPP

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <cstdint>
#include <functional>
#include "gutr_parser.hpp"

namespace uzaleat {

enum class ValueType { NIL, BOOL, INT, FLOAT, STRING, TENSOR, FUNCTION, TUPLE, SECT };

struct Tensor {
    std::vector<size_t> shape;
    std::vector<float> data;
    std::vector<float> grad;
    bool requires_grad;
    Tensor();
    explicit Tensor(const std::vector<size_t>& shape, bool requires_grad = true);
    Tensor(const std::vector<size_t>& shape, float fill_value, bool requires_grad = true);
    size_t size() const;
    void zero_grad();
    void fill(float value);
    void random_normal(float mean = 0.0f, float stddev = 0.02f);
    void random_uniform(float min = -0.1f, float max = 0.1f);
    void from_array(const std::vector<float>& arr);
};

struct GUTRValue {
    ValueType type;
    bool bool_val;
    int64_t int_val;
    double float_val;
    std::string string_val;
    std::shared_ptr<Tensor> tensor_val;
    std::vector<GUTRValue> tuple_val;
    std::shared_ptr<struct GUTRFunction> func_val;
    std::unordered_map<std::string, GUTRValue> sect_fields;
    GUTRValue();
    static GUTRValue nil();
    static GUTRValue boolean(bool v);
    static GUTRValue integer(int64_t v);
    static GUTRValue real(double v);
    static GUTRValue string(const std::string& v);
    static GUTRValue tensor(std::shared_ptr<Tensor> t);
    static GUTRValue tuple(const std::vector<GUTRValue>& v);
    static GUTRValue function(std::shared_ptr<struct GUTRFunction> f);
    static GUTRValue sect(const std::unordered_map<std::string, GUTRValue>& fields);
};

struct GUTRFunction {
    virtual ~GUTRFunction() = default;
    virtual GUTRValue call(const std::vector<GUTRValue>& args) = 0;
};

struct GUTRBuiltinFunction : public GUTRFunction {
    using FuncType = std::function<GUTRValue(const std::vector<GUTRValue>&)>;
    FuncType impl;
    GUTRBuiltinFunction(FuncType f) : impl(f) {}
    GUTRValue call(const std::vector<GUTRValue>& args) override { return impl(args); }
};

struct GUTRUserFunction : public GUTRFunction {
    std::vector<std::string> param_names;
    std::shared_ptr<struct GUTRBlockStmt> body;
    std::shared_ptr<struct GUTRContext> closure;
    GUTRValue call(const std::vector<GUTRValue>& args) override;
};

struct GUTRExpr {
    virtual ~GUTRExpr() = default;
    virtual GUTRValue eval(std::shared_ptr<struct GUTRContext> ctx) = 0;
};

struct GUTRLiteralExpr : public GUTRExpr {
    GUTRValue value;
    explicit GUTRLiteralExpr(const GUTRValue& v);
    GUTRValue eval(std::shared_ptr<GUTRContext> ctx) override;
};

struct GUTRVarExpr : public GUTRExpr {
    std::string name;
    explicit GUTRVarExpr(const std::string& n);
    GUTRValue eval(std::shared_ptr<GUTRContext> ctx) override;
};

struct GUTRBinaryOpExpr : public GUTRExpr {
    enum Op { ADD, SUB, MUL, DIV, LT, GT, EQ, NE, LE, GE, AND, OR };
    Op op;
    std::shared_ptr<GUTRExpr> left, right;
    GUTRValue eval(std::shared_ptr<GUTRContext> ctx) override;
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
    GUTRValue eval(std::shared_ptr<GUTRContext> ctx) override;
};

struct GUTRBlockStmt : public GUTRExpr {
    std::vector<std::shared_ptr<GUTRExpr>> statements;
    GUTRValue eval(std::shared_ptr<GUTRContext> ctx) override;
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
    bool break_flag;
    bool continue_flag;
    GUTRValue return_value;
    explicit GUTRContext(std::shared_ptr<GUTRContext> p = nullptr);
    GUTRValue get(const std::string& name);
    void set(const std::string& name, const GUTRValue& value);
    void define_builtin(const std::string& name, GUTRBuiltinFunction::FuncType func);
};

class GUTRProgram {
public:
    void init(const std::string& config_json);
    void train_step(const std::vector<int>& input, const std::vector<int>& target, float lr);
    void save(const std::string& path);
    void load(const std::string& path);
    int generate(const std::vector<int>& prompt, std::vector<int>& output, int max_tokens, float temp, float top_p);
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
