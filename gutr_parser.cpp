// файл: gutr_parser.cpp
#include "gutr_parser.hpp"
#include "gutr_vm.hpp"
#include <iostream>
#include <sstream>
#include <cctype>
#include <vector>
#include <memory>
#include <stdexcept>
#include <fstream>

namespace uzaleat {

// ============================================================================
// Токены
// ============================================================================

enum class TokenType {
    END, IDENT, NUMBER, STRING, KEYWORD,
    LPAREN, RPAREN, LBRACE, RBRACE, LBRACK, RBRACK,
    COMMA, SEMICOLON, COLON, DOT,
    PLUS, MINUS, STAR, SLASH, EQ, NE, LT, GT, LE, GE, BANG,
    AND, OR, ASSIGN, ARROW
};

struct Token {
    TokenType type;
    std::string text;
    int line, col;
    Token() : type(TokenType::END), line(0), col(0) {}
    Token(TokenType t, const std::string& txt, int l, int c) : type(t), text(txt), line(l), col(c) {}
};

// ============================================================================
// Лексер с комментариями и позициями
// ============================================================================

class Lexer {
    std::string src;
    size_t pos;
    int line, col;

    void advance() {
        if (pos < src.size()) {
            if (src[pos] == '\n') { line++; col = 1; }
            else col++;
            pos++;
        }
    }

    char peek() const { return pos < src.size() ? src[pos] : '\0'; }

    void skipWhitespace() {
        while (pos < src.size() && std::isspace(src[pos])) {
            advance();
        }
    }

    void skipComment() {
        if (pos + 1 >= src.size()) return;
        if (src[pos] == '/' && src[pos+1] == '/') {
            while (pos < src.size() && src[pos] != '\n') advance();
            advance(); // пропустить \n
        }
        else if (src[pos] == '/' && src[pos+1] == '*') {
            advance(); advance();
            while (pos + 1 < src.size() && !(src[pos] == '*' && src[pos+1] == '/')) {
                advance();
            }
            advance(); advance();
        }
    }

    Token readString() {
        int start_line = line, start_col = col;
        advance(); // пропустить "
        std::string str;
        while (pos < src.size() && src[pos] != '"') {
            if (src[pos] == '\\') {
                advance();
                if (pos < src.size()) str += src[pos];
                advance();
            } else {
                str += src[pos];
                advance();
            }
        }
        if (pos >= src.size()) throw std::runtime_error("unclosed string");
        advance(); // пропустить закрывающую "
        return Token(TokenType::STRING, str, start_line, start_col);
    }

    Token readNumber() {
        int start_line = line, start_col = col;
        size_t start = pos;
        while (pos < src.size() && (std::isdigit(src[pos]) || src[pos] == '.')) advance();
        std::string num = src.substr(start, pos - start);
        return Token(TokenType::NUMBER, num, start_line, start_col);
    }

    Token readIdent() {
        int start_line = line, start_col = col;
        size_t start = pos;
        while (pos < src.size() && (std::isalnum(src[pos]) || src[pos] == '_')) advance();
        std::string ident = src.substr(start, pos - start);
        if (ident == "fn" || ident == "sect" || ident == "export" ||
            ident == "if" || ident == "else" || ident == "while" ||
            ident == "for" || ident == "return" || ident == "break" ||
            ident == "continue" || ident == "true" || ident == "false" ||
            ident == "nil" || ident == "u64" || ident == "f64" ||
            ident == "bool" || ident == "tensor" || ident == "list") {
            return Token(TokenType::KEYWORD, ident, start_line, start_col);
        }
        return Token(TokenType::IDENT, ident, start_line, start_col);
    }

public:
    Lexer(const std::string& s) : src(s), pos(0), line(1), col(1) {}

    Token next() {
        skipWhitespace();

        // Пропуск комментариев
        while (pos + 1 < src.size() && src[pos] == '/' && (src[pos+1] == '/' || src[pos+1] == '*')) {
            skipComment();
            skipWhitespace();
        }

        if (pos >= src.size()) {
            return Token(TokenType::END, "", line, col);
        }

        char c = src[pos];

        // Односимвольные токены
        if (c == '(') { advance(); return Token(TokenType::LPAREN, "(", line, col-1); }
        if (c == ')') { advance(); return Token(TokenType::RPAREN, ")", line, col-1); }
        if (c == '{') { advance(); return Token(TokenType::LBRACE, "{", line, col-1); }
        if (c == '}') { advance(); return Token(TokenType::RBRACE, "}", line, col-1); }
        if (c == '[') { advance(); return Token(TokenType::LBRACK, "[", line, col-1); }
        if (c == ']') { advance(); return Token(TokenType::RBRACK, "]", line, col-1); }
        if (c == ',') { advance(); return Token(TokenType::COMMA, ",", line, col-1); }
        if (c == ';') { advance(); return Token(TokenType::SEMICOLON, ";", line, col-1); }
        if (c == ':') { advance(); return Token(TokenType::COLON, ":", line, col-1); }
        if (c == '.') { advance(); return Token(TokenType::DOT, ".", line, col-1); }
        if (c == '+') { advance(); return Token(TokenType::PLUS, "+", line, col-1); }
        if (c == '-') {
            advance();
            if (c == '-' && peek() == '>') { advance(); return Token(TokenType::ARROW, "->", line, col-2); }
            return Token(TokenType::MINUS, "-", line, col-1);
        }
        if (c == '*') { advance(); return Token(TokenType::STAR, "*", line, col-1); }
        if (c == '/') { advance(); return Token(TokenType::SLASH, "/", line, col-1); }
        if (c == '=') {
            advance();
            if (peek() == '=') { advance(); return Token(TokenType::EQ, "==", line, col-2); }
            return Token(TokenType::ASSIGN, "=", line, col-1);
        }
        if (c == '<') {
            advance();
            if (peek() == '=') { advance(); return Token(TokenType::LE, "<=", line, col-2); }
            return Token(TokenType::LT, "<", line, col-1);
        }
        if (c == '>') {
            advance();
            if (peek() == '=') { advance(); return Token(TokenType::GE, ">=", line, col-2); }
            return Token(TokenType::GT, ">", line, col-1);
        }
        if (c == '!') {
            advance();
            if (peek() == '=') { advance(); return Token(TokenType::NE, "!=", line, col-2); }
            return Token(TokenType::BANG, "!", line, col-1);
        }
        if (c == '&' && peek() == '&') { advance(); advance(); return Token(TokenType::AND, "&&", line, col-2); }
        if (c == '|' && peek() == '|') { advance(); advance(); return Token(TokenType::OR, "||", line, col-2); }

        if (c == '"') return readString();
        if (std::isdigit(c) || (c == '.' && std::isdigit(peek()))) return readNumber();
        if (std::isalpha(c) || c == '_') return readIdent();

        throw std::runtime_error(std::string("unexpected character: ") + c);
    }
};

// ============================================================================
// Парсер с поддержкой присваивания (исправленный)
// ============================================================================

class Parser {
    Lexer lexer;
    Token current;
    int errors = 0;

    void next() {
        try {
            current = lexer.next();
        } catch (const std::exception& e) {
            error(e.what());
        }
    }

    bool match(TokenType t) {
        if (current.type == t) {
            next();
            return true;
        }
        return false;
    }

    void expect(TokenType t, const std::string& msg = "") {
        if (current.type != t) {
            std::string err = msg.empty() ?
                std::string("expected ") + token_to_string(t) : msg;
            error(err);
        }
        next();
    }

public:
    Parser(const std::string& src) : lexer(src) { next(); }

    static std::string token_to_string(TokenType t) {
        switch (t) {
            case TokenType::END: return "end of file";
            case TokenType::IDENT: return "identifier";
            case TokenType::NUMBER: return "number";
            case TokenType::STRING: return "string";
            case TokenType::KEYWORD: return "keyword";
            case TokenType::LPAREN: return "(";
            case TokenType::RPAREN: return ")";
            case TokenType::LBRACE: return "{";
            case TokenType::RBRACE: return "}";
            case TokenType::LBRACK: return "[";
            case TokenType::RBRACK: return "]";
            case TokenType::COMMA: return ",";
            case TokenType::SEMICOLON: return ";";
            case TokenType::COLON: return ":";
            case TokenType::DOT: return ".";
            case TokenType::PLUS: return "+";
            case TokenType::MINUS: return "-";
            case TokenType::STAR: return "*";
            case TokenType::SLASH: return "/";
            case TokenType::EQ: return "==";
            case TokenType::NE: return "!=";
            case TokenType::LT: return "<";
            case TokenType::GT: return ">";
            case TokenType::LE: return "<=";
            case TokenType::GE: return ">=";
            case TokenType::BANG: return "!";
            case TokenType::AND: return "&&";
            case TokenType::OR: return "||";
            case TokenType::ASSIGN: return "=";
            case TokenType::ARROW: return "->";
            default: return "unknown";
        }
    }

    void error(const std::string& msg) {
        std::cerr << "\n[GUTR PARSER ERROR] " << msg << std::endl;
        std::cerr << "  at line " << current.line << ", column " << current.col << std::endl;
        std::cerr << "  current token: " << token_to_string(current.type);
        if (!current.text.empty()) std::cerr << " ('" << current.text << "')";
        std::cerr << std::endl;
        errors++;
        throw std::runtime_error("Parse failed");
    }

    std::shared_ptr<GUTRExpr> parse() {
        auto prog = std::make_shared<GUTRBlockStmt>();
        while (current.type != TokenType::END) {
            prog->statements.push_back(parseStatement());
        }
        return prog;
    }

private:
    std::shared_ptr<GUTRExpr> parseStatement() {
        // ----- Ключевые слова -----
        if (current.type == TokenType::KEYWORD && current.text == "fn")
            return parseFunctionDef();
        if (current.type == TokenType::KEYWORD && current.text == "sect")
            return parseSectDef();
        if (current.type == TokenType::KEYWORD && current.text == "export") {
            next();
            auto stmt = parseStatement();
            return stmt;
        }
        if (current.type == TokenType::KEYWORD && current.text == "if")
            return parseIf();
        if (current.type == TokenType::KEYWORD && current.text == "while")
            return parseWhile();
        if (current.type == TokenType::KEYWORD && current.text == "for")
            return parseFor();
        if (current.type == TokenType::KEYWORD && current.text == "return")
            return parseReturn();
        if (current.type == TokenType::KEYWORD && current.text == "break") {
            next();
            return std::make_shared<GUTRBreakStmt>();
        }
        if (current.type == TokenType::KEYWORD && current.text == "continue") {
            next();
            return std::make_shared<GUTRContinueStmt>();
        }

        // ----- Блок { ... } -----
        if (current.type == TokenType::LBRACE) {
            next();
            auto block = std::make_shared<GUTRBlockStmt>();
            while (current.type != TokenType::RBRACE && current.type != TokenType::END) {
                block->statements.push_back(parseStatement());
            }
            expect(TokenType::RBRACE);
            return block;
        }

        // ----- ПРИСВАИВАНИЕ: проверяем, что это идентификатор, за которым следует '=' -----
        if (current.type == TokenType::IDENT) {
            // Сохраняем позицию для отката
            Token save = current;
            // Заглядываем на один токен вперёд
            next();
            if (current.type == TokenType::ASSIGN) {
                // Это присваивание
                std::string var_name = save.text;
                next(); // пропускаем '='
                auto expr = parseExpression();
                auto assign = std::make_shared<GUTRAssignExpr>();
                assign->var_name = var_name;
                assign->value = expr;
                // Точка с запятой необязательна
                if (current.type == TokenType::SEMICOLON) next();
                return assign;
            } else {
                // Не присваивание — возвращаемся и идём дальше
                current = save;
            }
        }

        // ----- Выражение как оператор (вызов функции, операция) -----
        auto expr = parseExpression();
        if (current.type == TokenType::SEMICOLON) next();
        return expr;
    }

    std::shared_ptr<GUTRExpr> parseExpression() {
        return parseBinary(0);
    }

    int precedence(TokenType t) {
        switch (t) {
            case TokenType::OR: return 1;
            case TokenType::AND: return 2;
            case TokenType::EQ: case TokenType::NE: return 3;
            case TokenType::LT: case TokenType::GT: case TokenType::LE: case TokenType::GE: return 4;
            case TokenType::PLUS: case TokenType::MINUS: return 5;
            case TokenType::STAR: case TokenType::SLASH: return 6;
            default: return 0;
        }
    }

    std::shared_ptr<GUTRExpr> parseBinary(int min_prec) {
        auto left = parsePrimary();
        while (true) {
            int prec = precedence(current.type);
            if (prec < min_prec) break;
            // Присваивание не может быть в бинарном выражении
            if (current.type == TokenType::ASSIGN) {
                error("assignment operator cannot be used in expressions (use separate statement)");
            }
            TokenType op = current.type;
            next();
            auto right = parseBinary(prec + 1);
            auto bin = std::make_shared<GUTRBinaryOpExpr>();
            switch (op) {
                case TokenType::PLUS: bin->op = GUTRBinaryOpExpr::ADD; break;
                case TokenType::MINUS: bin->op = GUTRBinaryOpExpr::SUB; break;
                case TokenType::STAR: bin->op = GUTRBinaryOpExpr::MUL; break;
                case TokenType::SLASH: bin->op = GUTRBinaryOpExpr::DIV; break;
                case TokenType::LT: bin->op = GUTRBinaryOpExpr::LT; break;
                case TokenType::GT: bin->op = GUTRBinaryOpExpr::GT; break;
                case TokenType::EQ: bin->op = GUTRBinaryOpExpr::EQ; break;
                case TokenType::NE: bin->op = GUTRBinaryOpExpr::NE; break;
                case TokenType::LE: bin->op = GUTRBinaryOpExpr::LE; break;
                case TokenType::GE: bin->op = GUTRBinaryOpExpr::GE; break;
                case TokenType::AND: bin->op = GUTRBinaryOpExpr::AND; break;
                case TokenType::OR: bin->op = GUTRBinaryOpExpr::OR; break;
                default: error("unknown binary operator");
            }
            bin->left = left;
            bin->right = right;
            left = bin;
        }
        return left;
    }

    std::shared_ptr<GUTRExpr> parsePrimary() {
        if (match(TokenType::LPAREN)) {
            auto expr = parseExpression();
            expect(TokenType::RPAREN);
            return expr;
        }
        if (match(TokenType::NUMBER)) {
            std::string num = current.text;
            if (num.find('.') != std::string::npos)
                return std::make_shared<GUTRLiteralExpr>(GUTRValue::real(std::stod(num)));
            else
                return std::make_shared<GUTRLiteralExpr>(GUTRValue::integer(std::stoll(num)));
        }
        if (match(TokenType::STRING))
            return std::make_shared<GUTRLiteralExpr>(GUTRValue::string(current.text));
        if (current.type == TokenType::KEYWORD) {
            if (current.text == "true") {
                next();
                return std::make_shared<GUTRLiteralExpr>(GUTRValue::boolean(true));
            }
            if (current.text == "false") {
                next();
                return std::make_shared<GUTRLiteralExpr>(GUTRValue::boolean(false));
            }
            if (current.text == "nil") {
                next();
                return std::make_shared<GUTRLiteralExpr>(GUTRValue::nil());
            }
        }
        if (current.type == TokenType::IDENT) {
            std::string id = current.text;
            next();
            if (current.type == TokenType::LPAREN) {
                auto call = std::make_shared<GUTRCallExpr>();
                call->callee = std::make_shared<GUTRVarExpr>(id);
                next();
                while (current.type != TokenType::RPAREN && current.type != TokenType::END) {
                    call->args.push_back(parseExpression());
                    if (current.type == TokenType::COMMA) next();
                }
                expect(TokenType::RPAREN);
                return call;
            }
            if (current.type == TokenType::LBRACK) {
                auto idx = std::make_shared<GUTRIndexExpr>();
                idx->object = std::make_shared<GUTRVarExpr>(id);
                next();
                idx->index = parseExpression();
                expect(TokenType::RBRACK);
                return idx;
            }
            return std::make_shared<GUTRVarExpr>(id);
        }
        if (match(TokenType::MINUS)) {
            auto unary = std::make_shared<GUTRUnaryOpExpr>();
            unary->op = GUTRUnaryOpExpr::NEG;
            unary->arg = parsePrimary();
            return unary;
        }
        if (match(TokenType::BANG)) {
            auto unary = std::make_shared<GUTRUnaryOpExpr>();
            unary->op = GUTRUnaryOpExpr::NOT;
            unary->arg = parsePrimary();
            return unary;
        }
        // Литерал массива [1, 2, 3]
        if (match(TokenType::LBRACK)) {
            auto array_node = std::make_shared<GUTRArrayLiteralExpr>();
            while (current.type != TokenType::RBRACK && current.type != TokenType::END) {
                array_node->elements.push_back(parseExpression());
                if (current.type == TokenType::COMMA) next();
            }
            expect(TokenType::RBRACK);
            return array_node;
        }
        error("unexpected token in expression");
        return nullptr;
    }

    std::shared_ptr<GUTRFunctionDefExpr> parseFunctionDef() {
        expect(TokenType::KEYWORD, "expected 'fn'");
        std::string name = current.text;
        expect(TokenType::IDENT, "expected function name");
        expect(TokenType::LPAREN, "expected '(' after function name");

        std::vector<std::string> params;
        while (current.type != TokenType::RPAREN && current.type != TokenType::END) {
            if (current.type == TokenType::KEYWORD &&
                (current.text == "u64" || current.text == "f64" ||
                 current.text == "bool" || current.text == "tensor")) {
                next();
            }
            params.push_back(current.text);
            expect(TokenType::IDENT, "expected parameter name");
            if (current.type == TokenType::COMMA) next();
        }
        expect(TokenType::RPAREN);

        if (current.type == TokenType::ARROW) {
            next();
            if (current.type == TokenType::LPAREN) {
                next();
                while (current.type != TokenType::RPAREN) next();
                next();
            } else {
                next();
            }
        }

        auto body = std::dynamic_pointer_cast<GUTRBlockStmt>(parseStatement());
        auto def = std::make_shared<GUTRFunctionDefExpr>();
        def->name = name;
        def->params = params;
        def->body = body;
        return def;
    }

    std::shared_ptr<GUTRSectDefExpr> parseSectDef() {
        expect(TokenType::KEYWORD, "expected 'sect'");
        std::string name = current.text;
        expect(TokenType::IDENT, "expected section name");
        expect(TokenType::LBRACE, "expected '{'");

        std::vector<std::pair<std::string, std::shared_ptr<GUTRExpr>>> fields;
        while (current.type != TokenType::RBRACE && current.type != TokenType::END) {
            std::string fname = current.text;
            expect(TokenType::IDENT, "expected field name");
            expect(TokenType::ASSIGN, "expected '='");
            auto expr = parseExpression();
            fields.emplace_back(fname, expr);
            if (current.type == TokenType::SEMICOLON) next();
        }
        expect(TokenType::RBRACE);

        auto sect = std::make_shared<GUTRSectDefExpr>();
        sect->name = name;
        sect->field_inits = fields;
        return sect;
    }

    std::shared_ptr<GUTRIfStmt> parseIf() {
        expect(TokenType::KEYWORD, "expected 'if'");
        auto cond = parseExpression();
        auto then_branch = parseStatement();
        std::shared_ptr<GUTRExpr> else_branch;
        if (current.type == TokenType::KEYWORD && current.text == "else") {
            next();
            else_branch = parseStatement();
        }
        auto ifstmt = std::make_shared<GUTRIfStmt>();
        ifstmt->condition = cond;
        ifstmt->then_branch = then_branch;
        ifstmt->else_branch = else_branch;
        return ifstmt;
    }

    std::shared_ptr<GUTRWhileStmt> parseWhile() {
        expect(TokenType::KEYWORD, "expected 'while'");
        auto cond = parseExpression();
        auto body = parseStatement();
        auto w = std::make_shared<GUTRWhileStmt>();
        w->condition = cond;
        w->body = body;
        return w;
    }

    std::shared_ptr<GUTRForStmt> parseFor() {
        expect(TokenType::KEYWORD, "expected 'for'");
        std::string var = current.text;
        expect(TokenType::IDENT, "expected loop variable");
        expect(TokenType::ASSIGN, "expected '='");
        auto start = parseExpression();
        expect(TokenType::SEMICOLON);
        auto end = parseExpression();
        expect(TokenType::SEMICOLON);
        auto step = parseExpression();
        auto body = parseStatement();
        auto f = std::make_shared<GUTRForStmt>();
        f->var_name = var;
        f->start_expr = start;
        f->end_expr = end;
        f->step_expr = step;
        f->body = body;
        return f;
    }

    std::shared_ptr<GUTRReturnStmt> parseReturn() {
        expect(TokenType::KEYWORD, "expected 'return'");
        auto r = std::make_shared<GUTRReturnStmt>();
        if (current.type != TokenType::RBRACE && current.type != TokenType::SEMICOLON &&
            current.type != TokenType::END) {
            r->value = parseExpression();
        }
        return r;
    }
};

// ============================================================================
// Основной парсер
// ============================================================================

bool GUTRParser::parse(const std::string& source, GUTRProgram& program) {
    try {
        Parser parser(source);
        auto ast = parser.parse();
        auto ctx = std::make_shared<GUTRContext>();
        register_builtins(ctx);
        ast->eval(ctx);

        auto get_func = [&](const std::string& name) -> std::shared_ptr<GUTRFunction> {
            GUTRValue v = ctx->get(name);
            if (v.type != ValueType::FUNCTION)
                throw std::runtime_error("function not found: " + name);
            return v.func_val;
        };

        program.init_func = get_func("init");
        program.train_step_func = get_func("train_step");
        program.save_func = get_func("save");
        program.load_func = get_func("load");
        program.generate_func = get_func("generate");
        program.global_ctx = ctx;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "\n[GUTR] Parse failed: " << e.what() << std::endl;
        return false;
    }
}

} // namespace uzaleat
