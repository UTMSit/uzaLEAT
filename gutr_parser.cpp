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
#include <unordered_set>

namespace uzaleat {

enum class TokenType {
    END, IDENT, NUMBER, STRING, KEYWORD,
    LPAREN, RPAREN, LBRACE, RBRACE, LBRACK, RBRACK,
    COMMA, SEMICOLON, COLON, DOT,
    PLUS, MINUS, STAR, SLASH, EQ, NE, LT, GT, LE, GE, BANG,
    AND, OR, ASSIGN, ARROW
};

struct Token {
    TokenType type = TokenType::END;
    std::string text;
    int line = 0;
    int col = 0;
    Token() : type(TokenType::END), line(0), col(0) {}
    Token(TokenType t, const std::string& txt, int l, int c) : type(t), text(txt), line(l), col(c) {}
};

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
        while (pos < src.size() && std::isspace(src[pos])) advance();
    }

    void skipComment() {
        if (pos + 1 >= src.size()) return;
        if (src[pos] == '/' && src[pos+1] == '/') {
            while (pos < src.size() && src[pos] != '\n') advance();
            if (pos < src.size()) advance();
        }
        else if (src[pos] == '/' && src[pos+1] == '*') {
            advance(); advance();
            while (pos + 1 < src.size() && !(src[pos] == '*' && src[pos+1] == '/')) advance();
            if (pos + 1 < src.size()) { advance(); advance(); }
        }
    }

    Token readString() {
        int start_line = line, start_col = col;
        advance();
        std::string str;
        while (pos < src.size()) {
            if (src[pos] == '"') { advance(); return Token(TokenType::STRING, str, start_line, start_col); }
            if (src[pos] == '\\') {
                advance();
                if (pos < src.size()) {
                    switch (src[pos]) {
                        case 'n': str += '\n'; break;
                        case 't': str += '\t'; break;
                        case 'r': str += '\r'; break;
                        case '\\': str += '\\'; break;
                        case '"': str += '"'; break;
                        default: str += src[pos]; break;
                    }
                    advance();
                }
            } else { str += src[pos]; advance(); }
        }
        throw std::runtime_error("unclosed string at line " + std::to_string(start_line));
    }

    Token readNumber() {
        int start_line = line, start_col = col;
        size_t start = pos;
        bool has_dot = false;
        while (pos < src.size() && (std::isdigit(src[pos]) || (src[pos] == '.' && !has_dot))) {
            if (src[pos] == '.') has_dot = true;
            advance();
        }
        if (pos < src.size() && (src[pos] == 'e' || src[pos] == 'E')) {
            advance();
            if (pos < src.size() && (src[pos] == '+' || src[pos] == '-')) advance();
            while (pos < src.size() && std::isdigit(src[pos])) advance();
        }
        std::string num = src.substr(start, pos - start);
        if (num.empty()) throw std::runtime_error("invalid number at line " + std::to_string(start_line));
        return Token(TokenType::NUMBER, num, start_line, start_col);
    }

    Token readIdent() {
        int start_line = line, start_col = col;
        size_t start = pos;
        while (pos < src.size() && (std::isalnum(src[pos]) || src[pos] == '_')) advance();
        std::string ident = src.substr(start, pos - start);
        if (ident.empty()) throw std::runtime_error("empty identifier at line " + std::to_string(start_line));

        static const std::unordered_set<std::string> keywords = {
            "fn", "sect", "export", "if", "else", "while", "for", "return",
            "break", "continue", "true", "false", "nil", "u64", "f64",
            "bool", "and", "or", "not", "let", "var"
        };

        if (keywords.find(ident) != keywords.end())
            return Token(TokenType::KEYWORD, ident, start_line, start_col);
        return Token(TokenType::IDENT, ident, start_line, start_col);
    }

public:
    Lexer(const std::string& s) : src(s), pos(0), line(1), col(1) {
        if (s.empty()) throw std::runtime_error("Empty source code");
    }

    Token next() {
        while (true) {
            skipWhitespace();
            if (pos + 1 < src.size() && src[pos] == '/' && (src[pos+1] == '/' || src[pos+1] == '*')) {
                skipComment();
                continue;
            }
            if (pos >= src.size()) return Token(TokenType::END, "", line, col);

            char c = src[pos];

            // Двухсимвольные токены
            if (c == '=' && pos+1 < src.size() && src[pos+1]=='=') { advance(); advance(); return Token(TokenType::EQ, "==", line, col-2); }
            if (c == '!' && pos+1 < src.size() && src[pos+1]=='=') { advance(); advance(); return Token(TokenType::NE, "!=", line, col-2); }
            if (c == '<' && pos+1 < src.size() && src[pos+1]=='=') { advance(); advance(); return Token(TokenType::LE, "<=", line, col-2); }
            if (c == '>' && pos+1 < src.size() && src[pos+1]=='=') { advance(); advance(); return Token(TokenType::GE, ">=", line, col-2); }
            if (c == '&' && pos+1 < src.size() && src[pos+1]=='&') { advance(); advance(); return Token(TokenType::AND, "&&", line, col-2); }
            if (c == '|' && pos+1 < src.size() && src[pos+1]=='|') { advance(); advance(); return Token(TokenType::OR, "||", line, col-2); }
            if (c == '-' && pos+1 < src.size() && src[pos+1]=='>') { advance(); advance(); return Token(TokenType::ARROW, "->", line, col-2); }

            // Одиночные токены
            switch (c) {
                case '(': advance(); return Token(TokenType::LPAREN, "(", line, col-1);
                case ')': advance(); return Token(TokenType::RPAREN, ")", line, col-1);
                case '{': advance(); return Token(TokenType::LBRACE, "{", line, col-1);
                case '}': advance(); return Token(TokenType::RBRACE, "}", line, col-1);
                case '[': advance(); return Token(TokenType::LBRACK, "[", line, col-1);
                case ']': advance(); return Token(TokenType::RBRACK, "]", line, col-1);
                case ',': advance(); return Token(TokenType::COMMA, ",", line, col-1);
                case ';': advance(); return Token(TokenType::SEMICOLON, ";", line, col-1);
                case ':': advance(); return Token(TokenType::COLON, ":", line, col-1);
                case '.': advance(); return Token(TokenType::DOT, ".", line, col-1);
                case '+': advance(); return Token(TokenType::PLUS, "+", line, col-1);
                case '-': advance(); return Token(TokenType::MINUS, "-", line, col-1);
                case '*': advance(); return Token(TokenType::STAR, "*", line, col-1);
                case '/': advance(); return Token(TokenType::SLASH, "/", line, col-1);
                case '=': advance(); return Token(TokenType::ASSIGN, "=", line, col-1);
                case '<': advance(); return Token(TokenType::LT, "<", line, col-1);
                case '>': advance(); return Token(TokenType::GT, ">", line, col-1);
                case '!': advance(); return Token(TokenType::BANG, "!", line, col-1);
                case '"': return readString();
            }

            if (std::isdigit(c) || (c == '.' && pos+1 < src.size() && std::isdigit(src[pos+1]))) return readNumber();
            if (std::isalpha(c) || c == '_') return readIdent();

            throw std::runtime_error(std::string("unexpected character: '") + c + "' at line " + std::to_string(line));
        }
    }
};

class Parser {
    Lexer lexer;
    Token current;
    Token peek_token;
    bool has_peek = false;
    int errors = 0;
    const int MAX_ERRORS = 100;

    Token peek() {
        if (!has_peek) {
            try { peek_token = lexer.next(); }
            catch (const std::exception& e) { error(e.what()); }
            has_peek = true;
        }
        return peek_token;
    }

    void next() {
        if (has_peek) {
            current = peek_token;
            has_peek = false;
        } else {
            try { current = lexer.next(); }
            catch (const std::exception& e) { error(e.what()); }
        }
    }

    bool match(TokenType t) {
        if (current.type == t) { next(); return true; }
        return false;
    }

    void expect(TokenType t, const std::string& msg = "") {
        if (current.type != t) {
            std::string err = msg.empty() ?
                std::string("expected ") + token_to_string(t) + " but got " + token_to_string(current.type) : msg;
            error(err);
        }
        if (current.type == t) next();
    }

public:
    Parser(const std::string& src) : lexer(src) {
        if (src.empty()) throw std::runtime_error("Empty source code");
        next();
    }

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
        if (errors >= MAX_ERRORS)
            throw std::runtime_error("Too many parse errors");
    }

    std::shared_ptr<GUTRBlockStmt> parse() {
        auto prog = std::make_shared<GUTRBlockStmt>();
        while (current.type != TokenType::END) {
            auto stmt = parseStatement();
            if (stmt) prog->statements.push_back(stmt);
            if (current.type == TokenType::RBRACE) break;
        }
        return prog;
    }

private:
    std::shared_ptr<GUTRExpr> parseStatement() {
        // Проверка на закрытие блока
        if (current.type == TokenType::RBRACE) return nullptr;
        if (current.type == TokenType::END) return nullptr;

        // Ключевые слова
        if (current.type == TokenType::KEYWORD) {
            std::string kw = current.text;
            if (kw == "fn") return parseFunctionDef();
            if (kw == "sect") return parseSectDef();
            if (kw == "export") { next(); return parseStatement(); }
            if (kw == "if") return parseIf();
            if (kw == "while") return parseWhile();
            if (kw == "for") return parseFor();
            if (kw == "return") return parseReturn();
            if (kw == "break") { next(); return std::make_shared<GUTRBreakStmt>(); }
            if (kw == "continue") { next(); return std::make_shared<GUTRContinueStmt>(); }
            // Пропускаем let/var и обрабатываем как присваивание
            if (kw == "let" || kw == "var") next();
        }

        // Блок
        if (current.type == TokenType::LBRACE) {
            next();
            auto block = std::make_shared<GUTRBlockStmt>();
            while (current.type != TokenType::RBRACE && current.type != TokenType::END) {
                auto stmt = parseStatement();
                if (stmt) block->statements.push_back(stmt);
            }
            expect(TokenType::RBRACE, "Expected '}' to close block");
            return block;
        }

        // Присваивание: IDENT = expr
        if (current.type == TokenType::IDENT) {
            Token save = current;
            next();
            if (current.type == TokenType::ASSIGN) {
                std::string var_name = save.text;
                next();
                auto expr = parseExpression();
                auto assign = std::make_shared<GUTRAssignExpr>();
                assign->var_name = var_name;
                assign->value = expr;
                if (current.type == TokenType::SEMICOLON) next();
                return assign;
            }
            // Возвращаем токен обратно
            current = save;
            // Продолжаем как выражение
        }

        // Выражение
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
        auto left = parseUnary();
        if (!left) return nullptr;

        while (true) {
            // Останавливаемся на разделителях
            if (current.type == TokenType::RBRACE || current.type == TokenType::END ||
                current.type == TokenType::SEMICOLON || current.type == TokenType::COMMA ||
                current.type == TokenType::RPAREN || current.type == TokenType::RBRACK)
                break;

            int prec = precedence(current.type);
            if (prec < min_prec) break;

            TokenType op = current.type;
            next();
            auto right = parseBinary(prec + 1);
            if (!right) { error("Missing right operand"); break; }

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
                default: break;
            }
            bin->left = left;
            bin->right = right;
            left = bin;
        }
        return left;
    }

    std::shared_ptr<GUTRExpr> parseUnary() {
        if (match(TokenType::MINUS)) {
            auto u = std::make_shared<GUTRUnaryOpExpr>();
            u->op = GUTRUnaryOpExpr::NEG;
            u->arg = parsePrimary();
            return u;
        }
        if (match(TokenType::BANG)) {
            auto u = std::make_shared<GUTRUnaryOpExpr>();
            u->op = GUTRUnaryOpExpr::NOT;
            u->arg = parsePrimary();
            return u;
        }
        return parsePrimary();
    }

    std::shared_ptr<GUTRExpr> parsePrimary() {
        if (current.type == TokenType::RBRACE || current.type == TokenType::END ||
            current.type == TokenType::SEMICOLON || current.type == TokenType::COMMA ||
            current.type == TokenType::RPAREN || current.type == TokenType::RBRACK) {
            return nullptr;
        }

        // Скобки
        if (match(TokenType::LPAREN)) {
            auto expr = parseExpression();
            expect(TokenType::RPAREN, "Expected ')'");
            return expr;
        }

        // Число
        if (current.type == TokenType::NUMBER) {
            std::string num = current.text; next();
            try {
                if (num.find('.') != std::string::npos || num.find('e') != std::string::npos || num.find('E') != std::string::npos)
                    return std::make_shared<GUTRLiteralExpr>(GUTRValue::real(std::stod(num)));
                else
                    return std::make_shared<GUTRLiteralExpr>(GUTRValue::integer(std::stoll(num)));
            } catch (...) {
                return std::make_shared<GUTRLiteralExpr>(GUTRValue::integer(0));
            }
        }

        // Строка
        if (current.type == TokenType::STRING) {
            std::string val = current.text; next();
            return std::make_shared<GUTRLiteralExpr>(GUTRValue::string(val));
        }

        // Ключевые слова-литералы
        if (current.type == TokenType::KEYWORD) {
            std::string kw = current.text;
            if (kw == "true") { next(); return std::make_shared<GUTRLiteralExpr>(GUTRValue::boolean(true)); }
            if (kw == "false") { next(); return std::make_shared<GUTRLiteralExpr>(GUTRValue::boolean(false)); }
            if (kw == "nil") { next(); return std::make_shared<GUTRLiteralExpr>(GUTRValue::nil()); }
            // Специальные ключевые слова как функции
            if (kw == "list" || kw == "tensor" || kw == "print" || kw == "sect") {
                next();
                if (current.type == TokenType::LPAREN) {
                    return parseCall(std::make_shared<GUTRVarExpr>(kw));
                }
                if (current.type == TokenType::LBRACE) {
                    // sect без скобок - определение
                    return parseSectBody(kw);
                }
            }
            error("Unexpected keyword: " + kw);
            next();
            return nullptr;
        }

        // Идентификатор
        if (current.type == TokenType::IDENT) {
            std::string id = current.text;
            Token save = current;
            next();
            if (current.type == TokenType::ASSIGN) {
                next();
                auto expr = parseExpression();
                auto assign = std::make_shared<GUTRAssignExpr>();
                assign->var_name = id;
                assign->value = expr;
                if (current.type == TokenType::SEMICOLON) next();
                return assign;
            }
            // не присваивание — парсим как выражение, начиная с идентификатора
            if (current.type == TokenType::LPAREN) {
                return parseCall(std::make_shared<GUTRVarExpr>(id));
            }
            if (current.type == TokenType::LBRACK) {
                auto idx = std::make_shared<GUTRIndexExpr>();
                idx->object = std::make_shared<GUTRVarExpr>(id);
                next();
                idx->index = parseExpression();
                expect(TokenType::RBRACK, "Expected ']'");
                return idx;
            }
            if (current.type == TokenType::DOT) {
                next();
                if (current.type == TokenType::IDENT) {
                    std::string field = current.text; next();
                    auto call = std::make_shared<GUTRCallExpr>();
                    call->callee = std::make_shared<GUTRVarExpr>("sect_get");
                    call->args.push_back(std::make_shared<GUTRVarExpr>(id));
                    call->args.push_back(std::make_shared<GUTRLiteralExpr>(GUTRValue::string(field)));
                    return call;
                }
                error("Expected field name after '.'");
            }
            return std::make_shared<GUTRVarExpr>(id);
        }

        // Массив []
        if (match(TokenType::LBRACK)) {
            auto arr = std::make_shared<GUTRArrayLiteralExpr>();
            if (current.type != TokenType::RBRACK) {
                while (true) {
                    auto elem = parseExpression();
                    if (elem) arr->elements.push_back(elem);
                    if (current.type == TokenType::COMMA) next();
                    else break;
                }
            }
            expect(TokenType::RBRACK, "Expected ']'");
            return arr;
        }

        error("Unexpected token: " + token_to_string(current.type));
        return nullptr;
    }

    // Парсинг вызова функции
    std::shared_ptr<GUTRExpr> parseCall(std::shared_ptr<GUTRExpr> callee) {
        auto call = std::make_shared<GUTRCallExpr>();
        call->callee = callee;
        next(); // пропускаем '('

        while (current.type != TokenType::RPAREN && current.type != TokenType::END) {
            auto arg = parseExpression();
            if (arg) {
                call->args.push_back(arg);
            }
            if (current.type == TokenType::COMMA) {
                next();
            } else if (current.type != TokenType::RPAREN) {
                error("Expected ',' or ')' in function call");
                break;
            }
        }
        expect(TokenType::RPAREN, "Expected ')'");
        return call;
    }

    // Парсинг тела секции
    std::shared_ptr<GUTRExpr> parseSectBody(const std::string& name) {
        next(); // пропускаем '{'
        std::vector<std::pair<std::string, std::shared_ptr<GUTRExpr>>> fields;
        while (current.type != TokenType::RBRACE && current.type != TokenType::END) {
            if (current.type != TokenType::IDENT) { error("Expected field name"); break; }
            std::string fname = current.text; next();
            expect(TokenType::ASSIGN, "Expected '='");
            auto expr = parseExpression();
            if (expr) fields.emplace_back(fname, expr);
            if (current.type == TokenType::COMMA || current.type == TokenType::SEMICOLON) next();
        }
        expect(TokenType::RBRACE, "Expected '}'");
        auto sect = std::make_shared<GUTRSectDefExpr>();
        sect->name = name;
        sect->field_inits = fields;
        return sect;
    }

    std::shared_ptr<GUTRFunctionDefExpr> parseFunctionDef() {
        next(); // пропускаем 'fn'
        if (current.type != TokenType::IDENT) { error("Expected function name"); return nullptr; }
        std::string name = current.text; next();

        expect(TokenType::LPAREN, "Expected '('");
        std::vector<std::string> params;
        if (current.type != TokenType::RPAREN) {
            while (true) {
                // Пропускаем аннотации типов
                if (current.type == TokenType::KEYWORD &&
                    (current.text == "u64" || current.text == "f64" || current.text == "bool" ||
                     current.text == "tensor" || current.text == "list" || current.text == "string"))
                    next();

                if (current.type != TokenType::IDENT) { error("Expected parameter name"); break; }
                params.push_back(current.text); next();
                if (current.type == TokenType::COMMA) next();
                else break;
            }
        }
        expect(TokenType::RPAREN, "Expected ')'");

        // Пропускаем возвращаемый тип
        if (current.type == TokenType::ARROW) {
            next();
            if (current.type == TokenType::LPAREN) {
                next();
                int depth = 1;
                while (current.type != TokenType::END && depth > 0) {
                    if (current.type == TokenType::LPAREN) depth++;
                    else if (current.type == TokenType::RPAREN) depth--;
                    next();
                }
            } else if (current.type == TokenType::IDENT || current.type == TokenType::KEYWORD) {
                next();
            }
        }

        auto body = std::dynamic_pointer_cast<GUTRBlockStmt>(parseStatement());
        if (!body) body = std::make_shared<GUTRBlockStmt>();

        auto def = std::make_shared<GUTRFunctionDefExpr>();
        def->name = name;
        def->params = params;
        def->body = body;
        return def;
    }

    std::shared_ptr<GUTRSectDefExpr> parseSectDef() {
        next(); // пропускаем 'sect'
        if (current.type != TokenType::IDENT) { error("Expected sect name"); return nullptr; }
        std::string name = current.text; next();
        return std::dynamic_pointer_cast<GUTRSectDefExpr>(parseSectBody(name));
    }

    std::shared_ptr<GUTRIfStmt> parseIf() {
        next(); // пропускаем 'if'
        auto cond = parseExpression();
        auto then_branch = parseStatement();
        std::shared_ptr<GUTRExpr> else_branch;
        if (current.type == TokenType::KEYWORD && current.text == "else") {
            next();
            else_branch = parseStatement();
        }
        auto stmt = std::make_shared<GUTRIfStmt>();
        stmt->condition = cond;
        stmt->then_branch = then_branch;
        stmt->else_branch = else_branch;
        return stmt;
    }

    std::shared_ptr<GUTRWhileStmt> parseWhile() {
        next(); // пропускаем 'while'
        auto cond = parseExpression();
        auto body = parseStatement();
        auto stmt = std::make_shared<GUTRWhileStmt>();
        stmt->condition = cond;
        stmt->body = body;
        return stmt;
    }

    std::shared_ptr<GUTRForStmt> parseFor() {
        next(); // пропускаем 'for'
        if (current.type != TokenType::IDENT) { error("Expected loop variable"); return nullptr; }
        std::string var = current.text; next();
        expect(TokenType::ASSIGN, "Expected '='");
        auto start = parseExpression();
        expect(TokenType::COMMA, "Expected ','");
        auto end = parseExpression();
        std::shared_ptr<GUTRExpr> step;
        if (current.type == TokenType::COMMA) {
            next();
            step = parseExpression();
        }
        auto body = parseStatement();
        auto stmt = std::make_shared<GUTRForStmt>();
        stmt->var_name = var;
        stmt->start_expr = start;
        stmt->end_expr = end;
        stmt->step_expr = step ? step : std::make_shared<GUTRLiteralExpr>(GUTRValue::integer(1));
        stmt->body = body;
        return stmt;
    }

    std::shared_ptr<GUTRReturnStmt> parseReturn() {
        next(); // пропускаем 'return'
        auto stmt = std::make_shared<GUTRReturnStmt>();
        if (current.type != TokenType::RBRACE && current.type != TokenType::END &&
            current.type != TokenType::SEMICOLON)
            stmt->value = parseExpression();
        return stmt;
    }
};

// ============================================================================
// GUTRParser::parse
// ============================================================================

bool GUTRParser::parse(const std::string& source, GUTRProgram& program) {
    if (source.empty()) {
        std::cerr << "\n[GUTR] Error: Empty source code" << std::endl;
        return false;
    }

    try {
        Parser parser(source);
        auto ast = parser.parse();
        if (!ast) {
            std::cerr << "\n[GUTR] Error: Failed to parse AST" << std::endl;
            return false;
        }

        auto ctx = std::make_shared<GUTRContext>();
        register_builtins(ctx);

        try {
            ast->eval(ctx);
        } catch (const std::exception& e) {
            std::cerr << "\n[GUTR] Runtime error: " << e.what() << std::endl;
            return false;
        }

        auto get_func = [&](const std::string& name) -> std::shared_ptr<GUTRFunction> {
            try {
                GUTRValue v = ctx->get(name);
                if (v.type != ValueType::FUNCTION || !v.func_val)
                    throw std::runtime_error("function not found: " + name);
                return v.func_val;
            } catch (const std::exception& e) {
                throw std::runtime_error("Required function '" + name + "': " + e.what());
            }
        };

        program.init_func = get_func("init");
        program.train_step_func = get_func("train_step");
        program.save_func = get_func("save");
        program.load_func = get_func("load");
        program.generate_func = get_func("generate");
        program.global_ctx = ctx;

        std::cout << "[GUTR] Plugin loaded successfully" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "\n[GUTR] Parse failed: " << e.what() << std::endl;
        return false;
    }
}

} // namespace uzaleat
