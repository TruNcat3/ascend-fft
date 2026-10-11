#pragma once

#include <cctype>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace bfly { namespace config {

struct Value {
    enum Type { Object, Array, String, Number, Boolean, Null } type = Null;
    std::map<std::string, Value> object;
    std::vector<Value> array;
    std::string text;
    const Value& at(const std::string& key) const {
        auto it = object.find(key);
        if (type != Object || it == object.end())
            throw std::runtime_error("missing config field: " + key);
        return it->second;
    }
    int integer() const {
        if (type != Number || text.find_first_of(".eE") != std::string::npos)
            throw std::runtime_error("config field must be an integer");
        size_t used = 0;
        long long n = std::stoll(text, &used);
        if (used != text.size() || n < std::numeric_limits<int>::min() ||
            n > std::numeric_limits<int>::max())
            throw std::runtime_error("config integer out of range");
        return static_cast<int>(n);
    }
    std::string string() const {
        if (type != String || text.empty())
            throw std::runtime_error("config field must be a nonempty string");
        return text;
    }
};

// Small strict JSON reader for host configuration, without a CANN dependency.
class Parser {
    const std::string& text_;
    size_t pos_ = 0;
    void skip() { while (pos_ < text_.size() && std::isspace((unsigned char)text_[pos_])) ++pos_; }
    bool take(char ch) { skip(); if (pos_ < text_.size() && text_[pos_] == ch) { ++pos_; return true; } return false; }
    [[noreturn]] void bad() const { throw std::runtime_error("invalid JSON at byte " + std::to_string(pos_)); }
    std::string quoted() {
        if (!take('"')) bad();
        std::string out;
        while (pos_ < text_.size()) {
            unsigned char ch = text_[pos_++];
            if (ch == '"') return out;
            if (ch < 32) bad();
            if (ch != '\\') { out += ch; continue; }
            if (pos_ == text_.size()) bad();
            ch = text_[pos_++];
            switch (ch) {
                case '"': case '\\': case '/': out += ch; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    unsigned code = 0;
                    for (int i = 0; i < 4; ++i) {
                        if (pos_ == text_.size()) bad();
                        char c = text_[pos_++];
                        if (!std::isxdigit((unsigned char)c)) bad();
                        code = code * 16 + (c <= '9' ? c - '0' : std::tolower(c) - 'a' + 10);
                    }
                    if (code < 0x80) out += static_cast<char>(code);
                    else if (code < 0x800) { out += static_cast<char>(0xc0 | (code >> 6)); out += static_cast<char>(0x80 | (code & 63)); }
                    else { out += static_cast<char>(0xe0 | (code >> 12)); out += static_cast<char>(0x80 | ((code >> 6) & 63)); out += static_cast<char>(0x80 | (code & 63)); }
                    break;
                }
                default: bad();
            }
        }
        bad();
    }
    Value value(unsigned depth) {
        if (depth > 64) bad();
        skip();
        Value v;
        if (take('{')) {
            v.type = Value::Object;
            if (take('}')) return v;
            do {
                std::string key = quoted();
                if (!take(':') || v.object.count(key)) bad();
                v.object.emplace(key, value(depth + 1));
                if (take('}')) return v;
            } while (take(','));
            bad();
        }
        if (take('[')) {
            v.type = Value::Array;
            if (take(']')) return v;
            do { v.array.push_back(value(depth + 1)); if (take(']')) return v; } while (take(','));
            bad();
        }
        if (pos_ < text_.size() && text_[pos_] == '"') { v.type = Value::String; v.text = quoted(); return v; }
        for (const char* literal : {"true", "false", "null"}) {
            size_t n = std::char_traits<char>::length(literal);
            if (text_.compare(pos_, n, literal) == 0) {
                pos_ += n; v.type = literal[0] == 'n' ? Value::Null : Value::Boolean; v.text = literal; return v;
            }
        }
        size_t start = pos_;
        if (pos_ < text_.size() && text_[pos_] == '-') ++pos_;
        if (pos_ == text_.size() || !std::isdigit((unsigned char)text_[pos_])) bad();
        if (text_[pos_] == '0') ++pos_;
        else while (pos_ < text_.size() && std::isdigit((unsigned char)text_[pos_])) ++pos_;
        if (pos_ < text_.size() && text_[pos_] == '.') {
            ++pos_; size_t digits = pos_;
            while (pos_ < text_.size() && std::isdigit((unsigned char)text_[pos_])) ++pos_;
            if (digits == pos_) bad();
        }
        if (pos_ < text_.size() && (text_[pos_] == 'e' || text_[pos_] == 'E')) {
            ++pos_;
            if (pos_ < text_.size() && (text_[pos_] == '+' || text_[pos_] == '-')) ++pos_;
            size_t digits = pos_;
            while (pos_ < text_.size() && std::isdigit((unsigned char)text_[pos_])) ++pos_;
            if (digits == pos_) bad();
        }
        v.type = Value::Number; v.text = text_.substr(start, pos_ - start);
        return v;
    }
public:
    explicit Parser(const std::string& text) : text_(text) {}
    Value parse() { Value v = value(0); skip(); if (pos_ != text_.size() || v.type != Value::Object) bad(); return v; }
};

}}  // namespace bfly::config
