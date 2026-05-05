#ifndef GUTR_PARSER_HPP
#define GUTR_PARSER_HPP

#include <string>

namespace uzaleat {

class GUTRProgram;

class GUTRParser {
public:
    bool parse(const std::string& source, GUTRProgram& program);
};

} // namespace uzaleat

#endif
