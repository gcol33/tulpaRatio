// enum_names.h
// The names R sends for an enumerated choice, and the one lookup that reads
// them. A name a table does not hold is an error rather than a default, since
// a default would fit a model other than the one asked for.

#ifndef RATIOD_ENUM_NAMES_H
#define RATIOD_ENUM_NAMES_H

#include <cstddef>
#include <string>
#include <vector>
#include <Rcpp.h>

namespace ratiod_enum {

template <typename E>
struct EnumName {
  const char* name;
  E value;
};

// Every name a table accepts, in table order
template <typename E, std::size_t N>
inline std::vector<std::string> enum_names(const EnumName<E> (&table)[N]) {
  std::vector<std::string> out;
  out.reserve(N);
  for (const auto& e : table) out.emplace_back(e.name);
  return out;
}

// `what` names the choice in the error, e.g. "temporal structure"
template <typename E, std::size_t N>
inline E parse_enum(const EnumName<E> (&table)[N], const std::string& s,
                    const char* what) {
  for (const auto& e : table) {
    if (s == e.name) return e.value;
  }
  std::string valid;
  for (const auto& e : table) {
    if (!valid.empty()) valid += ", ";
    valid += "'" + std::string(e.name) + "'";
  }
  Rcpp::stop("Unknown %s '%s'. Expected one of %s.", what, s, valid);
}

// The first name a table gives a value
template <typename E, std::size_t N>
inline const char* enum_name(const EnumName<E> (&table)[N], E value) {
  for (const auto& e : table) {
    if (e.value == value) return e.name;
  }
  return "?";
}

}  // namespace ratiod_enum

#endif  // RATIOD_ENUM_NAMES_H
