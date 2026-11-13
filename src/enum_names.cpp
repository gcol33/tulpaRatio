// enum_names.cpp
// The C++ name tables, read from R so its own lists of choices come from, or
// are tested against, the strings the sampler parses.

#include <Rcpp.h>
#include "enum_names.h"
#include "engine_options.h"
#include "covariance.h"
#include "ratio_margins.h"
#include "gibbs_spatial.h"

// [[Rcpp::export]]
std::vector<std::string> cpp_enum_names(std::string kind) {
  using ratiod_enum::enum_names;
  if (kind == "temporal") return enum_names(tulpaRatio::TEMPORAL_NAMES);
  if (kind == "tvc") return enum_names(tulpaRatio::TVC_NAMES);
  if (kind == "cov") return enum_names(ratiod_cov::COV_TYPE_NAMES);
  if (kind == "gradient_mode") return tulpa::gradient_mode_names();
  if (kind == "metric") return enum_names(tulpaRatio::METRIC_NAMES);
  if (kind == "msgp_sampler") return enum_names(tulpaRatio::MSGP_NAMES);
  if (kind == "zi") return enum_names(ratiod_zi::ZI_TYPE_NAMES);
  if (kind == "gibbs_family") return enum_names(gibbs::GIBBS_FAMILY_NAMES);
  Rcpp::stop("Unknown enum kind '%s'.", kind);
}
