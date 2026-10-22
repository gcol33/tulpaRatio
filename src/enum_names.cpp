// enum_names.cpp
// The C++ name tables, read from R so its own lists of choices come from, or
// are tested against, the strings the sampler parses.

#include <Rcpp.h>
#include "enum_names.h"
#include "hmc_sampler.h"
#include "hmc_tvc.h"
#include "hmc_gp.h"
#include "gibbs_spatial.h"

// [[Rcpp::export]]
std::vector<std::string> cpp_enum_names(std::string kind) {
  using ratiod_enum::enum_names;
  if (kind == "temporal") return enum_names(ratiod_temporal::TEMPORAL_TYPE_NAMES);
  if (kind == "tvc") return enum_names(ratiod_tvc::TVC_STRUCTURE_NAMES);
  if (kind == "cov") return enum_names(ratiod_cov::COV_TYPE_NAMES);
  if (kind == "gradient_mode") return enum_names(ratiod_hmc::GRADIENT_MODE_NAMES);
  if (kind == "metric") return enum_names(ratiod_hmc::METRIC_NAMES);
  if (kind == "msgp_sampler") return enum_names(ratiod_gp::MSGP_SAMPLER_NAMES);
  if (kind == "zi") return enum_names(ratiod_zi::ZI_TYPE_NAMES);
  if (kind == "gibbs_family") return enum_names(gibbs::GIBBS_FAMILY_NAMES);
  Rcpp::stop("Unknown enum kind '%s'.", kind);
}
