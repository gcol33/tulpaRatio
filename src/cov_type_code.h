#ifndef RATIOD_COV_TYPE_CODE_H
#define RATIOD_COV_TYPE_CODE_H

#include <Rcpp.h>

#include "hmc_cov.h"

namespace ratiod_cov {

// The integer R sends for a cov = choice: its position in COV_TYPE_NAMES, the
// order R/spatial.R's cov_type_code() maps the documented names in. A code
// outside the table is refused rather than resolved to whichever kernel a
// fallthrough happens to reach.
inline CovType cov_type_from_int(int cov_type) {
  constexpr int n = static_cast<int>(sizeof(COV_TYPE_NAMES) / sizeof(COV_TYPE_NAMES[0]));
  if (cov_type < 0 || cov_type >= n) {
    Rcpp::stop("Unknown covariance code %d. Expected 0 to %d.", cov_type, n - 1);
  }
  return COV_TYPE_NAMES[cov_type].value;
}

}  // namespace ratiod_cov

#endif  // RATIOD_COV_TYPE_CODE_H
