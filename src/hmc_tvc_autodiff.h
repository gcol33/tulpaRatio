// hmc_tvc_autodiff.h
// Templated TVC (Temporally-Varying Coefficients) functions
// Works with both double and ad::Var for automatic differentiation

#ifndef RATIOD_HMC_TVC_AUTODIFF_H
#define RATIOD_HMC_TVC_AUTODIFF_H

#include <vector>
#include <cmath>
#include "autodiff_utils.h"
#include "hmc_tvc.h"  // For TVCData and TemporalType

namespace ratiod_tvc_ad {

using ratiod_tvc::TVCData;
using ratiod_temporal::TemporalType;
using namespace ratiod::math;

// The TVC quadratic forms, per-term prior, full prior and eta assembly are
// templated in hmc_tvc.h, next to the TVCData they read, and named here so the
// autodiff density keeps reading them under this namespace. One implementation
// serves both densities: the copies that used to live here had drifted from
// hmc_tvc.h on the cyclic wrap-around edge.
using ratiod_temporal::rw1_quadratic_form_t;
using ratiod_temporal::rw2_quadratic_form_t;
using ratiod_temporal::ar1_log_density_t;
using ratiod_tvc::tvc_term_log_prior;
using ratiod_tvc::tvc_log_prior;
using ratiod_tvc::compute_tvc_eta;


// The sum-to-zero constraint is ratiod_tvc::tvc_sum_to_zero_penalty in
// hmc_tvc.h, templated over the scalar type and shared with the plain sampler.

} // namespace ratiod_tvc_ad

#endif // RATIOD_HMC_TVC_AUTODIFF_H
