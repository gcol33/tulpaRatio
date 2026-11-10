#include <RcppEigen.h>
#include "hmc_grad_internal.h"
#include "log_post_impl.h"
#include "hmc_field_coordinates.h"

using namespace Rcpp;

namespace ratiod_hmc {

GradientMode g_gradient_mode = GradientMode::AUTO;


// Set global gradient mode (called at start of sampling)
void set_gradient_mode(GradientMode mode) {
    g_gradient_mode = mode;
}

void compute_gradient_field_coordinates(
    const std::vector<double>& z, const ModelData& data, const ParamLayout& layout,
    std::vector<double>& grad, double* log_post_out
) {
    const FieldCoordinates& coordinates = *field_coordinates;
    RATIOD_TLS_WORKSPACE(std::vector<double>, x);
    const double jacobian = coordinates.forward(z, x);
    FieldCoordinateScope raw(nullptr);
    coordinates.gradient(x, data, layout, grad, log_post_out);
    coordinates.pullback(z, x, grad);
    if (log_post_out) *log_post_out += jacobian;
}

void compute_gradient(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out
) {
    // Delegate to resolve_gradient_fn() — single source of truth for dispatch logic.
    // For hot paths (NUTS leapfrog), callers should resolve once via resolve_gradient_fn()
    // and reuse the pointer. This convenience wrapper is for cold-path callers.
    GradientFn fn = resolve_gradient_fn(g_gradient_mode, data, layout);
    fn(params, data, layout, grad, log_post_out);
}

// =====================================================================
// The latent blocks a model carries, as one mask. A specialized gradient
// declares the blocks it writes; a block the model carries outside that set is
// one the function leaves at zero while every other block is differentiated at
// a linear predictor missing it.
//
// The temporal margin contributes exactly one bit: a GP margin is
// GF_TEMPORAL_GP and not GF_TEMPORAL, the two being different blocks with
// different gradients, so a function writing one does not write the other.
// =====================================================================

enum GradFeature : unsigned {
  GF_AREAL          = 1u << 0,   // intrinsic ICAR / BYM2 field
  GF_GP             = 1u << 1,
  GF_MULTISCALE_GP  = 1u << 2,
  GF_HSGP           = 1u << 3,
  GF_SVC            = 1u << 4,
  GF_TEMPORAL       = 1u << 5,   // RW1 / RW2 / AR1 margin
  GF_TEMPORAL_GP    = 1u << 6,
  GF_MS_TEMPORAL    = 1u << 7,
  GF_TVC            = 1u << 8,
  GF_SPATIOTEMPORAL = 1u << 9,
  GF_LATENT         = 1u << 10,
  GF_RE_SLOPES      = 1u << 11,  // no specialized function writes the slope block
  // A collapsed field is a marginal, not a latent block: it is its own bit
  // rather than GF_AREAL / GF_GP so that a function written for the sampled
  // field cannot be selected for it.
  GF_AREAL_COLLAPSED = 1u << 12,
  GF_GP_COLLAPSED    = 1u << 13,
  // Crossed / nested RE terms. The specialized functions read the legacy
  // single-term block (data.re_group, layout.re_start), which holds one term
  // of several, so a model carrying more than one needs its own bit to fall
  // through to the composite on.
  GF_RE_MULTI        = 1u << 14,
  // Proper CAR carries a rho of its own and a log-determinant that moves with
  // it, and its full-rank precision leaves the field's mean in the likelihood
  // where an intrinsic field's is removed. A function written for the
  // intrinsic field writes neither, so proper CAR gets its own bit rather than
  // GF_AREAL and reaches only what names it.
  GF_AREAL_PROPER    = 1u << 15,
  // Zero / one inflation or a hurdle. The specialized functions take their
  // residuals from the vectorized kernels, which carry the plain families
  // only, and write no beta_zi / beta_oi block.
  GF_INFLATION       = 1u << 16
};

inline unsigned model_grad_features(const ModelData& data, const ParamLayout& layout) {
  unsigned f = 0u;
  const bool areal_collapsed = layout.is_icar_collapsed || layout.is_bym2_collapsed;
  if (layout.has_spatial)
    f |= areal_collapsed  ? GF_AREAL_COLLAPSED
       : layout.is_car_proper ? GF_AREAL_PROPER
                              : GF_AREAL;
  if (layout.is_gp && data.has_gp)
    f |= (layout.is_gp_collapsed && data.gp_collapsed) ? GF_GP_COLLAPSED : GF_GP;
  if (layout.is_multiscale_gp && data.has_multiscale_gp) f |= GF_MULTISCALE_GP;
  if (layout.is_hsgp && data.has_hsgp)                   f |= GF_HSGP;
  if (layout.has_svc && data.has_svc)                    f |= GF_SVC;
  if (layout.is_temporal_gp)                             f |= GF_TEMPORAL_GP;
  else if (layout.has_temporal)                          f |= GF_TEMPORAL;
  if (layout.has_multiscale_temporal)                    f |= GF_MS_TEMPORAL;
  if (layout.has_tvc && data.has_tvc)                    f |= GF_TVC;
  if (layout.has_spatiotemporal)                         f |= GF_SPATIOTEMPORAL;
  if (layout.has_latent && data.latent_n_factors > 0)    f |= GF_LATENT;
  if (layout.has_re_slopes)                              f |= GF_RE_SLOPES;
  if (data.n_re_terms > 1)                               f |= GF_RE_MULTI;
  if (layout.has_zi || layout.has_oi)                    f |= GF_INFLATION;
  return f;
}

// =====================================================================
// Resolve gradient function pointer — SINGLE SOURCE OF TRUTH for dispatch.
// Called once at sampling start, returns a function pointer to eliminate
// per-call branching during leapfrog steps. compute_gradient() delegates here.
// =====================================================================


namespace {
bool default_gradient_mode(GradientMode mode) {
  return mode != GradientMode::NUMERICAL && mode != GradientMode::AUTODIFF_ARENA &&
         mode != GradientMode::AUTODIFF_FORWARD;
}

bool needs_arena(const ModelData& data, const ParamLayout& layout) {
  return data.rsr.active || data.rtr.active ||
      (layout.has_spatiotemporal && !data.spatiotemporal_data.spatial_partition.trivial()) ||
      (layout.has_spatiotemporal &&
       ratiod_spatiotemporal::st_noncentered_gp(data.st_parameterization,
                                               data.st_is_hsgp,
                                               data.spatiotemporal_data));
}

bool specialized_likelihood(GradientMode mode, const ModelData& data,
                             const ParamLayout& layout) {
  return default_gradient_mode(mode) && !needs_arena(data, layout) &&
         handcoded_likelihood(data);
}

struct GradientEntry {
  const char* name;
  GradientFn fn;
  unsigned covers_mask;
  bool (*predicate)(GradientMode, const ModelData&, const ParamLayout&, unsigned);
};

const GradientEntry gradient_registry[] = {
  {"field_coordinates", compute_gradient_field_coordinates, ~0u,
   [](GradientMode, const ModelData&, const ParamLayout&, unsigned) {
     return field_coordinates && !field_coordinates->empty();
   }},
  {"numerical", compute_gradient_numerical, ~0u,
   [](GradientMode m, const ModelData& d, const ParamLayout& l, unsigned f) {
     if (m == GradientMode::NUMERICAL) return true;
     if (!default_gradient_mode(m) || needs_arena(d,l) ||
         !(f & (GF_AREAL_COLLAPSED | GF_GP_COLLAPSED))) return false;
     const bool gp = l.is_gp_collapsed && d.has_gp && d.gp_collapsed &&
                     (f & ~GF_GP_COLLAPSED) == 0u;
     const bool areal = (l.is_icar_collapsed || l.is_bym2_collapsed) &&
                        (f & ~(GF_AREAL_COLLAPSED | GF_TEMPORAL)) == 0u;
     return !handcoded_likelihood(d) || !(gp || areal);
   }},
  {"autodiff_arena", compute_gradient_arena, ~0u,
   [](GradientMode m, const ModelData& d, const ParamLayout& l, unsigned) {
     return m == GradientMode::AUTODIFF_ARENA ||
            (default_gradient_mode(m) && needs_arena(d,l));
   }},
  {"autodiff_forward", compute_gradient_forward, ~0u,
   [](GradientMode m, const ModelData&, const ParamLayout&, unsigned) {
     return m == GradientMode::AUTODIFF_FORWARD;
   }},
  {"analytical", compute_gradient_analytical, ~0u,
   [](GradientMode m, const ModelData& d, const ParamLayout& l, unsigned) {
     return specialized_likelihood(m,d,l) && can_use_analytical_gradient(d,l);
   }},
  {"hsgp", compute_gradient_hsgp, GF_HSGP | GF_TEMPORAL,
   [](GradientMode m, const ModelData& d, const ParamLayout& l, unsigned) {
     return specialized_likelihood(m,d,l) && (l.is_hsgp && d.has_hsgp);
   }},
  {"gp_collapsed", compute_gradient_gp_collapsed, GF_GP_COLLAPSED,
   [](GradientMode m, const ModelData& d, const ParamLayout& l, unsigned) {
     return specialized_likelihood(m,d,l) && (l.is_gp_collapsed && d.has_gp && d.gp_collapsed);
   }},
  {"icar_collapsed", compute_gradient_icar_collapsed, GF_AREAL_COLLAPSED | GF_TEMPORAL,
   [](GradientMode m, const ModelData& d, const ParamLayout& l, unsigned) {
     return specialized_likelihood(m,d,l) && (l.is_icar_collapsed || l.is_bym2_collapsed);
   }},
  {"gp", compute_gradient_gp_handcoded, GF_GP,
   [](GradientMode m, const ModelData& d, const ParamLayout& l, unsigned) {
     return specialized_likelihood(m,d,l) && (l.is_gp && d.has_gp);
   }},
  {"msgp_hsgp", compute_gradient_msgp_hsgp, GF_MULTISCALE_GP | GF_TEMPORAL,
   [](GradientMode m, const ModelData& d, const ParamLayout& l, unsigned) {
     return specialized_likelihood(m,d,l) && (l.is_multiscale_gp && d.has_multiscale_gp);
   }},
  {"gp_temporal", compute_gradient_gp_plus_temporal_gmrf, GF_GP | GF_TEMPORAL,
   [](GradientMode m, const ModelData& d, const ParamLayout& l, unsigned) {
     return specialized_likelihood(m,d,l) && (l.is_gp);
   }},
  {"svc_hsgp", compute_gradient_svc_hsgp_handcoded, GF_SVC,
   [](GradientMode m, const ModelData& d, const ParamLayout& l, unsigned) {
     return specialized_likelihood(m,d,l) && (l.has_svc && d.has_svc && d.svc_is_hsgp);
   }},
  {"svc", compute_gradient_svc_handcoded, GF_SVC,
   [](GradientMode m, const ModelData& d, const ParamLayout& l, unsigned) {
     return specialized_likelihood(m,d,l) && (l.has_svc && d.has_svc);
   }},
  {"tvc", compute_gradient_tvc_handcoded, GF_TVC,
   [](GradientMode m, const ModelData& d, const ParamLayout& l, unsigned) {
     return specialized_likelihood(m,d,l) && (l.has_tvc && d.has_tvc);
   }},
  {"spatiotemporal", compute_gradient_spatiotemporal_handcoded, GF_AREAL | GF_TEMPORAL | GF_SPATIOTEMPORAL,
   [](GradientMode m, const ModelData& d, const ParamLayout& l, unsigned) {
     return specialized_likelihood(m,d,l) && (l.has_spatiotemporal && d.spatiotemporal_data.type != STType::NONE && l.st_delta_start >= 0 && l.log_tau_st_idx >= 0);
   }},
  {"temporal_gp", compute_gradient_temporal_gp, GF_TEMPORAL_GP,
   [](GradientMode m, const ModelData& d, const ParamLayout& l, unsigned) {
     return specialized_likelihood(m,d,l) && (l.is_temporal_gp && l.has_temporal);
   }},
  {"ms_temporal", compute_gradient_ms_temporal_handcoded, GF_AREAL | GF_MS_TEMPORAL,
   [](GradientMode m, const ModelData& d, const ParamLayout& l, unsigned) {
     return specialized_likelihood(m,d,l) && (l.has_multiscale_temporal);
   }},
  {"latent", compute_gradient_latent_handcoded, GF_LATENT,
   [](GradientMode m, const ModelData& d, const ParamLayout& l, unsigned) {
     return specialized_likelihood(m,d,l) && (l.has_latent && d.latent_n_factors > 0);
   }},
  {"composite", compute_gradient_composite, ~0u,
   [](GradientMode, const ModelData&, const ParamLayout&, unsigned) { return true; }}
};
}  // namespace

GradientFn resolve_gradient_fn(GradientMode mode, const ModelData& data,
                                const ParamLayout& layout) {
  const unsigned features = model_grad_features(data, layout);
  for (const auto& entry : gradient_registry) {
    if ((features & ~entry.covers_mask) == 0u &&
        entry.predicate(mode, data, layout, features)) return entry.fn;
  }
  return nullptr;
}

const char* gradient_fn_name(GradientFn fn) {
  for (const auto& entry : gradient_registry) if (entry.fn == fn) return entry.name;
  return "unknown";
}

// Refuse a model its gradient cannot be computed for: an autodiff gradient
// mode on a model the templated log posterior cannot express, and a
// collapsed GP interaction outside the analytic path that carries it. Called
// on the R thread before sampling starts, because resolve_gradient_fn runs on
// OpenMP workers where a throw cannot cross the region boundary.
void require_gradient_supported(const ModelData& data, const ParamLayout& layout) {
  // A collapsed spatial field locates its mode with per-location curvature
  // written for the families below; with any other likelihood, or with
  // inflation, its gradient is not the gradient of its density.
  if (data.gp_collapsed || layout.is_icar_collapsed || layout.is_bym2_collapsed) {
    const ModelType mt = data.model_type;
    const bool family_ok =
        mt == ModelType::BINOMIAL || mt == ModelType::POISSON_GAMMA ||
        mt == ModelType::NEGBIN_NEGBIN ||
        (mt == ModelType::NEGBIN_GAMMA && !data.gp_collapsed);
    if (!family_ok || layout.has_zi || layout.has_oi) {
      Rcpp::stop(std::string("The collapsed ") +
                 (data.gp_collapsed ? "GP" : "areal") +
                 " field is written for the binomial, Poisson-gamma and "
                 "negative binomial families" +
                 (data.gp_collapsed ? "" : " (and negative binomial-gamma)") +
                 " without inflation. Use parameterization = \"centered\" or "
                 "\"noncentered\" for this model.");
    }
  }

  if (layout.st_collapsed) {
    // The Laplace step reads W = -d^2 l / d delta^2 as a precision, which a
    // log-concave likelihood keeps positive; the composite gradient is where
    // its adjoint residuals enter, and it carries the hand-coded families.
    if (!handcoded_likelihood(data) ||
        data.model_type == ModelType::BETA_BINOMIAL) {
      Rcpp::stop("The collapsed GP interaction needs a log-concave family with "
                 "its canonical links: binomial, Poisson-gamma, negative "
                 "binomial, gamma-gamma or lognormal.");
    }
    if (layout.has_zi || layout.has_oi) {
      Rcpp::stop("The collapsed GP interaction does not take zero or one "
                 "inflation, whose likelihood is not log-concave.");
    }
    if (data.rsr.active || data.rtr.active) {
      Rcpp::stop("The collapsed GP interaction does not take a restricted "
                 "spatial field or temporal effect.");
    }
  }

  if (g_gradient_mode != GradientMode::AUTODIFF_ARENA &&
      g_gradient_mode != GradientMode::AUTODIFF_FORWARD) return;

  const char* gap = ratiod::log_post_impl_gap(data, layout);
  if (gap == nullptr) return;

  const char* mode_name =
      (g_gradient_mode == GradientMode::AUTODIFF_ARENA)   ? "A_r" :
      "A";
  Rcpp::stop(std::string("gradient_mode '") + mode_name +
             "' differentiates the templated log posterior, which does not "
             "express the " + gap + " parameterization: that field is "
             "marginalized out by locating its mode and adding a Laplace "
             "correction, which is not a closed-form function of the "
             "parameters. Use gradient_mode = \"H\" or \"N\".");
}


}  // namespace ratiod_hmc
