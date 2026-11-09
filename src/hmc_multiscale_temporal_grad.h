// hmc_multiscale_temporal_grad.h
// Hand-coded gradients for Multi-scale Temporal decomposition
// Provides O(n) analytical gradients for trend + seasonal + short-term components

#ifndef RATIOD_HMC_MULTISCALE_TEMPORAL_GRAD_H
#define RATIOD_HMC_MULTISCALE_TEMPORAL_GRAD_H

#include <vector>
#include <cmath>
#include <cstring>
#include "hmc_temporal_multiscale.h"

#include <tulpa/sum_to_zero.h>  // s2z_aug_quad / _quad_grad / _rank

namespace ratiod_temporal_grad {

using ratiod_temporal::MultiscaleTemporalData;
using ratiod_temporal::TemporalType;

// The sum-to-zero augmentation, as the intrinsic VALUE functions apply it
// (ratiod_temporal::rw1_log_lik / rw2_log_lik with `augment`), split into the
// three places a gradient has to follow it: the normalizer's rank, the
// quadratic form the normalizer's tau multiplies, and the constant every
// coordinate picks up. The three live together so they cannot move apart.

// Rank of the normalizer: the augmentation fills exactly the component's
// constant direction, so it adds one pinned direction.
inline int aug_rank(int rank_Q, bool augment) {
    return augment ? tulpa::s2z_aug_rank(rank_Q, 1) : rank_Q;
}

// The quadratic form the normalizer's tau multiplies: the structure's own plus
// the augmentation's squared component sum over the component size.
inline double aug_quad(const double* phi, int n, double quad, bool augment) {
    if (!augment) return quad;
    return quad + tulpa::s2z_aug_quad(phi, 0, n, 1.0);
}

// The augmentation's contribution to d(log p)/d(phi_t): the same constant at
// every coordinate, since the augmentation couples the component through its
// sum and through nothing else.
inline void add_s2z_aug_grad(const double* phi, int n, double tau,
                             double* grad_phi, bool augment) {
    if (!augment || n < 1) return;
    const double g = tulpa::s2z_aug_quad_grad(phi, 0, n, tau);
    for (int t = 0; t < n; t++) grad_phi[t] -= g;
}

// Structure to hold multiscale temporal gradient results
// Pre-allocate with init() and reuse across iterations to avoid heap churn
struct MultiscaleTemporalGradients {
    std::vector<double> grad_trend;           // Gradient w.r.t. trend effects
    std::vector<double> grad_seasonal;        // Gradient w.r.t. seasonal effects
    std::vector<double> grad_short_term;      // Gradient w.r.t. short-term effects
    double grad_log_sigma2_trend = 0.0;
    double grad_log_sigma2_seasonal = 0.0;
    double grad_log_sigma2_short = 0.0;
    double grad_logit_rho_short = 0.0;

    void init(int n_trend, int n_seasonal, int n_short) {
        if ((int)grad_trend.size() != n_trend) grad_trend.resize(n_trend);
        if ((int)grad_seasonal.size() != n_seasonal) grad_seasonal.resize(n_seasonal);
        if ((int)grad_short_term.size() != n_short) grad_short_term.resize(n_short);
    }
};

// =============================================================================
// RW1 gradients (non-cyclic)
// =============================================================================

// RW1: log p(phi|sigma2) = -0.5 * (T-1) * log(2*pi*sigma2)
//                        - 0.5 / sigma2 * sum((phi[t] - phi[t-1])^2)
inline void rw1_grad_phi(const double* phi, int n, double sigma2,
                         double* grad_phi, bool augment = false) {
    double inv_sigma2 = 1.0 / (sigma2 + 1e-10);

    for (int t = 0; t < n; t++) {
        if (t == 0) {
            grad_phi[t] = -inv_sigma2 * (phi[0] - phi[1]);
        } else if (t == n - 1) {
            grad_phi[t] = -inv_sigma2 * (phi[t] - phi[t-1]);
        } else {
            grad_phi[t] = -inv_sigma2 * (2.0 * phi[t] - phi[t-1] - phi[t+1]);
        }
    }
    add_s2z_aug_grad(phi, n, inv_sigma2, grad_phi, augment);
}

inline double rw1_grad_log_sigma2(const double* phi, int n, double sigma2,
                                  bool augment = false) {
    double quad = 0.0;
    for (int t = 1; t < n; t++) {
        double diff = phi[t] - phi[t-1];
        quad += diff * diff;
    }
    // d/d(sigma2) [log_GMRF] = -0.5*rank/sigma2 + 0.5*quad/sigma2^2
    // Chain rule: d/d(log_sigma2) = d/d(sigma2) * sigma2
    //           = -0.5*rank + 0.5*quad/sigma2
    return -0.5 * aug_rank(tulpa::rw1_rank(n, false), augment)
         + 0.5 * aug_quad(phi, n, quad, augment) / (sigma2 + 1e-10);
}

// =============================================================================
// RW1 gradients (cyclic - for seasonal)
// =============================================================================

// Cyclic RW1: adds connection from last to first
inline void rw1_cyclic_grad_phi(const double* phi, int n, double sigma2,
                                double* grad_phi, bool augment = false) {
    if (n < 2) {
        for (int t = 0; t < n; t++) grad_phi[t] = 0.0;
        return;
    }

    double inv_sigma2 = 1.0 / (sigma2 + 1e-10);

    // Each diff d[t] = phi[t] - phi[t-1] contributes:
    //   grad_phi[t]   -= inv_sigma2 * d[t]   (from its own diff)
    //   grad_phi[t-1] += inv_sigma2 * d[t]   (from next diff)
    // Accumulate in-place without modulo by handling boundaries explicitly.
    std::memset(grad_phi, 0, n * sizeof(double));

    // Interior diffs: t = 1..n-1
    for (int t = 1; t < n; t++) {
        double neg_inv_d = -inv_sigma2 * (phi[t] - phi[t-1]);
        grad_phi[t]   += neg_inv_d;
        grad_phi[t-1] -= neg_inv_d;
    }
    // Cyclic wrap: diff from phi[0] - phi[n-1]
    {
        double neg_inv_d = -inv_sigma2 * (phi[0] - phi[n-1]);
        grad_phi[0]   += neg_inv_d;
        grad_phi[n-1] -= neg_inv_d;
    }

    add_s2z_aug_grad(phi, n, inv_sigma2, grad_phi, augment);
}

inline double rw1_cyclic_grad_log_sigma2(const double* phi, int n, double sigma2,
                                         bool augment = false) {
    double quad = 0.0;
    // Interior diffs
    for (int t = 1; t < n; t++) {
        double diff = phi[t] - phi[t-1];
        quad += diff * diff;
    }
    // Cyclic wrap
    {
        double diff = phi[0] - phi[n-1];
        quad += diff * diff;
    }
    return -0.5 * aug_rank(tulpa::rw1_rank(n, true), augment)
         + 0.5 * aug_quad(phi, n, quad, augment) / (sigma2 + 1e-10);
}

// =============================================================================
// RW2 gradients
// =============================================================================

inline void rw2_grad_phi(const double* phi, int n, double sigma2,
                         double* grad_phi, bool augment = false) {
    if (n < 3) {
        for (int t = 0; t < n; t++) grad_phi[t] = 0.0;
        return;
    }

    double inv_sigma2 = 1.0 / (sigma2 + 1e-10);

    // RW2: log p = -0.5/sigma2 * sum_{t=2}^{n-1} (phi[t] - 2*phi[t-1] + phi[t-2])^2
    // d/d(phi[k]) sums contributions from d[k] (coef=1), d[k+1] (coef=-2), d[k+2] (coef=1)
    // Compute in-place using a sliding window of 3 second-differences
    std::memset(grad_phi, 0, n * sizeof(double));

    for (int t = 2; t < n; t++) {
        double d_t = phi[t] - 2.0 * phi[t-1] + phi[t-2];
        double neg_inv_d = -inv_sigma2 * d_t;
        grad_phi[t]   += neg_inv_d;          // coef = 1
        grad_phi[t-1] += neg_inv_d * (-2.0); // coef = -2
        grad_phi[t-2] += neg_inv_d;          // coef = 1  (note: -inv*d * 1 = same sign reversal)
    }

    add_s2z_aug_grad(phi, n, inv_sigma2, grad_phi, augment);
}

inline double rw2_grad_log_sigma2(const double* phi, int n, double sigma2,
                                  bool augment = false) {
    double quad = 0.0;
    for (int t = 2; t < n; t++) {
        double d = phi[t] - 2.0 * phi[t-1] + phi[t-2];
        quad += d * d;
    }
    // d/d(sigma2) = -0.5*rank/sigma2 + 0.5*quad/sigma2^2
    // Chain rule: d/d(log_sigma2) = d/d(sigma2) * sigma2
    return -0.5 * aug_rank(tulpa::rw2_rank(n, false), augment)
         + 0.5 * aug_quad(phi, n, quad, augment) / (sigma2 + 1e-10);
}

// =============================================================================
// AR1 gradients
// =============================================================================

inline void ar1_grad_phi(const double* phi, int n, double sigma2, double rho, double* grad_phi) {
    double inv_sigma2 = 1.0 / (sigma2 + 1e-10);
    double one_m_rho2 = ratiod_ar1::one_minus_rho2(rho);
    double inv_var_stationary = one_m_rho2 * inv_sigma2;

    if (n == 1) {
        grad_phi[0] = -inv_var_stationary * phi[0];
        return;
    }

    // First time point
    double resid_1 = phi[1] - rho * phi[0];
    grad_phi[0] = -inv_var_stationary * phi[0] + inv_sigma2 * rho * resid_1;

    // Interior time points
    for (int t = 1; t < n - 1; t++) {
        double resid_t = phi[t] - rho * phi[t-1];
        double resid_tp1 = phi[t+1] - rho * phi[t];
        grad_phi[t] = -inv_sigma2 * resid_t + inv_sigma2 * rho * resid_tp1;
    }

    // Last time point
    double resid_T = phi[n-1] - rho * phi[n-2];
    grad_phi[n-1] = -inv_sigma2 * resid_T;
}

inline double ar1_grad_log_sigma2(const double* phi, int n, double sigma2, double rho) {
    double one_m_rho2 = ratiod_ar1::one_minus_rho2(rho);

    // AR1 has n terms: 1 marginal + (n-1) conditional
    // d/d(sigma2) [log_AR1] = -0.5*n/sigma2 + 0.5*quad/sigma2^2
    double grad = -0.5 * n;

    // Quadratic form: (1-rho^2)*phi[0]^2 + sum(resid^2)
    double quad = one_m_rho2 * phi[0] * phi[0];
    for (int t = 1; t < n; t++) {
        double resid = phi[t] - rho * phi[t-1];
        quad += resid * resid;
    }
    grad += 0.5 * quad / (sigma2 + 1e-10);

    // Chain rule: d/d(log_sigma2) = d/d(sigma2) * sigma2
    // Already applied: -0.5*n + 0.5*quad/sigma2 IS the result after chain rule
    return grad;
}

inline double ar1_grad_logit_rho(const double* phi, int n, double sigma2, double rho) {
    double inv_sigma2 = 1.0 / (sigma2 + 1e-10);
    double one_m_rho2 = ratiod_ar1::one_minus_rho2(rho);

    // AR1 log-lik = -0.5*log(sigma2/(1-rho^2)) - 0.5*(1-rho^2)/sigma2 * phi[0]^2
    //             + sum[-0.5*log(sigma2) - 0.5/sigma2 * (phi[t] - rho*phi[t-1])^2]
    //
    // d/d(rho) from normalization -0.5*log(sigma2/(1-rho^2)):
    //   = -rho / (1 - rho^2)
    double grad_rho = -rho / one_m_rho2;

    // d/d(rho) from stationary quadratic -0.5*(1-rho^2)/sigma2 * phi[0]^2:
    //   = rho * phi[0]^2 / sigma2
    grad_rho += rho * inv_sigma2 * phi[0] * phi[0];

    // d/d(rho) from AR terms: sum 1/sigma2 * (phi[t] - rho*phi[t-1]) * phi[t-1]
    for (int t = 1; t < n; t++) {
        double resid = phi[t] - rho * phi[t-1];
        grad_rho += inv_sigma2 * resid * phi[t-1];
    }

    // Transform to logit_rho
    // u = (rho + 1) / 2, logit_rho = logit(u)
    // d(rho)/d(logit_rho) = 2 * u * (1-u)
    double u = (rho + 1.0) / 2.0;
    double d_rho_d_logit = 2.0 * u * (1.0 - u);

    return grad_rho * d_rho_d_logit;
}

// =============================================================================
// IID gradients
// =============================================================================

inline void iid_grad_phi(const double* phi, int n, double sigma2, double* grad_phi) {
    double inv_sigma2 = 1.0 / (sigma2 + 1e-10);
    for (int t = 0; t < n; t++) {
        grad_phi[t] = -inv_sigma2 * phi[t];
    }
}

inline double iid_grad_log_sigma2(const double* phi, int n, double sigma2) {
    double quad = 0.0;
    for (int t = 0; t < n; t++) {
        quad += phi[t] * phi[t];
    }
    // d/d(sigma2) = -0.5*n/sigma2 + 0.5*quad/sigma2^2
    // Chain rule: d/d(log_sigma2) = d/d(sigma2) * sigma2
    return -0.5 * n + 0.5 * quad / (sigma2 + 1e-10);
}

// =============================================================================
// PC prior gradient for sigma2 (on log_sigma2 scale)
// =============================================================================

// PC prior: log p(sigma) = log(rate) - rate * sigma - log(2*sigma)
// where rate = -log(alpha) / U
// d/d(log_sigma2) = d/d(sigma) * d(sigma)/d(log_sigma2)
//                 = (-rate - 1/sigma) * (sigma / 2)
//                 = -0.5 * rate * sigma - 0.5


// =============================================================================
// Full multiscale temporal prior gradients
// =============================================================================

inline void multiscale_temporal_prior_gradients(
    const double* trend, int n_trend,
    const double* seasonal, int n_seasonal,
    const double* short_term, int n_short,
    double sigma2_trend, double sigma2_seasonal, double sigma2_short,
    double rho_short,
    const MultiscaleTemporalData& temp_data,
    MultiscaleTemporalGradients& grads
) {
    // Initialize gradients (reuse allocated memory)
    grads.init(n_trend, n_seasonal, n_short);
    if (n_trend > 0) std::memset(grads.grad_trend.data(), 0, n_trend * sizeof(double));
    if (n_seasonal > 0) std::memset(grads.grad_seasonal.data(), 0, n_seasonal * sizeof(double));
    if (n_short > 0) std::memset(grads.grad_short_term.data(), 0, n_short * sizeof(double));
    grads.grad_log_sigma2_trend = 0.0;
    grads.grad_log_sigma2_seasonal = 0.0;
    grads.grad_log_sigma2_short = 0.0;
    grads.grad_logit_rho_short = 0.0;

    // Trend component. The augmentation multiscale_temporal_log_lik applies to
    // each intrinsic arm is differentiated here, over the same component sum
    // and at the same precision, so the analytic gradient and the log-posterior
    // carry the same terms.
    // In the non-centred coordinate the arm's prior IS N(0, I) on the
    // coordinates that carry it -- the quadratic form and the transform's
    // Jacobian have already cancelled -- so its gradient is -z there, nothing
    // on the RW2's free linear direction, and nothing at all on sigma2, which
    // now reaches the field only through the transform.
    const bool nc = temp_data.noncentered;

    if (n_trend > 0) {
        if (nc) {
            const int order = ms_arm_order(temp_data.trend_type);
            const int d = ratiod_temporal_nc::nc_normal_dim(n_trend, order, false);
            for (int k = 0; k < d; k++) grads.grad_trend[k] = -trend[k];
        } else if (temp_data.trend_type == TemporalType::RW1) {
            rw1_grad_phi(trend, n_trend, sigma2_trend, grads.grad_trend.data(), true);
            grads.grad_log_sigma2_trend = rw1_grad_log_sigma2(trend, n_trend, sigma2_trend, true);
        } else if (temp_data.trend_type == TemporalType::RW2) {
            rw2_grad_phi(trend, n_trend, sigma2_trend, grads.grad_trend.data(), true);
            grads.grad_log_sigma2_trend = rw2_grad_log_sigma2(trend, n_trend, sigma2_trend, true);
        }
    }

    // Seasonal component (cyclic RW1)
    if (n_seasonal > 0 && temp_data.seasonal_period > 0) {
        if (nc) {
            const int d = ratiod_temporal_nc::nc_normal_dim(n_seasonal, 1, true);
            for (int k = 0; k < d; k++) grads.grad_seasonal[k] = -seasonal[k];
        } else {
            rw1_cyclic_grad_phi(seasonal, n_seasonal, sigma2_seasonal, grads.grad_seasonal.data(), true);
            grads.grad_log_sigma2_seasonal = rw1_cyclic_grad_log_sigma2(seasonal, n_seasonal, sigma2_seasonal, true);
        }
    }

    // Short-term component
    if (n_short > 0) {
        if (temp_data.short_term_type == TemporalType::AR1) {
            ar1_grad_phi(short_term, n_short, sigma2_short, rho_short, grads.grad_short_term.data());
            grads.grad_log_sigma2_short = ar1_grad_log_sigma2(short_term, n_short, sigma2_short, rho_short);
            grads.grad_logit_rho_short = ar1_grad_logit_rho(short_term, n_short, sigma2_short, rho_short);
        } else if (temp_data.short_term_type == TemporalType::IID) {
            iid_grad_phi(short_term, n_short, sigma2_short, grads.grad_short_term.data());
            grads.grad_log_sigma2_short = iid_grad_log_sigma2(short_term, n_short, sigma2_short);
        }
    }
}

// =============================================================================
// Likelihood-side projection for the centred intrinsic arms
// =============================================================================

// compute_ms_effect_by_time subtracts each intrinsic arm's mean on the way into
// eta, so eta depends on that arm's coefficients only through phi - mean(phi):
//
//     d eta_i / d phi[k] = delta(k, t_i) - 1/n.
//
// The gradient with respect to the sampled coefficients is therefore the
// projection (I - 11'/n) of the gradient the observation loop accumulated,
// which is that gradient minus its own mean. Applied to the arms
// multiscale_temporal_prior_gradients augments and to no others -- the
// short-term arm enters uncentred and its gradient passes through.
//
// This is the likelihood half of the same construction the augmentation is the
// prior half of; it lives beside multiscale_temporal_prior_gradients so a path
// cannot pick up one and miss the other.
// What one intrinsic arm needs for the second half of the projection: the
// block the sampler moves in, the effects the forward pass built, and where the
// transform's own sigma2 derivative lands. Left at its defaults in the centred
// coordinate, where there is no transform and none of it is read.
struct MSArmCoordinate {
    const double* block = nullptr;
    const double* effects = nullptr;
    double sigma2 = 1.0;
    double* grad_log_sigma2 = nullptr;
};

inline void project_ms_lik_gradients(
    double* grad_trend_lik, int n_trend,
    double* grad_seasonal_lik, int n_seasonal,
    const MultiscaleTemporalData& temp_data,
    const MSArmCoordinate& trend = MSArmCoordinate(),
    const MSArmCoordinate& seasonal = MSArmCoordinate()
) {
    const bool trend_centred = n_trend > 0 &&
        (temp_data.trend_type == TemporalType::RW1 ||
         temp_data.trend_type == TemporalType::RW2);
    if (trend_centred) {
        (void)tulpa::s2z_centre_component(grad_trend_lik, 0, n_trend);
    }
    const bool has_seasonal = n_seasonal > 0 && temp_data.seasonal_period > 0;
    if (has_seasonal) {
        (void)tulpa::s2z_centre_component(grad_seasonal_lik, 0, n_seasonal);
    }
    if (!temp_data.noncentered) return;

    // eta reads the centred effects and the effects are a linear map of the
    // block, so the gradient with respect to what the sampler moves in is the
    // projection above carried through that map. sigma2 reaches eta the same
    // way and picks up its own term here rather than in the arm's prior.
    auto carry = [](double* g, int n, int order, bool cyclic,
                    const MSArmCoordinate& arm) {
        if (n <= 0 || arm.block == nullptr) return;
        std::vector<double> gz(n, 0.0);
        ratiod_temporal_nc::rw_nc_backward(
            g, arm.effects, arm.block, n, order, cyclic,
            std::sqrt(arm.sigma2), gz.data(), arm.grad_log_sigma2);
        for (int k = 0; k < n; k++) g[k] = gz[k];
    };
    if (trend_centred) {
        carry(grad_trend_lik, n_trend, ms_arm_order(temp_data.trend_type),
              false, trend);
    }
    if (has_seasonal) {
        carry(grad_seasonal_lik, n_seasonal, 1, true, seasonal);
    }
}

} // namespace ratiod_temporal_grad

#endif // RATIOD_HMC_MULTISCALE_TEMPORAL_GRAD_H
