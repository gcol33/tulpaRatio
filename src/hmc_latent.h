// hmc_latent.h
// Observation-level latent factors with process-specific loadings.
//
// Each observation carries K factor scores f_i ~ N(0, I_K), and the P
// processes (numerator and denominator when the factors are shared, the
// numerator alone otherwise) load on them through a P x K matrix Lambda:
//
//     eta_num_i   += sum_k Lambda[0, k] f_ik
//     eta_denom_i += sum_k Lambda[1, k] f_ik
//
// The factors then add a residual covariance Lambda Lambda' between the two
// linear predictors of an observation. That covariance has P (P + 1) / 2 free
// entries, so at most K = P factors are identified, and Lambda is identified
// only up to an orthogonal rotation of the factors. Fixing Lambda lower
// triangular with a positive diagonal removes the rotation: Lambda Lambda' is
// then a Cholesky factorisation, which is unique.
//
// Parameter block: the loadings, factor by factor (factor k loads on processes
// k, ..., P - 1, its diagonal loading stored on the log scale), followed by
// the raw scores, observation-major (f_ik at i * K + k). The raw scores carry
// the N(0, I) prior; the linear predictors read them constrained (centred per
// factor, or shifted so the first observation's score is zero), which removes
// the direction a factor shares with the intercept.

#ifndef RATIOD_HMC_LATENT_H
#define RATIOD_HMC_LATENT_H

#include <vector>
#include <cmath>

#include "autodiff_utils.h"
#include <tulpa/bilinear_factor.h>

namespace ratiod_latent {



// Processes a factor can load on.
inline int n_processes(bool shared) { return shared ? 2 : 1; }

// Each process's latent contribution to its linear predictor.
template <typename T>
inline void process_effects(const std::vector<T>& f, const std::vector<T>& lam,
                            int N, int P, int K, std::vector<T>& eta_num,
                            std::vector<T>& eta_denom) {
  eta_num.assign(N, T(0.0));
  eta_denom.assign(P > 1 ? N : 0, T(0.0));
  for (int i = 0; i < N; i++) {
    for (int k = 0; k < K; k++) {
      eta_num[i] = eta_num[i] + lam[k] * f[i * K + k];
      if (P > 1) eta_denom[i] = eta_denom[i] + lam[K + k] * f[i * K + k];
    }
  }
}

// The latent block at a parameter vector, for the gradient paths: the scores
// and loadings as the linear predictors read them, each process's effect, and
// the backward pass that carries dL/d(eta) onto the block.
struct LatentView {
  int N = 0, P = 0, K = 0;
  int loading_start = -1, score_start = -1;
  tulpa::bilinear::LatentConstraint constraint = tulpa::bilinear::LatentConstraint::SUM_TO_ZERO;
  std::vector<double> lam, f, eta_num, eta_denom;

  bool active() const { return K > 0; }

  void read(const double* params, int loading_start_, int score_start_,
            int N_, int K_, bool shared, tulpa::bilinear::LatentConstraint constraint_) {
    N = N_;
    K = K_;
    P = n_processes(shared);
    loading_start = loading_start_;
    score_start = score_start_;
    constraint = constraint_;
    tulpa::bilinear::loadings(params + loading_start, P, K, lam);
    tulpa::bilinear::constrain_scores(params + score_start, N, K, constraint, f);
    process_effects(f, lam, N, P, K, eta_num, eta_denom);
  }

  // Effect on observation i's denominator predictor; zero when the factors
  // load on the numerator alone.
  double denom(int i) const { return P > 1 ? eta_denom[i] : 0.0; }

  // Adds to `grad` the likelihood's gradient through the block, given
  // dL/d(eta_num) and dL/d(eta_denom) per observation, and the gradient of
  // the block's prior.
  void add_gradient(const double* params, const std::vector<double>& d_num,
                    const std::vector<double>& d_denom, double rate,
                    double sd_off, double* grad) const {
    const double* theta = params + loading_start;
    const double* raw = params + score_start;
    double* g_theta = grad + loading_start;
    double* g_raw = grad + score_start;

    // dL/dLambda[p, k] = sum_i dL/d(eta_p,i) f_ik, carried to the stored
    // coordinate (log scale on the diagonal).
    int j = 0;
    for (int k = 0; k < K; k++) {
      for (int p = k; p < P; p++, j++) {
        const std::vector<double>& d = (p == 0) ? d_num : d_denom;
        double s = 0.0;
        for (int i = 0; i < N; i++) s += d[i] * f[i * K + k];
        g_theta[j] += (p == k) ? s * lam[p * K + k] : s;
      }
    }
    tulpa::bilinear::loadings_log_prior_grad(theta, P, K, rate, sd_off, g_theta);

    // dL/df_ik = sum_p Lambda[p, k] dL/d(eta_p,i), then through the
    // constraint: centring projects out each factor's mean, the first-zero
    // shift sends minus the factor's total to the first observation.
    std::vector<double> gf(static_cast<size_t>(N) * K, 0.0);
    for (int i = 0; i < N; i++) {
      for (int k = 0; k < K; k++) {
        double v = lam[k] * d_num[i];
        if (P > 1) v += lam[K + k] * d_denom[i];
        gf[i * K + k] = v;
      }
    }
    for (int k = 0; k < K; k++) {
      double total = 0.0;
      for (int i = 0; i < N; i++) total += gf[i * K + k];
      if (constraint == tulpa::bilinear::LatentConstraint::SUM_TO_ZERO) {
        const double mean = total / static_cast<double>(N);
        for (int i = 0; i < N; i++) g_raw[i * K + k] += gf[i * K + k] - mean;
      } else {
        for (int i = 0; i < N; i++) g_raw[i * K + k] += gf[i * K + k];
        g_raw[k] -= total;
      }
    }
    for (int jj = 0; jj < N * K; jj++) g_raw[jj] -= raw[jj];
  }
};

}  // namespace ratiod_latent

#endif  // RATIOD_HMC_LATENT_H
