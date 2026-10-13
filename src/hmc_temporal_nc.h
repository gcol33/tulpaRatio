// hmc_temporal_nc.h
// Non-centred coordinate for the intrinsic temporal walks.
//
// An augmented intrinsic precision factors as Q + 11'/n = A'A, where A stacks
// the walk's difference operator on 1'/sqrt(n): the augmentation IS that extra
// row. Sampling z ~ N(0, I) and setting phi = sigma * A^{-1} z therefore gives
// phi exactly the density the centred coordinate carries,
//
//     -0.5 * phi' (Q + 11'/n) phi / sigma2 - 0.5 * rank * log(2*pi*sigma2),
//
// because |A phi|^2 = sigma2 * |z|^2 and the Jacobian sigma^rank cancels the
// normalizer term for term. What changes is that sigma reaches the field
// through the transform rather than through a quadratic form, so the step size
// that holds the field stable no longer has to shrink with it.
//
// A non-cyclic RW2 annihilates a linear ramp as well as the constant, and only
// the constant is augmented (see rw2_log_lik), so A is (n-1) x n and one
// coordinate is left a free direction under the flat prior it already had.
// Q L = 0 makes the quadratic form blind to that direction, so it drops out of
// the identity above rather than complicating it.
//
// A^{-1} is a cumulative sum, once for RW1 and twice for RW2, so both the
// transform and its adjoint are O(n).

#ifndef RATIOD_HMC_TEMPORAL_NC_H
#define RATIOD_HMC_TEMPORAL_NC_H

#include <cmath>
#include <vector>

#include <tulpa/sum_to_zero.h>

namespace ratiod_temporal_nc {

// The linear null direction of a non-cyclic RW2, centred so that it is
// orthogonal to the constant and the sum coordinate below reads only the
// constant.
inline double lin_basis(int t, int n) {
  return static_cast<double>(t) - 0.5 * (static_cast<double>(n) - 1.0);
}

// Whether a walk of this length and order has a non-centred coordinate at all.
// Below it the centred quadratic form is empty and the two coordinates are the
// same block.
inline bool nc_applies(int n, int order, bool cyclic) {
  if (order == 1) return cyclic ? (n >= 3) : (n >= 2);
  return !cyclic && n >= 3;
}

// How many of the n coordinates carry the N(0, I) density. The remainder is
// the RW2's free linear direction, which keeps its flat prior.
inline int nc_normal_dim(int n, int order, bool cyclic) {
  if (!nc_applies(n, order, cyclic)) return 0;
  if (order == 1) return cyclic ? n : n;
  return n - 1;
}

// phi = sigma * A^{-1} z, with the RW2's linear coordinate added unscaled.
//
// Coordinate order, for a block of length n:
//   RW1 non-cyclic: z[0 .. n-2] increments,        z[n-1] sum
//   RW1 cyclic:     z[0 .. n-1] all of B^{-1}'s coordinates
//   RW2 non-cyclic: z[0 .. n-3] second differences, z[n-2] sum,
//                   z[n-1] the free linear direction (flat, unscaled)
template <typename T>
inline void rw_nc_forward(const T* z, int n, int order, bool cyclic,
                          const T& sigma, T* phi) {
  if (!nc_applies(n, order, cyclic)) {
    for (int t = 0; t < n; t++) phi[t] = z[t];
    return;
  }
  const double dn = static_cast<double>(n);
  const double sqrt_n = std::sqrt(dn);

  if (order == 1 && !cyclic) {
    // u_t = sum_{j < t} z_j, then the constant fixed by the sum coordinate.
    T run = T(0.0), sum_u = T(0.0);
    for (int t = 0; t < n; t++) {
      if (t > 0) run = run + z[t - 1];
      phi[t] = run;
      sum_u = sum_u + run;
    }
    const T a = (T(sqrt_n) * z[n - 1] - sum_u) / T(dn);
    for (int t = 0; t < n; t++) phi[t] = sigma * (phi[t] + a);
    return;
  }

  if (order == 1 && cyclic) {
    // B = C + 11'/n maps the constant to itself and acts as the circular
    // difference on the complement, so the constant part of z is the field's
    // mean and the rest integrates.
    T zbar = T(0.0);
    for (int t = 0; t < n; t++) zbar = zbar + z[t];
    zbar = zbar / T(dn);

    T run = T(0.0), sum_w = T(0.0);
    for (int t = 0; t < n; t++) {
      if (t > 0) run = run + (z[t] - zbar);
      phi[t] = run;
      sum_w = sum_w + run;
    }
    const T wbar = sum_w / T(dn);
    for (int t = 0; t < n; t++) phi[t] = sigma * (phi[t] - wbar + zbar);
    return;
  }

  // RW2, non-cyclic: v is the double integral with v_0 = v_1 = 0, the constant
  // comes from the sum coordinate, and the linear direction is carried by
  // z[n-1] at its own scale.
  //
  // Any right inverse of A reproduces the density, since A L = 0 makes the
  // quadratic form blind to what the map puts along L. It is not blind to the
  // CONDITIONING though: the double integral's particular solution carries a
  // large ramp, which z[n-1] can undo, so the two coordinates would describe
  // nearly the same direction of eta and the sampler would have to travel a
  // ridge between them. Projecting the ramp out leaves A v alone and the sum
  // alone (L is orthogonal to the constant) and makes z[n-1] the only
  // coordinate carrying it.
  T sum_v = T(0.0);
  for (int t = 0; t < n; t++) {
    if (t < 2) {
      phi[t] = T(0.0);
    } else {
      phi[t] = T(2.0) * phi[t - 1] - phi[t - 2] + z[t - 2];
    }
    sum_v = sum_v + phi[t];
  }
  T vL = T(0.0);
  double LL = 0.0;
  for (int t = 0; t < n; t++) {
    const double L = lin_basis(t, n);
    vL = vL + phi[t] * T(L);
    LL += L * L;
  }
  const T cL = vL / T(LL);
  for (int t = 0; t < n; t++) phi[t] = phi[t] - cL * T(lin_basis(t, n));

  const T a = (T(sqrt_n) * z[n - 2] - sum_v) / T(dn);
  for (int t = 0; t < n; t++) {
    phi[t] = sigma * (phi[t] + a) + z[n - 1] * T(lin_basis(t, n));
  }
}

// dL/dz and dL/d(log sigma2) from dL/dphi.
//
// `g` is already whatever projection the caller's eta assembly implies, so this
// is the transform's adjoint alone. `phi` is the field the forward pass built,
// which the sigma derivative needs: phi is sigma-linear except in the free
// linear direction, so d(phi)/d(log sigma2) = 0.5 * (phi - linear part).
inline void rw_nc_backward(const double* g, const double* phi, const double* z,
                           int n, int order, bool cyclic, double sigma,
                           double* gz, double* g_log_sigma2) {
  if (!nc_applies(n, order, cyclic)) {
    for (int t = 0; t < n; t++) gz[t] += g[t];
    return;
  }
  const double dn = static_cast<double>(n);
  const double sqrt_n = std::sqrt(dn);

  double gsum = 0.0;
  for (int t = 0; t < n; t++) gsum += g[t];

  if (order == 1 && !cyclic) {
    // S1(j) = sum_{t >= j} g_t, so d(u_t)/d(z_k) = 1[t > k] gives S1(k+1), and
    // the constant's share is the count (n - k - 1) spread over n.
    double tail = 0.0;
    std::vector<double> S1(n + 1, 0.0);
    for (int t = n - 1; t >= 0; t--) { tail += g[t]; S1[t] = tail; }
    for (int k = 0; k <= n - 2; k++) {
      gz[k] += sigma * (S1[k + 1] - (dn - k - 1.0) / dn * gsum);
    }
    gz[n - 1] += sigma * gsum / sqrt_n;
  } else if (order == 1 && cyclic) {
    // Solve B' y = g: the constant part of y is mean(g), and on the complement
    // (C' y)_k = y_k - y_{k+1} integrates backwards.
    const double gbar = gsum / dn;
    std::vector<double> y(n, 0.0);
    double run = 0.0;
    for (int k = n - 2; k >= 0; k--) { run += (g[k] - gbar); y[k] = run; }
    double ybar = 0.0;
    for (int k = 0; k < n; k++) ybar += y[k];
    ybar /= dn;
    for (int k = 0; k < n; k++) gz[k] += sigma * (y[k] - ybar + gbar);
  } else {
    // RW2: d(v_t)/d(z_k) = (t - k - 1) for t >= k + 2, a ramp, so the weighted
    // tail sums S1 and S2 give it in one pass rather than n.
    std::vector<double> S1(n + 2, 0.0), S2(n + 2, 0.0);
    for (int t = n - 1; t >= 0; t--) {
      S1[t] = S1[t + 1] + g[t];
      S2[t] = S2[t + 1] + static_cast<double>(t) * g[t];
    }
    // The ramp's own projection onto L, which the forward pass removed: its
    // inner products need the same two tail sums taken against L.
    std::vector<double> L1(n + 2, 0.0), L2(n + 2, 0.0);
    double LL = 0.0, gL = 0.0;
    for (int t = n - 1; t >= 0; t--) {
      const double L = lin_basis(t, n);
      L1[t] = L1[t + 1] + L;
      L2[t] = L2[t + 1] + static_cast<double>(t) * L;
      LL += L * L;
      gL += g[t] * L;
    }
    for (int k = 0; k <= n - 3; k++) {
      const int j = k + 2;
      const double ramp_g = S2[j] - (k + 1.0) * S1[j];
      // <d(v)/d(z_k), L>, the coefficient the forward pass subtracted
      const double ramp_L = L2[j] - (k + 1.0) * L1[j];
      // sum_t d(v_t)/d(z_k) = M(M+1)/2 with M = n - k - 2, unchanged by the
      // projection because L sums to zero
      const double m = dn - k - 2.0;
      const double ramp_sum = 0.5 * m * (m + 1.0);
      gz[k] += sigma * (ramp_g - ramp_L / LL * gL - ramp_sum / dn * gsum);
    }
    gz[n - 2] += sigma * gsum / sqrt_n;
    double glin = 0.0;
    for (int t = 0; t < n; t++) glin += g[t] * lin_basis(t, n);
    gz[n - 1] += glin;
  }

  if (g_log_sigma2 != nullptr) {
    // phi is sigma-linear apart from the free linear direction, and
    // d(sigma)/d(log sigma2) = 0.5 * sigma.
    double dot = 0.0;
    const bool has_lin = (order == 2 && !cyclic);
    for (int t = 0; t < n; t++) {
      const double scaled =
          has_lin ? (phi[t] - z[n - 1] * lin_basis(t, n)) : phi[t];
      dot += g[t] * scaled;
    }
    *g_log_sigma2 += 0.5 * dot;
  }
}

// log N(z; 0, I) over the coordinates that carry it. The free linear direction
// of a non-cyclic RW2 keeps the flat prior the centred coordinate gave it, so
// it contributes nothing here.
template <typename T>
inline T rw_nc_log_prior(const T* z, int n, int order, bool cyclic) {
  const int d = nc_normal_dim(n, order, cyclic);
  T quad = T(0.0);
  for (int k = 0; k < d; k++) quad = quad + z[k] * z[k];
  return T(-0.5) * quad - T(0.5 * d) * T(std::log(2.0 * M_PI));
}

// =====================================================================
// One walk per group
// =====================================================================
//
// With G temporal groups each group's walk is its own intrinsic field: it is
// centred on its own way into eta and its own constant is augmented, so the
// prior is blockdiag(Q_g + 11'/n). A group's level then belongs to a group
// intercept, which the walk cannot trade against, and the grouped transform is
// the single-walk primitive applied block by block.

// phi = sigma * A^{-1} z for each of G blocks of length n.
template <typename T>
inline void rw_nc_grouped_forward(const T* z, int n, int n_groups, int order,
                                  bool cyclic, const T& sigma, T* phi) {
  for (int g = 0; g < n_groups; g++) {
    rw_nc_forward(z + g * n, n, order, cyclic, sigma, phi + g * n);
  }
}

// log N(z; 0, I) over every block's coordinates that carry it.
template <typename T>
inline T rw_nc_grouped_log_prior(const T* z, int n, int n_groups, int order,
                                 bool cyclic) {
  T lp = T(0.0);
  for (int g = 0; g < n_groups; g++) {
    lp = lp + rw_nc_log_prior(z + g * n, n, order, cyclic);
  }
  return lp;
}

// d(rw_nc_grouped_log_prior)/dz, added to `gz`.
inline void rw_nc_grouped_log_prior_grad(const double* z, int n, int n_groups,
                                         int order, bool cyclic, double* gz) {
  const int d = nc_normal_dim(n, order, cyclic);
  for (int g = 0; g < n_groups; g++) {
    for (int k = 0; k < d; k++) gz[g * n + k] -= z[g * n + k];
  }
}

// dL/dz and dL/d(log sigma2) from dL/dphi, block by block. Each block's field
// is rebuilt for the sigma derivative rather than read from the caller, whose
// copy has been centred.
inline void rw_nc_grouped_backward(const double* g, const double* z, int n,
                                   int n_groups, int order, bool cyclic,
                                   double sigma, double* gz,
                                   double* g_log_sigma2) {
  std::vector<double> phig(n, 0.0);
  for (int gg = 0; gg < n_groups; gg++) {
    rw_nc_forward(z + gg * n, n, order, cyclic, sigma, phig.data());
    rw_nc_backward(g + gg * n, phig.data(), z + gg * n, n, order, cyclic,
                   sigma, gz + gg * n, g_log_sigma2);
  }
}

}  // namespace ratiod_temporal_nc

#endif  // RATIOD_HMC_TEMPORAL_NC_H
