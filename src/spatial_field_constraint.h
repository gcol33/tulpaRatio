// Hard sum-to-zero constraint for intrinsic (rank-deficient) spatial fields.
//
// An ICAR precision Q satisfies Q1 = 0, so the field's mean is unidentified
// and trades off against the intercept. Penalising sum(phi) softly cannot
// resolve this: the penalty puts precision lambda*J on the constant direction
// while the field's own directions sit at tau, and the resulting stiffness
// forces the NUTS step size down by sqrt(lambda*J/tau). Tight enough to
// identify and loose enough to sample are incompatible requirements.
//
// Instead the constant direction is removed rather than penalised. The
// sampled vector phi_raw is centred before it reaches the likelihood, so the
// likelihood cannot see its mean, and the intercept carries the level exactly.
// The freed direction is then given the field's own scale by using the proper
// precision tau*(Q + 11'/J) in place of tau*Q:
//
//   phi_raw' (Q + 11'/J) phi_raw = phi_raw' Q phi_raw + (sum phi_raw)^2 / J
//
// Q1 = 0 makes the quadratic form identical for phi_raw and its centred
// version, so only the second term is new. That matrix is full rank with
// log-determinant contribution J/2 * log(tau) rather than (J-1)/2, and its
// constant-direction precision is tau -- the same order as everything else,
// which is what removes the stiffness.
//
// This is the construction behind INLA's constr=TRUE and Stan's
// sum_to_zero_vector.

#ifndef RATIOD_SPATIAL_FIELD_CONSTRAINT_H
#define RATIOD_SPATIAL_FIELD_CONSTRAINT_H

#include <vector>

#include <tulpa/sum_to_zero.h>  // the construction's primitives

namespace ratiod_constraints {

// The engine owns the construction (tulpa/sum_to_zero.h); what lives here is
// the shape this package's gradient paths need it in -- a raw sum kept for the
// augmented quadratic, and the RAII binding below.

// Centres the spatial block of a parameter vector in place and returns the raw
// sum. Applied once at a function's boundary, so that every downstream reader
// sees the constrained field -- including vectorized fast paths that index
// params directly rather than through a bound pointer. Centring at individual
// binding sites instead leaves those paths reading raw values.
template <typename T>
inline T center_spatial_block(std::vector<T>& params, int start, int n) {
  const T sum = tulpa::s2z_component_sum(params.data(), start, n);
  (void)tulpa::s2z_centre_component(params.data(), start, n);
  return sum;
}

// Quadratic-form contribution of the freed constant direction: sum^2 / n.
template <typename T>
inline T free_direction_quad(const T& raw_sum, int n) {
  return tulpa::s2z_aug_coef(T(1), n) * raw_sum * raw_sum;
}

// d/d(phi_raw[s]) of -0.5 * scale * sum^2 / n, identical for every s.
template <typename T>
inline T free_direction_grad(const T& raw_sum, int n, const T& scale) {
  return -tulpa::s2z_aug_coef(scale, n) * raw_sum;
}

// The blocks an areal field's sum-to-zero acts on: one per connected
// component of two or more units (Freni-Sterrantino, Ventrucci and Rue 2018,
// INLA's constr = TRUE on a disconnected graph). An island carries a proper
// prior and no constraint. Empty `members` means one block over the whole
// field, the connected case.
struct ArealComponents {
  std::vector<std::vector<int>> members;  // constrained components, 0-based units
  std::vector<int> block_of;              // per unit: index into members, -1 for none
  bool whole() const { return members.empty(); }
  int block_of_unit(int i) const { return whole() ? 0 : block_of[i]; }
  int block_size(int b, int n) const {
    return whole() ? n : static_cast<int>(members[b].size());
  }
  int n_blocks() const { return whole() ? 1 : static_cast<int>(members.size()); }
};

// Centres each block of the spatial field in place and returns the raw sum of
// each, in block order.
template <typename T>
inline std::vector<T> center_areal_blocks(std::vector<T>& params, int start, int n,
                                          const ArealComponents& comps) {
  if (comps.whole()) return {center_spatial_block(params, start, n)};
  std::vector<T> sums;
  sums.reserve(comps.members.size());
  for (const auto& m : comps.members) {
    const int size = static_cast<int>(m.size());
    const T s = tulpa::s2z_component_sum(params.data(), start, m.data(), size);
    const T mean = s / T(static_cast<double>(size));
    for (int u : m) params[start + u] = params[start + u] - mean;
    sums.push_back(s);
  }
  return sums;
}

// Quadratic-form contribution of the freed constant directions:
// sum over blocks of sum_b^2 / J_b.
template <typename T>
inline T areal_free_direction_quad(const std::vector<T>& sums, int n,
                                   const ArealComponents& comps) {
  T q = T(0.0);
  for (int b = 0; b < static_cast<int>(sums.size()); b++) {
    q = q + tulpa::s2z_aug_coef(T(1), comps.block_size(b, n)) * sums[b] * sums[b];
  }
  return q;
}

// d/d(phi_raw[i]) of -0.5 * scale * sum_b^2 / J_b, for the block unit i is in;
// zero for an unconstrained island.
inline double areal_free_direction_grad(const std::vector<double>& sums, int n,
                                        const ArealComponents& comps, int i,
                                        double scale) {
  const int b = comps.block_of_unit(i);
  if (b < 0) return 0.0;
  return -tulpa::s2z_aug_coef(scale, comps.block_size(b, n)) * sums[b];
}

// The likelihood sees each block centred, so its gradient with respect to the
// raw field is the projection that removes each block's mean.
inline void project_areal_gradient(double* g, int n, const ArealComponents& comps) {
  if (comps.whole()) {
    double mean = 0.0;
    for (int i = 0; i < n; i++) mean += g[i];
    mean /= static_cast<double>(n);
    for (int i = 0; i < n; i++) g[i] -= mean;
    return;
  }
  for (const auto& m : comps.members) {
    double mean = 0.0;
    for (int u : m) mean += g[u];
    mean /= static_cast<double>(m.size());
    for (int u : m) g[u] -= mean;
  }
}

// Binds the parameter vector that every downstream reader of a gradient
// function should see. When the spatial block is intrinsic it is centred into
// an owned copy and raw_sum() carries the freed constant direction; otherwise
// the caller's vector is passed through untouched and raw_sum() is zero.
//
// Applied at a function's entry rather than at each binding site, so that the
// vectorized paths which index the vector directly see the constrained field
// too. Gradients produced under it are with respect to the raw parameters, so
// the spatial likelihood gradient must be projected.
class CenteredSpatialParams {
 public:
  CenteredSpatialParams(const std::vector<double>& params_in, bool center,
                        int start, int n, const ArealComponents& comps)
      : in_(&params_in), active_(center && start >= 0 && n > 0) {
    if (active_) {
      storage_ = params_in;
      raw_sums_ = center_areal_blocks(storage_, start, n, comps);
    }
  }

  const std::vector<double>& params() const { return active_ ? storage_ : *in_; }
  const std::vector<double>& raw_sums() const { return raw_sums_; }
  bool active() const { return active_; }

 private:
  const std::vector<double>* in_;
  std::vector<double> storage_;
  std::vector<double> raw_sums_;
  bool active_;
};

}  // namespace ratiod_constraints

#endif  // RATIOD_SPATIAL_FIELD_CONSTRAINT_H
