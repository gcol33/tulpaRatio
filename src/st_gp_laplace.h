// st_gp_laplace.h
// The GP space-time interaction integrated out inside HMC.
//
// The interaction delta has one value per space-time cell and the prior
// N(0, Q(theta)^-1), Q the nearest-neighbour GP precision of its covariance;
// the soft sum-to-zero on each spatial and temporal margin adds a constant
// precision P. Conditional on everything else in the model, which reaches
// each observation as fixed offsets on its two linear predictors, the
// sampler is handed the Laplace approximation of the marginal likelihood
// (Rue, Martino and Chopin 2009; inside HMC, Margossian et al. 2020):
//
//   log p(y | rest) ~ f(dh) - 0.5 log|H|,
//   f(delta) = sum_i l_i(eta_i + delta_c(i)) + 0.5 log|Q| - 0.5 delta'(Q + P)delta,
//   H = Q + P + W(dh),   W_c = -sum_{i in c} d^2 l_i / d delta_c^2,
//
// dh the mode of f. The likelihood terms and every derivative of them come from
// ratio_obs_log_lik evaluated on Taylor jets, so the family code is the one
// the sampler evaluates.
//
// P couples every cell of a time slice and every cell of a region, which would
// fill the Cholesky factor of H almost completely. It is low rank instead:
// P = B B' with one sparse column per time slice and per region, and H is the
// Schur complement of the quasi-definite augmented matrix
//
//   K = [ Q + W   B ]
//       [ B'     -I ],
//
// whose LDL' factorization exists under any symmetric ordering (Vanderbei
// 1995). One sparse factorization of K then gives every solve with H, log|H|
// as the sum of log|D_ii|, and H^-1 as the leading block of K^-1.
//
// The gradient (laplace_gradient) uses the implicit function theorem with one
// adjoint solve: with Z = H^-1 and a_c = 0.5 Z_cc sum_{i in c} l_i''', v = H^-1 a,
//
//   dL/d theta = (partial L / partial theta at dh) + v' (partial grad_delta f / partial theta),
//
// and Z is needed only on the sparsity pattern of the factor of K, which
// Takahashi's recursion (Takahashi, Fagan and Chen 1973) delivers.
//
// NOTE: include AFTER hmc_sampler.h.

#ifndef RATIOD_ST_GP_LAPLACE_H
#define RATIOD_ST_GP_LAPLACE_H

#include <vector>
#include <cmath>
#include <cstdint>
#include <limits>
#include <algorithm>
#include <RcppEigen.h>

#include <tulpa/soft_sum_to_zero.h>
#include "hmc_cov.h"
#include "hmc_spatiotemporal.h"
#include "taylor_jet.h"
#include "ratio_obs_lik.h"
#include "tls_workspace.h"

namespace ratiod_st_laplace {

using ratiod_hmc::ModelData;
using ratiod_hmc::ParamLayout;
using ratiod_spatiotemporal::SpatiotemporalData;
using Jet = ratiod::jet::Jet3<double>;
using JetD = ratiod::jet::Jet3<fwd::Dual>;

// =====================================================================
// The Vecchia factors of Q and their derivatives
// =====================================================================

// Q = (I - B)' D^-1 (I - B): row k of B holds the regression of delta_k on
// its neighbours, D the conditional variances. Each is stored with its
// derivative in the three hyperparameters the sampler moves, all on the log
// scale: log sigma2 (= -log tau), log phi_space, log phi_time.
struct Factors {
  int n = 0;
  std::vector<int> nb_ptr;  // CSR over cells: neighbours of cell k
  std::vector<int> nb;      // neighbour cells
  std::vector<double> B;    // regression coefficients, aligned with nb
  std::vector<double> dB_space, dB_time;
  std::vector<double> d;    // conditional variances
  std::vector<double> dd_sigma2, dd_space, dd_time;
};

// The neighbour structure, a property of the model: cell k's neighbours, in
// the NNGP order the scan visits them.
inline void build_structure(const SpatiotemporalData& st, Factors& f) {
  const int n = st.n_params;
  const int nn = st.nn;
  f.n = n;
  std::vector<int> count(n, 0);
  for (int i = 1; i < n; i++) {
    int c = 0;
    for (int j = 0; j < nn; j++) {
      if (st.nn_idx[static_cast<size_t>(i) * nn + j] > 0) c++;
    }
    count[st.nn_order[i]] = c;
  }
  f.nb_ptr.assign(n + 1, 0);
  for (int k = 0; k < n; k++) f.nb_ptr[k + 1] = f.nb_ptr[k] + count[k];
  const int tot = f.nb_ptr[n];
  f.nb.assign(tot, 0);
  for (int i = 1; i < n; i++) {
    const int k = st.nn_order[i];
    for (int j = 0; j < count[k]; j++) {
      f.nb[f.nb_ptr[k] + j] = st.nn_order[st.nn_idx[static_cast<size_t>(i) * nn + j] - 1];
    }
  }
  f.B.assign(tot, 0.0);
  f.dB_space.assign(tot, 0.0);
  f.dB_time.assign(tot, 0.0);
  f.d.assign(n, 0.0);
  f.dd_sigma2.assign(n, 0.0);
  f.dd_space.assign(n, 0.0);
  f.dd_time.assign(n, 0.0);
}

// Fills `f`'s values at (sigma2, phi_space, phi_time); its structure is built
// once per model. False where a neighbour block cannot be factorized, a
// rejected parameter state for the caller.
inline bool build_factors(const SpatiotemporalData& st, double sigma2,
                          double phi_space, double phi_time, Factors& f) {
  const int nn = st.nn;
  if (f.n != st.n_params || f.nb_ptr.empty()) build_structure(st, f);
  const int n = f.n;

  std::vector<double> zero(n, 0.0);
  std::vector<double> dc_s(nn), dc_t(nn), dC_s(static_cast<size_t>(nn) * nn),
      dC_t(static_cast<size_t>(nn) * nn), rhs(nn), y(nn);
  const bool ok = ratiod_spatiotemporal::st_gp_nngp_scan<double>(
      zero.data(), sigma2, phi_space, phi_time, st,
      [&](int i, int k, const int* nb, int n_nb, const double* L,
          const double* /*c_vec*/, const double* alpha, double /*cond_mean*/,
          double cond_var, bool floored) {
        f.d[k] = cond_var;
        f.dd_sigma2[k] = floored ? 0.0 : cond_var;
        f.dd_space[k] = 0.0;
        f.dd_time[k] = 0.0;
        if (n_nb == 0) return;
        const int off = f.nb_ptr[k];
        for (int j = 0; j < n_nb; j++) f.B[off + j] = alpha[j];

        // d c / d phi and d C / d phi, from the correlation derivatives
        for (int j = 0; j < n_nb; j++) {
          const size_t pos = static_cast<size_t>(i) * nn + j;
          double a, b;
          ratiod_spatiotemporal::st_corr_dphi(st.nn_dist_space[pos],
                                              st.nn_dist_time[pos], phi_space,
                                              phi_time, st, &a, &b);
          dc_s[j] = sigma2 * a;
          dc_t[j] = sigma2 * b;
        }
        for (int j1 = 0; j1 < n_nb; j1++) {
          dC_s[j1 * n_nb + j1] = 0.0;
          dC_t[j1 * n_nb + j1] = 0.0;
          for (int j2 = j1 + 1; j2 < n_nb; j2++) {
            double h, u, a, b;
            ratiod_spatiotemporal::st_pair_separation(st, nb[j1], nb[j2], &h, &u);
            ratiod_spatiotemporal::st_corr_dphi(h, u, phi_space, phi_time, st,
                                                &a, &b);
            dC_s[j1 * n_nb + j2] = dC_s[j2 * n_nb + j1] = sigma2 * a;
            dC_t[j1 * n_nb + j2] = dC_t[j2 * n_nb + j1] = sigma2 * b;
          }
        }

        // alpha = C^-1 c, so d alpha = C^-1 (dc - dC alpha), and
        // d = sigma2 - c' alpha gives dd = -2 dc' alpha + alpha' dC alpha.
        // Both are carried onto log phi.
        auto derive = [&](const double* dc, const double* dC, double phi,
                          double* dB_out, double& dd_out) {
          double quad = 0.0, lin = 0.0;
          for (int j1 = 0; j1 < n_nb; j1++) {
            double s = 0.0;
            for (int j2 = 0; j2 < n_nb; j2++) s += dC[j1 * n_nb + j2] * alpha[j2];
            rhs[j1] = dc[j1] - s;
            quad += alpha[j1] * s;
            lin += dc[j1] * alpha[j1];
          }
          ratiod_cov::nngp_forward_solve(L, n_nb, rhs.data(), y.data());
          ratiod_cov::nngp_back_solve(L, n_nb, y.data(), dB_out);
          for (int j = 0; j < n_nb; j++) dB_out[j] *= phi;
          dd_out = floored ? 0.0 : (-2.0 * lin + quad) * phi;
        };
        derive(dc_s.data(), dC_s.data(), phi_space, &f.dB_space[off], f.dd_space[k]);
        derive(dc_t.data(), dC_t.data(), phi_time, &f.dB_time[off], f.dd_time[k]);
      });
  return ok;
}

// r_k' x = x_k - sum_j B_kj x_nb(j)
inline double row_dot(const Factors& f, int k, const double* x) {
  double s = x[k];
  for (int p = f.nb_ptr[k]; p < f.nb_ptr[k + 1]; p++) s -= f.B[p] * x[f.nb[p]];
  return s;
}

// =====================================================================
// The margin penalty P
// =====================================================================

// P delta for the soft sum-to-zero over each spatial and each temporal
// margin, cell k = s * T + t (ratiod_spatiotemporal::st_sum_to_zero_penalty).
inline void margin_penalty_apply(const SpatiotemporalData& st, const double* x,
                                 double* out) {
  const int S = st.n_spatial, T = st.n_times;
  const double lam_s = tulpa::s2z_precision(S);
  const double lam_t = tulpa::s2z_precision(T);
  std::vector<double> col(T, 0.0), row(S, 0.0);
  for (int s = 0; s < S; s++) {
    for (int t = 0; t < T; t++) {
      col[t] += x[s * T + t];
      row[s] += x[s * T + t];
    }
  }
  for (int s = 0; s < S; s++) {
    for (int t = 0; t < T; t++) out[s * T + t] = lam_s * col[t] + lam_t * row[s];
  }
}

// The margin penalty's factor B (P = B B'), n x (T + S): column t has
// sqrt(lambda_s) on every cell of time slice t, column T + s sqrt(lambda_t)
// on every cell of region s. Emitted as (cell, column, value).
template <typename Emit>
inline void emit_margin_factor(const SpatiotemporalData& st, Emit&& emit) {
  const int S = st.n_spatial, T = st.n_times;
  const double rs = std::sqrt(tulpa::s2z_precision(S));
  const double rt = std::sqrt(tulpa::s2z_precision(T));
  for (int s = 0; s < S; s++) {
    for (int t = 0; t < T; t++) {
      emit(s * T + t, t, rs);
      emit(s * T + t, T + s, rt);
    }
  }
}

// =====================================================================
// Workspace
// =====================================================================

struct Workspace {
  std::uint64_t model_id = 0;
  bool pattern_ready = false;
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ldlt;
  Eigen::SparseMatrix<double> K;
  // K's entries in the fixed order assemble_K emits them, and where each one
  // lands among K's stored values
  std::vector<Eigen::Triplet<double>> trip;
  std::vector<int> trip_pos;
  // where each entry of H^-1 the hyperparameter gradient reads lies in the
  // selected inverse, in the order it reads them
  std::vector<int> z_pos;
  // where each entry Takahashi's recursion reads lies, in the order it reads
  // them; the pattern is fixed, so this is recorded on the first pass
  std::vector<int> taka_pos;

  // observations per cell, and the cell of each observation (-1 for none)
  std::vector<int> cell_ptr, cell_obs, obs_cell;

  // warm start for the mode
  std::vector<double> delta;
  bool have_delta = false;

  // the last evaluation, so that the log posterior and its gradient at the
  // same parameters share one solve
  std::vector<double> last_params;
  double last_log_marginal = 0.0;
  bool last_ok = false;

  // state at the mode, read by the gradient and the draw
  Factors fac;
  std::vector<double> W, v;
  double log_marginal = 0.0;

  void bind(std::uint64_t id) {
    if (id == model_id) return;
    model_id = id;
    pattern_ready = false;
    z_pos.clear();
    taka_pos.clear();
    fac = Factors();
    have_delta = false;
    last_params.clear();
    cell_ptr.clear();
  }
};

inline Workspace& workspace() {
  RATIOD_TLS_WORKSPACE(Workspace, ws);
  return ws;
}

// Forget the warm start and the last evaluation, so the next mode is found
// from zero and the gradient is a function of the point alone.
inline void reset_warm_start() {
  Workspace& ws = workspace();
  ws.have_delta = false;
  ws.last_ok = false;
}

inline void build_cell_map(const ModelData& data, Workspace& ws) {
  if (!ws.cell_ptr.empty()) return;
  const auto& st = data.spatiotemporal_data;
  const int n = st.n_params;
  ws.obs_cell.assign(data.N, -1);
  std::vector<int> cnt(n, 0);
  for (int i = 0; i < data.N; i++) {
    const int c = st.st_flat[i] - 1;
    if (c >= 0 && c < n) {
      ws.obs_cell[i] = c;
      cnt[c]++;
    }
  }
  ws.cell_ptr.assign(n + 1, 0);
  for (int c = 0; c < n; c++) ws.cell_ptr[c + 1] = ws.cell_ptr[c] + cnt[c];
  ws.cell_obs.assign(ws.cell_ptr[n], 0);
  std::fill(cnt.begin(), cnt.end(), 0);
  for (int i = 0; i < data.N; i++) {
    const int c = ws.obs_cell[i];
    if (c >= 0) ws.cell_obs[ws.cell_ptr[c] + cnt[c]++] = i;
  }
}

// K at W with a fixed sparsity pattern: every entry Q, W, B or the trailing -I
// can reach is emitted, zeros included, in one fixed order. The first call
// builds the matrix and records where each entry lands; later calls add the
// entries into those positions, and analyzePattern runs once.
template <typename Emit>
inline void emit_K(const SpatiotemporalData& st, const Factors& f,
                   const std::vector<double>& W, Emit&& emit) {
  const int n = f.n;
  for (int k = 0; k < n; k++) {
    const double inv_d = 1.0 / f.d[k];
    const int p0 = f.nb_ptr[k], p1 = f.nb_ptr[k + 1];
    emit(k, k, inv_d);
    for (int p = p0; p < p1; p++) {
      const double v = -f.B[p] * inv_d;
      emit(k, f.nb[p], v);
      emit(f.nb[p], k, v);
      for (int q = p0; q < p1; q++) emit(f.nb[p], f.nb[q], f.B[p] * f.B[q] * inv_d);
    }
  }
  for (int k = 0; k < n; k++) emit(k, k, W[k]);
  emit_margin_factor(st, [&](int cell, int col, double v) {
    emit(cell, n + col, v);
    emit(n + col, cell, v);
  });
  const int m = st.n_spatial + st.n_times;
  for (int j = 0; j < m; j++) emit(n + j, n + j, -1.0);
}

inline void assemble_K(const SpatiotemporalData& st, const Factors& f,
                       const std::vector<double>& W, Workspace& ws) {
  if (!ws.pattern_ready) {
    const int size = f.n + st.n_spatial + st.n_times;
    ws.trip.clear();
    emit_K(st, f, W, [&](int r, int c, double v) { ws.trip.emplace_back(r, c, v); });
    ws.K.resize(size, size);
    ws.K.setFromTriplets(ws.trip.begin(), ws.trip.end());
    const int* outer = ws.K.outerIndexPtr();
    const int* inner = ws.K.innerIndexPtr();
    ws.trip_pos.resize(ws.trip.size());
    for (size_t e = 0; e < ws.trip.size(); e++) {
      const int r = ws.trip[e].row(), c = ws.trip[e].col();
      const int* hit = std::lower_bound(inner + outer[c], inner + outer[c + 1], r);
      ws.trip_pos[e] = static_cast<int>(hit - inner);
    }
    ws.ldlt.analyzePattern(ws.K);
    ws.pattern_ready = true;
    return;
  }
  double* val = ws.K.valuePtr();
  std::fill(val, val + ws.K.nonZeros(), 0.0);
  size_t e = 0;
  emit_K(st, f, W, [&](int, int, double v) { val[ws.trip_pos[e++]] += v; });
}

// Factor K at W. False where the factorization fails.
inline bool factor_H(const ModelData& data, const Factors& f,
                     const std::vector<double>& W, Workspace& ws) {
  assemble_K(data.spatiotemporal_data, f, W, ws);
  ws.ldlt.factorize(ws.K);
  return ws.ldlt.info() == Eigen::Success;
}

// H^-1 b, the leading block of K^-1 [b; 0]
inline Eigen::VectorXd solve_H(const Workspace& ws, const Eigen::VectorXd& b) {
  const int n = static_cast<int>(b.size());
  Eigen::VectorXd rhs = Eigen::VectorXd::Zero(ws.K.rows());
  rhs.head(n) = b;
  return ws.ldlt.solve(rhs).head(n);
}

// =====================================================================
// One observation's jets
// =====================================================================

// The shift direction of the interaction in (eta_num, eta_denom): both
// predictors when the interaction is shared, the numerator alone otherwise.
inline double denom_slope(const ModelData& data) {
  const bool binomial = (data.model_type == ratiod_hmc::ModelType::BINOMIAL ||
                         data.model_type == ratiod_hmc::ModelType::BETA_BINOMIAL);
  return (data.spatiotemporal_data.shared && !binomial) ? 1.0 : 0.0;
}

// l_i along the line (eta_num, eta_denom) + t (a, b)
inline Jet obs_jet(const ModelData& data, const ParamLayout& layout, int i,
                   double eta_num, double eta_denom, double a, double b,
                   double phi_num, double phi_denom) {
  return ratiod::ratio_obs_log_lik<Jet>(
      data, layout, i, Jet::line(eta_num, a), Jet::line(eta_denom, b),
      Jet(phi_num), Jet(phi_denom), Jet(0.0), Jet(0.0));
}

// =====================================================================
// The mode and the marginal
// =====================================================================

// The quadratic form delta'(Q + P)delta and (Q + P) delta
inline double prior_quad(const ModelData& data, const Factors& f,
                         const double* x, double* QPx) {
  const int n = f.n;
  std::fill(QPx, QPx + n, 0.0);
  double quad = 0.0;
  for (int k = 0; k < n; k++) {
    const double r = row_dot(f, k, x);
    const double s = r / f.d[k];
    quad += r * s;
    QPx[k] += s;
    for (int p = f.nb_ptr[k]; p < f.nb_ptr[k + 1]; p++) QPx[f.nb[p]] -= f.B[p] * s;
  }
  std::vector<double> Px(n);
  margin_penalty_apply(data.spatiotemporal_data, x, Px.data());
  for (int k = 0; k < n; k++) {
    QPx[k] += Px[k];
    quad += x[k] * Px[k];
  }
  return quad;
}

// sum_i l_i at delta, and per cell the first and second derivative sums
inline double lik_at(const ModelData& data, const ParamLayout& layout,
                     const double* off_num, const double* off_denom,
                     double phi_num, double phi_denom, const double* delta,
                     const Workspace& ws, double* g, double* W) {
  const double b = denom_slope(data);
  const int n = data.spatiotemporal_data.n_params;
  if (g) std::fill(g, g + n, 0.0);
  if (W) std::fill(W, W + n, 0.0);
  double ll = 0.0;
  for (int i = 0; i < data.N; i++) {
    const int c = ws.obs_cell[i];
    const double dc = c >= 0 ? delta[c] : 0.0;
    const Jet l = obs_jet(data, layout, i, off_num[i] + dc, off_denom[i] + b * dc,
                          1.0, b, phi_num, phi_denom);
    ll += l.v;
    if (c >= 0) {
      if (g) g[c] += l.d1;
      if (W) W[c] -= l.d2;
    }
  }
  return ll;
}

// log|H| = log|det K| = sum_i log|D_ii|
inline double log_det_factor(const Workspace& ws) {
  return ws.ldlt.vectorD().array().abs().log().sum();
}

// The Laplace log marginal at the given offsets and hyperparameters, with
// the mode, W, the factor of H and the selected inverse left in `ws`.
// Returns -inf where a factorization fails.
inline double laplace_log_marginal(const ModelData& data, const ParamLayout& layout,
                                   const double* off_num, const double* off_denom,
                                   double phi_num, double phi_denom,
                                   double sigma2, double phi_space, double phi_time,
                                   Workspace& ws) {
  const double neg_inf = -std::numeric_limits<double>::infinity();
  const auto& st = data.spatiotemporal_data;
  ws.bind(data.unique_id);
  build_cell_map(data, ws);
  if (!build_factors(st, sigma2, phi_space, phi_time, ws.fac)) return neg_inf;
  const Factors& f = ws.fac;
  const int n = f.n;

  if (!ws.have_delta || static_cast<int>(ws.delta.size()) != n) {
    ws.delta.assign(n, 0.0);
  }
  std::vector<double>& delta = ws.delta;
  std::vector<double> g(n), W(n), QPd(n), grad(n), trial(n);

  auto objective = [&](const double* x, double* gx, double* Wx, double* QPx) {
    const double ll = lik_at(data, layout, off_num, off_denom, phi_num, phi_denom,
                             x, ws, gx, Wx);
    return ll - 0.5 * prior_quad(data, f, x, QPx);
  };

  double fval = objective(delta.data(), g.data(), W.data(), QPd.data());
  if (!std::isfinite(fval)) {
    std::fill(delta.begin(), delta.end(), 0.0);
    fval = objective(delta.data(), g.data(), W.data(), QPd.data());
    if (!std::isfinite(fval)) return neg_inf;
  }

  // Damped Newton. H is factored at the current point before each step, and
  // the factor at the last point is the one the marginal and the gradient
  // read, so the mode costs one factorization per step taken plus one.
  if (!factor_H(data, f, W, ws)) return neg_inf;
  for (int it = 0; it < 50; it++) {
    double gmax = 0.0;
    for (int k = 0; k < n; k++) {
      grad[k] = g[k] - QPd[k];
      gmax = std::max(gmax, std::abs(grad[k]));
    }
    if (gmax < 1e-7) break;
    Eigen::Map<Eigen::VectorXd> gv(grad.data(), n);
    Eigen::VectorXd step = solve_H(ws, gv);

    // Halve the step until the objective does not decrease
    double t = 1.0;
    double fnew = neg_inf;
    std::vector<double> gn(n), Wn(n), QPn(n);
    for (int h = 0; h < 30; h++) {
      for (int k = 0; k < n; k++) trial[k] = delta[k] + t * step[k];
      fnew = objective(trial.data(), gn.data(), Wn.data(), QPn.data());
      if (std::isfinite(fnew) && fnew >= fval - 1e-12 * std::abs(fval)) break;
      t *= 0.5;
    }
    if (!std::isfinite(fnew)) return neg_inf;
    const bool stalled = std::abs(fnew - fval) < 1e-14 * std::max(1.0, std::abs(fval));
    delta = trial;
    g = gn;
    W = Wn;
    QPd = QPn;
    fval = fnew;
    if (!factor_H(data, f, W, ws)) return neg_inf;
    if (stalled) break;
  }
  ws.have_delta = true;
  ws.W = W;

  double log_det_Q = 0.0;
  for (int k = 0; k < n; k++) log_det_Q -= std::log(f.d[k]);
  ws.log_marginal = fval + 0.5 * log_det_Q - 0.5 * log_det_factor(ws);
  return ws.log_marginal;
}

// =====================================================================
// The selected inverse
// =====================================================================

// K^-1 on the sparsity pattern of its factor (P K P' = L D L', L unit lower
// triangular), by Takahashi's recursion, column by column from the last:
// for i > j in column j,
//   Z_ij = -sum_{k in col j, k > j} L_kj Z_ki,
//   Z_jj = 1/D_jj - sum_{k in col j, k > j} L_kj Z_kj.
// Every Z_ki the sum reads lies on the pattern, which is chordal. The leading
// n x n block of K^-1 is H^-1.
struct SelectedInverse {
  Eigen::SparseMatrix<double> L;  // strictly lower part, permuted order
  std::vector<double> Z;          // aligned with L's stored values
  std::vector<double> Zd;         // the diagonal, permuted order
  std::vector<int> perm;          // original index -> permuted index

  // Where entry (a, b) of K^-1 is kept, in permuted indices: a stored
  // position of Z, -(a + 2) for the diagonal entry a, -1 for none.
  int locate(int a, int b) const {
    if (a == b) return -(a + 2);
    if (a < b) std::swap(a, b);  // row a of column b
    const int* outer = L.outerIndexPtr();
    const int* inner = L.innerIndexPtr();
    const int* hit = std::lower_bound(inner + outer[b], inner + outer[b + 1], a);
    return (hit != inner + outer[b + 1] && *hit == a) ? static_cast<int>(hit - inner)
                                                     : -1;
  }
  double value(int code) const {
    if (code >= 0) return Z[code];
    if (code <= -2) return Zd[-code - 2];
    return 0.0;
  }
  // H^-1 at original indices (a, b), and where it is kept
  int position(int a, int b) const { return locate(perm[a], perm[b]); }
  double operator()(int a, int b) const { return value(position(a, b)); }
};

inline void selected_inverse(Workspace& ws, SelectedInverse& si) {
  {
    Eigen::SparseMatrix<double> Lfull = ws.ldlt.matrixL();
    si.L = Lfull.triangularView<Eigen::StrictlyLower>();
    si.L.makeCompressed();
  }
  const int size = si.L.cols();
  const int* outer = si.L.outerIndexPtr();
  const int* inner = si.L.innerIndexPtr();
  const double* val = si.L.valuePtr();
  const Eigen::VectorXd& D = ws.ldlt.vectorD();
  si.Z.assign(si.L.nonZeros(), 0.0);
  si.Zd.assign(size, 0.0);
  const auto& P = ws.ldlt.permutationP();
  si.perm.assign(size, 0);
  for (int i = 0; i < size; i++) si.perm[i] = P.indices()[i];

  const bool record = ws.taka_pos.empty();
  size_t tp = 0;
  for (int j = size - 1; j >= 0; j--) {
    const int p0 = outer[j], p1 = outer[j + 1];
    // off-diagonal entries of column j, from the bottom
    for (int p = p1 - 1; p >= p0; p--) {
      const int i = inner[p];
      double s = 0.0;
      for (int q = p0; q < p1; q++) {
        if (record) ws.taka_pos.push_back(si.locate(inner[q], i));
        s += val[q] * si.value(ws.taka_pos[tp++]);
      }
      si.Z[p] = -s;
    }
    double s = 0.0;
    for (int q = p0; q < p1; q++) s += val[q] * si.Z[q];
    si.Zd[j] = 1.0 / D[j] - s;
  }
}

// =====================================================================
// The gradient
// =====================================================================

struct Gradient {
  std::vector<double> e_num, e_denom;  // dL/d(eta offset), per observation
  double phi_num = 0.0, phi_denom = 0.0;  // dL/d(phi)
  double log_sigma2 = 0.0, log_phi_space = 0.0, log_phi_time = 0.0;
};

// The gradient of the log marginal at the state laplace_log_marginal left in
// `ws`. has_phi_num / has_phi_denom say which dispersions are parameters.
inline void laplace_gradient(const ModelData& data, const ParamLayout& layout,
                             const double* off_num, const double* off_denom,
                             double phi_num, double phi_denom,
                             bool has_phi_num, bool has_phi_denom,
                             Workspace& ws, Gradient& out) {
  const Factors& f = ws.fac;
  const int n = f.n;
  const double b = denom_slope(data);
  const std::vector<double>& delta = ws.delta;

  SelectedInverse Z;
  selected_inverse(ws, Z);
  std::vector<double> Zcc(n);
  for (int c = 0; c < n; c++) Zcc[c] = Z(c, c);

  // a_c = 0.5 Z_cc sum_{i in c} l_i''' along the shift; v = H^-1 a
  std::vector<double> a(n, 0.0);
  for (int i = 0; i < data.N; i++) {
    const int c = ws.obs_cell[i];
    if (c < 0) continue;
    const Jet l = obs_jet(data, layout, i, off_num[i] + delta[c],
                          off_denom[i] + b * delta[c], 1.0, b, phi_num, phi_denom);
    a[c] += 0.5 * Zcc[c] * l.d3;
  }
  Eigen::Map<Eigen::VectorXd> av(a.data(), n);
  Eigen::VectorXd vv = solve_H(ws, av);
  ws.v.assign(vv.data(), vv.data() + n);
  const std::vector<double>& v = ws.v;

  // Observations: for an offset on predictor e, with u = (1, b) the shift,
  //   dL/d eta_e = l_e + v_c T2(u, e) + 0.5 Z_cc T3(u, u, e),
  // the mixed derivatives read off directional jets by polarization:
  //   T2(u, e) = [D^2_{u+e} - D^2_{u-e}] / 4,
  //   T3(u, u, e) = [D^3_{u+e} - D^3_{u-e} - 2 D^3_e] / 6.
  out.e_num.assign(data.N, 0.0);
  out.e_denom.assign(data.N, 0.0);
  for (int i = 0; i < data.N; i++) {
    const int c = ws.obs_cell[i];
    const double dc = c >= 0 ? delta[c] : 0.0;
    const double en = off_num[i] + dc, ed = off_denom[i] + b * dc;
    auto along = [&](double x, double y) {
      return obs_jet(data, layout, i, en, ed, x, y, phi_num, phi_denom);
    };
    const Jet j_num = along(1.0, 0.0);
    const Jet j_den = along(0.0, 1.0);
    out.e_num[i] = j_num.d1;
    out.e_denom[i] = j_den.d1;
    if (c < 0) continue;
    const Jet up_n = along(1.0 + 1.0, b), dn_n = along(1.0 - 1.0, b);
    const Jet up_d = along(1.0, b + 1.0), dn_d = along(1.0, b - 1.0);
    const double T2n = (up_n.d2 - dn_n.d2) / 4.0;
    const double T3n = (up_n.d3 - dn_n.d3 - 2.0 * j_num.d3) / 6.0;
    const double T2d = (up_d.d2 - dn_d.d2) / 4.0;
    const double T3d = (up_d.d3 - dn_d.d3 - 2.0 * j_den.d3) / 6.0;
    out.e_num[i] += v[c] * T2n + 0.5 * Zcc[c] * T3n;
    out.e_denom[i] += v[c] * T2d + 0.5 * Zcc[c] * T3d;
  }

  // Dispersions: l, D_u l and D^2_u l differentiated in phi on jets whose
  // coefficients are forward duals
  auto dispersion = [&](bool num) {
    double total = 0.0;
    for (int i = 0; i < data.N; i++) {
      const int c = ws.obs_cell[i];
      const double dc = c >= 0 ? delta[c] : 0.0;
      const fwd::Dual pn(phi_num, num ? 1.0 : 0.0);
      const fwd::Dual pd(phi_denom, num ? 0.0 : 1.0);
      const JetD l = ratiod::ratio_obs_log_lik<JetD>(
          data, layout, i,
          JetD(fwd::Dual(off_num[i] + dc), fwd::Dual(1.0), fwd::Dual(0.0), fwd::Dual(0.0)),
          JetD(fwd::Dual(off_denom[i] + b * dc), fwd::Dual(b), fwd::Dual(0.0), fwd::Dual(0.0)),
          JetD::constant(pn), JetD::constant(pd), JetD(0.0), JetD(0.0));
      total += l.v.grad;
      if (c >= 0) total += v[c] * l.d1.grad + 0.5 * Zcc[c] * l.d2.grad;
    }
    return total;
  };
  out.phi_num = has_phi_num ? dispersion(true) : 0.0;
  out.phi_denom = has_phi_denom ? dispersion(false) : 0.0;

  // Hyperparameters, through Q alone:
  //   d/dp [0.5 log|Q| - 0.5 d'Qd] - 0.5 tr(Z dQ) - v' dQ d,
  // with Q = sum_k r_k r_k' / d_k and dr_k = -sum_j dB_kj e_nb(j).
  // The H^-1 entries are read at positions recorded on the first call.
  if (ws.z_pos.empty()) {
    for (int k = 0; k < n; k++) {
      const int p0 = f.nb_ptr[k], p1 = f.nb_ptr[k + 1];
      ws.z_pos.push_back(Z.position(k, k));
      for (int p = p0; p < p1; p++) ws.z_pos.push_back(Z.position(k, f.nb[p]));
      for (int p = p0; p < p1; p++) {
        for (int r = p0; r < p1; r++) ws.z_pos.push_back(Z.position(f.nb[r], f.nb[p]));
      }
    }
  }
  std::vector<double> w;
  size_t zp = 0;
  double g_sigma2 = 0.0, g_space = 0.0, g_time = 0.0;
  for (int k = 0; k < n; k++) {
    const int p0 = f.nb_ptr[k], p1 = f.nb_ptr[k + 1];
    const double dk = f.d[k];
    const double q = row_dot(f, k, delta.data());
    const double vr = row_dot(f, k, v.data());

    // r_k' Z r_k and w_j = Z_{k,nb j} - sum_l B_kl Z_{nb l, nb j}, so that
    // r_k' Z dr_k = -sum_j dB_kj w_j
    double rZr = Z.value(ws.z_pos[zp++]);
    w.assign(p1 - p0, 0.0);
    for (int p = p0; p < p1; p++) {
      const double zkj = Z.value(ws.z_pos[zp++]);
      rZr -= 2.0 * f.B[p] * zkj;
      w[p - p0] = zkj;
    }
    for (int p = p0; p < p1; p++) {
      double s = 0.0;
      for (int r = p0; r < p1; r++) s += f.B[r] * Z.value(ws.z_pos[zp++]);
      rZr += f.B[p] * s;
      w[p - p0] -= s;
    }

    auto term = [&](double ddk, const std::vector<double>* dB) {
      double dr_delta = 0.0, dr_v = 0.0, rZdr = 0.0;
      if (dB) {
        for (int p = p0; p < p1; p++) {
          const double db = (*dB)[p];
          dr_delta -= db * delta[f.nb[p]];
          dr_v -= db * v[f.nb[p]];
          rZdr -= db * w[p - p0];
        }
      }
      const double inv_d = 1.0 / dk, inv_d2 = inv_d * inv_d;
      return -0.5 * ddk * inv_d - q * dr_delta * inv_d + 0.5 * q * q * ddk * inv_d2
             + 0.5 * ddk * inv_d2 * rZr - inv_d * rZdr
             + ddk * inv_d2 * vr * q - inv_d * (dr_v * q + vr * dr_delta);
    };
    g_sigma2 += term(f.dd_sigma2[k], nullptr);
    g_space += term(f.dd_space[k], &f.dB_space);
    g_time += term(f.dd_time[k], &f.dB_time);
  }
  out.log_sigma2 = g_sigma2;
  out.log_phi_space = g_space;
  out.log_phi_time = g_time;
}

// =====================================================================
// Entry points
// =====================================================================

// The log marginal at `params`. The last evaluation is kept, so the density
// and the gradient at the same parameters share one solve; the offsets and
// dispersions are functions of `params`, which is the whole key.
inline double evaluate(const ModelData& data, const ParamLayout& layout,
                       const std::vector<double>& params,
                       const double* off_num, const double* off_denom,
                       double phi_num, double phi_denom) {
  Workspace& ws = workspace();
  ws.bind(data.unique_id);
  if (ws.last_ok && ws.last_params == params) return ws.last_log_marginal;
  const double sigma2 = std::exp(-params[layout.log_tau_st_idx]);
  const double phi_space = std::exp(params[layout.log_phi_st_space_idx]);
  const double phi_time = std::exp(params[layout.log_phi_st_time_idx]);
  const double lm = laplace_log_marginal(data, layout, off_num, off_denom,
                                         phi_num, phi_denom, sigma2, phi_space,
                                         phi_time, ws);
  ws.last_params = params;
  ws.last_log_marginal = lm;
  ws.last_ok = true;
  return lm;
}

// The log marginal and its gradient at `params`. False where the marginal is
// not finite, a rejected parameter state.
inline bool gradient(const ModelData& data, const ParamLayout& layout,
                     const std::vector<double>& params,
                     const double* off_num, const double* off_denom,
                     double phi_num, double phi_denom, bool has_phi_num,
                     bool has_phi_denom, Gradient& out) {
  const double lm = evaluate(data, layout, params, off_num, off_denom, phi_num,
                             phi_denom);
  if (!std::isfinite(lm)) return false;
  laplace_gradient(data, layout, off_num, off_denom, phi_num, phi_denom,
                   has_phi_num, has_phi_denom, workspace(), out);
  return true;
}

// =====================================================================
// A draw of the interaction
// =====================================================================

// delta ~ N(dh, H^-1). With Q = R'R from the Vecchia rows, R = D^-1/2 (I - B),
// y = R' z1 + W^1/2 z2 + B_P z3 has covariance Q + W + P = H, so H^-1 y has
// covariance H^-1.
template <typename Rng>
inline void draw(const ModelData& data, const Workspace& ws, Rng& rng,
                 double* out) {
  const Factors& f = ws.fac;
  const int n = f.n;
  std::normal_distribution<double> rnorm(0.0, 1.0);
  Eigen::VectorXd y = Eigen::VectorXd::Zero(n);
  for (int k = 0; k < n; k++) {
    const double zk = rnorm(rng) / std::sqrt(f.d[k]);
    y[k] += zk;
    for (int p = f.nb_ptr[k]; p < f.nb_ptr[k + 1]; p++) y[f.nb[p]] -= f.B[p] * zk;
  }
  for (int k = 0; k < n; k++) y[k] += std::sqrt(std::max(ws.W[k], 0.0)) * rnorm(rng);
  const int m = data.spatiotemporal_data.n_spatial + data.spatiotemporal_data.n_times;
  std::vector<double> z3(m);
  for (int j = 0; j < m; j++) z3[j] = rnorm(rng);
  emit_margin_factor(data.spatiotemporal_data,
                     [&](int cell, int col, double v) { y[cell] += v * z3[col]; });
  Eigen::VectorXd x = solve_H(ws, y);
  for (int k = 0; k < n; k++) out[k] = ws.delta[k] + x[k];
}

}  // namespace ratiod_st_laplace

#endif  // RATIOD_ST_GP_LAPLACE_H
