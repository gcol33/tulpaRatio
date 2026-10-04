// hmc_svc_field.h
// The NNGP SVC terms at the coordinate they are sampled in.
//
// Centred, each term's block holds the field w_j itself and carries the NNGP
// prior directly. Non-centred, the block holds z_j ~ N(0, I) and the field is
// w_j = L(sigma2_j, phi_j) z_j through the NNGP autoregression the spatial GP
// uses (ratiod_gp::nngp_nc_forward), so sigma2_j reaches the field through the
// transform and the field's level carries unit prior scale. Every density, both
// gradient paths and the draw store read the field through this file.

#ifndef RATIOD_HMC_SVC_FIELD_H
#define RATIOD_HMC_SVC_FIELD_H

#include <vector>
#include <cmath>
#include <algorithm>

#include "hmc_svc.h"
#include "hmc_gp.h"
#include "tls_workspace.h"

namespace ratiod_svc {

// The SVC neighbour structure as single-scale GPData, so the non-centred
// transform written against one NNGP field serves each SVC term. Built once
// when the model is set up. The pairwise neighbour distances are the ones the
// centred density computes from the coordinates (nngp_log_lik), so both
// coordinates place the same NNGP on w.
inline ratiod_gp::GPData make_svc_gp_view(const SVCData& s) {
  ratiod_gp::GPData g;
  const int N = s.n_obs;
  const int nn = s.nn;
  g.n_obs = N;
  g.nn = nn;
  g.coords = s.coords;
  g.nn_idx = s.nn_idx;
  g.nn_dist = s.nn_dist;
  g.nn_order = s.nn_order;
  g.nn_order_inv = s.nn_order_inv;
  g.obs_to_loc.resize(N);
  for (int i = 0; i < N; i++) g.obs_to_loc[i] = i;
  g.cov_type = s.cov_type;
  g.nu = 1.5;  // ratiod_cov's Matern is the 3/2 member
  g.shared = s.shared;

  g.nn_neighbor_dist.assign(static_cast<size_t>(N) * nn * nn, 0.0);
  for (int i = 0; i < N; i++) {
    int n_nb = 0;
    while (n_nb < nn && s.nn_idx[i * nn + n_nb] > 0) n_nb++;
    for (int j1 = 0; j1 < n_nb; j1++) {
      const int l1 = s.nn_order[s.nn_idx[i * nn + j1] - 1];
      for (int j2 = 0; j2 < n_nb; j2++) {
        if (j1 == j2) continue;
        const int l2 = s.nn_order[s.nn_idx[i * nn + j2] - 1];
        const double dx = s.coords[l1 * 2] - s.coords[l2 * 2];
        const double dy = s.coords[l1 * 2 + 1] - s.coords[l2 * 2 + 1];
        g.nn_neighbor_dist[static_cast<size_t>(i) * nn * nn + j1 * nn + j2] =
            std::sqrt(dx * dx + dy * dy);
      }
    }
  }
  return g;
}

// One transform workspace per term: the adjoint of term j reads the factors
// its own forward pass cached, after every term's forward pass has run. The
// density reads one slot and a gradient pass another, so a density evaluated
// between a gradient's forward and backward passes leaves the factors the
// backward pass reads in place.
constexpr int SVC_DENSITY_SLOT = 0;
constexpr int SVC_GRAD_SLOT = 1;

template <typename T, int Slot>
inline std::vector<ratiod_gp::NNGPNCWorkspaceT<T>>& svc_nc_ws() {
  RATIOD_TLS_WORKSPACE(std::vector<ratiod_gp::NNGPNCWorkspaceT<T>>, ws);
  return ws;
}

// The term-major field (n_svc x n_obs) the sampled block stands for, written
// to `w_out`. `sigma2` and `phi` are per term.
template <int Slot, typename T>
inline void svc_field(const T* block, const T* sigma2, const T* phi,
                      const SVCData& svc, const ratiod_gp::GPData& view,
                      bool noncentered, T* w_out) {
  const int n_obs = svc.n_obs;
  const int n_svc = svc.n_svc;
  if (!noncentered) {
    std::copy(block, block + static_cast<size_t>(n_obs) * n_svc, w_out);
    return;
  }
  std::vector<ratiod_gp::NNGPNCWorkspaceT<T>>& ws = svc_nc_ws<T, Slot>();
  if ((int)ws.size() < n_svc) ws.resize(n_svc);
  for (int j = 0; j < n_svc; j++) {
    const size_t off = static_cast<size_t>(j) * n_obs;
    ratiod_gp::nngp_nc_forward(block + off, sigma2[j], phi[j], view, ws[j]);
    std::copy(ws[j].w.begin(), ws[j].w.begin() + n_obs, w_out + off);
  }
}

// The field's likelihood gradient dL/dw (term-major, already projected by
// svc_center_terms) onto the sampled block and, non-centred, onto each
// term's log sigma2 and log phi. Non-centred, the block's N(0, I) prior is
// seeded here too (nngp_nc_backward returns prior + likelihood), so a caller
// adds no field prior of its own. Reads the factors the last svc_field() call
// on this thread and slot cached.
template <int Slot>
inline void svc_field_accumulate(const double* block, const double* sigma2,
                                 const double* phi, const SVCData& svc,
                                 const ratiod_gp::GPData& view,
                                 bool noncentered, const double* dL_dw,
                                 int w_start, int log_sigma2_start,
                                 int log_phi_start, double* grad) {
  const int n_obs = svc.n_obs;
  const int n_svc = svc.n_svc;
  if (!noncentered) {
    for (int k = 0; k < n_obs * n_svc; k++) grad[w_start + k] += dL_dw[k];
    return;
  }
  std::vector<ratiod_gp::NNGPNCWorkspace>& ws = svc_nc_ws<double, Slot>();
  std::vector<double> grad_z(n_obs);
  for (int j = 0; j < n_svc; j++) {
    const size_t off = static_cast<size_t>(j) * n_obs;
    double g_sigma2 = 0.0, g_phi = 0.0, g_phi_jac = 0.0;
    ratiod_gp::nngp_nc_backward(block + off, sigma2[j], phi[j], view, ws[j],
                                dL_dw + off, grad_z.data(), g_sigma2, g_phi,
                                g_phi_jac);
    for (int i = 0; i < n_obs; i++) grad[w_start + off + i] += grad_z[i];
    grad[log_sigma2_start + j] += g_sigma2;
    grad[log_phi_start + j] += g_phi;
  }
}

}  // namespace ratiod_svc

#endif  // RATIOD_HMC_SVC_FIELD_H
