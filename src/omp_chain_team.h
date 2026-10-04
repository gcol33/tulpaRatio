#ifndef TULPARATIO_OMP_CHAIN_TEAM_H
#define TULPARATIO_OMP_CHAIN_TEAM_H

#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ratiod_omp {

// The team used for chain-parallel work never shrinks within a session.
//
// Sizing it per fit lets it shrink between fits, and a shrink makes libgomp
// destroy the surplus workers, which runs whatever thread-exit hooks the
// process has registered. Per-thread scratch is held so that a dying worker
// frees nothing (tls_workspace.h); a team that only ever grows keeps the
// surplus workers alive as well.
//
// The size is the widest budget any fit has asked for so far, never
// omp_get_max_threads(). The nthreads-var is not a fixed quantity: the
// Polya-Gamma samplers move it through omp_set_num_threads(), so one of them
// ahead of the first chain fit would pin the team at 1 and serialize every
// chain fit for the rest of the session.
//
// Called only from the thread that launches a fit, which is R's, so the running
// maximum needs no synchronization.
inline int chain_team_size(int wanted) {
#ifdef _OPENMP
  static int team = 1;
  if (wanted > team) team = wanted;
  return team;
#else
  (void)wanted;
  return 1;
#endif
}

// Runs fn(c) for chains 0..n_chains-1, at most max_concurrent at a time.
//
// Concurrency is bounded by how many chains are offered to each region, not by
// resizing the team: a fit narrower than an earlier one leaves the surplus
// threads without an iteration and they sit out. That keeps `cores` meaning
// what it says while the team itself only ever grows.
template <typename Fn>
inline void for_each_chain(int n_chains, int max_concurrent, Fn fn) {
#ifdef _OPENMP
  // A per-fit thread count set via ScopedThreadCount (or the ambient
  // nthreads-var read directly, as compute_gradient_analytical's fallback
  // path does) is inherited into this team's workers as their own nthreads-var,
  // but OpenMP still runs any parallel region a worker starts at width 1
  // unless nesting is enabled: the default max-active-levels is 1. Without
  // this, a chain running concurrently with others could never also get
  // within-chain width, and `cores` could not express a budget split across
  // both dimensions at once.
  omp_set_max_active_levels(2);
  const int width = std::max(1, std::min(max_concurrent, n_chains));
  const int team = chain_team_size(width);
  for (int base = 0; base < n_chains; base += width) {
    const int stop = std::min(base + width, n_chains);
    #pragma omp parallel for schedule(static) num_threads(team)
    for (int c = base; c < stop; c++) {
      fn(c);
    }
  }
#else
  (void)max_concurrent;
  for (int c = 0; c < n_chains; c++) {
    fn(c);
  }
#endif
}

}  // namespace ratiod_omp

#endif  // TULPARATIO_OMP_CHAIN_TEAM_H
