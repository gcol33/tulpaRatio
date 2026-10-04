// Exercises the workspace shape RATIOD_TLS_WORKSPACE produces against the team
// schedule that destroys workers.
//
// A `static thread_local Workspace ws;` holding heap buffers dies here: 0 of 5
// runs survive this schedule, because narrowing a team makes libgomp destroy
// the surplus workers and their thread-local destructors free through emutls
// storage that a separate thread-exit hook may already have released. The
// macro's shape carries no destructor, so it comes back with a finite number.

#include "tls_workspace.h"
#include "omp_sum.h"

#include <Rcpp.h>

#include <cstddef>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

struct TlsProbeWorkspace {
  std::vector<double> a, b;

  void size_to(std::size_t n) {
    if (a.size() < n) {
      a.assign(n, 1.0);
      b.assign(n, 2.0);
    }
  }
};

double touch(std::size_t n, int seed) {
  RATIOD_TLS_WORKSPACE(TlsProbeWorkspace, ws);
  ws.size_to(n);
  double s = 0.0;
  for (std::size_t i = 0; i < n; i++) {
    ws.a[i] = ws.b[i] * static_cast<double>(i + seed);
    s += ws.a[i];
  }
  return s;
}

// A free written through a dangling destructor surfaces at a later unrelated
// free, so each cycle ends by exercising the allocator on every worker.
double churn(int rounds) {
  return ratiod_omp::sum_range(rounds * 8, 0, [&](int r) {
    const std::size_t sz = static_cast<std::size_t>(1 + (r * 37) % 512);
    double* p = new double[sz];
    p[0] = static_cast<double>(sz);
    p[sz - 1] = 1.0;
    const double v = p[0] + p[sz - 1];
    delete[] p;
    return v;
  });
}

}  // namespace

// Alternates wide and narrow teams so every other cycle is a shrink, touching a
// macro-held workspace on every worker of every team.
// [[Rcpp::export]]
double cpp_tls_workspace_shrink(int cycles, int work, int n) {
  double acc = 0.0;
  const std::size_t len = static_cast<std::size_t>(n);
#ifdef _OPENMP
  const int widths[] = {16, 2, 12, 1, 8, 3, 32, 1, 4, 2};
  const int n_widths = 10;
  for (int cy = 0; cy < cycles; cy++) {
    const double local = ratiod_omp::sum_range(
        work, widths[cy % n_widths],
        [&](int i) { return touch(len, i + cy); });
    acc += local + churn(32);
  }
#else
  for (int cy = 0; cy < cycles; cy++) {
    for (int i = 0; i < work; i++) acc += touch(len, i + cy);
  }
#endif
  return acc;
}
