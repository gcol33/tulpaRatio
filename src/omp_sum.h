// omp_sum.h
// Floating-point sums over an OpenMP team without the `reduction` clause.
//
// libomp's tree reduction on aarch64-w64-mingw (llvm-mingw clang 19, Rtools45
// aarch64) drops partial sums once the team is as wide as the machine: a
// 4-thread `reduction(+:double)` over a plain loop returned the serial value in
// 0 of 500 repeats on a 4-core runner while 1 and 2 threads were exact. Each
// thread here sums one contiguous chunk into a slot of its own and the slots
// are added serially in thread order, so nothing is combined by the runtime and
// the result is a function of the team size alone.

#ifndef TULPARATIO_OMP_SUM_H
#define TULPARATIO_OMP_SUM_H

#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ratiod_omp {

// Sum of f(i) for i in [0, n). `n_threads` <= 0 takes the runtime's maximum.
// f is called from several threads at once and must not write shared state.
template <typename F>
inline double sum_range(int n, int n_threads, F&& f) {
#ifdef _OPENMP
  const int cap = n_threads > 0 ? n_threads : omp_get_max_threads();
  if (cap > 1 && n > 1) {
    struct alignas(64) Slot {
      double v = 0.0;
    };
    std::vector<Slot> part(static_cast<size_t>(cap));
#pragma omp parallel num_threads(cap)
    {
      const int t = omp_get_thread_num();
      const int team = omp_get_num_threads();
      const long long lo = static_cast<long long>(n) * t / team;
      const long long hi = static_cast<long long>(n) * (t + 1) / team;
      double acc = 0.0;
      for (long long i = lo; i < hi; i++) acc += f(static_cast<int>(i));
      part[static_cast<size_t>(t)].v = acc;
    }
    double total = 0.0;
    for (const Slot& s : part) total += s.v;
    return total;
  }
#else
  (void)n_threads;
#endif
  double total = 0.0;
  for (int i = 0; i < n; i++) total += f(i);
  return total;
}

}  // namespace ratiod_omp

#endif  // TULPARATIO_OMP_SUM_H
