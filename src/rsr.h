#ifndef TULPARATIO_RSR_H
#define TULPARATIO_RSR_H

#include <vector>

// Restricted spatial regression. A field restricted to the orthogonal
// complement of the restricted covariates reaches the linear predictor as
// P f, where f is the field's value at each observation and P = I - Q Q',
// Q the N x r orthonormal basis of the covariates held row-major. The field's
// own prior reads f; only the predictor reads P f.
namespace ratiod_rsr {

template <typename T>
inline void project_out(std::vector<T>& f, const std::vector<double>& Q, int r) {
  const int n = static_cast<int>(f.size());
  std::vector<T> c(r, T(0.0));
  for (int i = 0; i < n; i++) {
    for (int k = 0; k < r; k++) c[k] = c[k] + T(Q[i * r + k]) * f[i];
  }
  for (int i = 0; i < n; i++) {
    T s = T(0.0);
    for (int k = 0; k < r; k++) s = s + T(Q[i * r + k]) * c[k];
    f[i] = f[i] - s;
  }
}

}  // namespace ratiod_rsr

#endif
