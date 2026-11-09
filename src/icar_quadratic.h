#ifndef RATIOD_ICAR_QUADRATIC_H
#define RATIOD_ICAR_QUADRATIC_H

namespace ratiod_icar {
// Diagonal and edge weights describe Q; neighbours are indexed from zero.
template<class T, class Degree, class Neighbor, class Diagonal, class Edge>
inline T quadratic(const T* phi, int size, Degree degree, Neighbor neighbor,
                   Diagonal diagonal, Edge edge) {
  T value = T(0.0);
  for (int i = 0; i < size; ++i) {
    value = value + T(diagonal(i)) * phi[i] * phi[i];
    for (int k = 0; k < degree(i); ++k) {
      const int j = neighbor(i, k);
      const double weight = edge(i, j);
      if (weight != 0.0) value = value - T(weight) * phi[i] * phi[j];
    }
  }
  return value;
}
}
#endif
