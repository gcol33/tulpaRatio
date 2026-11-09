#ifndef TULPARATIO_HMC_INTEGRATOR_H
#define TULPARATIO_HMC_INTEGRATOR_H

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>
#include "linalg_fast.h"

namespace ratiod_hmc {

template<class Metric>
inline void advance_metric_position(const Metric& metric, double* q, const double* p,
                                     int n, double epsilon, double* displacement) {
  metric.inv_mass_times_p(p,displacement,epsilon);
  for (int i = 0; i < n; ++i) q[i] += displacement[i];
}

struct IdentityMetric {
  int n;
  void sample_momentum(double* p, std::mt19937& rng) const {
    std::normal_distribution<double> normal(0.0,1.0);
    for (int i = 0; i < n; ++i) p[i] = normal(rng);
  }
  double kinetic_energy(const double* p) const {
    return 0.5*ratiod_linalg::norm_squared(p,n);
  }
  void inv_mass_times_p(const double* p, double* out, double scale = 1.0) const {
    for (int i = 0; i < n; ++i) out[i] = scale*p[i];
  }
  void advance_position(double* q, const double* p, double epsilon, double* scratch) const {
    advance_metric_position(*this,q,p,n,epsilon,scratch);
  }
};

struct DiagonalMetric {
  const std::vector<double>& inverse_mass;
  void sample_momentum(double* p, std::mt19937& rng) const {
    std::normal_distribution<double> normal(0.0,1.0);
    for (size_t i = 0; i < inverse_mass.size(); ++i)
      p[i] = normal(rng)/std::sqrt(inverse_mass[i]);
  }
  double kinetic_energy(const double* p) const {
    double sum = 0.0;
    for (size_t i = 0; i < inverse_mass.size(); ++i) sum += p[i]*p[i]*inverse_mass[i];
    return 0.5*sum;
  }
  void inv_mass_times_p(const double* p, double* out, double scale = 1.0) const {
    for (size_t i = 0; i < inverse_mass.size(); ++i) out[i] = scale*inverse_mass[i]*p[i];
  }
  void advance_position(double* q, const double* p, double epsilon, double* scratch) const {
    advance_metric_position(*this,q,p,inverse_mass.size(),epsilon,scratch);
  }
};

// The gradient buffer holds the score at q on entry and at the new q on exit.
// The metric scales its product before the position addition.
template<class Metric, class Evaluate>
inline bool metric_leapfrog(double* q, double* p, double* grad, int n,
                            double epsilon, const Metric& metric,
                            double* displacement, Evaluate evaluate, double& log_prob) {
  ratiod_linalg::axpy(0.5*epsilon,grad,p,n);
  metric.advance_position(q,p,epsilon,displacement);
  evaluate(q,grad,log_prob);
  ratiod_linalg::axpy(0.5*epsilon,grad,p,n);
  if (!std::isfinite(log_prob)) return true;
  for (int i = 0; i < n; ++i)
    if (std::abs(q[i]) > 1e10 || !std::isfinite(q[i])) return true;
  return false;
}

template<class Metric, class Evaluate>
double metric_find_epsilon(const std::vector<double>& q, const Metric& metric,
                           std::mt19937& rng, Evaluate evaluate) {
  const int n = q.size();
  std::vector<double> p(n), grad(n), displacement(n), initial_grad(n);
  metric.sample_momentum(p.data(),rng);
  double initial_lp;
  evaluate(q.data(),initial_grad.data(),initial_lp);
  const double initial_H = -initial_lp + metric.kinetic_energy(p.data());
  std::vector<double> qt(n), pt(n);
  double lp = 0.0;
  const auto trial = [&](double epsilon) {
    qt = q; pt = p; grad = initial_grad;
    const bool divergent = metric_leapfrog(qt.data(),pt.data(),grad.data(),n,
        epsilon,metric,displacement.data(),evaluate,lp);
    return divergent ? std::numeric_limits<double>::infinity()
                     : -lp + metric.kinetic_energy(pt.data()) - initial_H;
  };
  double epsilon = 1.0;
  double delta = trial(epsilon);
  const int direction = (!std::isfinite(delta) || delta > std::log(2.0)) ? -1 : 1;
  for (int iter = 0; iter < 50; ++iter) {
    epsilon *= direction == 1 ? 2.0 : 0.5;
    if (epsilon < 1e-10 || epsilon > 1e5) break;
    delta = trial(epsilon);
    if (direction == 1 && (!std::isfinite(delta) || delta > std::log(2.0))) break;
    if (direction == -1 && std::isfinite(delta) && delta < std::log(2.0)) break;
  }
  return std::max(1e-10,std::min(epsilon,1e3));
}

}
#endif
