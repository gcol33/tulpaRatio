#ifndef TULPARATIO_HMC_INTEGRATOR_H
#define TULPARATIO_HMC_INTEGRATOR_H

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>
#include "linalg_fast.h"
#include "hmc_sampler.h"

namespace ratiod_hmc {

// Gradient implementations may replace their output vector. Only scratch
// storage is passed to them; the integrator's score buffer keeps its address.
struct GradientEvaluator {
  const ModelData& data;
  const ParamLayout& layout;
  std::vector<double>& params;
  std::vector<double>& scratch;
  GradientFn gradient_fn = compute_gradient;

  void operator()(const double* q, double* grad, double& log_prob) const {
    if (q != params.data()) std::copy_n(q, params.size(), params.data());
    gradient_fn(params, data, layout, scratch, &log_prob);
    std::copy_n(scratch.data(), params.size(), grad);
  }
};

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
// evaluate writes through grad without resizing its owning storage.
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

template<class Metric>
LeapfrogResultWithGrad integrate_copy(const std::vector<double>& q,
    const std::vector<double>& p, double epsilon, const Metric& metric,
    const ModelData& data, const ParamLayout& layout,
    const std::vector<double>* initial_grad = nullptr,
    GradientFn gradient_fn = compute_gradient) {
  LeapfrogResultWithGrad result;
  result.q = q; result.p = p;
  result.grad.resize(q.size());
  std::vector<double> scratch(q.size()), displacement(q.size());
  const GradientEvaluator evaluate{data,layout,result.q,scratch,gradient_fn};
  if (initial_grad) result.grad = *initial_grad;
  else evaluate(q.data(),result.grad.data(),result.log_prob);
  result.divergent = metric_leapfrog(result.q.data(),result.p.data(),result.grad.data(),
      q.size(),epsilon,metric,displacement.data(),evaluate,result.log_prob);
  return result;
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

template<class Metric>
double find_epsilon(const std::vector<double>& q, const ModelData& data,
    const ParamLayout& layout, std::mt19937& rng, const Metric& metric,
    GradientFn gradient_fn = compute_gradient) {
  std::vector<double> params(q.size()), scratch(q.size());
  const GradientEvaluator evaluate{data,layout,params,scratch,gradient_fn};
  return metric_find_epsilon(q,metric,rng,evaluate);
}

}
#endif
