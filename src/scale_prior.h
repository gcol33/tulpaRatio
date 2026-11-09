#ifndef RATIOD_SCALE_PRIOR_H
#define RATIOD_SCALE_PRIOR_H

#include "autodiff_utils.h"
#include <cmath>

namespace ratiod_scale {

inline double log_tau_conditional(double x, double quad, int rank,
                                  double U, double alpha) {
    return ratiod::math::pc_log_density(x, -0.5, U, alpha)
         + 0.5 * rank * x - 0.5 * quad * std::exp(x);
}

// Slice sampling on log precision retains the PC prior's nonconjugate form.
template<class Uniform>
inline double sample_tau(double tau, double quad, int rank,
                          double U, double alpha, Uniform uniform) {
    auto positive_uniform = [&]() {
        double u;
        do { u = uniform(); } while (u <= 0.0 || u >= 1.0);
        return u;
    };
    auto density = [&](double x) {
        return log_tau_conditional(x, quad, rank, U, alpha);
    };
    const double x = std::log(tau);
    const double height = density(x) + std::log(positive_uniform());
    double left = x - positive_uniform(), right = left + 1.0;
    while (density(left) > height) left -= 1.0;
    while (density(right) > height) right += 1.0;
    for (;;) {
        const double proposal = left + (right - left) * positive_uniform();
        if (density(proposal) >= height) return std::exp(proposal);
        if (proposal < x) left = proposal;
        else right = proposal;
    }
}

}
#endif
