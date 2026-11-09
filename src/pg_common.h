#ifndef RATIOD_PG_COMMON_H
#define RATIOD_PG_COMMON_H

#include <RcppEigen.h>
#include <functional>
#include <vector>
#include "pg_rng.h"

namespace ratiod {

inline int pg_n_save(int iterations, int warmup, int thin) {
  if (thin < 1 || warmup < 0 || iterations <= warmup)
    Rcpp::stop("PG sampling requires thin > 0 and 0 <= warmup < iterations.");
  return 1 + (iterations - warmup - 1) / thin;
}

inline Eigen::LLT<Eigen::MatrixXd> pg_factor(const Eigen::MatrixXd& precision) {
  if (precision.rows() != precision.cols() || !precision.allFinite())
    Rcpp::stop("PG conditional precision is not positive definite.");
  Eigen::LLT<Eigen::MatrixXd> factor(precision);
  if (factor.info() != Eigen::Success)
    Rcpp::stop("PG conditional precision is not positive definite.");
  return factor;
}

inline Rcpp::NumericMatrix chol_decomp(const Rcpp::NumericMatrix& precision) {
  const Eigen::Map<const Eigen::MatrixXd> matrix(precision.begin(), precision.nrow(), precision.ncol());
  Eigen::MatrixXd lower = pg_factor(matrix).matrixL();
  return Rcpp::wrap(lower);
}

inline Rcpp::NumericVector update_beta(const Rcpp::NumericVector& kappa,
    const Rcpp::NumericVector& omega, const Rcpp::NumericMatrix& X,
    const Rcpp::NumericVector& offset, double prior_sd) {
  const int n = X.nrow(), p = X.ncol();
  Eigen::MatrixXd precision(p, p);
  Eigen::VectorXd rhs(p);
  for (int j = 0; j < p; ++j) {
    for (int k = j; k < p; ++k) {
      double sum = 0.0;
      for (int i = 0; i < n; ++i) sum += X(i, j) * omega[i] * X(i, k);
      precision(j, k) = precision(k, j) = sum;
    }
    precision(j, j) += 1.0 / (prior_sd * prior_sd);
    double sum = 0.0;
    for (int i = 0; i < n; ++i)
      sum += omega[i] == 0.0 ? X(i, j) * kappa[i] :
        X(i, j) * omega[i] * (kappa[i] / omega[i] - offset[i]);
    rhs[j] = sum;
  }
  auto factor = pg_factor(precision);
  Eigen::VectorXd z(p);
  for (int j = 0; j < p; ++j) z[j] = R::rnorm(0.0, 1.0);
  // Q = L L': the covariance of L'^{-1} z is Q^{-1}.
  const Eigen::VectorXd draw = factor.solve(rhs) + factor.matrixU().solve(z);
  return Rcpp::wrap(draw);
}

inline Rcpp::NumericVector update_re(const Rcpp::NumericVector& kappa,
    const Rcpp::NumericVector& omega, const Rcpp::NumericVector& offset,
    const Rcpp::IntegerVector& group, int n_groups, double sigma_re) {
  Rcpp::NumericVector sum_omega(n_groups), sum_resid(n_groups), re(n_groups);
  const double prior_prec = 1.0 / (sigma_re * sigma_re + 1e-10);
  for (int i = 0; i < kappa.size(); ++i) {
    const int g = group[i] - 1;
    sum_omega[g] += omega[i];
    sum_resid[g] += kappa[i] - omega[i] * offset[i];
  }
  for (int g = 0; g < n_groups; ++g) {
    const double variance = 1.0 / (sum_omega[g] + prior_prec);
    re[g] = R::rnorm(variance * sum_resid[g], std::sqrt(variance));
  }
  return re;
}

struct HalfCauchyScale {
  double sigma = 1.0;
  double aux = 1.0;
  double update(const Rcpp::NumericVector& effects, double scale) {
    double ss = 0.0;
    for (int j = 0; j < effects.size(); ++j) ss += effects[j] * effects[j];
    const double sigma_sq = 1.0 / R::rgamma((effects.size() + 1.0) / 2.0,
      1.0 / (ss / 2.0 + 1.0 / aux));
    aux = 1.0 / R::rgamma(1.0, 1.0 / (1.0 / (scale * scale) + 1.0 / sigma_sq));
    return sigma = std::sqrt(sigma_sq);
  }
};

class PGComponent {
public:
  virtual ~PGComponent() = default;
  virtual const Rcpp::NumericVector& contribution() const = 0;
  virtual void update(const Rcpp::NumericVector& kappa,
                      const Rcpp::NumericVector& omega,
                      const Rcpp::NumericVector& offset) = 0;
  virtual void save(int row) = 0;
};

class PGBeta : public PGComponent {
  const Rcpp::NumericMatrix& X_;
  Rcpp::NumericVector& beta_;
  Rcpp::NumericVector& eta_;
  Rcpp::NumericMatrix& draws_;
  double prior_sd_;
public:
  PGBeta(const Rcpp::NumericMatrix& X, Rcpp::NumericVector& beta,
         Rcpp::NumericVector& eta, Rcpp::NumericMatrix& draws, double prior_sd)
    : X_(X), beta_(beta), eta_(eta), draws_(draws), prior_sd_(prior_sd) {}
  const Rcpp::NumericVector& contribution() const override { return eta_; }
  void update(const Rcpp::NumericVector& kappa, const Rcpp::NumericVector& omega,
              const Rcpp::NumericVector& offset) override {
    beta_ = update_beta(kappa, omega, X_, offset, prior_sd_);
    for (int i = 0; i < X_.nrow(); ++i) {
      eta_[i] = 0.0;
      for (int j = 0; j < X_.ncol(); ++j) eta_[i] += X_(i, j) * beta_[j];
    }
  }
  void save(int row) override {
    for (int j = 0; j < beta_.size(); ++j) draws_(row, j) = beta_[j];
  }
};

class PGRe : public PGComponent {
  const Rcpp::IntegerVector& group_;
  Rcpp::NumericVector& re_;
  Rcpp::NumericVector& eta_;
  Rcpp::NumericMatrix& draws_;
  Rcpp::NumericVector& scale_draws_;
  double& sigma_;
  double prior_scale_;
  HalfCauchyScale prior_;
public:
  PGRe(const Rcpp::IntegerVector& group, Rcpp::NumericVector& re,
       Rcpp::NumericVector& eta, Rcpp::NumericMatrix& draws,
       Rcpp::NumericVector& scale_draws, double& sigma, double prior_scale)
    : group_(group), re_(re), eta_(eta), draws_(draws), scale_draws_(scale_draws),
      sigma_(sigma), prior_scale_(prior_scale) {}
  const Rcpp::NumericVector& contribution() const override { return eta_; }
  void update(const Rcpp::NumericVector& kappa, const Rcpp::NumericVector& omega,
              const Rcpp::NumericVector& offset) override {
    if (re_.size() == 0) return;
    re_ = update_re(kappa, omega, offset, group_, re_.size(), sigma_);
    sigma_ = prior_.update(re_, prior_scale_);
    for (int i = 0; i < eta_.size(); ++i) eta_[i] = re_[group_[i] - 1];
  }
  void save(int row) override {
    for (int j = 0; j < re_.size(); ++j) draws_(row, j) = re_[j];
    scale_draws_[row] = sigma_;
  }
};

class PGField : public PGComponent {
  const Rcpp::NumericVector& eta_;
  std::function<void(const Rcpp::NumericVector&, const Rcpp::NumericVector&,
                     const Rcpp::NumericVector&)> update_;
  std::function<void(int)> save_;
public:
  template<class Update, class Save>
  PGField(const Rcpp::NumericVector& eta, Update update, Save save)
    : eta_(eta), update_(update), save_(save) {}
  const Rcpp::NumericVector& contribution() const override { return eta_; }
  void update(const Rcpp::NumericVector& kappa, const Rcpp::NumericVector& omega,
              const Rcpp::NumericVector& offset) override { update_(kappa, omega, offset); }
  void save(int row) override { save_(row); }
};

template<class Augment>
inline void pg_run(int n_iter, int n_warmup, int thin, bool verbose,
                   const std::vector<PGComponent*>& components,
                   Rcpp::NumericVector& kappa, Rcpp::NumericVector& omega,
                   Rcpp::NumericMatrix& eta_draws, bool store_eta, Augment augment) {
  const int N = kappa.size();
  Rcpp::NumericVector eta(N), offset(N);
  auto predictor = [&](int excluded, Rcpp::NumericVector& out) {
    std::fill(out.begin(), out.end(), 0.0);
    for (int c = 0; c < static_cast<int>(components.size()); ++c) {
      if (c == excluded) continue;
      const auto& value = components[c]->contribution();
      for (int i = 0; i < N; ++i) out[i] += value[i];
    }
  };
  int row = 0;
  for (int iter = 0; iter < n_iter; ++iter) {
    predictor(-1, eta);
    augment(eta, kappa, omega);
    for (int c = 0; c < static_cast<int>(components.size()); ++c) {
      predictor(c, offset);
      components[c]->update(kappa, omega, offset);
    }
    if (iter >= n_warmup && (iter - n_warmup) % thin == 0) {
      for (auto* component : components) component->save(row);
      if (store_eta) {
        predictor(-1, eta);
        for (int i = 0; i < N; ++i) eta_draws(row, i) = eta[i];
      }
      ++row;
    }
    if (verbose && (iter + 1) % 500 == 0)
      Rcpp::Rcout << "Iteration " << iter + 1 << "/" << n_iter << std::endl;
    if ((iter + 1) % 100 == 0) Rcpp::checkUserInterrupt();
  }
}

inline void pg_run_binomial(const Rcpp::IntegerVector& trials, int n_iter,
    int n_warmup, int thin, bool verbose, const std::vector<PGComponent*>& components,
    Rcpp::NumericVector& kappa, Rcpp::NumericVector& omega,
    Rcpp::NumericMatrix& eta_draws, bool store_eta) {
  pg_run(n_iter, n_warmup, thin, verbose, components, kappa, omega, eta_draws,
    store_eta, [&](const Rcpp::NumericVector& eta, Rcpp::NumericVector&,
                   Rcpp::NumericVector& weights) {
      for (int i = 0; i < trials.size(); ++i) weights[i] = rpg_int(trials[i], eta[i]);
    });
}

}
#endif
