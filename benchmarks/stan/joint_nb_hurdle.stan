// Joint NegBin-NegBin model with SHARED random effects + Hurdle
// Matches numdenom ratiod_hurdle_negbin() with (1|site)
// Row 47 in gradient_methods.md
//
// Hurdle NegBin for numerator:
// P(Y=0) = 1 - theta
// P(Y=y) = theta * TruncNegBin(y|mu,phi), y > 0
// where TruncNegBin is NegBin conditional on Y > 0

data {
  int<lower=1> N;
  array[N] int<lower=0> y_num;
  array[N] int<lower=0> y_denom;
  int<lower=1> p;
  matrix[N, p] X;

  // Random effects
  int<lower=1> n_groups;
  array[N] int<lower=1,upper=n_groups> group_idx;
}

parameters {
  vector[p] beta_num;
  vector[p] beta_denom;
  real<lower=0> phi_num;
  real<lower=0> phi_denom;

  // SHARED random effects
  vector[n_groups] z_re;
  real<lower=0> sigma_re;

  // Hurdle parameter
  real logit_theta;
}

transformed parameters {
  vector[n_groups] re;
  re = sigma_re * z_re;
}

model {
  vector[N] eta_num;
  vector[N] eta_denom;
  vector[N] mu_num;
  vector[N] mu_denom;

  eta_num = X * beta_num;
  eta_denom = X * beta_denom;

  for (n in 1:N) {
    eta_num[n] += re[group_idx[n]];
    eta_denom[n] += re[group_idx[n]];
  }

  // Priors
  beta_num ~ normal(0, 10);
  beta_denom ~ normal(0, 10);
  phi_num ~ gamma(2, 0.1);
  phi_denom ~ gamma(2, 0.1);

  // RE priors
  z_re ~ std_normal();
  sigma_re ~ cauchy(0, 2.5);

  // Hurdle prior
  logit_theta ~ normal(0, 1.5);

  // Means
  mu_num = exp(eta_num);
  mu_denom = exp(eta_denom);

  // Hurdle-NegBin likelihood for numerator
  {
    real log_theta = log_inv_logit(logit_theta);
    real log_1m_theta = log1m_inv_logit(logit_theta);

    for (n in 1:N) {
      // P(Y=0|NegBin) = (phi/(phi+mu))^phi
      real log_p0_nb = phi_num * log(phi_num / (phi_num + mu_num[n]));
      real log_1m_p0 = log1m_exp(log_p0_nb);  // log(1 - p0)

      if (y_num[n] == 0) {
        target += log_1m_theta;
      } else {
        // theta * NegBin(y|mu,phi) / (1 - p0)
        target += log_theta + neg_binomial_2_lpmf(y_num[n] | mu_num[n], phi_num) - log_1m_p0;
      }
    }
  }

  // Standard NegBin for denominator
  y_denom ~ neg_binomial_2(mu_denom, phi_denom);
}

generated quantities {
  vector[N] log_lik;
  real log_theta = log_inv_logit(logit_theta);
  real log_1m_theta = log1m_inv_logit(logit_theta);

  for (n in 1:N) {
    real eta_num_n = X[n] * beta_num + re[group_idx[n]];
    real eta_denom_n = X[n] * beta_denom + re[group_idx[n]];
    real mu_num_n = exp(eta_num_n);
    real mu_denom_n = exp(eta_denom_n);
    real log_p0_nb = phi_num * log(phi_num / (phi_num + mu_num_n));
    real log_1m_p0 = log1m_exp(log_p0_nb);

    if (y_num[n] == 0) {
      log_lik[n] = log_1m_theta;
    } else {
      log_lik[n] = log_theta + neg_binomial_2_lpmf(y_num[n] | mu_num_n, phi_num) - log_1m_p0;
    }

    log_lik[n] += neg_binomial_2_lpmf(y_denom[n] | mu_denom_n, phi_denom);
  }
}
