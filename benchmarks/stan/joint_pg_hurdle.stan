// Joint Poisson-Gamma model with SHARED random effects + Hurdle
// Matches numdenom ratiod_hurdle_poisson() with (1|site)
// Row 17 in gradient_methods.md
//
// KEY INSIGHT: Hurdle applies to numerator (count) only, denom (effort) is Gamma
//
// Hurdle Poisson:
// P(Y=0) = 1 - theta
// P(Y=y) = theta * TruncPoisson(y|mu), y > 0
// where TruncPoisson is Poisson conditional on Y > 0
//
// theta is modeled via logit link: logit(theta) = gamma_0

data {
  int<lower=1> N;
  array[N] int<lower=0> y_num;         // Numerator counts (Hurdle-Poisson)
  vector<lower=0>[N] y_denom;          // Denominator (Gamma-distributed effort)
  int<lower=1> p;                      // Number of predictors
  matrix[N, p] X;                      // Design matrix (shared)

  // Random effects
  int<lower=1> n_groups;               // Number of RE groups (sites)
  array[N] int<lower=1,upper=n_groups> group_idx;  // Group assignment
}

parameters {
  vector[p] beta_num;                  // Numerator coefficients
  vector[p] beta_denom;                // Denominator coefficients
  real<lower=0> shape;                 // Gamma shape parameter

  // SHARED random effects (non-centered)
  vector[n_groups] z_re;               // Standard normal RE
  real<lower=0> sigma_re;              // RE standard deviation

  // Hurdle parameter
  real logit_theta;                    // logit(theta) = logit(P(Y > 0))
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

  // Linear predictors with SHARED RE
  eta_num = X * beta_num;
  eta_denom = X * beta_denom;

  for (n in 1:N) {
    eta_num[n] += re[group_idx[n]];
    eta_denom[n] += re[group_idx[n]];
  }

  // Priors matching numdenom defaults
  beta_num ~ normal(0, 10);
  beta_denom ~ normal(0, 10);
  shape ~ gamma(2, 0.1);

  // RE priors
  z_re ~ std_normal();
  sigma_re ~ cauchy(0, 2.5);

  // Hurdle prior (logit scale)
  logit_theta ~ normal(0, 1.5);

  // Means
  mu_num = exp(eta_num);
  mu_denom = exp(eta_denom);

  // Hurdle-Poisson likelihood for numerator
  {
    real log_theta = log_inv_logit(logit_theta);        // log(theta)
    real log_1m_theta = log1m_inv_logit(logit_theta);   // log(1-theta)

    for (n in 1:N) {
      if (y_num[n] == 0) {
        // P(Y=0) = 1 - theta
        target += log_1m_theta;
      } else {
        // P(Y=y) = theta * Poisson(y|mu) / (1 - exp(-mu))
        real log_normalizer = log1m_exp(-mu_num[n]);  // log(1 - exp(-mu))
        target += log_theta + poisson_lpmf(y_num[n] | mu_num[n]) - log_normalizer;
      }
    }
  }

  // Gamma likelihood for denominator
  for (n in 1:N) {
    if (y_denom[n] > 0) {
      target += gamma_lpdf(y_denom[n] | shape, shape / mu_denom[n]);
    }
  }
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

    // Hurdle-Poisson
    if (y_num[n] == 0) {
      log_lik[n] = log_1m_theta;
    } else {
      real log_normalizer = log1m_exp(-mu_num_n);
      log_lik[n] = log_theta + poisson_lpmf(y_num[n] | mu_num_n) - log_normalizer;
    }

    // Gamma
    if (y_denom[n] > 0) {
      log_lik[n] += gamma_lpdf(y_denom[n] | shape, shape / mu_denom_n);
    }
  }
}
