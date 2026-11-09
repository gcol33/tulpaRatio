// Joint Poisson-Gamma model with SHARED RE + ICAR spatial + Zero-Inflation
// Matches numdenom ratiod_zi_poisson() with (1|site) + spatial_car()
// Row 24 in gradient_methods.md
//
// KEY: ZI applies to numerator (count) only, denom (effort) is Gamma
// Both RE and spatial effects are SHARED between num and denom

functions {
  // ICAR log prior - pairwise differences formulation
  real icar_lpdf(vector phi, int J, array[] int n_neighbors,
                 int n_edges, array[] int edge1, array[] int edge2, real tau) {
    real quad_form = 0;
    for (e in 1:n_edges) {
      quad_form += square(phi[edge1[e]] - phi[edge2[e]]);
    }
    return 0.5 * (J - 1) * log(tau) - 0.5 * tau * quad_form;
  }
}

data {
  int<lower=1> N;
  array[N] int<lower=0> y_num;         // Numerator counts (ZI-Poisson)
  vector<lower=0>[N] y_denom;          // Denominator (Gamma-distributed effort)
  int<lower=1> p;                      // Number of predictors
  matrix[N, p] X;                      // Design matrix (shared)

  // Random effects
  int<lower=1> n_groups;               // Number of RE groups (sites)
  array[N] int<lower=1,upper=n_groups> group_idx;  // Group assignment

  // Spatial structure
  int<lower=1> J;                      // Number of spatial units
  array[N] int<lower=1,upper=J> spatial_idx;       // Spatial unit for each obs
  array[J] int<lower=0> n_neighbors;   // Number of neighbors per unit
  int<lower=0> n_edges;                // Total number of edges
  array[n_edges] int<lower=1,upper=J> edge1;       // First node of each edge
  array[n_edges] int<lower=1,upper=J> edge2;       // Second node of each edge
}

parameters {
  vector[p] beta_num;                  // Numerator coefficients
  vector[p] beta_denom;                // Denominator coefficients
  real<lower=0> shape;                 // Gamma shape parameter

  // SHARED random effects (non-centered)
  vector[n_groups] z_re;               // Standard normal RE
  real<lower=0> sigma_re;              // RE standard deviation

  // SHARED spatial effects
  vector[J] phi_spatial;               // Spatial effects (sum-to-zero constrained)
  real<lower=0> tau_spatial;           // Spatial precision

  // Zero-inflation parameter
  real logit_zi;                       // logit(pi) - intercept only
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

  // Linear predictors with SHARED RE and SHARED spatial
  eta_num = X * beta_num;
  eta_denom = X * beta_denom;

  for (n in 1:N) {
    eta_num[n] += re[group_idx[n]] + phi_spatial[spatial_idx[n]];
    eta_denom[n] += re[group_idx[n]] + phi_spatial[spatial_idx[n]];
  }

  // Priors matching numdenom defaults
  beta_num ~ normal(0, 10);
  beta_denom ~ normal(0, 10);
  shape ~ gamma(2, 0.1);

  // RE priors
  z_re ~ std_normal();
  sigma_re ~ cauchy(0, 2.5);

  // Spatial priors (matching numdenom: tau ~ Gamma(1, 0.01))
  tau_spatial ~ gamma(1, 0.01);
  phi_spatial ~ icar(J, n_neighbors, n_edges, edge1, edge2, tau_spatial);

  // Soft sum-to-zero constraint on phi_spatial
  sum(phi_spatial) ~ normal(0, 0.001 * J);

  // ZI prior (logit scale, N(0, 1.5) is weakly informative)
  logit_zi ~ normal(0, 1.5);

  // Means
  mu_num = exp(eta_num);
  mu_denom = exp(eta_denom);

  // ZI-Poisson likelihood for numerator
  {
    real zi_prob = inv_logit(logit_zi);
    for (n in 1:N) {
      if (y_num[n] == 0) {
        target += log_sum_exp(
          log(zi_prob),
          log1m(zi_prob) - mu_num[n]
        );
      } else {
        target += log1m(zi_prob) + poisson_lpmf(y_num[n] | mu_num[n]);
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
  real zi_prob = inv_logit(logit_zi);

  for (n in 1:N) {
    real eta_num_n = X[n] * beta_num + re[group_idx[n]] + phi_spatial[spatial_idx[n]];
    real eta_denom_n = X[n] * beta_denom + re[group_idx[n]] + phi_spatial[spatial_idx[n]];
    real mu_num_n = exp(eta_num_n);
    real mu_denom_n = exp(eta_denom_n);

    // ZI-Poisson
    if (y_num[n] == 0) {
      log_lik[n] = log_sum_exp(log(zi_prob), log1m(zi_prob) - mu_num_n);
    } else {
      log_lik[n] = log1m(zi_prob) + poisson_lpmf(y_num[n] | mu_num_n);
    }

    // Gamma
    if (y_denom[n] > 0) {
      log_lik[n] += gamma_lpdf(y_denom[n] | shape, shape / mu_denom_n);
    }
  }
}
