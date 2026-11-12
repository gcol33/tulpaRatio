test_that("production inverse logit matches the logistic CDF", {
  x <- c(-1000, -40, -10, -1, 0, 1, 10, 40, 1000)
  expect_equal(tulpaRatio:::cpp_test_inverse_logit(x), plogis(x), tolerance = 1e-15)
})

test_that("production leapfrog and Hamiltonian match a Gaussian reference", {
  q <- c(1, 0.5, -0.3)
  p <- c(0.2, -0.4, 0.6)
  epsilon <- 0.1
  result <- tulpa:::cpp_test_integrator(q, p, epsilon)
  half <- p - epsilon * q / 2
  expected_q <- q + epsilon * half
  expected_p <- half - epsilon * expected_q / 2
  expect_equal(result$q, expected_q, tolerance = 1e-10)
  expect_equal(result$p, expected_p, tolerance = 1e-10)
  expect_equal(result$hamiltonian, sum(q^2 + p^2) / 2, tolerance = 1e-10)
  reverse <- tulpa:::cpp_test_integrator(result$q, -result$p, epsilon)
  expect_equal(reverse$q, q, tolerance = 1e-10)
  expect_equal(reverse$p, -p, tolerance = 1e-10)
})

test_that("production Cholesky matches R on positive definite matrices", {
  for (a in list(diag(3), matrix(c(4, 2, 2, 3), 2),
                 matrix(c(4, 2, 1, 2, 5, 3, 1, 3, 6), 3))) {
    expect_equal(as.matrix(tulpaRatio:::cpp_test_cholesky(a)),
                 t(chol(a)), tolerance = 1e-10)
  }
})

test_that("autodiff unary operations match exact values and derivatives", {
  cases <- list(
    exp_chain = list(x = c(0, 0.5, 1, -0.5, -1), f = function(x) exp(x^2),
                     d = function(x) 2 * x * exp(x^2)),
    lgamma = list(x = c(0.5, 1, 2, 5, 10), f = lgamma, d = digamma),
    softplus = list(x = c(-50, -5, -1, 0, 1, 5, 50),
                    f = function(x) log1p(exp(x)), d = plogis),
    inv_logit = list(x = c(-3, -1, 0, 1, 3), f = plogis,
                     d = function(x) plogis(x) * plogis(-x)),
    log = list(x = c(0.5, 1, 2, 5, 10), f = log, d = function(x) 1 / x),
    sqrt = list(x = c(0.25, 1, 4, 9, 16), f = sqrt,
                d = function(x) 1 / (2 * sqrt(x))),
    log1p = list(x = c(0, 0.5, 1, 5, 10), f = log1p,
                 d = function(x) 1 / (1 + x)),
    logit = list(x = c(0.1, 0.3, 0.5, 0.7, 0.9), f = qlogis,
                 d = function(x) 1 / (x * (1 - x))))
  for (op in names(cases)) {
    case <- cases[[op]]
    for (x in case$x) {
      result <- tulpaRatio:::cpp_test_autodiff_unary(op, x)
      expect_equal(result$value, case$f(x), tolerance = 1e-10)
      expect_equal(result$gradient, case$d(x), tolerance = 1e-8)
    }
  }
})

test_that("autodiff division, log sums and constant powers match exact derivatives", {
  for (pair in list(c(6, 2), c(1, 3), c(-4, 2), c(5, -2))) {
    a <- pair[1]; b <- pair[2]
    result <- tulpaRatio:::cpp_test_autodiff_binary("division", a, b)
    expect_equal(result$value, a / b, tolerance = 1e-10)
    expect_equal(result$gradient, c(1 / b, -a / b^2), tolerance = 1e-10)
  }
  for (pair in list(c(1, 2), c(-1, 1), c(100, 101), c(-100, -99))) {
    expected <- exp(pair - max(pair)); expected <- expected / sum(expected)
    result <- tulpaRatio:::cpp_test_autodiff_binary("log_sum_exp", pair[1], pair[2])
    expect_equal(result$value, max(pair) + log(sum(exp(pair - max(pair)))), tolerance = 1e-10)
    expect_equal(result$gradient, expected, tolerance = 1e-10)
  }
  for (pair in list(c(2, 2), c(3, 0.5), c(4, -1), c(2, 3))) {
    x <- pair[1]; p <- pair[2]
    result <- tulpaRatio:::cpp_test_autodiff_unary("pow", x, p)
    expect_equal(result$value, x^p, tolerance = 1e-10)
    expect_equal(result$gradient, p * x^(p - 1), tolerance = 1e-8)
  }
})

test_that("autodiff vector operations match exact densities and derivatives", {
  for (x in list(c(1, 2, 3, -1.5), c(0, 0), c(10, 20, 50), 3)) {
    result <- tulpaRatio:::cpp_test_autodiff_vector("squares", x, integer())
    expect_equal(result$value, sum(x^2), tolerance = 1e-10)
    expect_equal(result$gradient, 2 * x, tolerance = 1e-10)
  }
  y <- c(0L, 1L, 3L, 5L, 10L)
  eta <- c(-0.5, 0, 0.5, 1, 2)
  result <- tulpaRatio:::cpp_test_autodiff_vector("poisson", eta, y)
  expect_equal(result$value, sum(dpois(y, exp(eta), log = TRUE)), tolerance = 1e-10)
  expect_equal(result$gradient, y - exp(eta), tolerance = 1e-10)
  y <- c(2L, 5L, 8L); mu <- c(3, 4, 6); phi <- 2
  result <- tulpaRatio:::cpp_test_autodiff_vector("negbin", mu, y, phi)
  expect_equal(result$value, sum(dnbinom(y, mu = mu, size = phi, log = TRUE)), tolerance = 1e-10)
  expect_equal(result$gradient, y / mu - (y + phi) / (mu + phi), tolerance = 1e-10)
  expected_phi <- sum(digamma(y + phi) - digamma(phi) + log(phi / (phi + mu)) +
                        1 - (phi + y) / (phi + mu))
  expect_equal(result$gradient_phi, expected_phi, tolerance = 1e-8)
})

test_that("production softmax and binomial density match exact references", {
  x <- c(1, 3, 2)
  expect_equal(tulpaRatio:::cpp_test_softmax_inplace(x), exp(x) / sum(exp(x)), tolerance = 1e-10)
  for (eta in c(-3, 0, 2)) {
    expect_equal(tulpaRatio:::cpp_test_binomial_lpmf(3L, 10L, eta, 0, 0, "none"),
                 dbinom(3, 10, plogis(eta), log = TRUE), tolerance = 1e-10)
  }
})

# ---------------------------------------------------------------------------
# Pólya-Gamma RNG tests
# ---------------------------------------------------------------------------

test_that("cpp_rpg1 returns correct length", {
  z <- c(0.0, 1.0, -1.0, 2.0)
  result <- tulpaRatio:::cpp_rpg1(z)
  expect_length(result, length(z))
})

test_that("cpp_rpg1 returns positive values", {
  set.seed(123)
  z <- rnorm(100)
  result <- tulpaRatio:::cpp_rpg1(z)
  expect_true(all(result > 0))
})

test_that("cpp_rpg1 mean approximates E[PG(1,z)] = tanh(z/2)/(2z)", {
  set.seed(42)
  z <- 1.0
  n <- 1000
  samples <- replicate(n, tulpaRatio:::cpp_rpg1(z))

  # E[PG(1,z)] = tanh(z/2) / (2z) for z != 0
  expected_mean <- tanh(z / 2) / (2 * z)

  # Check mean is approximately correct (allow for Monte Carlo error)
  expect_equal(mean(samples), expected_mean, tolerance = 0.05)
})

test_that("cpp_rpg handles vector inputs", {
  b <- as.integer(c(1, 2, 3, 4))
  z <- c(0.5, 1.0, 1.5, 2.0)

  result <- tulpaRatio:::cpp_rpg(b, z)
  expect_length(result, length(b))
  expect_true(all(result > 0))
})

test_that("cpp_rpg with b=1 matches cpp_rpg1",
{
  set.seed(123)
  z <- c(0.5, 1.0, 2.0)
  result1 <- tulpaRatio:::cpp_rpg(as.integer(rep(1, 3)), z)

  set.seed(123)
  result2 <- tulpaRatio:::cpp_rpg1(z)

  expect_equal(result1, result2)
})

# ---------------------------------------------------------------------------
# PG Binomial update function tests (pg_binomial.cpp)
# ---------------------------------------------------------------------------

test_that("pg update_beta returns correct length", {
  n <- 20
  p <- 3
  kappa <- rnorm(n)
  omega <- abs(rnorm(n)) + 0.1
  X <- matrix(rnorm(n * p), n, p)
  re_contrib <- rep(0, n)
  prior_sd <- 10.0

  result <- tulpaRatio:::cpp_test_pg_update_beta(kappa, omega, X, re_contrib, prior_sd)
  expect_length(result, p)
})

test_that("pg update_beta returns finite values", {
  set.seed(123)
  n <- 30
  p <- 4
  kappa <- rnorm(n)
  omega <- abs(rnorm(n)) + 0.1
  X <- matrix(rnorm(n * p), n, p)
  re_contrib <- rnorm(n, sd = 0.5)
  prior_sd <- 5.0

  result <- tulpaRatio:::cpp_test_pg_update_beta(kappa, omega, X, re_contrib, prior_sd)
  expect_true(all(is.finite(result)))
})

test_that("pg update_re returns correct length", {
  set.seed(456)
  n <- 50
  n_groups <- 5

  kappa <- rnorm(n)
  omega <- abs(rnorm(n)) + 0.1
  X_beta <- rnorm(n)
  group <- as.integer(sample(1:n_groups, n, replace = TRUE))
  sigma_re <- 1.0

  result <- tulpaRatio:::cpp_test_pg_update_re(kappa, omega, X_beta, group, n_groups, sigma_re)
  expect_length(result, n_groups)
})

test_that("pg update_re returns finite values", {
  set.seed(789)
  n <- 100
  n_groups <- 10

  kappa <- rnorm(n)
  omega <- abs(rnorm(n)) + 0.1
  X_beta <- rnorm(n)
  group <- as.integer(sample(1:n_groups, n, replace = TRUE))
  sigma_re <- 2.0

  result <- tulpaRatio:::cpp_test_pg_update_re(kappa, omega, X_beta, group, n_groups, sigma_re)
  expect_true(all(is.finite(result)))
})

test_that("PG sigma update samples the half-Cauchy prior when there are no effects", {
  set.seed(101)
  scale <- 2.5
  draws <- tulpaRatio:::cpp_test_pg_sigma_halfcauchy_chain(numeric(0), scale, 40000L)
  # half-Cauchy(0, scale): quartiles scale * tan(pi/8), scale, scale * tan(3 pi/8)
  q <- quantile(draws, c(0.25, 0.5, 0.75))
  expect_equal(unname(q), scale * tan(pi * c(1, 2, 3) / 8), tolerance = 0.06)
})

test_that("PG sigma update matches the half-Cauchy posterior by quadrature", {
  set.seed(202)
  scale <- 2.5
  re <- rnorm(8, sd = 1.5)
  draws <- tulpaRatio:::cpp_test_pg_sigma_halfcauchy_chain(re, scale, 40000L)

  log_post <- function(s) {
    sum(dnorm(re, 0, s, log = TRUE)) - log1p((s / scale)^2)
  }
  grid <- seq(1e-3, 15, length.out = 6000)
  w <- exp(vapply(grid, log_post, numeric(1)) - max(vapply(grid, log_post, numeric(1))))
  post_mean <- sum(grid * w) / sum(w)
  expect_equal(mean(draws), post_mean, tolerance = 0.03)
})

# ---------------------------------------------------------------------------
# linalg_fast.h tests - fast linear algebra operations
# ---------------------------------------------------------------------------

test_that("dot_product matches sum(x*y)", {
  x <- c(1, 2, 3, 4, 5)
  y <- c(2, 3, 4, 5, 6)

  result <- tulpaRatio:::cpp_test_dot_product(x, y)
  expect_equal(result, sum(x * y), tolerance = 1e-10)
})

test_that("dot_product handles unrolled loop correctly", {
  # Test cases for different lengths (testing loop unrolling)
  for (n in c(1, 2, 3, 4, 5, 7, 8, 15, 16, 17)) {
    x <- rnorm(n)
    y <- rnorm(n)
    result <- tulpaRatio:::cpp_test_dot_product(x, y)
    expect_equal(result, sum(x * y), tolerance = 1e-10)
  }
})

test_that("norm_squared matches sum(x^2)", {
  x <- c(3, 4, 0, -2)
  result <- tulpaRatio:::cpp_test_norm_squared(x)
  expect_equal(result, sum(x^2), tolerance = 1e-10)
})

test_that("vector_sum matches sum()", {
  x <- c(1, 2, 3, -4, 5.5, -0.5)
  result <- tulpaRatio:::cpp_test_vector_sum(x)
  expect_equal(result, sum(x), tolerance = 1e-10)
})

test_that("axpy computes y = a*x + y correctly", {
  a <- 2.5
  x <- c(1, 2, 3)
  y <- c(10, 20, 30)

  result <- tulpaRatio:::cpp_test_axpy(a, x, y)
  expected <- a * x + y
  expect_equal(result, expected, tolerance = 1e-10)
})

test_that("scale computes x = a*x correctly", {
  a <- 3.0
  x <- c(1, 2, 3, 4)

  result <- tulpaRatio:::cpp_test_scale(a, x)
  expect_equal(result, a * x, tolerance = 1e-10)
})

test_that("linalg_matvec matches X %*% beta", {
  set.seed(123)
  X <- matrix(rnorm(20), nrow = 5, ncol = 4)
  beta <- rnorm(4)

  result <- tulpaRatio:::cpp_test_linalg_matvec(X, beta)
  expected <- as.vector(X %*% beta)
  expect_equal(result, expected, tolerance = 1e-10)
})

test_that("linalg_matvec_add computes y + X*beta correctly", {
  set.seed(456)
  X <- matrix(rnorm(12), nrow = 3, ncol = 4)
  beta <- rnorm(4)
  y_init <- c(1, 2, 3)

  result <- tulpaRatio:::cpp_test_linalg_matvec_add(X, beta, y_init)
  expected <- y_init + as.vector(X %*% beta)
  expect_equal(result, expected, tolerance = 1e-10)
})

test_that("linalg_matvec_transpose matches t(X) %*% x", {
  set.seed(789)
  X <- matrix(rnorm(20), nrow = 5, ncol = 4)
  x <- rnorm(5)

  result <- tulpaRatio:::cpp_test_linalg_matvec_transpose(X, x)
  expected <- as.vector(t(X) %*% x)
  expect_equal(result, expected, tolerance = 1e-10)
})

test_that("sparse_laplacian_quadform computes graph Laplacian quadratic form", {
  # Simple chain graph: 1 -- 2 -- 3 -- 4
  # Adjacency: (1,2), (2,1), (2,3), (3,2), (3,4), (4,3)
  # CSR format
  row_ptr <- c(0L, 1L, 3L, 5L, 6L)  # Node degrees: 1,2,2,1

col_idx <- c(1L, 0L, 2L, 1L, 3L, 2L)  # 0-based neighbors

  x <- c(1, 2, 4, 8)

  result <- tulpaRatio:::cpp_test_sparse_laplacian_quadform(row_ptr, col_idx, x)

  # x'Lx = sum over edges (x_i - x_j)^2
  # Edge (0,1): (1-2)^2 = 1
  # Edge (1,2): (2-4)^2 = 4
  # Edge (2,3): (4-8)^2 = 16
  expected <- 1 + 4 + 16
  expect_equal(result, expected, tolerance = 1e-10)
})

test_that("linalg_log_sum_exp is numerically stable", {
  # Normal case
  expect_equal(
    tulpaRatio:::cpp_test_linalg_log_sum_exp(1.0, 2.0),
    log(exp(1) + exp(2)),
    tolerance = 1e-10
  )

  # Large values that would overflow
  result <- tulpaRatio:::cpp_test_linalg_log_sum_exp(700, 701)
  expected <- 701 + log(exp(-1) + 1)
  expect_equal(result, expected, tolerance = 1e-10)

  # Negative values
  result <- tulpaRatio:::cpp_test_linalg_log_sum_exp(-100, -99)
  expected <- -99 + log(exp(-1) + 1)
  expect_equal(result, expected, tolerance = 1e-10)
})

test_that("linalg_log_sum_exp_vec handles vectors", {
  x <- c(1, 2, 3)
  result <- tulpaRatio:::cpp_test_linalg_log_sum_exp_vec(x)
  expected <- log(sum(exp(x)))
  expect_equal(result, expected, tolerance = 1e-10)

  # Large values
  x_large <- c(700, 701, 702)
  result_large <- tulpaRatio:::cpp_test_linalg_log_sum_exp_vec(x_large)
  expected_large <- 702 + log(exp(-2) + exp(-1) + 1)
  expect_equal(result_large, expected_large, tolerance = 1e-10)
})

test_that("softmax_inplace produces valid probabilities", {
  x <- c(1, 2, 3, 4)
  result <- tulpaRatio:::cpp_test_softmax_inplace(x)

  expect_equal(sum(result), 1.0, tolerance = 1e-10)
  expect_true(all(result > 0))
  expect_true(all(result < 1))
  # Should preserve ordering
  expect_true(all(diff(result) > 0))
})

test_that("compute_linear_predictors works correctly", {
  set.seed(111)
  N <- 10
  X_num <- matrix(rnorm(N * 3), nrow = N, ncol = 3)
  X_denom <- matrix(rnorm(N * 2), nrow = N, ncol = 2)
  beta_num <- rnorm(3)
  beta_denom <- rnorm(2)

  result <- tulpaRatio:::cpp_test_compute_linear_predictors(
    X_num, beta_num, X_denom, beta_denom, 1L
  )

  expect_equal(result$eta_num, as.vector(X_num %*% beta_num), tolerance = 1e-10)
  expect_equal(result$eta_denom, as.vector(X_denom %*% beta_denom), tolerance = 1e-10)
})

# ---------------------------------------------------------------------------
# hmc_temporal.h tests - temporal random effects
# ---------------------------------------------------------------------------

test_that("rw1_quadratic_form computes sum of squared differences", {
  phi <- c(1, 2, 5, 7, 8)

  # Non-cyclic: sum((phi[t+1] - phi[t])^2) for t=1..T-1
  result <- tulpa:::cpp_test_rw1_quadratic_form(phi, FALSE)
  expected <- (2-1)^2 + (5-2)^2 + (7-5)^2 + (8-7)^2
  expect_equal(result, expected, tolerance = 1e-10)
})

test_that("rw1_quadratic_form handles cyclic case", {
  phi <- c(1, 2, 5, 7, 8)

  # Cyclic: adds (phi[1] - phi[T])^2
  result <- tulpa:::cpp_test_rw1_quadratic_form(phi, TRUE)
  expected <- (2-1)^2 + (5-2)^2 + (7-5)^2 + (8-7)^2 + (1-8)^2
  expect_equal(result, expected, tolerance = 1e-10)
})

test_that("rw2_quadratic_form computes sum of squared second differences", {
  phi <- c(1, 3, 4, 6, 10)

  # Non-cyclic: sum((phi[t] - 2*phi[t+1] + phi[t+2])^2)
  result <- tulpa:::cpp_test_rw2_quadratic_form(phi, FALSE)
  d1 <- 1 - 2*3 + 4   # = -1
  d2 <- 3 - 2*4 + 6   # = 1
  d3 <- 4 - 2*6 + 10  # = 2
  expected <- d1^2 + d2^2 + d3^2
  expect_equal(result, expected, tolerance = 1e-10)
})

test_that("rw2_quadratic_form returns 0 for T < 3", {
  expect_equal(tulpa:::cpp_test_rw2_quadratic_form(c(1, 2), FALSE), 0)
  expect_equal(tulpa:::cpp_test_rw2_quadratic_form(c(1), FALSE), 0)
})

test_that("ar1_log_density is finite for valid inputs", {
  phi <- c(0.5, 0.8, 0.3, -0.2, 0.1)
  rho <- 0.7
  tau <- 2.0

  result <- tulpa:::cpp_test_ar1_log_density(phi, rho, tau)
  expect_true(is.finite(result))
  expect_true(result < 0)  # Log-density should be negative
})

test_that("ar1_log_density matches manual computation", {
  # AR1: phi[0] ~ N(0, sigma^2/(1-rho^2)), phi[t]|phi[t-1] ~ N(rho*phi[t-1], sigma^2)
  phi <- c(0.5, 0.8, 0.3)
  rho <- 0.7
  tau <- 2.0
  sigma2 <- 1.0 / tau

  result <- tulpa:::cpp_test_ar1_log_density(phi, rho, tau)

  # Manual computation
  T <- length(phi)
  marginal_var <- sigma2 / (1 - rho^2)

  # First obs: N(0, marginal_var)
  log_dens <- dnorm(phi[1], 0, sqrt(marginal_var), log = TRUE)

  # Conditional: phi[t] | phi[t-1] ~ N(rho * phi[t-1], sigma^2)
  for (t in 2:T) {
    log_dens <- log_dens + dnorm(phi[t], rho * phi[t-1], sqrt(sigma2), log = TRUE)
  }

  expect_equal(result, log_dens, tolerance = 1e-10)
})

test_that("temporal_log_prior RW1 matches manual computation", {
  phi <- c(1, 2, 4)
  tau <- 2.0

  result <- tulpa:::cpp_test_temporal_log_prior(phi, "rw1", tau, 0, FALSE)

  # Manual: 0.5 * rank * log(tau) - 0.5 * tau * quad
  quad <- (2-1)^2 + (4-2)^2  # = 1 + 4 = 5
  rank <- 2  # T - 1 for non-cyclic
  expected <- 0.5 * rank * log(tau / (2 * pi)) - 0.5 * tau * quad
  expect_equal(result, expected, tolerance = 1e-10)
})

test_that("the intrinsic arms' augmentation carries both the quadratic and the rank", {
  # Q -> Q + 11'/n gives the constant direction the field's own precision tau,
  # which adds (sum phi)^2 / n to the quadratic form AND one to the rank. A
  # path that took one and not the other would make the tau-marginal wrong.
  sigma2 <- 0.7
  phi <- c(1, 2, 3)  # sum = 6

  quad <- (2 - 1)^2 + (3 - 2)^2                      # 2
  quad_aug <- quad + 6^2 / 3                         # 14
  rank <- 2                                          # rw1_rank(3, FALSE)

  plain <- tulpa:::cpp_test_ms_rw_log_lik(phi, sigma2, 1L, FALSE, FALSE)
  expect_equal(plain,
               -0.5 * quad / sigma2 - 0.5 * rank * log(2 * pi * sigma2),
               tolerance = 1e-10)

  augmented <- tulpa:::cpp_test_ms_rw_log_lik(phi, sigma2, 1L, FALSE, TRUE)
  expect_equal(augmented,
               -0.5 * quad_aug / sigma2 - 0.5 * (rank + 1) * log(2 * pi * sigma2),
               tolerance = 1e-10)

  # On the sum-to-zero subspace the quadratic is untouched, so the whole
  # difference is the one extra rank the full-rank normalizer carries.
  centred <- c(-1, 0, 1)
  expect_equal(
    tulpa:::cpp_test_ms_rw_log_lik(centred, sigma2, 1L, FALSE, TRUE) -
      tulpa:::cpp_test_ms_rw_log_lik(centred, sigma2, 1L, FALSE, FALSE),
    -0.5 * log(2 * pi * sigma2),
    tolerance = 1e-10)
})

test_that("a non-cyclic RW2 stays deficient by one after the augmentation", {
  # The augmentation fills the constant direction and nothing else. A non-cyclic
  # RW2 also annihilates a linear ramp, so its rank goes T-2 -> T-1, not to T;
  # passing the field length there would bias the variance component.
  sigma2 <- 1.3
  phi <- c(0.2, -0.5, 1.1, 0.4)
  n <- length(phi)

  d2 <- diff(diff(phi))
  quad_aug <- sum(d2^2) + sum(phi)^2 / n

  expect_equal(
    tulpa:::cpp_test_ms_rw_log_lik(phi, sigma2, 2L, FALSE, TRUE),
    -0.5 * quad_aug / sigma2 - 0.5 * (n - 1) * log(2 * pi * sigma2),
    tolerance = 1e-10)
})

# ---------------------------------------------------------------------------
# hmc_zi.h tests - zero-inflation functions
# ---------------------------------------------------------------------------

test_that("log1pexp is numerically stable", {
  # Normal values
  expect_equal(tulpaRatio:::cpp_test_log1pexp(0), log(2), tolerance = 1e-10)
  expect_equal(tulpaRatio:::cpp_test_log1pexp(1), log(1 + exp(1)), tolerance = 1e-10)

  # Large positive (should return x)
  expect_equal(tulpaRatio:::cpp_test_log1pexp(50), 50, tolerance = 1e-10)

  # Large negative: log1p(exp(x)), not exp(x)
  expect_equal(tulpaRatio:::cpp_test_log1pexp(-20), log1p(exp(-20)), tolerance = 1e-15)
  expect_equal(tulpaRatio:::cpp_test_log1pexp(-800), 0)
})

test_that("log_logistic is numerically stable", {
  # log(sigmoid(x))
  expect_equal(tulpaRatio:::cpp_test_log_logistic(0), log(0.5), tolerance = 1e-10)

  # Large positive: log(sigmoid(x)) ≈ 0
  expect_equal(tulpaRatio:::cpp_test_log_logistic(50), 0, tolerance = 1e-10)

  # Large negative: log(sigmoid(x)) ≈ x
  expect_equal(tulpaRatio:::cpp_test_log_logistic(-50), -50, tolerance = 1e-10)
})

test_that("log1m_logistic is numerically stable", {
  # log(1 - sigmoid(x)) = log(sigmoid(-x))
  expect_equal(tulpaRatio:::cpp_test_log1m_logistic(0), log(0.5), tolerance = 1e-10)

  # Large positive: log(1-sigmoid(x)) ≈ -x
  expect_equal(tulpaRatio:::cpp_test_log1m_logistic(50), -50, tolerance = 1e-10)

  # Large negative: log(1-sigmoid(x)) ≈ 0
  expect_equal(tulpaRatio:::cpp_test_log1m_logistic(-50), 0, tolerance = 1e-10)
})

test_that("zi_poisson_lpmf matches dpois", {
  test_cases <- list(
    list(y = 0L, mu = 2.0),
    list(y = 3L, mu = 2.0),
    list(y = 10L, mu = 5.0)
  )

  for (tc in test_cases) {
    result <- tulpaRatio:::cpp_test_zi_poisson_lpmf(tc$y, tc$mu)
    expected <- dpois(tc$y, tc$mu, log = TRUE)
    expect_equal(result, expected, tolerance = 1e-10)
  }
})

test_that("zi_negbin_lpmf matches dnbinom", {
  test_cases <- list(
    list(y = 0L, mu = 3.0, phi = 2.0),
    list(y = 5L, mu = 3.0, phi = 2.0),
    list(y = 10L, mu = 5.0, phi = 10.0)
  )

  for (tc in test_cases) {
    result <- tulpaRatio:::cpp_test_zi_negbin_lpmf(tc$y, tc$mu, tc$phi)
    expected <- dnbinom(tc$y, size = tc$phi, mu = tc$mu, log = TRUE)
    expect_equal(result, expected, tolerance = 1e-8)
  }
})

test_that("zi_poisson_lpmf_logit handles y=0 correctly", {
  mu <- 3.0
  logit_zi <- 0.5
  zi <- plogis(logit_zi)

  result <- tulpaRatio:::cpp_test_zi_poisson_lpmf_logit(0L, mu, logit_zi)

  # P(Y=0) = zi + (1-zi)*exp(-mu)
  expected <- log(zi + (1 - zi) * exp(-mu))
  expect_equal(result, expected, tolerance = 1e-10)
})

test_that("zi_poisson_lpmf_logit handles y>0 correctly", {
  mu <- 3.0
  logit_zi <- -0.5
  zi <- plogis(logit_zi)

  result <- tulpaRatio:::cpp_test_zi_poisson_lpmf_logit(5L, mu, logit_zi)

  # P(Y=y) = (1-zi) * Poisson(y|mu)
  expected <- log(1 - zi) + dpois(5, mu, log = TRUE)
  expect_equal(result, expected, tolerance = 1e-10)
})

test_that("truncated_poisson_lpmf is correct for y>0", {
  mu <- 3.0

  for (y in 1:10) {
    result <- tulpaRatio:::cpp_test_truncated_poisson_lpmf(y, mu)
    # P(Y=y|Y>0) = Poisson(y) / (1 - exp(-mu))
    expected <- dpois(y, mu, log = TRUE) - log(1 - exp(-mu))
    expect_equal(result, expected, tolerance = 1e-10)
  }
})

test_that("hurdle_poisson_lpmf_logit handles y=0 correctly", {
  logit_theta <- 0.5

  result <- tulpaRatio:::cpp_test_hurdle_poisson_lpmf_logit(0L, 3.0, logit_theta)
  # P(Y=0) = 1 - theta
  expected <- log(1 - plogis(logit_theta))
  expect_equal(result, expected, tolerance = 1e-10)
})

test_that("hurdle_poisson_lpmf_logit handles y>0 correctly", {
  mu <- 3.0
  logit_theta <- 0.5
  theta <- plogis(logit_theta)

  for (y in 1:5) {
    result <- tulpaRatio:::cpp_test_hurdle_poisson_lpmf_logit(as.integer(y), mu, logit_theta)
    # P(Y=y) = theta * truncated_poisson(y)
    trunc_lpmf <- dpois(y, mu, log = TRUE) - log(1 - exp(-mu))
    expected <- log(theta) + trunc_lpmf
    expect_equal(result, expected, tolerance = 1e-10)
  }
})

test_that("zi_log_likelihood dispatches correctly", {
  mu <- 3.0
  phi <- 2.0
  logit_zi <- 0.5

  # None: should return standard negbin
  result_none <- tulpaRatio:::cpp_test_zi_log_likelihood(5L, mu, phi, logit_zi, "none")
  expected_none <- dnbinom(5, size = phi, mu = mu, log = TRUE)
  expect_equal(result_none, expected_none, tolerance = 1e-8)

  # ZI Poisson
  result_zip <- tulpaRatio:::cpp_test_zi_log_likelihood(0L, mu, 0, logit_zi, "zi_poisson")
  expected_zip <- tulpaRatio:::cpp_test_zi_poisson_lpmf_logit(0L, mu, logit_zi)
  expect_equal(result_zip, expected_zip, tolerance = 1e-10)

  # Hurdle Poisson
  result_hp <- tulpaRatio:::cpp_test_zi_log_likelihood(3L, mu, 0, logit_zi, "hurdle_poisson")
  expected_hp <- tulpaRatio:::cpp_test_hurdle_poisson_lpmf_logit(3L, mu, logit_zi)
  expect_equal(result_hp, expected_hp, tolerance = 1e-10)
})

test_that("zi_negbin_lpmf_logit reaches the Poisson limit", {
  mu <- 2.0
  logit_zi <- -0.3
  # The zero mass phi log(phi / (phi + mu)) -> -mu, past where the ratio rounds to 1
  expect_equal(tulpaRatio:::cpp_test_zi_negbin_lpmf_logit(0L, mu, 1e20, logit_zi),
               tulpaRatio:::cpp_test_zi_poisson_lpmf_logit(0L, mu, logit_zi),
               tolerance = 1e-12)
  expect_equal(tulpaRatio:::cpp_test_zi_negbin_lpmf_logit(3L, mu, 1e8, logit_zi),
               tulpaRatio:::cpp_test_zi_poisson_lpmf_logit(3L, mu, logit_zi),
               tolerance = 1e-6)
})

test_that("binomial-type densities stay finite at extreme logits", {
  q <- 40
  p_lo <- -40
  lp <- plogis(q, log.p = TRUE)
  l1mp <- plogis(q, lower.tail = FALSE, log.p = TRUE)
  expect_equal(tulpaRatio:::cpp_test_binomial_lpmf(3L, 10L, q, 0, 0, "none"),
               lchoose(10, 3) + 3 * lp + 7 * l1mp, tolerance = 1e-12)
  expect_equal(tulpaRatio:::cpp_test_binomial_lpmf(3L, 10L, p_lo, 0, 0, "none"),
               dbinom(3, 10, plogis(p_lo), log = TRUE), tolerance = 1e-12)
  for (type in c("zi_binomial", "oi_binomial", "zoib", "hurdle_binomial")) {
    for (y in c(0L, 3L, 10L)) {
      expect_true(is.finite(tulpaRatio:::cpp_test_binomial_lpmf(y, 10L, q, 0.2, -0.4, type)),
                  label = paste(type, y))
      expect_true(is.finite(tulpaRatio:::cpp_test_binomial_lpmf(y, 10L, -q, 0.2, -0.4, type)),
                  label = paste(type, y))
    }
  }
  expect_true(is.finite(tulpaRatio:::cpp_test_beta_binomial_lpmf(3L, 10L, q, 5)))
})

# ---------------------------------------------------------------------------
# OpenMP parallel execution tests
# ---------------------------------------------------------------------------

test_that("OpenMP max threads is at least 1", {
  max_threads <- tulpaRatio:::cpp_get_max_threads()
  expect_true(max_threads >= 1)
})

test_that("parallel dot products produce correct results", {
  set.seed(555)
  N <- 100
  p <- 5
  X <- matrix(rnorm(N * p), nrow = N, ncol = p)
  y <- rnorm(p)

  for (n_threads in c(1, 2, 4)) {
    result <- tulpaRatio:::cpp_test_parallel_dot_products(X, y, n_threads)

    # Verify individual dot products
    expected_results <- as.vector(X %*% y)
    expect_equal(result$results, expected_results, tolerance = 1e-10)

    # Verify reduction
    expect_equal(result$total_sum, sum(expected_results), tolerance = 1e-10)
  }
})

test_that("parallel likelihood reduction is correct", {
  set.seed(666)
  N <- 200
  y <- rpois(N, lambda = 5)
  mu <- runif(N, 2, 8)

  # Expected log-likelihood
  expected_ll <- sum(dpois(y, mu, log = TRUE))

  for (n_threads in c(1, 2, 4)) {
    result <- tulpaRatio:::cpp_test_parallel_likelihood(y, mu, n_threads)
    expect_equal(result$log_lik, expected_ll, tolerance = 1e-8)
  }
})

test_that("compute_linear_predictors is consistent across thread counts", {
  set.seed(777)
  N <- 100
  X_num <- matrix(rnorm(N * 3), nrow = N, ncol = 3)
  X_denom <- matrix(rnorm(N * 2), nrow = N, ncol = 2)
  beta_num <- rnorm(3)
  beta_denom <- rnorm(2)

  result_1t <- tulpaRatio:::cpp_test_compute_linear_predictors(
    X_num, beta_num, X_denom, beta_denom, 1L
  )

  for (n_threads in c(2, 4)) {
    result_mt <- tulpaRatio:::cpp_test_compute_linear_predictors(
      X_num, beta_num, X_denom, beta_denom, n_threads
    )

    expect_equal(result_1t$eta_num, result_mt$eta_num, tolerance = 1e-10)
    expect_equal(result_1t$eta_denom, result_mt$eta_denom, tolerance = 1e-10)
  }
})
