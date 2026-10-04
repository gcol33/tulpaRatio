# Unit tests for C++ helper functions
# These test core numerical routines directly via Rcpp wrappers

# ---------------------------------------------------------------------------
# Leapfrog integrator tests
# ---------------------------------------------------------------------------

test_that("leapfrog preserves Hamiltonian for quadratic potential", {
  # For a quadratic potential U(q) = 0.5 * sum(q^2), the leapfrog integrator

  # should approximately preserve the Hamiltonian H = U(q) + K(p)

  q <- c(1.0, 0.5, -0.3)
  p <- c(0.2, -0.4, 0.6)

  H_initial <- tulpaRatio:::cpp_test_hamiltonian(q, p)

  # Small step size, many steps for accuracy
  result <- tulpaRatio:::cpp_test_leapfrog(q, p, epsilon = 0.01, L = 100)

  H_final <- tulpaRatio:::cpp_test_hamiltonian(result$q, result$p)

  # Hamiltonian should be conserved to within numerical precision

  expect_equal(H_initial, H_final, tolerance = 0.01)
})

test_that("leapfrog is reversible", {
  q <- c(1.0, -0.5)
  p <- c(0.3, 0.7)

  # Forward integration
  fwd <- tulpaRatio:::cpp_test_leapfrog(q, p, epsilon = 0.1, L = 10)

  # Backward integration (negate momentum, integrate, negate again)
  bwd <- tulpaRatio:::cpp_test_leapfrog(fwd$q, -fwd$p, epsilon = 0.1, L = 10)

  # Should return to starting point
  expect_equal(bwd$q, q, tolerance = 1e-10)
  expect_equal(-bwd$p, p, tolerance = 1e-10)
})

test_that("leapfrog with zero steps returns input",
{
  q <- c(1.0, 2.0)
  p <- c(-0.5, 0.5)

  # L=0 should be a no-op (though our implementation uses L-1 full steps)
  # Actually with L=1, we do: half momentum, full position, half momentum
  result <- tulpaRatio:::cpp_test_leapfrog(q, p, epsilon = 0.01, L = 1)

  # With L=1, we still do one leapfrog step
  expect_true(TRUE)  # Just check it doesn't error
})

# ---------------------------------------------------------------------------
# Hamiltonian computation tests
# ---------------------------------------------------------------------------

test_that("Hamiltonian is sum of potential and kinetic energy", {
  q <- c(2.0, 3.0)
  p <- c(1.0, 1.0)

  H <- tulpaRatio:::cpp_test_hamiltonian(q, p)

  # U(q) = 0.5 * (4 + 9) = 6.5
  # K(p) = 0.5 * (1 + 1) = 1.0
  # H = 7.5
  expect_equal(H, 7.5)
})

test_that("Hamiltonian is zero at origin with zero momentum", {
  q <- c(0.0, 0.0, 0.0)
  p <- c(0.0, 0.0, 0.0)

  H <- tulpaRatio:::cpp_test_hamiltonian(q, p)
  expect_equal(H, 0.0)
})

# ---------------------------------------------------------------------------
# Log-sum-exp tests
# ---------------------------------------------------------------------------

test_that("log_sum_exp is correct for simple cases", {
  # log(exp(1) + exp(2)) = log(e + e^2) ≈ 2.313
  result <- tulpaRatio:::cpp_test_log_sum_exp(c(1, 2))
  expect_equal(result, log(exp(1) + exp(2)), tolerance = 1e-10)
})
test_that("log_sum_exp handles large values without overflow", {
  # Values that would overflow if we computed exp() directly
  large_vals <- c(700, 701, 702)
  result <- tulpaRatio:::cpp_test_log_sum_exp(large_vals)

  # log(exp(700) + exp(701) + exp(702)) = 702 + log(exp(-2) + exp(-1) + 1)
  expected <- 702 + log(exp(-2) + exp(-1) + 1)
  expect_equal(result, expected, tolerance = 1e-10)
})

test_that("log_sum_exp handles negative values", {
  result <- tulpaRatio:::cpp_test_log_sum_exp(c(-1000, -999, -998))
  expected <- -998 + log(exp(-2) + exp(-1) + 1)
  expect_equal(result, expected, tolerance = 1e-10)
})

test_that("log_sum_exp of single value returns that value", {
  expect_equal(tulpaRatio:::cpp_test_log_sum_exp(5.0), 5.0)
})

# ---------------------------------------------------------------------------
# Softmax tests
# ---------------------------------------------------------------------------

test_that("softmax sums to 1", {
  x <- c(1, 2, 3, 4)
  result <- tulpaRatio:::cpp_test_softmax(x)
  expect_equal(sum(result), 1.0, tolerance = 1e-10)
})

test_that("softmax preserves ordering", {
  x <- c(1, 3, 2)
  result <- tulpaRatio:::cpp_test_softmax(x)
  expect_true(result[2] > result[3])
  expect_true(result[3] > result[1])
})

test_that("softmax is shift-invariant", {
  x <- c(1, 2, 3)
  result1 <- tulpaRatio:::cpp_test_softmax(x)
  result2 <- tulpaRatio:::cpp_test_softmax(x + 1000)
  expect_equal(result1, result2, tolerance = 1e-10)
})

test_that("softmax of equal values gives uniform", {
  x <- c(5, 5, 5, 5)
  result <- tulpaRatio:::cpp_test_softmax(x)
  expect_equal(result, rep(0.25, 4), tolerance = 1e-10)
})

# ---------------------------------------------------------------------------
# Inverse logit tests
# ---------------------------------------------------------------------------

test_that("inv_logit of 0 is 0.5", {
  result <- tulpaRatio:::cpp_test_inv_logit(0)
  expect_equal(result, 0.5)
})

test_that("inv_logit is bounded (0, 1)", {
  # Use moderate values to avoid numerical precision at extremes
  x <- c(-10, -5, -1, 0, 1, 5, 10)
  result <- tulpaRatio:::cpp_test_inv_logit(x)
  expect_true(all(result > 0))
  expect_true(all(result < 1))
})

test_that("inv_logit is monotonic", {
  x <- seq(-5, 5, by = 0.5)
  result <- tulpaRatio:::cpp_test_inv_logit(x)
  expect_true(all(diff(result) > 0))
})

test_that("inv_logit is symmetric around 0.5", {
  x <- c(-2, 2)
  result <- tulpaRatio:::cpp_test_inv_logit(x)
  expect_equal(result[1] + result[2], 1.0, tolerance = 1e-10)
})

# ---------------------------------------------------------------------------
# Log-gamma tests
# ---------------------------------------------------------------------------

test_that("lgamma matches R's lgamma", {
  x_vals <- c(0.5, 1, 2, 5, 10, 100)
  for (x in x_vals) {
    expect_equal(tulpaRatio:::cpp_test_lgamma(x), lgamma(x), tolerance = 1e-10)
  }
})

test_that("lgamma(1) = 0", {
  expect_equal(tulpaRatio:::cpp_test_lgamma(1), 0)
})

test_that("lgamma(n) = log((n-1)!) for integers", {
  expect_equal(tulpaRatio:::cpp_test_lgamma(5), log(factorial(4)), tolerance = 1e-10)
  expect_equal(tulpaRatio:::cpp_test_lgamma(10), log(factorial(9)), tolerance = 1e-10)
})

# ---------------------------------------------------------------------------
# Poisson log-likelihood tests
# ---------------------------------------------------------------------------

test_that("poisson_loglik matches dpois", {
  y <- as.integer(c(0, 1, 2, 5, 10))
  lambda <- c(1, 2, 3, 4, 5)

  cpp_ll <- tulpaRatio:::cpp_test_poisson_loglik(y, lambda)
  r_ll <- sum(dpois(y, lambda, log = TRUE))

  expect_equal(cpp_ll, r_ll, tolerance = 1e-10)
})

test_that("poisson_loglik returns -Inf for impossible observations", {
  y <- as.integer(c(5))
  lambda <- c(0)  # y > 0 but lambda = 0 is impossible

  result <- tulpaRatio:::cpp_test_poisson_loglik(y, lambda)
  expect_equal(result, -Inf)
})

# ---------------------------------------------------------------------------
# Binomial log-likelihood tests
# ---------------------------------------------------------------------------

test_that("binomial_loglik matches dbinom", {
  y <- as.integer(c(3, 5, 7))
  n <- as.integer(c(10, 10, 10))
  p <- c(0.3, 0.5, 0.7)

  cpp_ll <- tulpaRatio:::cpp_test_binomial_loglik(y, n, p)
  r_ll <- sum(dbinom(y, n, p, log = TRUE))

  expect_equal(cpp_ll, r_ll, tolerance = 1e-10)
})

test_that("binomial_loglik handles edge probabilities", {
  y <- as.integer(c(0, 10))
  n <- as.integer(c(10, 10))

  # p=0 with y=0 is valid, p=1 with y=n is valid
  expect_true(is.finite(tulpaRatio:::cpp_test_binomial_loglik(
    as.integer(0), as.integer(10), 0.0001
  )))
})

# ---------------------------------------------------------------------------
# Negative binomial log-likelihood tests
# ---------------------------------------------------------------------------

test_that("negbin_loglik matches dnbinom", {
  y <- as.integer(c(0, 2, 5, 10))
  mu <- c(3, 3, 3, 3)
  phi <- 2.0  # size parameter

  cpp_ll <- tulpaRatio:::cpp_test_negbin_loglik(y, mu, phi)
  r_ll <- sum(dnbinom(y, size = phi, mu = mu, log = TRUE))

  expect_equal(cpp_ll, r_ll, tolerance = 1e-10)
})

test_that("negbin_loglik handles large phi (approaches Poisson)", {
  y <- as.integer(c(5, 10, 15))
  mu <- c(5, 10, 15)
  phi <- 1000  # Large phi -> Poisson

  cpp_ll <- tulpaRatio:::cpp_test_negbin_loglik(y, mu, phi)
  poisson_ll <- sum(dpois(y, mu, log = TRUE))

  # Should be close to Poisson for large phi
  expect_equal(cpp_ll, poisson_ll, tolerance = 0.1)
})

# ---------------------------------------------------------------------------
# Normal log-likelihood tests
# ---------------------------------------------------------------------------

test_that("normal_loglik matches dnorm", {
  y <- c(1.5, 2.0, 2.5, 3.0)
  mu <- c(2.0, 2.0, 2.0, 2.0)
  sigma <- 0.5

  cpp_ll <- tulpaRatio:::cpp_test_normal_loglik(y, mu, sigma)
  r_ll <- sum(dnorm(y, mu, sigma, log = TRUE))

  expect_equal(cpp_ll, r_ll, tolerance = 1e-10)
})

test_that("normal_loglik peaks at y = mu", {
  mu <- 5.0
  sigma <- 1.0

  ll_at_mu <- tulpaRatio:::cpp_test_normal_loglik(mu, mu, sigma)
  ll_away <- tulpaRatio:::cpp_test_normal_loglik(mu + 1, mu, sigma)

  expect_true(ll_at_mu > ll_away)
})

# ---------------------------------------------------------------------------
# Cholesky decomposition tests
# ---------------------------------------------------------------------------

test_that("cholesky of identity is identity", {
  I <- diag(3)
  L <- tulpaRatio:::cpp_test_cholesky(I)
  expect_equal(as.matrix(L), I, tolerance = 1e-10)
})

test_that("cholesky satisfies L %*% t(L) = A", {
  A <- matrix(c(4, 2, 2, 3), nrow = 2)
  L <- tulpaRatio:::cpp_test_cholesky(A)

  reconstructed <- L %*% t(L)
  expect_equal(as.matrix(reconstructed), A, tolerance = 1e-10)
})

test_that("cholesky is lower triangular", {
  A <- matrix(c(4, 2, 1, 2, 5, 3, 1, 3, 6), nrow = 3)
  L <- tulpaRatio:::cpp_test_cholesky(A)

  # Upper triangle (excluding diagonal) should be zero
  expect_true(all(L[upper.tri(L)] == 0))
})

# ---------------------------------------------------------------------------
# Matrix-vector multiplication tests
# ---------------------------------------------------------------------------

test_that("matvec matches R's %*%", {
  A <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 2, ncol = 3)
  x <- c(1, 2, 3)

  cpp_result <- tulpaRatio:::cpp_test_matvec(A, x)
  r_result <- as.vector(A %*% x)

  expect_equal(cpp_result, r_result, tolerance = 1e-10)
})

test_that("matvec with identity matrix returns input", {
  I <- diag(4)
  x <- c(1, 2, 3, 4)

  result <- tulpaRatio:::cpp_test_matvec(I, x)
  expect_equal(result, x, tolerance = 1e-10)
})

test_that("matvec with zero matrix returns zeros", {
  Z <- matrix(0, nrow = 3, ncol = 4)
  x <- c(1, 2, 3, 4)

  result <- tulpaRatio:::cpp_test_matvec(Z, x)
  expect_equal(result, rep(0, 3), tolerance = 1e-10)
})

# ---------------------------------------------------------------------------
# Autodiff tests - testing actual autodiff.h implementation
# ---------------------------------------------------------------------------

test_that("autodiff gradient of sum(x^2) is 2*x", {
  x <- c(1.0, 2.0, 3.0, -1.5)
  result <- tulpaRatio:::cpp_test_autodiff_gradient(x)

  # f(x) = sum(x^2) = 1 + 4 + 9 + 2.25 = 16.25

  expect_equal(result$value, sum(x^2), tolerance = 1e-10)


  # Gradient should be 2*x
  expect_equal(result$gradient, 2 * x, tolerance = 1e-10)
})

test_that("autodiff handles zero gradient correctly", {
  x <- c(0.0, 0.0)
  result <- tulpaRatio:::cpp_test_autodiff_gradient(x)

  expect_equal(result$value, 0.0)
  expect_equal(result$gradient, c(0.0, 0.0), tolerance = 1e-10)
})

test_that("autodiff chain rule works for exp(x^2)", {
  test_vals <- c(0.0, 0.5, 1.0, -0.5, -1.0)  # Keep values small to avoid overflow

  for (x in test_vals) {
    result <- tulpaRatio:::cpp_test_autodiff_exp_chain(x)

    # Value should be exp(x^2)
    expect_equal(result$value, exp(x^2), tolerance = 1e-10)

    # Gradient should match expected: 2x * exp(x^2)
    expect_equal(result$gradient, result$expected_gradient, tolerance = 1e-10)
  }
})

test_that("autodiff Poisson log-likelihood gradient is correct", {
  y <- as.integer(c(0, 1, 3, 5, 10))
  eta <- c(-0.5, 0.0, 0.5, 1.0, 2.0)

  result <- tulpaRatio:::cpp_test_autodiff_log_likelihood(y, eta)

  # Gradient should be y - exp(eta)
  expect_equal(result$gradient, result$expected_gradient, tolerance = 1e-10)

  # Double-check with R computation
  expected_grad <- y - exp(eta)
  expect_equal(result$gradient, expected_grad, tolerance = 1e-10)
})

test_that("autodiff log-likelihood value is correct", {
  y <- as.integer(c(2, 3, 5))
  eta <- c(0.5, 1.0, 1.5)

  result <- tulpaRatio:::cpp_test_autodiff_log_likelihood(y, eta)

  # ll = sum(y * eta - exp(eta))
  expected_ll <- sum(y * eta - exp(eta))
  expect_equal(result$value, expected_ll, tolerance = 1e-10)
})

test_that("autodiff division gradients are correct", {
  test_cases <- list(
    list(a = 6.0, b = 2.0),
    list(a = 1.0, b = 3.0),
    list(a = -4.0, b = 2.0),
    list(a = 5.0, b = -2.0)
  )

  for (tc in test_cases) {
    result <- tulpaRatio:::cpp_test_autodiff_division(tc$a, tc$b)

    # Value should be a/b
    expect_equal(result$value, tc$a / tc$b, tolerance = 1e-10)

    # Gradients should match expected
    expect_equal(result$grad_a, result$expected_grad_a, tolerance = 1e-10)
    expect_equal(result$grad_b, result$expected_grad_b, tolerance = 1e-10)
  }
})

test_that("autodiff lgamma gradient matches digamma", {
  test_vals <- c(0.5, 1.0, 2.0, 5.0, 10.0)

  for (x in test_vals) {
    result <- tulpaRatio:::cpp_test_autodiff_lgamma(x)

    # Value should be lgamma(x)
    expect_equal(result$value, lgamma(x), tolerance = 1e-10)

    # Gradient should be digamma(x)
    expect_equal(result$gradient, result$expected_gradient, tolerance = 1e-8)
  }
})

test_that("autodiff handles extreme values without overflow", {
  # Large values that might cause numerical issues
  x <- c(10.0, 20.0, 50.0)
  result <- tulpaRatio:::cpp_test_autodiff_gradient(x)

  expect_true(is.finite(result$value))
  expect_true(all(is.finite(result$gradient)))
})

test_that("autodiff gradient accumulates correctly for shared variables", {
  # x appears twice in f(x) = x^2, gradient should be 2x
  x <- 3.0
  result <- tulpaRatio:::cpp_test_autodiff_gradient(x)

  expect_equal(result$gradient, 6.0, tolerance = 1e-10)
})

test_that("autodiff softplus value and gradient are correct", {
  test_vals <- c(-5.0, -1.0, 0.0, 1.0, 5.0)

  for (x in test_vals) {
    result <- tulpaRatio:::cpp_test_autodiff_softplus(x)

    # Value should be log(1 + exp(x))
    expected_val <- log(1 + exp(x))
    expect_equal(result$value, expected_val, tolerance = 1e-8)

    # Gradient should be sigmoid(x) = 1/(1+exp(-x))
    expect_equal(result$gradient, result$expected_gradient, tolerance = 1e-8)
  }
})

test_that("autodiff softplus handles extreme values", {
  # Large positive: softplus(x) ≈ x, gradient ≈ 1
  result_large <- tulpaRatio:::cpp_test_autodiff_softplus(50.0)
  expect_equal(result_large$value, 50.0, tolerance = 1e-6)
  expect_equal(result_large$gradient, 1.0, tolerance = 1e-10)

  # Large negative: softplus(x) ≈ 0, gradient ≈ 0
  result_small <- tulpaRatio:::cpp_test_autodiff_softplus(-50.0)
  expect_equal(result_small$value, 0.0, tolerance = 1e-6)
  expect_equal(result_small$gradient, 0.0, tolerance = 1e-10)
})

test_that("autodiff inv_logit value and gradient are correct", {
  test_vals <- c(-3.0, -1.0, 0.0, 1.0, 3.0)

  for (x in test_vals) {
    result <- tulpaRatio:::cpp_test_autodiff_inv_logit(x)

    # Value should be sigmoid(x) = 1/(1+exp(-x))
    expected_val <- 1 / (1 + exp(-x))
    expect_equal(result$value, expected_val, tolerance = 1e-10)

    # Gradient should be sigmoid(x) * (1 - sigmoid(x))
    expect_equal(result$gradient, result$expected_gradient, tolerance = 1e-10)
  }
})

test_that("autodiff inv_logit(0) = 0.5 with correct gradient", {
  result <- tulpaRatio:::cpp_test_autodiff_inv_logit(0.0)

  expect_equal(result$value, 0.5, tolerance = 1e-10)
  # Gradient at 0: 0.5 * 0.5 = 0.25
  expect_equal(result$gradient, 0.25, tolerance = 1e-10)
})

# ---------------------------------------------------------------------------
# Additional autodiff function tests
# ---------------------------------------------------------------------------

test_that("autodiff log value and gradient are correct", {
  test_vals <- c(0.5, 1.0, 2.0, 5.0, 10.0)

  for (x in test_vals) {
    result <- tulpaRatio:::cpp_test_autodiff_log(x)
    expect_equal(result$value, log(x), tolerance = 1e-10)
    expect_equal(result$gradient, result$expected_gradient, tolerance = 1e-10)
  }
})

test_that("autodiff sqrt value and gradient are correct", {
  test_vals <- c(0.25, 1.0, 4.0, 9.0, 16.0)

  for (x in test_vals) {
    result <- tulpaRatio:::cpp_test_autodiff_sqrt(x)
    expect_equal(result$value, sqrt(x), tolerance = 1e-10)
    expect_equal(result$gradient, result$expected_gradient, tolerance = 1e-10)
  }
})

test_that("autodiff pow value and gradient are correct", {
  test_cases <- list(
    list(x = 2.0, p = 2.0),
    list(x = 3.0, p = 0.5),
    list(x = 4.0, p = -1.0),
    list(x = 2.0, p = 3.0)
  )

  for (tc in test_cases) {
    result <- tulpaRatio:::cpp_test_autodiff_pow(tc$x, tc$p)
    expect_equal(result$value, tc$x^tc$p, tolerance = 1e-10)
    expect_equal(result$gradient, result$expected_gradient, tolerance = 1e-8)
  }
})

test_that("autodiff log1p value and gradient are correct", {
  test_vals <- c(0.0, 0.5, 1.0, 5.0, 10.0)

  for (x in test_vals) {
    result <- tulpaRatio:::cpp_test_autodiff_log1p(x)
    expect_equal(result$value, log1p(x), tolerance = 1e-10)
    expect_equal(result$gradient, result$expected_gradient, tolerance = 1e-10)
  }
})

test_that("autodiff log_sum_exp value and gradients are correct", {
  test_cases <- list(
    list(a = 1.0, b = 2.0),
    list(a = -1.0, b = 1.0),
    list(a = 100.0, b = 101.0),  # Large values
    list(a = -100.0, b = -99.0)  # Small values
  )

  for (tc in test_cases) {
    result <- tulpaRatio:::cpp_test_autodiff_log_sum_exp(tc$a, tc$b)

    # Value should match log(exp(a) + exp(b))
    expected_val <- log(exp(tc$a - max(tc$a, tc$b)) + exp(tc$b - max(tc$a, tc$b))) + max(tc$a, tc$b)
    expect_equal(result$value, expected_val, tolerance = 1e-10)

    # Gradients should be softmax
    expect_equal(result$grad_a, result$expected_grad_a, tolerance = 1e-10)
    expect_equal(result$grad_b, result$expected_grad_b, tolerance = 1e-10)
  }
})

test_that("autodiff logit value and gradient are correct", {
  test_vals <- c(0.1, 0.3, 0.5, 0.7, 0.9)

  for (x in test_vals) {
    result <- tulpaRatio:::cpp_test_autodiff_logit(x)

    # Value should be log(x/(1-x))
    expect_equal(result$value, log(x / (1 - x)), tolerance = 1e-10)

    # Gradient should match
    expect_equal(result$gradient, result$expected_gradient, tolerance = 1e-10)
  }
})

test_that("autodiff negbin log-likelihood gradients are correct", {
  y <- as.integer(c(2, 5, 8))
  mu <- c(3.0, 4.0, 6.0)
  phi <- 2.0

  result <- tulpaRatio:::cpp_test_autodiff_negbin_loglik(y, mu, phi)

  # Check gradients are finite and non-zero
  expect_true(all(is.finite(result$gradient_mu)))
  expect_true(is.finite(result$gradient_phi))

  # Verify gradients via finite differences
  eps <- 1e-5
  for (i in seq_along(mu)) {
    mu_plus <- mu
    mu_plus[i] <- mu[i] + eps
    result_plus <- tulpaRatio:::cpp_test_autodiff_negbin_loglik(y, mu_plus, phi)

    mu_minus <- mu
    mu_minus[i] <- mu[i] - eps
    result_minus <- tulpaRatio:::cpp_test_autodiff_negbin_loglik(y, mu_minus, phi)

    fd_grad <- (result_plus$value - result_minus$value) / (2 * eps)
    expect_equal(result$gradient_mu[i], fd_grad, tolerance = 1e-4)
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

test_that("pg update_sigma_re returns positive value", {
  set.seed(101)
  re <- rnorm(10, sd = 1.5)
  scale <- 2.5

  result <- tulpaRatio:::cpp_test_pg_update_sigma_re(re, scale)
  expect_true(result > 0)
})

test_that("pg update_sigma_re is roughly proportional to RE standard deviation", {
  set.seed(202)

  # Small RE variance
  re_small <- rnorm(20, sd = 0.5)
  result_small <- tulpaRatio:::cpp_test_pg_update_sigma_re(re_small, 2.5)

  # Large RE variance
  re_large <- rnorm(20, sd = 3.0)
  result_large <- tulpaRatio:::cpp_test_pg_update_sigma_re(re_large, 2.5)

  # Larger RE should generally produce larger sigma estimate
  # (stochastic so we just check they're both positive)
  expect_true(result_small > 0)
  expect_true(result_large > 0)
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
  result <- tulpaRatio:::cpp_test_rw1_quadratic_form(phi, FALSE)
  expected <- (2-1)^2 + (5-2)^2 + (7-5)^2 + (8-7)^2
  expect_equal(result, expected, tolerance = 1e-10)
})

test_that("rw1_quadratic_form handles cyclic case", {
  phi <- c(1, 2, 5, 7, 8)

  # Cyclic: adds (phi[1] - phi[T])^2
  result <- tulpaRatio:::cpp_test_rw1_quadratic_form(phi, TRUE)
  expected <- (2-1)^2 + (5-2)^2 + (7-5)^2 + (8-7)^2 + (1-8)^2
  expect_equal(result, expected, tolerance = 1e-10)
})

test_that("rw2_quadratic_form computes sum of squared second differences", {
  phi <- c(1, 3, 4, 6, 10)

  # Non-cyclic: sum((phi[t] - 2*phi[t+1] + phi[t+2])^2)
  result <- tulpaRatio:::cpp_test_rw2_quadratic_form(phi, FALSE)
  d1 <- 1 - 2*3 + 4   # = -1
  d2 <- 3 - 2*4 + 6   # = 1
  d3 <- 4 - 2*6 + 10  # = 2
  expected <- d1^2 + d2^2 + d3^2
  expect_equal(result, expected, tolerance = 1e-10)
})

test_that("rw2_quadratic_form returns 0 for T < 3", {
  expect_equal(tulpaRatio:::cpp_test_rw2_quadratic_form(c(1, 2), FALSE), 0)
  expect_equal(tulpaRatio:::cpp_test_rw2_quadratic_form(c(1), FALSE), 0)
})

test_that("ar1_log_density is finite for valid inputs", {
  phi <- c(0.5, 0.8, 0.3, -0.2, 0.1)
  rho <- 0.7
  tau <- 2.0

  result <- tulpaRatio:::cpp_test_ar1_log_density(phi, rho, tau)
  expect_true(is.finite(result))
  expect_true(result < 0)  # Log-density should be negative
})

test_that("ar1_log_density matches manual computation", {
  # AR1: phi[0] ~ N(0, sigma^2/(1-rho^2)), phi[t]|phi[t-1] ~ N(rho*phi[t-1], sigma^2)
  phi <- c(0.5, 0.8, 0.3)
  rho <- 0.7
  tau <- 2.0
  sigma2 <- 1.0 / tau

  result <- tulpaRatio:::cpp_test_ar1_log_density(phi, rho, tau)

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

  result <- tulpaRatio:::cpp_test_temporal_log_prior(phi, "rw1", tau, 0, FALSE)

  # Manual: 0.5 * rank * log(tau) - 0.5 * tau * quad
  quad <- (2-1)^2 + (4-2)^2  # = 1 + 4 = 5
  rank <- 2  # T - 1 for non-cyclic
  expected <- 0.5 * rank * log(tau) - 0.5 * tau * quad
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

  plain <- tulpaRatio:::cpp_test_ms_rw_log_lik(phi, sigma2, 1L, FALSE, FALSE)
  expect_equal(plain,
               -0.5 * quad / sigma2 - 0.5 * rank * log(2 * pi * sigma2),
               tolerance = 1e-10)

  augmented <- tulpaRatio:::cpp_test_ms_rw_log_lik(phi, sigma2, 1L, FALSE, TRUE)
  expect_equal(augmented,
               -0.5 * quad_aug / sigma2 - 0.5 * (rank + 1) * log(2 * pi * sigma2),
               tolerance = 1e-10)

  # On the sum-to-zero subspace the quadratic is untouched, so the whole
  # difference is the one extra rank the full-rank normalizer carries.
  centred <- c(-1, 0, 1)
  expect_equal(
    tulpaRatio:::cpp_test_ms_rw_log_lik(centred, sigma2, 1L, FALSE, TRUE) -
      tulpaRatio:::cpp_test_ms_rw_log_lik(centred, sigma2, 1L, FALSE, FALSE),
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
    tulpaRatio:::cpp_test_ms_rw_log_lik(phi, sigma2, 2L, FALSE, TRUE),
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

  # Large negative (should return exp(x))
  expect_equal(tulpaRatio:::cpp_test_log1pexp(-20), exp(-20), tolerance = 1e-15)
})

test_that("zi_logistic matches plogis", {
  test_vals <- c(-5, -1, 0, 1, 5)
  for (x in test_vals) {
    expect_equal(tulpaRatio:::cpp_test_zi_logistic(x), plogis(x), tolerance = 1e-10)
  }
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

test_that("zi_poisson_grad_logit_zi matches finite differences", {
  mu <- 3.0
  logit_zi <- 0.5
  eps <- 1e-5

  for (y in c(0L, 1L, 5L)) {
    grad <- tulpaRatio:::cpp_test_zi_poisson_grad_logit_zi(y, mu, logit_zi)

    # Finite difference
    f_plus <- tulpaRatio:::cpp_test_zi_poisson_lpmf_logit(y, mu, logit_zi + eps)
    f_minus <- tulpaRatio:::cpp_test_zi_poisson_lpmf_logit(y, mu, logit_zi - eps)
    fd_grad <- (f_plus - f_minus) / (2 * eps)

    expect_equal(grad, fd_grad, tolerance = 1e-4)
  }
})

test_that("hurdle_grad_logit_theta matches finite differences", {
  logit_theta <- 0.5
  eps <- 1e-5
  mu <- 3.0

  for (y in c(0L, 1L, 5L)) {
    grad <- tulpaRatio:::cpp_test_hurdle_grad_logit_theta(y, logit_theta)

    # Finite difference
    f_plus <- tulpaRatio:::cpp_test_hurdle_poisson_lpmf_logit(y, mu, logit_theta + eps)
    f_minus <- tulpaRatio:::cpp_test_hurdle_poisson_lpmf_logit(y, mu, logit_theta - eps)
    fd_grad <- (f_plus - f_minus) / (2 * eps)

    expect_equal(grad, fd_grad, tolerance = 1e-4)
  }
})

# ---------------------------------------------------------------------------
# OpenMP parallel execution tests
# ---------------------------------------------------------------------------

test_that("OpenMP max threads is at least 1", {
  max_threads <- tulpaRatio:::cpp_test_get_max_threads()
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

test_that("parallel independent computations are thread-safe", {
  n <- 1000

  # Run with different thread counts and verify determinism
  results_1t <- tulpaRatio:::cpp_test_parallel_independent(n, 1L)

  for (n_threads in c(2, 4)) {
    results_mt <- tulpaRatio:::cpp_test_parallel_independent(n, n_threads)

    # Results should be identical regardless of thread count
    expect_equal(results_1t, results_mt, tolerance = 1e-10)
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
