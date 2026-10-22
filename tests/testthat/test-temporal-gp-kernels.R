# Each temporal_gp() kernel reaches the density the samplers evaluate. The
# reference is the multivariate normal of the kernel's covariance, written out
# here independently of the C++.

kernel_r <- function(d, phi, kernel, period = 1) {
  switch(kernel,
    exponential = exp(-d / phi),
    matern32 = {
      u <- sqrt(3) * d / phi
      (1 + u) * exp(-u)
    },
    matern52 = {
      u <- sqrt(5) * d / phi
      (1 + u + u^2 / 3) * exp(-u)
    },
    gaussian = exp(-0.5 * (d / phi)^2),
    periodic = exp(-2 * sin(pi * d / period)^2 / phi^2)
  )
}

tgp_bundle <- function(times, cov, nu = 1.5, period = 1, param = "centered") {
  list(n_times = length(times), n_groups = 1L, shared = TRUE,
       time_values = times, cov_type = cov, nu = nu, period = period,
       gp_sigma2_prior_U = 1, gp_sigma2_prior_alpha = 0.01,
       gp_phi_prior_lower = 0.01, gp_phi_prior_upper = 10,
       gp_parameterization = param)
}

KERNELS <- list(
  exponential = list(cov = "exponential", nu = 1.5),
  matern32 = list(cov = "matern", nu = 1.5),
  matern52 = list(cov = "matern", nu = 2.5),
  gaussian = list(cov = "gaussian", nu = 1.5),
  periodic = list(cov = "periodic", nu = 1.5)
)

times <- c(0, 0.7, 1.1, 2.4, 3.0, 4.6, 5.2)
sigma2 <- 0.8
phi <- 1.3
period <- 2.5
logit_phi <- qlogis((phi - 0.01) / (10 - 0.01))

# The correlation the sampler factors: the exponential kernel goes through its
# exact AR(1) recursion, the others carry a 1e-6 nugget.
corr_r <- function(kernel) {
  R <- kernel_r(abs(outer(times, times, "-")), phi, kernel, period)
  if (kernel == "exponential") R else R + diag(1e-6, length(times))
}

# PC prior on sigma and uniform prior on phi, on (log sigma2, logit phi)
hyper_log_prior <- function() {
  rate <- -log(0.01) / 1
  sigma <- sqrt(sigma2)
  log(rate) - rate * sigma - log(2 * sigma) + log(sigma2) +
    log(phi - 0.01) + log(10 - phi) - log(10 - 0.01)
}

test_that("each kernel's centred density is the normal of its covariance", {
  set.seed(1)
  f <- rnorm(length(times))
  for (k in names(KERNELS)) {
    p <- tgp_bundle(times, KERNELS[[k]]$cov, KERNELS[[k]]$nu, period)
    K <- sigma2 * corr_r(k)
    ref <- -0.5 * (length(f) * log(2 * pi) +
                   as.numeric(determinant(K)$modulus) + sum(f * solve(K, f)))
    got <- tulpaRatio:::cpp_temporal_gp_log_prior(f, log(sigma2), logit_phi, p)
    expect_equal(got$log_prior, hyper_log_prior() + ref, tolerance = 1e-8, label = k)
    expect_equal(got$f, f)
  }
})

test_that("the non-centred block is N(0, I) and maps to sigma L z", {
  set.seed(2)
  z <- rnorm(length(times))
  for (k in names(KERNELS)) {
    p <- tgp_bundle(times, KERNELS[[k]]$cov, KERNELS[[k]]$nu, period,
                    param = "noncentered")
    got <- tulpaRatio:::cpp_temporal_gp_log_prior(z, log(sigma2), logit_phi, p)
    expect_equal(got$log_prior, hyper_log_prior() - 0.5 * sum(z^2),
                 tolerance = 1e-10, label = k)
    expect_equal(got$f, as.vector(sqrt(sigma2) * t(chol(corr_r(k))) %*% z),
                 tolerance = 1e-8, label = k)
  }
})

test_that("the draws' effects come from the sampler's kernel", {
  set.seed(3)
  z <- matrix(rnorm(3 * length(times)), 3)
  p <- tgp_bundle(times, "matern", 2.5, param = "noncentered")
  f <- tulpaRatio:::cpp_temporal_gp_field(z, rep(log(sigma2), 3),
                                          rep(logit_phi, 3), p)
  L <- t(chol(corr_r("matern52")))
  expect_equal(f, t(sqrt(sigma2) * L %*% t(z)), tolerance = 1e-8)
})

test_that("temporal_gp() takes the Matern smoothnesses with a closed form", {
  expect_error(temporal_gp("t", cov = "matern", nu = 1), "0.5, 1.5 or 2.5")
  expect_silent(temporal_gp("t", cov = "matern", nu = 0.5))
})

test_that("a period in the data's units is rescaled with the time values", {
  d <- data.frame(month = rep(1:36, 2))
  spec <- tulpaRatio:::validate_temporal_gp(
    temporal_gp("month", cov = "periodic", period = 12), d)
  info <- tulpaRatio:::prepare_temporal_for_hmc(spec, d, nrow(d))
  expect_equal(info$period, 12 / stats::sd(d$month))
  # One period apart in months is one period apart in the scaled times.
  tv <- info$time_values
  expect_equal(tv[13] - tv[1], info$period)
})

test_that("the kernel changes the fit", {
  skip_on_cran()
  set.seed(4)
  d <- data.frame(time = rep(1:15, each = 4), x = rnorm(60))
  d$n <- 20L
  d$y <- rbinom(60, d$n, plogis(0.3 * d$x + sin(d$time / 2)))
  ctrl <- list(iter = 200, warmup = 100, chains = 1, verbose = FALSE, seed = 9)
  sig <- vapply(c("exponential", "gaussian"), function(cv) {
    fit <- suppressWarnings(tratio(y | n ~ x, data = d, family = ratiod_binomial(),
                                   temporal = temporal_gp("time", cov = cv),
                                   control = ctrl))
    dr <- posterior::as_draws_df(fit$draws)
    mean(dr[[grep("phi_temporal", posterior::variables(dr), value = TRUE)[1]]])
  }, numeric(1))
  expect_false(isTRUE(all.equal(sig[[1]], sig[[2]])))
})
