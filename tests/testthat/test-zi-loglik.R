# The pointwise likelihood of every inflation type is a probability mass
# function: summed over the support it is one.

pmf_sum <- function(zi_type, n_trials, support, eta = 0.3, logit_zi = -0.4,
                    logit_oi = 0.5, phi = 4) {
  N <- length(support)
  fit <- list(zi = list(type = zi_type, X_zi = matrix(1, N, 1),
                        X_oi = matrix(1, N, 1)))
  draws <- matrix(c(logit_zi, logit_oi), 1, 2,
                  dimnames = list(NULL, c("beta_zi[1]", "beta_oi[1]")))
  hmc_data <- list(y_num = support, y_denom = rep(n_trials, N))
  ll <- tulpaRatio:::.ratiod_inflated_num_loglik(
    fit, draws, matrix(eta, 1, N), list(phi_num = phi), hmc_data, zi_type)
  sum(exp(ll))
}

test_that("every inflated pmf sums to one", {
  for (type in c("zi_binomial", "hurdle_binomial", "oi_binomial", "zoib")) {
    expect_equal(pmf_sum(type, 6, 0:6), 1, tolerance = 1e-12, info = type)
  }
  for (type in c("zi_poisson", "hurdle_poisson", "zi_negbin", "hurdle_negbin")) {
    expect_equal(pmf_sum(type, 0, 0:400), 1, tolerance = 1e-10, info = type)
  }
})

test_that("loo() and posterior_predict() run on zero-inflated fits", {
  skip_on_cran()
  set.seed(8)
  n <- 60
  d <- data.frame(y = ifelse(runif(n) < 0.3, 0, rpois(n, 5)),
                  e = rgamma(n, 4, 1), x = rnorm(n),
                  k = rbinom(n, 8, 0.5), trials = 8L)
  d$k[1:8] <- 0L
  d$k[9:14] <- 8L
  fits <- list(
    zipois = tratio(y | e ~ x, data = d, family = ratiod_zipois(),
                    control = list(iter = 300, chains = 1, seed = 1)),
    zoib = tratio(k | trials ~ x, data = d, family = ratiod_zoibinomial(),
                  control = list(iter = 300, chains = 1, seed = 1))
  )
  for (nm in names(fits)) {
    l <- suppressWarnings(loo(fits[[nm]]))
    expect_true(is.finite(l$estimates["elpd_loo", "Estimate"]), info = nm)
    pp <- posterior_predict(fits[[nm]], ndraws = 20)
    expect_equal(dim(pp$y_num_rep), c(20L, n), info = nm)
    expect_true(any(pp$y_num_rep == 0), info = nm)
  }
})

test_that("a hurdle fit has no region where the truncation adds mass", {
  # Where a small mean puts P(Y = 0) within rounding of 1, log(1 - P0) used to
  # come out as the -1e10 sentinel, which the truncation then subtracted: the
  # log-posterior gained 1e10 per observation there and a chain that wandered
  # in stayed.
  skip_on_cran()
  set.seed(123)
  n <- 120
  d <- data.frame(site = factor(rep(1:12, each = 10)),
                  habitat = factor(rep(c("a", "b"), 60)), trials = 10L)
  d$detected <- ifelse(runif(n) < 0.7, pmax(1L, rbinom(n, 10, 0.4)), 0L)
  fit <- tratio(detected | trials ~ habitat + (1 | site),
                zi = hurdle_binomial(~ habitat), data = d,
                family = ratiod_hurdle_binomial(),
                control = list(iter = 300, chains = 2, cores = 1, seed = 1,
                               gradient_mode = "A"))
  expect_true(all(fit$log_prob < 0))
  expect_equal(fit$diagnostics$n_divergent, 0)
})

test_that("a zero-inflated negative binomial fits a Gamma denominator", {
  skip_on_cran()
  set.seed(3)
  n <- 80
  d <- data.frame(x = rnorm(n), e = rgamma(n, 4, 1))
  d$y <- ifelse(runif(n) < 0.3, 0L,
                rnbinom(n, size = 3, mu = d$e * exp(0.5 + 0.4 * d$x)))
  fit <- tratio(y | e ~ x, data = d,
                family = ratiod_zinegbin(denom_family = "gamma"),
                control = list(iter = 300, chains = 1, seed = 1))
  l <- suppressWarnings(loo(fit))
  expect_true(is.finite(l$estimates["elpd_loo", "Estimate"]))
  pp <- posterior_predict(fit, ndraws = 5)
  expect_true(any(pp$y_denom_rep != round(pp$y_denom_rep)))
})
