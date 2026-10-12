# A known exposure scales the numerator mean, and a Poisson numerator pairs
# with a negative binomial denominator. Each pair has the gradient of its log
# posterior, recovers the model it was simulated from, and reports the rate as
# its ratio.

exposure_rel_dev <- function(r, atol = 1e-6) {
  max(pmax(0, abs(r$analytic - r$finite_diff) - atol) /
        pmax(1e-6, pmax(abs(r$analytic), abs(r$finite_diff))))
}

test_that("the new pairs have the gradient of their log posterior", {
  for (family in c("poisson_fixed", "negbin_fixed", "lognormal_fixed",
                   "poisson_negbin")) {
    for (field in c("none", "re", "icar")) {
      r <- tulpaRatio:::cpp_gradient_check(field, family = family)
      expect_lt(exposure_rel_dev(r), 1e-4, label = paste(family, field))
    }
  }
  inflated <- list(poisson_fixed = c("zi_poisson", "hurdle_poisson"),
                   negbin_fixed = c("zi_negbin", "hurdle_negbin"),
                   poisson_negbin = c("zi_poisson", "hurdle_poisson"))
  for (family in names(inflated)) {
    for (zi in inflated[[family]]) {
      r <- tulpaRatio:::cpp_gradient_check("none", family = family, zi = zi)
      expect_lt(exposure_rel_dev(r), 1e-4, label = paste(family, zi))
    }
  }
})

exposure_data <- function(n = 400, seed = 31) {
  set.seed(seed)
  x <- rnorm(n)
  exposure <- runif(n, 0.5, 4)
  rate <- exp(0.4 + 0.5 * x)
  structural_zero <- runif(n) < 0.25
  data.frame(
    x = x, exposure = exposure,
    count = ifelse(structural_zero, 0L, rpois(n, rate * exposure)),
    trapped = rnbinom(n, size = 6, mu = exp(2 + 0.3 * x)),
    mass = exposure * exp(0.4 + 0.5 * x + rnorm(n, sd = 0.3))
  )
}

test_that("a zero-inflated Poisson over a known exposure recovers its rate", {
  skip_on_cran()
  d <- exposure_data()
  fit <- tratio(count | exposure ~ x, data = d,
                family = ratiod_zipois(denom_family = "fixed"),
                control = list(iter = 1000, chains = 2, seed = 1))
  b <- coef(fit)
  expect_false(any(grepl("^denom:", names(b))))
  expect_equal(unname(b[c("num:(Intercept)", "num:x")]), c(0.4, 0.5),
               tolerance = 0.15)

  # The ratio is the rate; the numerator mean is the rate times the exposure
  fv <- fitted(fit, summary = FALSE)
  expect_equal(fv$numerator, fv$ratio * fv$denominator)
  expect_equal(fv$denominator[1, ], d$exposure)
  r <- ratio(fit, summary = FALSE)
  expect_equal(unname(colMeans(r$draws)), unname(colMeans(fv$ratio)), tolerance = 1e-8)

  rep <- posterior_predict(fit, ndraws = 20)
  expect_equal(dim(rep$y_num_rep), c(20L, nrow(d)))
  expect_equal(rep$y_denom_rep[1, ], d$exposure)
})

test_that("a log-normal numerator over a known exposure is least squares on log(y / E)", {
  skip_on_cran()
  d <- exposure_data()
  fit <- tratio(mass | exposure ~ x, data = d,
                family = ratiod_lognormal(denom_fixed = TRUE),
                control = list(iter = 1000, chains = 2, seed = 1))
  ols <- stats::coef(stats::lm(log(mass / exposure) ~ x, data = d))
  expect_equal(unname(coef(fit)[c("num:(Intercept)", "num:x")]), unname(ols),
               tolerance = 0.02)
})

test_that("a Poisson numerator with a negative binomial denominator is recovered", {
  skip_on_cran()
  d <- exposure_data()
  fit <- tratio(count | trapped ~ x, data = d,
                family = ratiod_zipois(denom_family = "negbin"),
                control = list(iter = 1000, chains = 2, seed = 1))
  b <- coef(fit)
  expect_equal(unname(b[c("denom:(Intercept)", "denom:x")]), c(2, 0.3),
               tolerance = 0.1)
  expect_true(all(is.finite(loo(fit)$estimates)))
})

test_that("a known exposure must be positive", {
  d <- exposure_data(20)
  d$exposure[3] <- 0
  expect_error(
    tratio(count | exposure ~ x, data = d,
           family = ratiod_zipois(denom_family = "fixed"),
           control = list(iter = 50, chains = 1, seed = 1)),
    "must be positive")
})

test_that("samplers with likelihoods of their own refuse the new pairs", {
  d <- exposure_data(40)
  expect_error(
    tratio(count | exposure ~ x, data = d,
           family = ratiod_zipois(denom_family = "fixed"), mode = "laplace"),
    "likelihood")
})
