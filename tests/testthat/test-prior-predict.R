pp_data <- function(n = 40, seed = 1) {
  set.seed(seed)
  data.frame(
    count = rep(0L, n),
    total = rep(0L, n),
    x = rnorm(n),
    site = factor(rep(seq_len(n / 5), each = 5))
  )
}

pp_trial_data <- function(n = 40, seed = 2) {
  set.seed(seed)
  data.frame(
    detections = rep(0L, n),
    trials = rep(10L, n),
    x = rnorm(n),
    site = factor(rep(seq_len(n / 5), each = 5))
  )
}

test_that("prior_predict returns the documented shape for every count family", {
  df <- pp_data()
  fams <- list(
    ratiod_negbin_negbin(),
    ratiod_poisson_gamma(),
    ratiod_negbin_gamma(),
    ratiod_gamma_gamma(),
    ratiod_lognormal()
  )

  for (fam in fams) {
    set.seed(11)
    pp <- prior_predict(count | total ~ x + (1 | site), family = fam,
                        data = df, n = 20)

    expect_s3_class(pp, "ratiod_priorpred")
    for (nm in c("y_num", "y_denom", "ratio", "mu_num", "mu_denom")) {
      expect_equal(dim(pp[[nm]]), c(20L, 40L), info = paste(fam$name, nm))
    }
    expect_equal(dim(pp$beta_num), c(20L, 2L), info = fam$name)
    expect_equal(colnames(pp$beta_num), c("(Intercept)", "x"), info = fam$name)
    expect_true(all(is.finite(pp$ratio)), info = fam$name)
    expect_true(all(pp$y_num >= 0), info = fam$name)
  }
})

test_that("the two arms draw their random-effect scales separately", {
  set.seed(12)
  pp <- prior_predict(count | total ~ x + (1 | site),
                      family = ratiod_negbin_negbin(),
                      data = pp_data(), n = 40)

  expect_equal(colnames(pp$sigma), c("num.site.sigma", "denom.site.sigma"))
  # Independent draws, so the two columns must not coincide.
  expect_false(isTRUE(all.equal(pp$sigma[, 1], pp$sigma[, 2])))
  expect_true(all(pp$sigma > 0))
})

test_that("a slope term contributes one scale per column", {
  set.seed(13)
  pp <- prior_predict(count | total ~ x + (1 + x | site),
                      family = ratiod_negbin_negbin(),
                      data = pp_data(), n = 10)

  expect_equal(colnames(pp$sigma),
               c("num.site.sigma", "num.site.x.sigma",
                 "denom.site.sigma", "denom.site.x.sigma"))
})

test_that("the response column need not be present in data", {
  df <- pp_data()[, c("x", "site")]
  set.seed(14)
  pp <- prior_predict(count | total ~ x + (1 | site),
                      family = ratiod_negbin_negbin(), data = df, n = 10)
  expect_equal(dim(pp$ratio), c(10L, 40L))
})

test_that("a tighter coefficient prior narrows the predicted ratios", {
  df <- pp_data()

  set.seed(15)
  wide <- prior_predict(count | total ~ x, family = ratiod_negbin_negbin(),
                        data = df, priors = ratiod_priors(beta = prior_normal(0, 2.5)),
                        n = 400)
  set.seed(15)
  tight <- prior_predict(count | total ~ x, family = ratiod_negbin_negbin(),
                         data = df, priors = ratiod_priors(beta = prior_normal(0, 0.25)),
                         n = 400)

  expect_lt(sd(log(as.vector(tight$ratio))), sd(log(as.vector(wide$ratio))))
})

test_that("the fixed-denominator families use the trial counts as data", {
  df <- pp_trial_data()

  set.seed(16)
  pp <- prior_predict(detections | trials ~ x + (1 | site),
                      family = ratiod_binomial(), data = df, n = 20)

  expect_true(all(pp$y_denom == 10))
  expect_true(all(pp$y_num >= 0 & pp$y_num <= 10))
  # The reported ratio is the modelled probability, not successes over trials.
  expect_true(all(pp$ratio >= 0 & pp$ratio <= 1))
  expect_false(isTRUE(all.equal(pp$ratio, pp$y_num / pp$y_denom)))
  expect_equal(ncol(pp$beta_denom), 0L)

  set.seed(17)
  ppb <- prior_predict(detections | trials ~ x + (1 | site),
                       family = ratiod_beta_binomial(), data = df, n = 20)
  expect_true(all(ppb$y_num >= 0 & ppb$y_num <= 10))
})

test_that("a binomial family without trial counts is refused by name", {
  df <- pp_trial_data()[, c("detections", "x", "site")]
  expect_error(
    prior_predict(detections | trials ~ x, family = ratiod_binomial(),
                  data = df, n = 5),
    "Trial counts"
  )
})

test_that("the draws are reproducible under a seed", {
  df <- pp_data()
  set.seed(18)
  a <- prior_predict(count | total ~ x + (1 | site),
                     family = ratiod_negbin_negbin(), data = df, n = 15)
  set.seed(18)
  b <- prior_predict(count | total ~ x + (1 | site),
                     family = ratiod_negbin_negbin(), data = df, n = 15)
  expect_identical(a$y_num, b$y_num)
  expect_identical(a$sigma, b$sigma)
})

test_that("prior_predict validates its arguments", {
  df <- pp_data()
  expect_error(prior_predict(count | total ~ x, family = "negbin", data = df),
               "ratiod_family")
  expect_error(prior_predict(count | total ~ x, family = ratiod_negbin_negbin(),
                             data = list(x = 1)), "data frame")
  expect_error(prior_predict(count | total ~ x, family = ratiod_negbin_negbin(),
                             data = df, n = 0), "positive integer")
  expect_error(prior_predict(count | total ~ x, family = ratiod_negbin_negbin(),
                             data = df, priors = list(beta = 1)),
               "ratiod_priors")
})

test_that("prior_predict names the arm a formula leaves unread", {
  df <- pp_data()
  # A single-response formula names the numerator and leaves the denominator
  # to `formula_denom`; a one-sided formula names neither.
  expect_error(
    prior_predict(count ~ x, family = ratiod_negbin_negbin(), data = df, n = 5),
    "formula_denom"
  )
  expect_error(
    prior_predict(~ x, family = ratiod_negbin_negbin(), data = df, n = 5),
    "response and predictors"
  )
})

test_that("every prior_*() constructor can be drawn from", {
  priors <- list(
    prior_normal(1, 2), prior_half_normal(1), prior_half_cauchy(2),
    prior_gamma(2, 0.5), prior_exponential(1), prior_beta(2, 3),
    prior_pc(1, 0.01)
  )
  for (p in priors) {
    d <- pp_draw_prior(p, 50)
    expect_length(d, 50)
    expect_true(all(is.finite(d)), info = p$distribution)
  }
  # The positive-support priors must stay positive.
  for (p in priors[-1]) {
    expect_true(all(pp_draw_prior(p, 50) >= 0), info = p$distribution)
  }
})

test_that("each inverse link maps to its own range", {
  eta <- seq(-3, 3, length.out = 25)
  expect_true(all(pp_linkinv("log")(eta) > 0))
  expect_equal(pp_linkinv("identity")(eta), eta)
  for (lk in c("logit", "probit", "cloglog")) {
    p <- pp_linkinv(lk)(eta)
    expect_true(all(p > 0 & p < 1), info = lk)
    expect_true(all(diff(p) > 0), info = lk)
  }
  expect_error(pp_linkinv("sqrt"), "Unsupported link")
})

test_that("the response sampler recovers the mean it was given", {
  set.seed(19)
  mu <- rep(4, 20000)

  expect_equal(mean(pp_rdist("poisson", mu, NA)), 4, tolerance = 0.05)
  expect_equal(mean(pp_rdist("neg_binomial_2", mu, 5)), 4, tolerance = 0.1)
  expect_equal(mean(pp_rdist("gamma", mu, 3)), 4, tolerance = 0.1)

  p <- rep(0.3, 20000)
  expect_equal(mean(pp_rdist("binomial", p, NA, trials = rep(10L, 20000))) / 10,
               0.3, tolerance = 0.02)
  expect_equal(mean(pp_rdist("beta_binomial", p, 20,
                             trials = rep(10L, 20000))) / 10,
               0.3, tolerance = 0.02)

  expect_error(pp_rdist("weibull", mu, 1), "Cannot simulate")
})

test_that("print reports the family, the draw count and the ratio quantiles", {
  set.seed(20)
  pp <- prior_predict(count | total ~ x, family = ratiod_negbin_negbin(),
                      data = pp_data(), n = 10)
  out <- capture.output(print(pp))
  expect_true(any(grepl("negbin_negbin", out)))
  expect_true(any(grepl("Implied ratio", out)))
  expect_invisible(print(pp))
})
