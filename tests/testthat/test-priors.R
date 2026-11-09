# Tests for prior specification (v0.9.0)

test_that("prior_normal creates valid prior", {
  p <- prior_normal(0, 2.5)

  expect_s3_class(p, "ratiod_prior")
  expect_equal(p$distribution, "normal")
  expect_equal(p$mean, 0)
  expect_equal(p$sd, 2.5)
})

test_that("prior_normal validates sd", {
  expect_error(prior_normal(0, -1), "positive")
  expect_error(prior_normal(0, 0), "positive")
})

test_that("prior_half_normal creates valid prior", {
  p <- prior_half_normal(1)

  expect_s3_class(p, "ratiod_prior")
  expect_equal(p$distribution, "half_normal")
  expect_equal(p$sd, 1)
})

test_that("prior_half_cauchy creates valid prior", {
  p <- prior_half_cauchy(2.5)

  expect_s3_class(p, "ratiod_prior")
  expect_equal(p$distribution, "half_cauchy")
  expect_equal(p$scale, 2.5)
})

test_that("prior_gamma creates valid prior", {
  p <- prior_gamma(2, 0.1)

  expect_s3_class(p, "ratiod_prior")
  expect_equal(p$distribution, "gamma")
  expect_equal(p$shape, 2)
  expect_equal(p$rate, 0.1)
})

test_that("prior_gamma validates parameters", {
  expect_error(prior_gamma(-1, 0.1), "positive")
  expect_error(prior_gamma(2, -0.1), "positive")
})

test_that("prior_exponential creates valid prior", {
  p <- prior_exponential(1)

  expect_s3_class(p, "ratiod_prior")
  expect_equal(p$distribution, "exponential")
  expect_equal(p$rate, 1)
})

test_that("prior_beta creates valid prior", {
  p <- prior_beta(2, 2)

  expect_s3_class(p, "ratiod_prior")
  expect_equal(p$distribution, "beta")
  expect_equal(p$alpha, 2)
  expect_equal(p$beta, 2)
})

test_that("prior_beta validates parameters", {
  expect_error(prior_beta(-1, 2), "positive")
  expect_error(prior_beta(2, 0), "positive")
})

test_that("prior_pc creates valid prior", {
  p <- prior_pc(U = 1, alpha = 0.01)

  expect_s3_class(p, "ratiod_prior")
  expect_equal(p$distribution, "pc")
  expect_equal(p$U, 1)
  expect_equal(p$alpha, 0.01)
  expect_equal(p$rate, -log(0.01) / 1)
})

test_that("prior_pc validates parameters", {
  expect_error(prior_pc(U = -1), "positive")
  expect_error(prior_pc(alpha = 0), "in \\(0, 1\\)")
  expect_error(prior_pc(alpha = 1), "in \\(0, 1\\)")
  expect_error(prior_pc(alpha = 1.5), "in \\(0, 1\\)")
})

test_that("ratiod_priors() leaves unspecified components to the defaults", {
  p <- ratiod_priors(beta = prior_normal(0, 1))
  expect_s3_class(p, "ratiod_priors")
  expect_equal(p$beta$sd, 1)
  expect_null(p$sigma)

  f <- tulpaRatio:::fill_priors(p, "poisson_gamma")
  expect_equal(f$beta$sd, 1)
  expect_equal(f$sigma$distribution, "half_cauchy")
  expect_equal(f$sigma$scale, 2.5)
  expect_equal(c(f$phi$shape, f$phi$rate), c(2, 0.5))
  expect_equal(c(f$rho_temporal$alpha, f$rho_temporal$beta), c(2, 2))
  expect_equal(c(f$rho_spatial$alpha, f$rho_spatial$beta), c(1, 1))
})

test_that("the dispersion default follows the family", {
  dp <- function(mt) {
    x <- tulpaRatio:::default_prior("phi", mt)
    c(x$shape, x$rate)
  }
  expect_equal(dp("lognormal"), c(2, 2))
  expect_equal(dp("gamma_gamma"), c(2, 0.5))
  expect_equal(dp("negbin_negbin"), c(2, 0.1))
  expect_equal(tulpaRatio:::default_prior("beta")$sd, 10)
})

test_that("sampler_priors() passes the families the samplers carry and refuses others", {
  sp <- tulpaRatio:::sampler_priors(
    ratiod_priors(beta = prior_normal(0, 3), sigma = prior_half_cauchy(1),
                  phi = prior_gamma(3, 1)),
    "negbin_negbin")
  expect_equal(sp[c("sigma_beta", "sigma_re_scale", "phi_shape", "phi_rate")],
               list(sigma_beta = 3, sigma_re_scale = 1, phi_shape = 3, phi_rate = 1))
  expect_error(tulpaRatio:::sampler_priors(ratiod_priors(sigma = prior_pc(1, 0.01)),
                                           "negbin_negbin"),
               "prior_half_cauchy")
  expect_error(tulpaRatio:::sampler_priors(ratiod_priors(beta = prior_normal(1, 2)),
                                           "negbin_negbin"),
               "prior_normal\\(0, sd\\)")
})

test_that("a prior given to tratio() is the prior the HMC fit runs under", {
  skip_on_cran()
  sim <- sim_ratiod(n = 40, family = ratiod_poisson_gamma(), seed = 5)
  fit_prior <- function(priors) {
    tratio(y_num | y_denom ~ x1, data = sim$data,
           family = ratiod_poisson_gamma(), priors = priors,
           control = list(iter = 300, chains = 1, seed = 1))
  }
  b_default <- coef(fit_prior(NULL))
  b_tight <- coef(fit_prior(ratiod_priors(beta = prior_normal(0, 0.01))))
  expect_true(all(abs(b_tight) < 0.05))
  expect_true(any(abs(b_default) > 0.2))
  expect_error(fit_prior(ratiod_priors(sigma = prior_pc(1, 0.01))), "half_cauchy")
})

test_that("ratiod_priors accepts custom priors", {
  p <- ratiod_priors(
    beta = prior_normal(0, 1),
    sigma = prior_half_cauchy(2.5),
    rho_temporal = prior_beta(5, 2)
  )

  expect_equal(p$beta$sd, 1)
  expect_equal(p$sigma$distribution, "half_cauchy")
  expect_equal(p$rho_temporal$alpha, 5)
})

test_that("ratiod_priors rejects invalid priors", {
  expect_error(
    ratiod_priors(beta = "normal(0, 1)"),
    "must be a prior object"
  )

  expect_error(
    ratiod_priors(sigma = list(distribution = "normal")),
    "must be a prior object"
  )
})

test_that("prior print methods work", {
  expect_output(print(prior_normal(0, 2.5)), "Normal")
  expect_output(print(prior_half_cauchy(2.5)), "Half-Cauchy")
  expect_output(print(prior_pc(1, 0.01)), "PC prior")
  expect_output(print(prior_beta(2, 2)), "Beta")
  expect_output(print(ratiod_priors()), "tulpaRatio prior specification")
})

test_that("each prior prints its parameters and calibration exactly", {
  priors <- list(prior_normal(1, 2), prior_half_normal(1.5), prior_half_cauchy(2),
                 prior_gamma(3, 0.5), prior_exponential(2), prior_beta(2, 3),
                 prior_pc(2, 0.05))
  expected <- list(
    "  Normal(1.00, 2.00)", "  Half-Normal(1.50)", "  Half-Cauchy(2.00)",
    "  Gamma(3.00, 0.50)  [mean = 6.00]", "  Exponential(2.00)  [mean = 0.50]",
    "  Beta(2.00, 3.00)  [mean = 0.40]",
    c("  PC prior: P(x > 2.00) = 0.050", "    => Exponential(1.498)"))
  for (i in seq_along(priors)) {
    expect_identical(capture.output(print_prior(priors[[i]], "  ")), expected[[i]])
  }
  expect_identical(capture.output(print_prior(NULL, "  ")), "  default")
})


test_that("priors_default() prints no dispersion prior for a family without one", {
  out <- capture.output(priors_default(ratiod_poisson()))
  expect_true(any(grepl("none (poisson_fixed)", out, fixed = TRUE)))
  out <- capture.output(priors_default(ratiod_negbin()))
  expect_false(any(grepl("none (", out, fixed = TRUE)))
})

test_that("prior_predict refuses a family it cannot simulate from", {
  expect_error(prior_predict(NULL, NULL, NULL), "ratiod_family")
})

test_that("prior_normal creates valid prior", {
  p <- prior_normal(0, 2.5)
  expect_s3_class(p, "ratiod_prior")
  expect_equal(p$dist, "normal")
  expect_equal(p$mean, 0)
  expect_equal(p$sd, 2.5)
})

test_that("prior_half_normal creates valid prior", {
  p <- prior_half_normal(2.5)
  expect_s3_class(p, "ratiod_prior")
  expect_equal(p$dist, "half_normal")
  expect_equal(p$sd, 2.5)
})

test_that("prior_half_cauchy creates valid prior", {
  p <- prior_half_cauchy(2.5)
  expect_s3_class(p, "ratiod_prior")
  expect_equal(p$dist, "half_cauchy")
  expect_equal(p$scale, 2.5)
})

test_that("prior_gamma creates valid prior", {
  p <- prior_gamma(2, 0.5)
  expect_s3_class(p, "ratiod_prior")
  expect_equal(p$dist, "gamma")
  expect_equal(p$shape, 2)
  expect_equal(p$rate, 0.5)
})

test_that("prior_beta creates valid prior", {
  p <- prior_beta(2, 2)
  expect_s3_class(p, "ratiod_prior")
  expect_equal(p$dist, "beta")
  expect_equal(p$alpha, 2)
  expect_equal(p$beta, 2)
})

test_that("prior_exponential creates valid prior", {
  p <- prior_exponential(1)
  expect_s3_class(p, "ratiod_prior")
  expect_equal(p$dist, "exponential")
  expect_equal(p$rate, 1)
})

test_that("prior_pc creates valid prior", {
  p <- prior_pc(U = 1.0, alpha = 0.01)
  expect_s3_class(p, "ratiod_prior")
  expect_equal(p$dist, "pc")
  expect_equal(p$U, 1.0)
  expect_equal(p$alpha, 0.01)
})

test_that("validate_prior rejects non-prior objects", {
  expect_error(
    tulpaRatio:::validate_prior(list(x = 1), "test"),
    "must be a prior object"
  )
})

test_that("ratiod_priors validates all priors", {
  expect_error(
    ratiod_priors(beta = list(x = 1)),
    "must be a prior object"
  )
})

test_that("temporal_ar1 creates valid object with a correlation prior", {
  temp <- temporal_ar1(time_var = "year", rho_prior = prior_beta(9, 1))
  expect_s3_class(temp$rho_prior, "ratiod_prior")
  expect_equal(temp$rho_prior$alpha, 9)
})

test_that("ratiod_priors creates valid prior object", {
  priors <- ratiod_priors()
  expect_s3_class(priors, "ratiod_priors")
})

test_that("ratiod_priors accepts custom values", {
  priors <- ratiod_priors(
    beta = prior_normal(0, 5),
    sigma = prior_pc(U = 1.5),
    phi = prior_gamma(2, 0.5)
  )

  expect_s3_class(priors, "ratiod_priors")
  expect_equal(priors$beta$dist, "normal")
  expect_equal(priors$sigma$dist, "pc")
})

test_that("print.ratiod_priors works", {
  priors <- ratiod_priors()
  output <- capture.output(print(priors))
  expect_true(any(grepl("tulpaRatio prior specification", output)))
})

test_that("temporal_ar1 accepts a Beta prior on its correlation", {
  temp <- temporal_ar1(time_var = "year", rho_prior = prior_beta(5, 2))
  expect_equal(temp$rho_prior$alpha, 5)
  expect_equal(temp$rho_prior$beta, 2)
  # `rho` is not an argument; it used to partial-match rho_prior and store a
  # bare number nothing read.
  expect_error(temporal_ar1(time_var = "year", rho = 0.8), "Beta prior")
})
