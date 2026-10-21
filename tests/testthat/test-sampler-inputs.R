# Every sampler on the HMC parameter vector builds its model from
# sampler_inputs(), so a structure either reaches the sampler or is refused.

temporal_binomial_data <- function() {
  set.seed(3)
  n_times <- 10L
  reps <- 8L
  time <- rep(seq_len(n_times), each = reps)
  x1 <- rnorm(length(time))
  nt <- sample(10:30, length(time), replace = TRUE)
  y <- rbinom(length(time), nt, plogis(-0.2 + 0.4 * x1 + sin(time / 2) * 0.8))
  data.frame(y = y, n_trials = nt, x1 = x1, time = time,
             g1 = factor(time %% 3), g2 = factor(time %% 4))
}

test_that("ESS, SGHMC and VI carry iid and GP temporal fields", {
  skip_on_cran()
  d <- temporal_binomial_data()
  ctrl <- list(iter = 60, warmup = 30, chains = 1, verbose = FALSE, seed = 1)
  for (m in c("ess", "sghmc", "vi")) {
    for (spec in list(temporal_iid("time"), temporal_gp("time"))) {
      fit <- suppressWarnings(tratio(y | n_trials ~ x1, data = d,
                                     family = ratiod_binomial(),
                                     temporal = spec, mode = m, control = ctrl))
      v <- posterior::variables(posterior::as_draws(fit$draws))
      expect_gte(sum(grepl("temporal", v)), 10L, label = paste(m, class(spec)[1]))
    }
  }
})

test_that("ESS, SGHMC, SGLD and VI refuse what their parameter vector lacks", {
  d <- temporal_binomial_data()
  d$lon <- runif(nrow(d))
  d$lat <- runif(nrow(d))
  for (b in c("ess", "sghmc", "sgld", "vi")) {
    expect_match(
      tulpaRatio:::unsupported_structures(
        b, list(temporal = temporal_multiscale("time")))[["temporal"]],
      "multi-scale")
    expect_match(
      tulpaRatio:::unsupported_structures(
        b, list(spatial = spatial_gp(~ lon + lat)))[["spatial"]],
      "Gaussian-process")
  }
  expect_error(
    tratio(y | n_trials ~ x1 + (1 | g1) + (1 | g2), data = d,
           family = ratiod_binomial(), mode = "vi",
           control = list(verbose = FALSE)),
    "one random intercept")
})

test_that("every backend reads the same parameter bundles as HMC", {
  d <- temporal_binomial_data()
  pri <- ratiod_priors(rho_temporal = prior_beta(3, 5))
  args <- list(tulpaRatio:::ratiod_formula(y | n_trials ~ x1, data = d),
               d, ratiod_binomial(), temporal = temporal_ar1("time"),
               priors = pri)
  hmc <- do.call(tulpaRatio:::sampler_inputs, c(args, re_param = "noncentered"))
  other <- do.call(tulpaRatio:::sampler_inputs, c(args, re_param = "centered"))
  b1 <- hmc$bundles
  b2 <- other$bundles
  b1$re_params$parameterization <- b2$re_params$parameterization <- NULL
  expect_identical(b1, b2)

  default <- tulpaRatio:::sampler_inputs(args[[1]], d, ratiod_binomial(),
                                         temporal = temporal_ar1("time"))
  expect_false(identical(b2$temporal_params[c("rho_prior_a", "rho_prior_b")],
                         default$bundles$temporal_params[c("rho_prior_a", "rho_prior_b")]))
})
