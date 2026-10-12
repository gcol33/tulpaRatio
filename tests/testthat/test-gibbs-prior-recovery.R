# With zero binomial trials the log-likelihood is 0*log(p) + 0*log(1-p) = 0 at
# every eta, so the posterior equals the prior exactly. Anything the Gibbs
# spatial sampler gets wrong in a hyperparameter's prior density or in a move's
# Jacobian shows up here as a sampled marginal that does not match the prior it
# was given -- this is what caught the BYM2 intercept variance inflating
# ~2000x under a tight prior in #15 (an unconditional, untested transfer of
# the field's level into the intercept).

gibbs_recovery_setup <- function(n_sites = 25, n = 200) {
  grid <- expand.grid(lon = 1:5, lat = 1:5)
  adj <- matrix(0, n_sites, n_sites)
  for (i in seq_len(n_sites)) for (j in seq_len(n_sites)) {
    dd <- sqrt((grid$lon[i] - grid$lon[j])^2 + (grid$lat[i] - grid$lat[j])^2)
    if (i != j && dd <= 1.5) adj[i, j] <- 1
  }
  site <- factor(rep(seq_len(n_sites), length.out = n))
  dimnames(adj) <- list(levels(site), levels(site))
  df <- data.frame(y = rep(0L, n), trials = rep(0L, n), x = rnorm(n), site = site)
  fspec <- tulpaRatio:::ratiod_formula(formula = y | trials ~ x, data = df)
  list(df = df, adj = adj, fspec = fspec)
}

gibbs_recovery_fit <- function(setup, spatial, priors, iter = 4000, warmup = 2000,
                               chains = 4, seed = 11) {
  fit <- tulpaRatio:::fit_gibbs(
    formula = setup$fspec, data = setup$df, family = ratiod_binomial(),
    spatial = spatial, priors = priors,
    iter = iter, warmup = warmup, chains = chains, seed = seed, verbose = FALSE)
  as.matrix(fit$draws)
}

test_that("ICAR tau recovers its Gamma(shape, rate) prior with zero-trials data", {
  skip_on_cran()
  s <- gibbs_recovery_setup()
  car <- spatial_car(s$adj, level = "group", group_var = "site")
  dr <- gibbs_recovery_fit(s, car, list())

  qs <- quantile(dr[, "tau_spatial"], c(0.25, 0.5, 0.75))
  target <- qgamma(c(0.25, 0.5, 0.75), shape = 1.0, rate = 0.01)
  expect_equal(unname(qs), target, tolerance = 0.25)
})

test_that("the intercept's posterior sd matches its own prior under ICAR", {
  # A tight prior makes any uncorrected drift from the spatial field into the
  # intercept visible immediately: the drift the field injects per sweep does
  # not shrink when the prior does, so an sd ratio far from 1 here is a wrong
  # Jacobian or a missing acceptance test on some move that touches the
  # intercept, not sampling noise.
  skip_on_cran()
  s <- gibbs_recovery_setup()
  car <- spatial_car(s$adj, level = "group", group_var = "site")
  dr <- gibbs_recovery_fit(s, car, ratiod_priors(beta = prior_normal(0, 0.5)))
  expect_equal(sd(dr[, "beta_num[1]"]), 0.5, tolerance = 0.3)
})

test_that("BYM2 sigma, rho and the intercept recover their priors with zero-trials data", {
  # The one-block update proposes the fixed effects, both field components
  # and the two hyperparameters together; a wrong proposal density or
  # constraint correction anywhere in it moves these marginals off their
  # priors.
  skip_on_cran()
  s <- gibbs_recovery_setup()
  bym <- spatial_bym2(s$adj, level = "group", group_var = "site")
  priors <- ratiod_priors(beta = prior_normal(0, 0.5))
  dr <- gibbs_recovery_fit(s, bym, priors)
  scale <- tulpaRatio:::sampler_priors(priors, "binomial")$sigma_re_scale
  p <- c(0.25, 0.5, 0.75)
  expect_equal(unname(quantile(dr[, "sigma_spatial"], p)), scale * tan(pi * p / 2),
               tolerance = 0.25)
  expect_equal(unname(quantile(dr[, "rho_spatial"], p)), p, tolerance = 0.2)
  expect_equal(sd(dr[, "beta_num[1]"]), 0.5, tolerance = 0.3)
})
