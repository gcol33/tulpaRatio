# test-specs-spatial.R
# B1d Step 2 parity test: ICAR + BYM2 (non-collapsed, group-level) routed
# through the LikelihoodSpec path produce posterior means within 4x within-MC
# noise of the legacy backend at the same seed.

fit_one_spatial <- function(sim, use_specs, seed_val, spatial_struct) {
  op <- options(tulpaRatio.use_specs = use_specs); on.exit(options(op), add = TRUE)
  fit <- tulpaRatio::tratio(
    formula = sim$formula, data = sim$data, family = sim$family,
    spatial = spatial_struct,
    mode = "hmc",
    control = list(iter = 3000L, warmup = 1000L, chains = 1L, seed = seed_val, verbose = FALSE, gradient_mode = "A_r")
  )
  colMeans(fit$draws)
}

test_that("B1d spec path matches legacy for ICAR (group-level)", {
  skip_on_cran()
  sim <- simulate_spatial_binomial()
  sp  <- tulpaRatio::spatial_car(adjacency = sim$adj, level = "group",
                                  group_var = "unit",
                                  parameterization = "standard")
  legacy42 <- fit_one_spatial(sim, FALSE, 42L, sp)
  specs42  <- fit_one_spatial(sim, TRUE,  42L, sp)
  legacy43 <- fit_one_spatial(sim, FALSE, 43L, sp)

  expect_named(specs42, names(legacy42))
  cross  <- max(abs(legacy42 - specs42))
  within <- max(abs(legacy42 - legacy43))
  expect_lt(cross, max(4 * within, 5e-3))
})

test_that("B1d spec path matches legacy for BYM2 (group-level)", {
  skip_on_cran()
  sim <- simulate_spatial_binomial()
  sp  <- tulpaRatio::spatial_bym2(adjacency = sim$adj, level = "group",
                                   group_var = "unit",
                                   parameterization = "standard")
  legacy42 <- fit_one_spatial(sim, FALSE, 42L, sp)
  specs42  <- fit_one_spatial(sim, TRUE,  42L, sp)
  legacy43 <- fit_one_spatial(sim, FALSE, 43L, sp)

  expect_named(specs42, names(legacy42))
  cross  <- max(abs(legacy42 - specs42))
  within <- max(abs(legacy42 - legacy43))
  expect_lt(cross, max(4 * within, 5e-3))
})
