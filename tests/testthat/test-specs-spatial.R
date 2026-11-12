fit_one_spatial <- function(sim, seed_val, spatial_struct) {
  fit <- tulpaRatio::tratio(
    formula = sim$formula, data = sim$data, family = sim$family,
    spatial = spatial_struct,
    mode = "hmc",
    control = list(iter = 3000L, warmup = 1000L, chains = 1L, seed = seed_val, verbose = FALSE, gradient_mode = "A_r")
  )
  fit
}

for (type in c("icar", "bym2")) local({
  field <- type
  test_that(paste("engine recovers group-level", field, "effects"), {
    skip_on_cran()
    sim <- simulate_spatial_binomial()
    constructor <- if (field == "icar") spatial_car else spatial_bym2
    sp <- constructor(sim$adj, spatial_level = "group", group_var = "unit",
      parameterization = "standard")
    fit <- fit_one_spatial(sim, 42L, sp)
    expect_lt(abs(mean(fit$draws[, "beta_num[2]"]) - .6), .2)
    expect_lt(sqrt(mean((colMeans(fitted(fit, summary = FALSE)$numerator) - sim$truth)^2)), .1)
  })
})
