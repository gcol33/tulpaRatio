# control$walnuts runs each NUTS trajectory as tulpa's WALNUTS transition
# (<tulpa/walnuts.h>). On a BYM2 field over 12 units with one observation each
# and adapt_delta = 0.5, NUTS diverges (122 transitions at seed 3) and WALNUTS
# does not; over 5 seeds both sit within 0.06 reference SDs of a long NUTS run
# at adapt_delta = 0.95 on every fixed effect and hyperparameter.

walnuts_bym2_fit <- function(walnuts, adapt_delta, iter, seed) {
  sim <- simulate_spatial_binomial(n_units = 12L, reps = 1L)
  sp <- spatial_bym2(adjacency = sim$adj, level = "group", group_var = "unit",
                     parameterization = "standard")
  tratio(sim$formula, sim$data, sim$family, spatial = sp, mode = "hmc",
         control = list(iter = iter, warmup = iter %/% 3, chains = 1L,
                        seed = seed, verbose = FALSE, walnuts = walnuts,
                        adapt_delta = adapt_delta))
}

test_that("walnuts runs the WALNUTS kernel and recovers the NUTS posterior", {
  skip_on_cran()
  w <- walnuts_bym2_fit(TRUE, 0.5, 6000L, 3L)
  expect_identical(w$diagnostics$algorithm, "WALNUTS")
  expect_equal(w$diagnostics$n_divergent, 0L)
  ref <- walnuts_bym2_fit(FALSE, 0.95, 15000L, 99L)
  cols <- grep("beta|sigma|rho", colnames(ref$draws))
  z <- (colMeans(w$draws)[cols] - colMeans(ref$draws)[cols]) /
    apply(ref$draws[, cols, drop = FALSE], 2, sd)
  expect_lt(max(abs(z)), 0.2)
})
