# A GP spatial main effect routes the whole model to cpp_hmc_fit_gp_v2. That
# entry used to be handed no spatiotemporal, latent, TVC or SVC bundle, so it
# built a ModelData without those blocks while q_init and hmc_param_layout()
# above it allocated and named their parameters. Every column from the dropped
# block onward was then read at the wrong offset and reported under the wrong
# name (gcol33/tulpaRatio#70).
#
# The arbiter is the prior support. A hyperparameter whose prior is uniform on
# (l, u) has a density of -Inf outside those bounds, so no draw of it can land
# there; a column that does is not that parameter. It reads the fit from the
# outside and needs no knowledge of the layout, which is what the mislabelling
# defeated.

gp_st_data <- function(S = 6L, T_ = 5L, seed = 20260826L) {
  set.seed(seed)
  grid <- expand.grid(unit = seq_len(S), year = seq_len(T_))
  d <- data.frame(
    unit = grid$unit,
    year = grid$year,
    lon  = ((grid$unit - 1L) %% 3L) + 0.0,
    lat  = ((grid$unit - 1L) %/% 3L) + 0.0,
    x    = rnorm(nrow(grid))
  )
  d$denom <- rpois(nrow(d), 60) + 5L
  d$num <- rpois(nrow(d), 0.3 * d$denom)
  d
}

# Bounds every fit below is checked against: the uniform ranges the densities
# refuse outside of. spatiotemporal()'s two GP ranges default to (0.01, 10) and
# spatial_gp()'s to (0.01, 100); see ratiod_priors().
SUPPORT_BOUNDS <- list(
  phi_st_space = c(0.01, 10),
  phi_st_time  = c(0.01, 10),
  phi_gp       = c(0.01, 100)
)

expect_draws_in_support <- function(fit) {
  dr <- as.matrix(fit$draws)
  checked <- 0L
  for (nm in names(SUPPORT_BOUNDS)) {
    b <- SUPPORT_BOUNDS[[nm]]
    for (col in grep(paste0("^", nm), colnames(dr), value = TRUE)) {
      v <- dr[, col]
      checked <- checked + 1L
      expect_true(
        all(v >= b[1] & v <= b[2]),
        label = sprintf("%s in (%g, %g): min %.4g, max %.4g",
                        col, b[1], b[2], min(v), max(v))
      )
    }
  }
  expect_gt(checked, 0L)
  invisible(dr)
}


test_that("a GP main effect carries a GP interaction, and its draws stay in support", {
  skip_on_cran()
  d <- gp_st_data()
  fit <- tratio(
    num | denom ~ x,
    data = d,
    family = ratiod_poisson_gamma(),
    mode = "hmc",
    spatial = spatial_gp(~ lon + lat, nn = 4),
    spatiotemporal = spatiotemporal(
      spatial = spatial_gp(~ lon + lat),
      temporal = temporal_rw1("year"),
      type = "separable",
      coords = ~ lon + lat,
      nn = 4
    ),
    control = list(iter = 200, warmup = 100, chains = 1, seed = 1,
                   verbose = FALSE)
  )
  dr <- expect_draws_in_support(fit)
  # The interaction's own hyperparameters are reported, which is what says the
  # block reached the sampler rather than being named over someone else's
  # columns.
  expect_true(all(c("tau_st", "phi_st_space", "phi_st_time") %in% colnames(dr)))
})


test_that("a GP main effect carries a Knorr-Held interaction", {
  skip_on_cran()
  d <- gp_st_data()
  adj <- matrix(0, 6, 6)
  for (i in 1:5) adj[i, i + 1L] <- adj[i + 1L, i] <- 1
  fit <- tratio(
    num | denom ~ x,
    data = d,
    family = ratiod_poisson_gamma(),
    mode = "hmc",
    spatial = spatial_gp(~ lon + lat, nn = 4),
    spatiotemporal = spatiotemporal(
      spatial = spatial_car(adj, level = "group", group_var = "unit"),
      temporal = temporal_rw1("year"),
      type = "IV"
    ),
    control = list(iter = 200, warmup = 100, chains = 1, seed = 2,
                   verbose = FALSE)
  )
  dr <- expect_draws_in_support(fit)
  expect_true("tau_st" %in% colnames(dr))
})


test_that("a GP main effect carries a latent factor", {
  skip_on_cran()
  d <- gp_st_data()
  fit <- tratio(
    num | denom ~ x,
    data = d,
    family = ratiod_poisson_gamma(),
    mode = "hmc",
    spatial = spatial_gp(~ lon + lat, nn = 4),
    latent = latent_factor(n_factors = 1),
    control = list(iter = 200, warmup = 100, chains = 1, seed = 3,
                   verbose = FALSE)
  )
  dr <- expect_draws_in_support(fit)
  expect_true(any(grepl("^sigma_latent|^latent", colnames(dr))))
})


# The same entry point declared its random-effect block single-term and
# slope-free (`data.n_re_terms = 0`, `has_re_slopes = false`) while
# prepare_hmc_data() and hmc_param_layout() above it parsed and named the slope
# and multi-term structure. The two layouts then disagreed on the size of the
# RE block and every column after it was read at the wrong offset
# (gcol33/tulpaRatio#72). Both bundles now go through the builder both entry
# points call.

test_that("a GP main effect carries a correlated random slope", {
  skip_on_cran()
  d <- gp_st_data()
  fit <- tratio(
    num | denom ~ x + (1 + x | unit),
    data = d,
    family = ratiod_poisson_gamma(),
    mode = "hmc",
    spatial = spatial_gp(~ lon + lat, nn = 3),
    control = list(iter = 200, warmup = 100, chains = 1, seed = 4,
                   verbose = FALSE)
  )
  dr <- expect_draws_in_support(fit)
  # The slope block reached the sampler: its own scale, its correlation and
  # a per-group slope are all reported.
  expect_true(any(colnames(dr) == "sigma_re[1,x]"))
  expect_true(any(startsWith(colnames(dr), "cor_re[")))
  expect_true(any(colnames(dr) == "re[1,1,x]"))
})


test_that("a GP main effect carries an uncorrelated random slope", {
  skip_on_cran()
  d <- gp_st_data()
  fit <- tratio(
    num | denom ~ x + (1 + x || unit),
    data = d,
    family = ratiod_poisson_gamma(),
    mode = "hmc",
    spatial = spatial_gp(~ lon + lat, nn = 3),
    control = list(iter = 200, warmup = 100, chains = 1, seed = 5,
                   verbose = FALSE)
  )
  dr <- expect_draws_in_support(fit)
  expect_true(any(colnames(dr) == "sigma_re[1,x]"))
  expect_false(any(startsWith(colnames(dr), "cor_re[")))
})


test_that("a GP main effect carries crossed random-effect terms", {
  skip_on_cran()
  d <- gp_st_data()
  d$block <- rep(1:2, length.out = nrow(d))
  fit <- tratio(
    num | denom ~ x + (1 | unit) + (1 | block),
    data = d,
    family = ratiod_poisson_gamma(),
    mode = "hmc",
    spatial = spatial_gp(~ lon + lat, nn = 3),
    control = list(iter = 200, warmup = 100, chains = 1, seed = 6,
                   verbose = FALSE)
  )
  dr <- expect_draws_in_support(fit)
  # Both terms' scales are reported, which is what says the second block was
  # allocated rather than named over the first one's columns.
  expect_gte(length(grep("^sigma_re", colnames(dr))), 2L)
})


# The temporal bundle reached the same entry with rho_prior_a / rho_prior_b on
# it, and the entry's own extraction stopped short of them: the values crossed
# the .Call boundary and the fit ran on ModelData's defaults
# (gcol33/tulpaRatio#73). Two priors have to give two posteriors, on both entry
# points -- the areal one is the control, and it moved before the fix.

test_that("the AR1 rho prior reaches both sampler entry points", {
  skip_on_cran()
  d <- gp_st_data(S = 6L, T_ = 10L)
  adj <- matrix(0, 6, 6)
  for (i in 1:5) adj[i, i + 1L] <- adj[i + 1L, i] <- 1

  fit_rho <- function(spatial_arg, prior) {
    tratio(
      num | denom ~ x,
      data = d,
      family = ratiod_poisson_gamma(),
      mode = "hmc",
      spatial = spatial_arg,
      temporal = temporal_ar1("year"),
      priors = ratiod_priors(rho_temporal = prior),
      control = list(iter = 200, warmup = 100, chains = 1, seed = 7,
                     verbose = FALSE)
    )
  }
  rho_draws <- function(fit) {
    dr <- as.matrix(fit$draws)
    cn <- grep("^rho_ar1", colnames(dr), value = TRUE)
    expect_gt(length(cn), 0L)
    dr[, cn[1], drop = TRUE]
  }

  arms <- list(
    gp = spatial_gp(~ lon + lat, nn = 4),
    areal = spatial_car(adj, level = "group", group_var = "unit")
  )
  for (nm in names(arms)) {
    flat <- rho_draws(fit_rho(arms[[nm]], prior_beta(2, 2)))
    tight <- rho_draws(fit_rho(arms[[nm]], prior_beta(50, 1)))
    expect_false(identical(flat, tight), label = sprintf("%s draws differ", nm))
    # Beta(50, 1) on u = (rho + 1) / 2 puts its mass at rho near 1.
    expect_gt(mean(tight), mean(flat) + 0.5)
  }
})
