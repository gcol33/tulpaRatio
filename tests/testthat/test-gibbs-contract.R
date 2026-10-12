# The Gibbs spatial backend must return the same fit contract as every other
# backend: the number of chains that was requested, a 2-D draws matrix carrying
# parameter names, and a per-chain samples list. Without those, split-Rhat is
# not computable and a spatial fit cannot be checked for convergence at all.

gibbs_test_data <- function(n = 300, n_sites = 25) {
  set.seed(123)
  x <- rnorm(n)
  site <- factor(rep(seq_len(n_sites), length.out = n))
  n_side <- ceiling(sqrt(n_sites))
  grid <- expand.grid(lon = seq_len(n_side), lat = seq_len(n_side))[seq_len(n_sites), ]
  adj <- matrix(0, n_sites, n_sites)
  for (i in seq_len(n_sites)) for (j in seq_len(n_sites)) {
    d <- sqrt((grid$lon[i] - grid$lon[j])^2 + (grid$lat[i] - grid$lat[j])^2)
    if (i != j && d <= 1.5) adj[i, j] <- 1
  }
  dimnames(adj) <- list(levels(site), levels(site))
  trials <- sample(10:50, n, replace = TRUE)
  y <- rbinom(n, trials, plogis(0.5 + 0.3 * x))
  list(df = data.frame(y = y, trials = trials, x = x, site = site,
                       spatial_site = site),
       adj = adj)
}

gibbs_fit <- function(d, chains = 4, seed = 99, spatial = NULL) {
  if (is.null(spatial)) {
    spatial <- spatial_car(d$adj, level = "group", group_var = "spatial_site")
  }
  tratio(y | trials ~ x + (1 | site), data = d$df,
         family = ratiod_binomial(), spatial = spatial,
         control = list(iter = 600, warmup = 300, chains = chains, seed = seed, verbose = FALSE))
}

test_that("Gibbs backend honours the requested number of chains", {
  skip_on_cran()
  d <- gibbs_test_data()
  fit <- gibbs_fit(d, chains = 4)

  expect_identical(fit$backend, "gibbs")
  expect_identical(as.integer(fit$chains), 4L)
  expect_length(fit$samples, 4L)
  expect_identical(nrow(as.matrix(fit$draws)),
                   as.integer(fit$n_save_per_chain * 4L))
})

test_that("Gibbs draws are a named 2-D matrix including the spatial field", {
  skip_on_cran()
  d <- gibbs_test_data()
  dr <- as.matrix(gibbs_fit(d, chains = 2)$draws)

  expect_length(dim(dr), 2L)
  expect_false(is.null(colnames(dr)))
  expect_true(all(c("beta_num[1]", "beta_num[2]", "tau_spatial") %in% colnames(dr)))
  # The spatial field is part of the posterior, not a side channel, and is
  # named as every other backend names it.
  expect_true(any(grepl("^phi_spatial\\[", colnames(dr))))
})

test_that("split-Rhat is computable on a Gibbs fit and the chains agree", {
  skip_on_cran()
  skip_if_not_installed("posterior")
  d <- gibbs_test_data()
  fit <- gibbs_fit(d, chains = 4)

  dr <- as.matrix(fit$draws)
  per <- nrow(dr) / fit$chains
  for (nm in c("beta_num[1]", "beta_num[2]")) {
    m <- matrix(dr[, nm], nrow = per, ncol = fit$chains)
    expect_lt(posterior::rhat(m), 1.05)
    expect_gt(posterior::ess_bulk(m), 100)
  }
})

test_that("Gibbs chains are independent but the fit is seed-reproducible", {
  skip_on_cran()
  d <- gibbs_test_data()
  fit <- gibbs_fit(d, chains = 3, seed = 7)

  # Distinct seeds per chain: no two chains may be the same draws.
  expect_false(identical(fit$samples[[1]], fit$samples[[2]]))
  expect_false(identical(fit$samples[[2]], fit$samples[[3]]))

  again <- gibbs_fit(d, chains = 3, seed = 7)
  expect_identical(as.matrix(fit$draws), as.matrix(again$draws))

  other <- gibbs_fit(d, chains = 3, seed = 8)
  expect_false(identical(as.matrix(fit$draws), as.matrix(other$draws)))
})

test_that("Gibbs recovers the fixed effect under a spatial field", {
  skip_on_cran()
  d <- gibbs_test_data()
  dr <- as.matrix(gibbs_fit(d, chains = 4)$draws)

  ci <- unname(quantile(dr[, "beta_num[2]"], c(0.025, 0.975)))
  expect_lt(ci[1], 0.3)
  expect_gt(ci[2], 0.3)
})

# An ICAR precision satisfies Q1 = 0, and under BYM2 sigma_u * mean(theta) is a
# level the intercept carries just as well. Both directions are invisible to the
# likelihood, so a sampler that leaves them to the intercept's own updates mixes
# the intercept far worse than the slope at the same problem size. The field here
# is strong enough for that gap to be measurable; at the milder field the earlier
# tests use, the intercept clears any threshold either way.
gibbs_field_data <- function(n = 400, n_sites = 25, amp = 3.0, seed = 2024) {
  set.seed(seed)
  n_side <- ceiling(sqrt(n_sites))
  grid <- expand.grid(lon = seq_len(n_side), lat = seq_len(n_side))[seq_len(n_sites), ]
  adj <- matrix(0, n_sites, n_sites)
  for (i in seq_len(n_sites)) for (j in seq_len(n_sites)) {
    d <- sqrt((grid$lon[i] - grid$lon[j])^2 + (grid$lat[i] - grid$lat[j])^2)
    if (i != j && d <= 1.5) adj[i, j] <- 1
  }
  site <- factor(rep(seq_len(n_sites), length.out = n))
  dimnames(adj) <- list(levels(site), levels(site))

  field <- amp * scale(grid$lon + grid$lat)[, 1]
  field <- field - mean(field)
  x <- rnorm(n)
  trials <- sample(10:50, n, replace = TRUE)
  y <- rbinom(n, trials, plogis(0.5 + 0.3 * x + field[as.integer(site)]))
  list(df = data.frame(y = y, trials = trials, x = x, site = site), adj = adj)
}

test_that("the intercept mixes under a spatial field", {
  skip_on_cran()
  skip_if_not_installed("posterior")
  d <- gibbs_field_data()

  intercept_ess <- function(spatial, iter, warmup) {
    fit <- tratio(y | trials ~ x, data = d$df, family = ratiod_binomial(),
                  spatial = spatial,
                  control = list(iter = iter, warmup = warmup, chains = 4, seed = 99, verbose = FALSE))
    dr <- as.matrix(fit$draws)
    posterior::ess_bulk(matrix(dr[, "beta_num[1]"],
                               nrow = nrow(dr) / fit$chains, ncol = fit$chains))
  }

  # Measured on this fit: 76 before the field was identified against the
  # intercept, 212 after (#13).
  expect_gt(intercept_ess(spatial_car(d$adj, level = "group", group_var = "site"), 1000, 500), 120)
})

test_that("a single-chain fit keeps the contract", {
  skip_on_cran()
  d <- gibbs_test_data()

  one <- gibbs_fit(d, chains = 1)
  expect_identical(as.integer(one$chains), 1L)
  expect_false(is.null(colnames(as.matrix(one$draws))))
})

test_that("Gibbs agrees with HMC on an ICAR model", {
  skip_on_cran()
  skip_if_not_installed("posterior")
  d <- gibbs_field_data()
  car <- spatial_car(d$adj, level = "group", group_var = "site")
  fit_mode <- function(mode) {
    tratio(y | trials ~ x, data = d$df, family = ratiod_binomial(),
           spatial = car, mode = mode,
           control = list(iter = 2000, warmup = 1000, chains = 4, seed = 99))
  }
  g <- as.matrix(fit_mode("gibbs")$draws)
  h <- as.matrix(fit_mode("hmc")$draws)
  for (nm in c("beta_num[1]", "beta_num[2]", "tau_spatial")) {
    expect_lt(abs(mean(g[, nm]) - mean(h[, nm])) / sd(h[, nm]), 0.25, label = nm)
  }
})

test_that("Gibbs agrees with HMC on a BYM2 model, and auto mode picks it", {
  # The BYM2 field moves as one block with its fixed effects and
  # hyperparameters. The component-wise sweep it replaced settled sigma near
  # 0.9 where HMC's is near 8, with the intercept's R-hat at 1.6 after 40000
  # iterations (#133).
  skip_on_cran()
  d <- gibbs_field_data()
  bym <- spatial_bym2(d$adj, level = "group", group_var = "site")
  fit_mode <- function(mode) {
    tratio(y | trials ~ x, data = d$df, family = ratiod_binomial(),
           spatial = bym, mode = mode,
           control = list(iter = 2000, warmup = 1000, chains = 4, seed = 99))
  }
  g <- as.matrix(fit_mode("gibbs")$draws)
  h <- as.matrix(fit_mode("hmc")$draws)
  for (nm in c("beta_num[1]", "beta_num[2]", "sigma_spatial", "rho_spatial")) {
    expect_lt(abs(mean(g[, nm]) - mean(h[, nm])) / sd(h[, nm]), 0.25, label = nm)
  }

  selection <- tulpaRatio:::select_inference_mode(
    mode = "auto", family = ratiod_binomial(), n_obs = nrow(d$df),
    structures = list(spatial = bym)
  )
  expect_equal(selection$backend, "gibbs")
})
