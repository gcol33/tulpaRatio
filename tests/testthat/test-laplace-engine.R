# The Laplace tier reads the engine's latent vector through per-block
# transforms. Each transform is pinned to the engine's own per-cell linear
# predictor, so a change in the engine's latent scale fails here rather than
# shifting every stored effect.

laplace_chain_adjacency <- function(S) {
  adj <- matrix(0L, S, S)
  for (i in seq_len(S - 1L)) {
    adj[i, i + 1L] <- 1L
    adj[i + 1L, i] <- 1L
  }
  adj
}

laplace_binomial_fixture <- function(seed = 3) {
  set.seed(seed)
  S <- 8L
  reps <- 6L
  n <- S * reps
  site <- rep(seq_len(S), each = reps)
  data.frame(
    s = rbinom(n, 12L, plogis(0.2 + cumsum(rnorm(S, 0, 0.3))[site])),
    n = 12L,
    x = rnorm(n),
    site = factor(site),
    yr = rep(seq_len(6L), times = S),
    lon = runif(n),
    lat = runif(n)
  )
}

expect_cell_eta_matches_engine <- function(spatial = NULL, temporal = NULL,
                                           formula = s | n ~ x) {
  df <- laplace_binomial_fixture()
  f <- ratiod_formula(formula, data = df)
  model <- tulpaRatio:::laplace_model(f, df, ratiod_binomial(), spatial, temporal)
  engine <- tulpaRatio:::laplace_engine(model, priors = NULL, cores = 1L)

  cell <- which.max(engine$weights)
  rebuilt <- tulpaRatio:::laplace_cell_eta(engine, model, cell)
  expect_equal(rebuilt, as.numeric(engine$fitted_eta[cell, ]), tolerance = 1e-8)
}

test_that("a structure-free model is its fixed effects alone", {
  df <- laplace_binomial_fixture()
  f <- ratiod_formula(s | n ~ x, data = df)
  model <- tulpaRatio:::laplace_model(f, df, ratiod_binomial(), NULL, NULL)
  expect_length(model$parts, 0L)
  expect_cell_eta_matches_engine()
})

test_that("iid blocks enter the predictor as sigma times the latent coordinate", {
  expect_cell_eta_matches_engine(formula = s | n ~ x + (1 | site))
  expect_cell_eta_matches_engine(formula = s | n ~ x + (1 | site) + (1 | yr))
})

test_that("an ICAR block enters the predictor as its latent coordinate", {
  expect_cell_eta_matches_engine(
    spatial = spatial_car(adjacency = laplace_chain_adjacency(8L), group_var = "site")
  )
})

test_that("a BYM2 block enters the predictor as sigma (sqrt(rho) a + sqrt(1 - rho) b)", {
  expect_cell_eta_matches_engine(
    spatial = spatial_bym2(adjacency = laplace_chain_adjacency(8L), group_var = "site")
  )
})

test_that("a proper CAR block enters the predictor as its latent coordinate", {
  expect_cell_eta_matches_engine(
    spatial = spatial_car(adjacency = laplace_chain_adjacency(8L),
                          group_var = "site", proper = TRUE)
  )
})

test_that("an HSGP block enters the predictor through sqrt(S_j) times the coefficient", {
  expect_cell_eta_matches_engine(spatial = spatial_hsgp(~ lon + lat, m = 4))
  expect_cell_eta_matches_engine(spatial = spatial_gp(~ lon + lat))
})

test_that("a restricted field enters the predictor through its projector", {
  adj <- laplace_chain_adjacency(8L)
  expect_cell_eta_matches_engine(
    spatial = spatial_rsr(spatial_car(adjacency = adj, group_var = "site"), ~ x)
  )
  expect_cell_eta_matches_engine(
    spatial = spatial_rsr(spatial_bym2(adjacency = adj, group_var = "site"), ~ x)
  )
  expect_cell_eta_matches_engine(
    spatial = spatial_rsr(spatial_hsgp(~ lon + lat, m = 4), ~ x)
  )
})

test_that("a restricted field is orthogonal to the restricted covariates", {
  skip_on_cran()
  df <- laplace_binomial_fixture()
  fit <- tratio(
    s | n ~ x, data = df, family = ratiod_binomial(),
    spatial = spatial_rsr(spatial_car(adjacency = laplace_chain_adjacency(8L),
                                      group_var = "site"), ~ x),
    mode = "laplace", control = list(verbose = FALSE)
  )
  part <- fit$.internal$layout$parts[[1L]]
  effect <- fit$.internal$samples[, part$cols, drop = FALSE]
  expect_equal(ncol(effect), nrow(df))
  X <- cbind(1, df$x)
  expect_lt(max(abs(effect %*% X)), 1e-8)
})

test_that("the trend, seasonal and short-term blocks enter as their coordinates", {
  expect_cell_eta_matches_engine(
    temporal = temporal_multiscale("yr", trend = "rw1", seasonal = 3,
                                   short_term = "iid")
  )
})
