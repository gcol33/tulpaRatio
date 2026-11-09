# The Laplace tier reads the engine's latent vector through per-block
# transforms. Each transform is pinned to the engine's own per-cell linear
# predictor, so a change in the engine's latent scale fails here rather than
# shifting every stored effect.

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
    spatial = spatial_car(adjacency = build_chain_adjacency(8L), group_var = "site")
  )
})

test_that("a BYM2 block enters the predictor as sigma (sqrt(rho) a + sqrt(1 - rho) b)", {
  expect_cell_eta_matches_engine(
    spatial = spatial_bym2(adjacency = build_chain_adjacency(8L), group_var = "site")
  )
})

test_that("a proper CAR block enters the predictor as its latent coordinate", {
  expect_cell_eta_matches_engine(
    spatial = spatial_car(adjacency = build_chain_adjacency(8L),
                          group_var = "site", proper = TRUE)
  )
})

test_that("an HSGP block enters the predictor through sqrt(S_j) times the coefficient", {
  expect_cell_eta_matches_engine(spatial = spatial_hsgp(~ lon + lat, m = 4))
  expect_cell_eta_matches_engine(spatial = spatial_gp(~ lon + lat))
})

test_that("a restricted field enters the predictor through its projector", {
  adj <- build_chain_adjacency(8L)
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
    spatial = spatial_rsr(spatial_car(adjacency = build_chain_adjacency(8L),
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

test_that('HSGP Laplace cells match the exact squared-exponential Gaussian posterior', {
  skip_on_cran()
  df <- expand.grid(lon = seq(-1, 1, length.out = 6),
                    lat = seq(-.8, .8, length.out = 5))
  X <- cbind(1, df$lon)
  y <- sin(2 * df$lon) + cos(3 * df$lat)
  noise <- .3
  responses <- list(num = list(y = y, X = X, family = 'gaussian', phi = noise))
  for (widths in list(4L, 6L, c(4L, 4L))) {
    bases <- lapply(widths, function(m)
      tulpaRatio:::hsgp_basis_2d(as.matrix(df), m = m, c_boundary = 1.5))
    parts <- lapply(seq_along(bases), function(b) {
      part <- tulpaRatio:::laplace_hsgp_part(
        b, 'gp', 'w_gp', bases[[b]], seq_len(nrow(df)), 1L)
      part$block$sigma2_grid <- c(.3, 1.2) / b
      part$block$lengthscale_grid <- c(.2, 1.1) / b
      part
    })
    fit <- tulpa::tulpa_nested_laplace_joint(
      responses = responses, prior = lapply(parts, `[[`, 'block'),
      control = list(integration = 'grid', axis_refine = 'none', prune = FALSE,
                     progress = FALSE, n_threads = 1L, tol = 1e-10))
    model <- list(parts = parts, arms = list(responses = responses))
    for (cell in seq_len(nrow(fit$theta_grid))) {
      loadings <- lapply(seq_along(bases), function(b) {
        sigma2 <- fit$theta_grid[cell, paste0('b', b, '.sigma2')]
        ell <- fit$theta_grid[cell, paste0('b', b, '.lengthscale')]
        basis <- bases[[b]]
        spectrum <- 2 * pi * sigma2 * ell^2 * exp(-ell^2 * basis$eigenvalues / 2)
        sweep(basis$phi, 2L, sqrt(spectrum), '*')
      })
      A <- cbind(X, do.call(cbind, loadings))
      Q <- crossprod(A) / noise + diag(c(rep(1 / 2.5^2, ncol(X)),
                                        rep(1, sum(widths^2))))
      V <- solve(Q)
      mu <- as.numeric(V %*% crossprod(A, y) / noise)
      eta <- as.numeric(A %*% mu)
      eta_var <- rowSums((A %*% V) * A)
      expect_equal(as.numeric(fit$modes[cell, ]), mu, tolerance = 1e-8)
      expect_equal(as.numeric(fit$fitted_eta[cell, ]), eta, tolerance = 1e-8)
      expect_equal(as.numeric(fit$fitted_eta_var[cell, ]), eta_var, tolerance = 1e-8)
      expect_equal(tulpaRatio:::laplace_cell_eta(fit, model, cell), eta,
                   tolerance = 1e-8)
    }
  }
})
