# test-predict-spatial.R
# Tests for predict() with spatial models

test_that("predict works for basic model (no spatial)", {
  skip_on_cran()

  set.seed(123)
  n <- 30
  df <- data.frame(
    y = rpois(n, 10),
    n_trials = rep(20L, n),
    x = rnorm(n),
    site = factor(rep(1:5, each = 6))
  )

  fit <- tratio(
    y | n_trials ~ x + (1 | site),
    data = df,
    family = ratiod_binomial(),
    control = list(iter = 100, warmup = 50, chains = 1)
  )

  # Basic prediction
  new_df <- data.frame(x = c(-1, 0, 1), site = factor(c(1, 2, 3)))
  pred <- predict(fit, newdata = new_df)

  expect_s3_class(pred, "ratiod_prediction")
  expect_equal(nrow(pred), 3)
  expect_true(all(c("mean", "sd", "q2.5", "q97.5") %in% names(pred)))

  # Population-level prediction
  pred_pop <- predict(fit, newdata = new_df, re_formula = NA)
  expect_equal(nrow(pred_pop), 3)
})

test_that("predict works with ICAR spatial (lookup)",
{
  skip_on_cran()

  set.seed(456)
  n_sites <- 6
  n <- 30

  # Create simple adjacency
  adj <- build_chain_adjacency(n_sites, integer = FALSE)

  df <- data.frame(
    y = rpois(n, 10),
    n_trials = rep(20L, n),
    x = rnorm(n),
    region = factor(rep(LETTERS[1:n_sites], length.out = n))
  )

  fit <- tratio(
    y | n_trials ~ x,
    data = df,
    family = ratiod_binomial(),
    spatial = spatial_car(adjacency = adj, group_var = "region"),
    control = list(iter = 100, warmup = 50, chains = 1)
  )

  # Predict for existing regions
  new_df <- data.frame(x = c(0, 0.5), region = factor(c("A", "C")))
  pred <- predict(fit, newdata = new_df)

  expect_equal(nrow(pred), 2)
  expect_true(all(pred$mean > 0 & pred$mean < 1))  # Probabilities
})

test_that("predict works with return_spatial = TRUE", {
  skip_on_cran()

  set.seed(789)
  n_sites <- 6
  n <- 30

  adj <- build_chain_adjacency(n_sites, integer = FALSE)

  df <- data.frame(
    y = rpois(n, 10),
    n_trials = rep(20L, n),
    x = rnorm(n),
    region = factor(rep(LETTERS[1:n_sites], length.out = n))
  )

  fit <- tratio(
    y | n_trials ~ x,
    data = df,
    family = ratiod_binomial(),
    spatial = spatial_car(adjacency = adj, group_var = "region"),
    control = list(iter = 100, warmup = 50, chains = 1)
  )

  new_df <- data.frame(x = c(0, 0.5, 1), region = factor(c("A", "B", "C")))
  pred <- predict(fit, newdata = new_df, return_spatial = TRUE)

  expect_type(pred, "list")
  expect_true("predictions" %in% names(pred))
  expect_true("w.0.samples" %in% names(pred))
  expect_equal(ncol(pred$w.0.samples), 3)  # 3 new observations
})

test_that("predict errors when coords.0 missing for GP", {
  skip_on_cran()

  set.seed(111)
  n <- 20
  df <- data.frame(
    y = rbinom(n, 20L, 0.4),
    n_trials = rep(20L, n),
    x = rnorm(n),
    lon = runif(n),
    lat = runif(n)
  )

  fit <- tratio(
    y | n_trials ~ x,
    data = df,
    family = ratiod_binomial(),
    spatial = spatial_gp(coords = c("lon", "lat"), nn = 5),
    control = list(iter = 50, warmup = 25, chains = 1)
  )

  new_df <- data.frame(x = c(0, 1))

  expect_error(
    predict(fit, newdata = new_df),
    "coords.0 required"
  )
})

# The four kernels, written out here rather than read off the package, so the
# kriging below is scored against a covariance family it does not share code
# with. phi is a lengthscale in all four.
krige_kernels <- list(
  exponential = function(d, s2, phi) s2 * exp(-d / phi),
  matern      = function(d, s2, phi) {
    u <- sqrt(3) * d / phi
    s2 * (1 + u) * exp(-u)
  },
  gaussian    = function(d, s2, phi) s2 * exp(-0.5 * (d / phi)^2),
  spherical   = function(d, s2, phi) {
    r <- d / phi
    ifelse(d >= phi, 0, s2 * (1 - 1.5 * r + 0.5 * r^3))
  }
)

# Nearest-neighbour kriging at one new location for one draw, from its
# definition: the k nearest training locations, the neighbour block plus the
# 1e-6 jitter the sampler paths use, and C_s0S C_SS^-1 w.
krige_reference <- function(f, coords_train, coords_new, w_train, s2, phi, k) {
  vapply(seq_len(nrow(coords_new)), function(i) {
    d0 <- sqrt(rowSums((coords_train -
                          matrix(coords_new[i, ], nrow(coords_train), 2,
                                 byrow = TRUE))^2))
    nb <- order(d0)[seq_len(min(k, nrow(coords_train)))]
    C_SS <- f(as.matrix(dist(coords_train[nb, , drop = FALSE])), s2, phi)
    diag(C_SS) <- s2 + 1e-6
    sum(solve(C_SS, f(d0[nb], s2, phi)) * w_train[nb])
  }, 0)
}

test_that("kriging reproduces its definition under every covariance", {
  set.seed(4)
  coords_train <- cbind(runif(14), runif(14))
  coords_new <- cbind(runif(5), runif(5))
  w_train <- rbind(rnorm(14), rnorm(14), rnorm(14))
  s2 <- c(0.7, 1.3, 0.4)
  phi <- c(0.25, 0.6, 0.9)

  codes <- c(exponential = 0L, matern = 1L, gaussian = 2L, spherical = 3L)
  for (cv in names(codes)) {
    got <- tulpaRatio:::cpp_kriging_predict(
      coords_train = coords_train, coords_new = coords_new,
      w_train = w_train, sigma2 = s2, phi = phi,
      cov_type = codes[[cv]], nn = 5L
    )
    want <- t(vapply(seq_along(s2), function(s) {
      krige_reference(krige_kernels[[cv]], coords_train, coords_new,
                      w_train[s, ], s2[s], phi[s], 5L)
    }, numeric(nrow(coords_new))))

    expect_equal(dim(got), c(length(s2), nrow(coords_new)), label = cv)
    expect_equal(got, want, tolerance = 1e-10, label = cv)
  }

  # The kernel is selected, not defaulted. Spherical used to reach an R copy of
  # the family that knew three kernels and answered the fourth with an
  # exponential one.
  krige <- function(code) tulpaRatio:::cpp_kriging_predict(
    coords_train = coords_train, coords_new = coords_new, w_train = w_train,
    sigma2 = s2, phi = phi, cov_type = code, nn = 5L)
  expect_false(isTRUE(all.equal(krige(0L), krige(3L), tolerance = 0)))
  expect_error(krige(4L), "Unknown covariance code")
})

test_that("kriging refuses a field that is not indexed by location", {
  set.seed(4)
  coords_train <- cbind(runif(8), runif(8))
  coords_new <- cbind(runif(2), runif(2))

  # One column per row of DATA rather than per location: what pairing the field
  # with the observation-order coordinate matrix produces.
  expect_error(
    tulpaRatio:::cpp_kriging_predict(
      coords_train = coords_train, coords_new = coords_new,
      w_train = matrix(rnorm(16), 1), sigma2 = 1, phi = 0.5,
      cov_type = 0L, nn = 5L),
    "indexed by location"
  )

  # Hyperparameter draws that do not line up with the field draws, which is
  # what reading them from one chain while the field spans several gives.
  expect_error(
    tulpaRatio:::cpp_kriging_predict(
      coords_train = coords_train, coords_new = coords_new,
      w_train = matrix(rnorm(16), 2), sigma2 = 1, phi = 0.5,
      cov_type = 0L, nn = 5L),
    "row-aligned draws"
  )
})

test_that("a PG GP fit predicts its field at new coordinates", {
  skip_on_cran()
  # The converter read the field draws off a name the sampler does not return,
  # so every PG GP fit carried none and this whole path returned NULL without
  # saying so.
  set.seed(3)
  n_loc <- 20L
  reps <- 4L
  co <- cbind(runif(n_loc), runif(n_loc))
  S <- 0.7 * exp(-as.matrix(dist(co)) / 0.3)
  w <- as.vector(t(chol(S + diag(1e-8, n_loc))) %*% rnorm(n_loc))
  loc <- rep(seq_len(n_loc), each = reps)
  N <- n_loc * reps
  x <- rnorm(N)
  trials <- rep(30L, N)
  df <- data.frame(
    y = rbinom(N, trials, plogis(-0.2 + 0.5 * x + w[loc])),
    n_trials = trials, x = x, lon = co[loc, 1], lat = co[loc, 2]
  )

  fit <- tratio(y | n_trials ~ x, data = df, family = ratiod_binomial(),
                mode = "pg", spatial = spatial_gp(~ lon + lat, nn = 5),
                control = list(iter = 200, warmup = 100, chains = 2,
                               verbose = FALSE, seed = 5))

  # One column per LOCATION, and the hyperparameters row-aligned with it across
  # both chains -- reading sigma2 from the first chain alone leaves the second
  # chain of field draws paired with nothing.
  w_gp <- fit$.internal$w_gp
  expect_equal(ncol(w_gp), n_loc)
  expect_equal(nrow(fit$.internal$gp_hyper), nrow(w_gp))

  # spatial_gp(scale_coords = TRUE) is the default, so the fitted field lives in
  # centred and scaled coordinates while coords.0 arrives raw. Putting the fit's
  # own locations through the conversion has to return the ones it was built on.
  expect_equal(tulpaRatio:::gp_scale_new_coords(fit$spatial, co),
               as.matrix(fit$.internal$gp_coords), tolerance = 1e-12)

  # Kriging back onto the training coordinates recovers the field it was fitted
  # to. Measured 0.95 here; reading coords.0 in its own raw units instead gives
  # -0.01, since the neighbours it picks are unrelated to the ones the fit has.
  back <- tulpaRatio:::predict_spatial_gp_pg(fit, co, return_spatial = TRUE)
  expect_equal(dim(back$w_pred), c(nrow(w_gp), n_loc))
  expect_gt(cor(colMeans(back$w_pred), w), 0.8)

  # And through the front door at coordinates the fit never saw.
  new_df <- data.frame(x = c(0, 0.5, 1))
  new_co <- cbind(c(0.2, 0.5, 0.8), c(0.3, 0.5, 0.7))
  pred <- predict(fit, newdata = new_df, coords.0 = new_co,
                  return_spatial = TRUE)
  expect_equal(ncol(pred$w.0.samples), 3L)
  expect_true(all(is.finite(pred$w.0.samples)))
})

test_that("detect_spatial_type identifies spatial types correctly", {
  # Mock objects
  obj_none <- list(spatial = NULL, .internal = list(hmc_data = list()))
  obj_gp <- list(
    spatial = structure(list(), class = "ratiod_gp"),
    .internal = list(hmc_data = list(gp_type = 1L))
  )
  obj_icar <- list(
    spatial = list(type = "icar"),
    .internal = list(hmc_data = list())
  )

  expect_null(detect_spatial_type(obj_none))
  expect_equal(detect_spatial_type(obj_gp), "gp")
  expect_equal(detect_spatial_type(obj_icar), "icar")
})

test_that("predict works for PG backend with spatial", {
  skip_on_cran()

  set.seed(222)
  n_sites <- 6
  n <- 30

  adj <- build_chain_adjacency(n_sites, integer = FALSE)

  df <- data.frame(
    y = rbinom(n, 20, 0.4),
    n_trials = rep(20L, n),
    x = rnorm(n),
    region = factor(rep(LETTERS[1:n_sites], length.out = n))
  )

  fit <- tratio(
    y | n_trials ~ x,
    data = df,
    family = ratiod_binomial(),
    spatial = spatial_car(adjacency = adj, group_var = "region"),
    mode = "pg",
    control = list(iter = 100, warmup = 50, chains = 1)
  )

  new_df <- data.frame(x = c(0, 0.5), region = factor(c("A", "C")))
  pred <- predict(fit, newdata = new_df)

  expect_equal(nrow(pred), 2)
})
