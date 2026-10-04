# Tests for spatial_rsr() - Restricted Spatial Regression

test_that("spatial_rsr wraps spatial_gp correctly", {
  gp <- spatial_gp(~ lon + lat)
  rsr <- spatial_rsr(gp, restrict_to = ~ x1 + x2)

  expect_s3_class(rsr, "ratiod_rsr")
  expect_s3_class(rsr, "ratiod_gp")
  expect_s3_class(rsr, "ratiod_spatial")
  expect_true(rsr$rsr)
  expect_equal(rsr$rsr_formula, ~ x1 + x2)
})

test_that("spatial_rsr wraps spatial_car correctly", {
  adj <- matrix(c(0, 1, 0, 1, 0, 1, 0, 1, 0), 3, 3)
  car <- spatial_car(adj, level = "obs")
  rsr <- spatial_rsr(car, restrict_to = ~ temp)

  expect_s3_class(rsr, "ratiod_rsr")
  expect_s3_class(rsr, "ratiod_spatial")
  expect_true(rsr$rsr)
})

test_that("spatial_rsr wraps spatial_multiscale correctly", {
  ms <- spatial_multiscale(~ lon + lat)
  rsr <- spatial_rsr(ms, restrict_to = ~ depth)

  expect_s3_class(rsr, "ratiod_rsr")
  expect_s3_class(rsr, "ratiod_multiscale")
  expect_true(rsr$rsr)
})

test_that("spatial_rsr rejects non-spatial input", {
  expect_error(spatial_rsr("not spatial", ~ x), "tulpaRatio spatial specification")
  expect_error(spatial_rsr(list(a = 1), ~ x), "tulpaRatio spatial specification")
})

test_that("spatial_rsr rejects non-formula restrict_to", {
  gp <- spatial_gp(~ lon + lat)
  expect_error(spatial_rsr(gp, "x1"), "must be a formula")
  expect_error(spatial_rsr(gp, c("x1", "x2")), "must be a formula")
})

test_that("spatial_rsr print method works", {
  gp <- spatial_gp(~ lon + lat)
  rsr <- spatial_rsr(gp, restrict_to = ~ depth + temp)

  output <- capture.output(print(rsr))

  expect_true(any(grepl("Gaussian Process", output)))
  expect_true(any(grepl("Restricted Spatial Regression", output)))
  expect_true(any(grepl("depth", output)))
  expect_true(any(grepl("temp", output)))
})

# Tests for compute_rsr_projection

test_that("compute_rsr_projection creates valid projection matrix", {
  # Simple design matrix
  set.seed(123)
  X <- cbind(1, rnorm(20), rnorm(20))

  P_perp <- compute_rsr_projection(X)

  expect_equal(dim(P_perp), c(20, 20))

  # P_perp should be idempotent: P_perp %*% P_perp = P_perp
  expect_equal(P_perp %*% P_perp, P_perp, tolerance = 1e-10)

  # P_perp should be symmetric
  expect_equal(P_perp, t(P_perp), tolerance = 1e-10)
})

test_that("compute_rsr_projection orthogonalizes correctly", {
  set.seed(123)
  n <- 30
  X <- cbind(1, rnorm(n), rnorm(n))

  P_perp <- compute_rsr_projection(X)

  # Random vector
  w <- rnorm(n)

  # Projected vector should be orthogonal to X
  w_projected <- P_perp %*% w

  # X' %*% w_projected should be (approximately) zero
  orthogonality <- t(X) %*% w_projected
  expect_equal(as.vector(orthogonality), rep(0, 3), tolerance = 1e-10)
})

test_that("compute_rsr_projection warns for high-dimensional case", {
  # More columns than rows
  X <- matrix(rnorm(50), nrow = 5, ncol = 10)

  expect_warning(compute_rsr_projection(X), "More covariates than observations")
})

# Tests for validate_rsr

test_that("validate_rsr computes projection matrix", {
  gp <- spatial_gp(~ x + y)
  rsr <- spatial_rsr(gp, restrict_to = ~ z1 + z2)

  df <- data.frame(
    x = rnorm(20),
    y = rnorm(20),
    z1 = rnorm(20),
    z2 = rnorm(20)
  )

  validated <- validate_rsr(rsr, df, NULL)

  expect_true(!is.null(validated$rsr_projection))
  expect_equal(dim(validated$rsr_projection), c(20, 20))
  expect_equal(validated$rsr_vars, c("z1", "z2"))
})

test_that("validate_rsr errors for missing variables", {
  gp <- spatial_gp(~ x + y)
  rsr <- spatial_rsr(gp, restrict_to = ~ missing_var)

  df <- data.frame(x = 1:10, y = 1:10)

  expect_error(validate_rsr(rsr, df, NULL), "not found in data")
})

test_that("validate_rsr returns non-RSR spatial unchanged", {
  gp <- spatial_gp(~ x + y)
  df <- data.frame(x = 1:10, y = 1:10)

  # Non-RSR spatial should pass through unchanged
  result <- validate_rsr(gp, df, NULL)
  expect_identical(result, gp)
})

# Tests for apply_rsr_projection

test_that("apply_rsr_projection applies projection correctly", {
  set.seed(123)
  n <- 20
  X <- cbind(1, rnorm(n))
  P_perp <- compute_rsr_projection(X)

  w <- rnorm(n)
  w_rsr <- apply_rsr_projection(w, P_perp)

  expect_length(w_rsr, n)

  # Result should be orthogonal to X
  expect_equal(as.vector(t(X) %*% w_rsr), c(0, 0), tolerance = 1e-10)
})

test_that("apply_rsr_projection is idempotent", {
  set.seed(456)
  n <- 15
  X <- cbind(1, rnorm(n), rnorm(n))
  P_perp <- compute_rsr_projection(X)

  w <- rnorm(n)

  # First projection
  w1 <- apply_rsr_projection(w, P_perp)

  # Second projection should give same result
  w2 <- apply_rsr_projection(w1, P_perp)

  expect_equal(w1, w2, tolerance = 1e-10)
})

test_that("rsr_basis_of spans the covariates at their rank", {
  set.seed(7)
  x <- rnorm(30)
  X <- cbind(1, x, 2 * x)
  Q <- rsr_basis_of(X)
  expect_equal(ncol(Q), 2L)
  expect_equal(crossprod(Q), diag(2), tolerance = 1e-12)
  expect_equal(compute_rsr_projection(X) %*% X, 0 * X, tolerance = 1e-10)
})

# ---------------------------------------------------------------------------
# Fitted restricted fields
# ---------------------------------------------------------------------------

rsr_fixture <- function(seed = 3, S = 8L, reps = 6L) {
  set.seed(seed)
  unit <- rep(seq_len(S), each = reps)
  x <- rnorm(S * reps) + 0.3 * unit
  data.frame(
    y = rbinom(S * reps, 15L, plogis(-0.2 + 0.4 * x + cumsum(rnorm(S, 0, 0.4))[unit])),
    trials = 15L, x = x, region = factor(unit),
    lon = runif(S * reps), lat = runif(S * reps)
  )
}

rsr_chain <- function(S = 8L) {
  A <- matrix(0L, S, S)
  for (i in seq_len(S - 1L)) {
    A[i, i + 1L] <- 1L
    A[i + 1L, i] <- 1L
  }
  A
}

rsr_fit <- function(spatial, mode, df = rsr_fixture(), iter = 200) {
  suppressWarnings(tratio(
    y | trials ~ x, data = df, family = ratiod_binomial(), spatial = spatial,
    mode = mode,
    control = list(iter = iter, warmup = iter / 2, chains = 1, seed = 1,
                   verbose = FALSE)
  ))
}

hmc_field_eta <- function(fit) {
  contribs <- hmc_structure_contributions(hmc_fit_unpack(fit), hmc_fit_design(fit))
  Filter(Negate(is.null), lapply(contribs, `[[`, "eta"))[[1L]]
}

test_that("HMC carries a restricted areal field orthogonal to the covariates", {
  skip_on_cran()
  df <- rsr_fixture()
  car <- spatial_car(rsr_chain(), group_var = "region")
  plain <- rsr_fit(car, "hmc", df)
  restricted <- rsr_fit(spatial_rsr(car, ~ x), "hmc", df)

  expect_false(isTRUE(all.equal(unclass(plain$draws), unclass(restricted$draws))))

  eta <- hmc_field_eta(restricted)
  expect_equal(dim(eta), c(nrow(restricted$draws), nrow(df)))
  expect_lt(max(abs(eta %*% cbind(1, df$x))), 1e-8)

  r <- ratio(restricted, summary = FALSE)$draws
  f <- fitted(restricted, component = "ratio", summary = FALSE)$ratio
  expect_equal(unname(f), unname(r))
})

test_that("HMC carries a restricted BYM2 and HSGP field", {
  skip_on_cran()
  df <- rsr_fixture()
  for (spatial in list(
    spatial_rsr(spatial_bym2(rsr_chain(), group_var = "region"), ~ x),
    spatial_rsr(spatial_hsgp(~ lon + lat, m = 4), ~ x)
  )) {
    fit <- rsr_fit(spatial, "hmc", df)
    expect_lt(max(abs(hmc_field_eta(fit) %*% cbind(1, df$x))), 1e-8)
  }
})

test_that("PG draws a restricted ICAR field from its exact conditional", {
  skip_on_cran()
  df <- rsr_fixture()
  spatial <- spatial_rsr(spatial_car(rsr_chain(), group_var = "region"), ~ x)
  pg <- rsr_fit(spatial, "pg", df, iter = 3000)
  field <- pg$.internal$spatial
  expect_equal(dim(field), c(nrow(pg$.internal$beta), nrow(df)))
  expect_lt(max(abs(field %*% cbind(1, df$x))), 1e-8)

  # The Laplace tier carries the same restricted model on the engine.
  lap <- rsr_fit(spatial, "laplace", df)
  pg_beta <- pg$.internal$beta
  lap_beta <- lap$draws[, lap$fixed_columns, drop = FALSE]
  expect_lt(max(abs(colMeans(pg_beta) - colMeans(lap_beta)) /
                apply(lap_beta, 2, sd)), 0.35)
})

test_that("a backend that does not carry RSR refuses it", {
  df <- rsr_fixture()
  car <- spatial_car(rsr_chain(), group_var = "region")
  restricted <- spatial_rsr(car, ~ x)

  for (backend in c("ess", "gibbs", "sghmc", "sgld", "vi")) {
    expect_match(unsupported_structures(backend, list(spatial = restricted))[["spatial"]],
                 "restricted spatial regression", info = backend)
  }
  expect_match(
    unsupported_structures("pg", list(
      spatial = spatial_rsr(spatial_bym2(rsr_chain(), group_var = "region"), ~ x)
    ))[["spatial"]],
    "intrinsic CAR"
  )
  expect_length(unsupported_structures("pg", list(spatial = restricted)), 0L)
  expect_length(unsupported_structures("hmc", list(spatial = restricted)), 0L)
  expect_length(unsupported_structures("laplace", list(spatial = restricted)), 0L)
})

test_that("a restricted field has no value at a new row", {
  skip_on_cran()
  df <- rsr_fixture()
  fit <- rsr_fit(spatial_rsr(spatial_car(rsr_chain(), group_var = "region"), ~ x),
                 "hmc", df, iter = 40)
  expect_error(predict(fit, newdata = df[1:3, ]), "no value at a new")
  expect_no_error(suppressWarnings(
    predict(fit, newdata = df[1:3, ], include_spatial = FALSE)
  ))
})
