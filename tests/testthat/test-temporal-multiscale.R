# Tests for temporal_multiscale()

test_that("temporal_multiscale creates valid object with defaults", {
  tm <- temporal_multiscale("year")

  expect_s3_class(tm, "ratiod_temporal_multiscale")
  expect_s3_class(tm, "ratiod_temporal")
  expect_equal(tm$type, "multiscale")
  expect_equal(tm$time_var, "year")
  expect_equal(tm$trend, "rw2")
  expect_equal(tm$short_term, "ar1")
  expect_null(tm$seasonal)
  expect_true(tm$shared)
})

test_that("temporal_multiscale accepts all component types", {
  # RW1 trend
  tm1 <- temporal_multiscale("year", trend = "rw1")
  expect_equal(tm1$trend, "rw1")

  # No trend
  tm2 <- temporal_multiscale("year", trend = "none", seasonal = 12)
  expect_equal(tm2$trend, "none")

  # IID short-term
  tm3 <- temporal_multiscale("year", short_term = "iid")
  expect_equal(tm3$short_term, "iid")

  # No short-term
 tm4 <- temporal_multiscale("year", short_term = "none")
  expect_equal(tm4$short_term, "none")
})

test_that("temporal_multiscale accepts seasonal specification", {
  tm <- temporal_multiscale("month", seasonal = 12)

  expect_equal(tm$seasonal, 12L)
  expect_true("seasonal" %in% tm$components)
})

test_that("temporal_multiscale components list is correct", {
  # All components
  tm1 <- temporal_multiscale("t", trend = "rw2", seasonal = 12, short_term = "ar1")
  expect_equal(tm1$components, c("trend", "seasonal", "short_term"))

  # Just trend
  tm2 <- temporal_multiscale("t", trend = "rw1", seasonal = NULL, short_term = "none")
  expect_equal(tm2$components, c("trend"))

  # Just seasonal
  tm3 <- temporal_multiscale("t", trend = "none", seasonal = 4, short_term = "none")
  expect_equal(tm3$components, c("seasonal"))

  # Trend + short-term
  tm4 <- temporal_multiscale("t", trend = "rw2", seasonal = NULL, short_term = "iid")
  expect_equal(tm4$components, c("trend", "short_term"))
})

test_that("temporal_multiscale rejects invalid time_var", {
  expect_error(temporal_multiscale(123), "character string")
  expect_error(temporal_multiscale(c("a", "b")), "single character string")
})

test_that("temporal_multiscale rejects invalid seasonal", {
  expect_error(temporal_multiscale("t", seasonal = 1), ">= 2")
  expect_error(temporal_multiscale("t", seasonal = 0), ">= 2")
  expect_error(temporal_multiscale("t", seasonal = -5), ">= 2")
})

test_that("temporal_multiscale rejects all-none components", {
  expect_error(
    temporal_multiscale("t", trend = "none", seasonal = NULL, short_term = "none"),
    "At least one"
  )
})

test_that("temporal_multiscale print method works", {
  tm <- temporal_multiscale("year", trend = "rw2", seasonal = 12, short_term = "ar1")
  output <- capture.output(print(tm))

  expect_true(any(grepl("Multi-Scale", output)))
  expect_true(any(grepl("Trend", output)))
  expect_true(any(grepl("RW2", output)))
  expect_true(any(grepl("Seasonal", output)))
  expect_true(any(grepl("12", output)))
  expect_true(any(grepl("AR\\(1\\)", output)))
})

# Tests for validate_temporal_multiscale

test_that("validate_temporal_multiscale computes indices", {
  tm <- temporal_multiscale("year", trend = "rw2")

  df <- data.frame(
    year = rep(2010:2020, each = 5),
    x = rnorm(55)
  )

  validated <- tulpaRatio:::validate_temporal_multiscale(tm, df)

  expect_equal(validated$n_times, 11)
  expect_equal(length(validated$time_index), 55)
  expect_equal(validated$time_levels, as.character(2010:2020))
})

test_that("validate_temporal_multiscale handles groups", {
  tm <- temporal_multiscale("year", group_var = "site")

  df <- data.frame(
    year = rep(2010:2015, 3),
    site = rep(c("A", "B", "C"), each = 6)
  )

  validated <- tulpaRatio:::validate_temporal_multiscale(tm, df)

  expect_equal(validated$n_times, 6)
  expect_equal(validated$n_groups, 3)
  expect_equal(validated$group_levels, c("A", "B", "C"))
})

test_that("validate_temporal_multiscale errors for missing time variable", {
  tm <- temporal_multiscale("year")
  df <- data.frame(time = 1:10)

  expect_error(tulpaRatio:::validate_temporal_multiscale(tm, df), "year.*not found")
})

test_that("validate_temporal_multiscale errors for RW2 with < 3 time points", {
  tm <- temporal_multiscale("year", trend = "rw2")
  df <- data.frame(year = c(2020, 2021))

  expect_error(tulpaRatio:::validate_temporal_multiscale(tm, df), "at least 3 time points")
})

test_that("validate_temporal_multiscale warns for short series with seasonal", {
  tm <- temporal_multiscale("month", trend = "none", seasonal = 12, short_term = "none")
  df <- data.frame(month = 1:6)

  expect_warning(tulpaRatio:::validate_temporal_multiscale(tm, df), "less than seasonal period")
})

test_that("validate_temporal_multiscale builds precision structures", {
  tm <- temporal_multiscale("year", trend = "rw2", seasonal = 12, short_term = "ar1")

  df <- data.frame(year = rep(1:20, 5))

  validated <- tulpaRatio:::validate_temporal_multiscale(tm, df)

  expect_true("trend" %in% names(validated$precision_structures))
  expect_true("seasonal" %in% names(validated$precision_structures))
  expect_true("short_term" %in% names(validated$precision_structures))

  expect_equal(validated$precision_structures$trend$type, "rw2")
  expect_equal(validated$precision_structures$seasonal$type, "rw1")
  expect_true(validated$precision_structures$seasonal$cyclic)
  expect_equal(validated$precision_structures$short_term$type, "ar1")
})

test_that("validate_temporal_multiscale calculates n_temporal_params correctly", {
  # Trend (T=10) + seasonal (P=4) + short-term (T=10) = 24
  tm <- temporal_multiscale("t", trend = "rw1", seasonal = 4, short_term = "iid")
  df <- data.frame(t = 1:10)

  validated <- tulpaRatio:::validate_temporal_multiscale(tm, df)
  expect_equal(validated$n_temporal_params, 10 + 4 + 10)

  # With 3 groups: 24 * 3 = 72
  tm2 <- temporal_multiscale("t", trend = "rw1", seasonal = 4, short_term = "iid",
                             group_var = "g")
  df2 <- data.frame(t = rep(1:10, 3), g = rep(c("A", "B", "C"), each = 10))

  validated2 <- tulpaRatio:::validate_temporal_multiscale(tm2, df2)
  expect_equal(validated2$n_temporal_params, (10 + 4 + 10) * 3)
})

test_that("validate_temporal_multiscale handles factor time variable", {
  tm <- temporal_multiscale("period")

  df <- data.frame(
    period = factor(rep(c("Q1", "Q2", "Q3", "Q4"), 5),
                    levels = c("Q1", "Q2", "Q3", "Q4"))
  )

  validated <- tulpaRatio:::validate_temporal_multiscale(tm, df)

  expect_equal(validated$n_times, 4)
  expect_equal(validated$time_levels, c("Q1", "Q2", "Q3", "Q4"))
})

# ============================================================================
# The multi-scale block under the non-GP sampler entry (#74)
# ============================================================================
#
# A multi-scale temporal term paired with an AREAL spatial field. Were
# data.has_multiscale_temporal cleared after the temporal bundle set it,
# compute_param_layout() would allocate none of the trend/seasonal/short-term
# blocks while R went on naming their columns, and every multi-scale column
# would be read at an offset holding something else.
#
# The arbiter is the prior support, as in test-gp-sampler-blocks.R: rho_short
# is logit-mapped onto the OPEN interval (-1, 1) and the variances are
# exp-transformed, so no draw of them can be 0, +/-1, infinite or NaN. Reading
# the fit from the outside this way needs no knowledge of the layout, which is
# exactly what the mislabelling defeated.

ms_areal_fixture <- function(S = 6L, T_ = 12L, seed = 3L) {
  set.seed(seed)
  grid <- expand.grid(unit = seq_len(S), year = seq_len(T_))
  d <- data.frame(unit = factor(grid$unit), year = grid$year,
                  x = rnorm(nrow(grid)))
  d$denom <- rpois(nrow(d), 60) + 5L
  d$num <- rpois(nrow(d), 0.3 * d$denom)
  side <- ceiling(sqrt(S))
  A <- matrix(0, S, S)
  for (i in seq_len(S)) for (j in seq_len(S)) {
    li <- c((i - 1L) %% side, (i - 1L) %/% side)
    lj <- c((j - 1L) %% side, (j - 1L) %/% side)
    if (i != j && sum((li - lj)^2) <= 1) A[i, j] <- 1
  }
  dimnames(A) <- list(levels(d$unit), levels(d$unit))
  list(data = d, adj = A)
}

test_that("an areal spatial field carries a multi-scale temporal block, and its draws stay in support", {
  skip_on_cran()
  fx <- ms_areal_fixture()
  fit <- tratio(
    num | denom ~ x,
    data = fx$data,
    family = ratiod_poisson_gamma(),
    mode = "hmc",
    spatial = spatial_car(fx$adj, level = "group", group_var = "unit"),
    temporal = temporal_multiscale("year", trend = "rw1", short_term = "ar1"),
    control = list(iter = 100, warmup = 50, chains = 1, seed = 3, verbose = FALSE)
  )
  dr <- as.matrix(fit$draws)

  # The block has to be named at all before anything below means something.
  field_cols <- grep("^(trend|short_term)\\[", colnames(dr), value = TRUE)
  expect_gt(length(field_cols), 0L)

  for (nm in c("sigma2_trend", "sigma2_short")) {
    expect_true(nm %in% colnames(dr))
    v <- dr[, nm]
    expect_true(all(is.finite(v)),
                label = sprintf("%s finite: min %.4g, max %.4g", nm, min(v), max(v)))
    expect_true(all(v > 0),
                label = sprintf("%s > 0: min %.4g", nm, min(v)))
  }

  expect_true("rho_short" %in% colnames(dr))
  rho <- dr[, "rho_short"]
  expect_true(all(is.finite(rho) & rho > -1 & rho < 1),
              label = sprintf("rho_short in (-1, 1): min %.4g, max %.4g",
                              min(rho), max(rho)))

  # A field under a proper prior does not wander to a standard deviation in the
  # thousands over 50 draws; the unallocated block did, on every column.
  for (nm in field_cols) {
    expect_lt(sd(dr[, nm]), 10,
              label = sprintf("%s sd %.4g", nm, sd(dr[, nm])))
  }

  # Each column has to be a draw, not one value repeated: an offset landing on
  # a constant slot would satisfy the bounds above while carrying no posterior.
  expect_gt(min(vapply(field_cols, function(nm) length(unique(dr[, nm])), integer(1))), 1L)
})
