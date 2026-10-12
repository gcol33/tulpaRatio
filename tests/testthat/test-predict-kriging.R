# Predicting the spatial field at the training locations has to return the
# field the fit drew there: the kriging interpolates its own data points, and
# the HSGP basis is the one the fit was built on.

kriging_data <- function(n = 25, seed = 21) {
  set.seed(seed)
  data.frame(
    count = rpois(n, 18),
    effort = rgamma(n, 9, 1),
    depth = rnorm(n),
    x = runif(n, 0, 50),
    y = runif(n, 100, 140)
  )
}

field_at_rows <- function(fit, rows) {
  w <- fit$.internal$spatial_draws
  loc <- fit$.internal$spatial_spec$obs_to_loc %||% seq_len(ncol(w))
  w[, loc[rows], drop = FALSE]
}

test_that("GP kriging at training locations returns the fitted field", {
  skip_on_cran()
  df <- kriging_data()
  fit <- tratio(count | effort ~ depth, data = df,
                spatial = spatial_gp(coords = c("x", "y")),
                family = ratiod_poisson_gamma(),
                control = list(iter = 200, chains = 1, seed = 1))
  rows <- 1:4
  p <- predict(fit, newdata = df[rows, ], coords.0 = as.matrix(df[rows, c("x", "y")]),
               return_spatial = TRUE)
  expect_equal(unname(p$w.0.samples), unname(field_at_rows(fit, rows)),
               tolerance = 1e-3)
  expect_true(all(is.finite(p$w.0.samples)))
})

test_that("HSGP prediction at training locations returns the fitted field", {
  skip_on_cran()
  df <- kriging_data()
  fit <- tratio(count | effort ~ depth, data = df,
                spatial = spatial_hsgp(coords = c("x", "y")),
                family = ratiod_poisson_gamma(),
                control = list(iter = 200, chains = 1, seed = 1))
  rows <- 1:4
  p <- predict(fit, newdata = df[rows, ], coords.0 = as.matrix(df[rows, c("x", "y")]),
               return_spatial = TRUE)
  expect_equal(unname(p$w.0.samples), unname(fit$.internal$spatial_draws[, rows]),
               tolerance = 1e-8)
})
