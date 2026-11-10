test_that("spatial covariance values and range gradients honor Matern smoothness", {
  distance <- c(0, .03, .3, 1, 3)
  variance <- 1.7; range <- .6
  kernel <- function(nu, ell) {
    z <- sqrt(2 * nu) * distance / ell
    if (nu == .5) variance * exp(-z)
    else if (nu == 1.5) variance * (1 + z) * exp(-z)
    else variance * (1 + z + z^2 / 3) * exp(-z)
  }
  for (nu in c(.5, 1.5, 2.5)) {
    value <- tulpaRatio:::cpp_test_spatial_covariance(distance, variance, range, 'matern', nu)
    expect_equal(value[, 1], kernel(nu, range), tolerance = 1e-13)
    eps <- 1e-6
    expect_equal(value[, 2], (kernel(nu, range + eps) - kernel(nu, range - eps)) / (2 * eps),
      tolerance = 1e-8)
  }
  expect_equal(tulpaRatio:::cov_type_code('matern', .5), 0L)
  expect_equal(tulpaRatio:::cov_type_code('matern', 1.5), 1L)
  expect_equal(tulpaRatio:::cov_type_code('matern', 2.5), 4L)
  expect_error(spatial_gp(~ x + y, cov = 'matern', nu = 1), 'supported smoothness')
  gaussian <- tulpaRatio:::cpp_test_spatial_covariance(distance, variance, range, 'gaussian')
  expect_equal(gaussian[, 1], variance * exp(-.5 * (distance / range)^2), tolerance = 1e-13)
})
