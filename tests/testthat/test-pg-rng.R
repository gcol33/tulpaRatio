test_that("integer PG draws match the analytic mean and Laplace transform", {
  skip_on_cran()
  set.seed(219)
  n <- 50000L
  for (z in c(0, 4, 12)) {
    draws <- tulpaRatio:::cpp_rpg1(rep(z, n))
    mu <- if (z == 0) 1 / 4 else tanh(z / 2) / (2 * z)
    variance <- if (z == 0) 1 / 24 else
      tanh(z / 2) / (2 * z^3) - 1 / (4 * z^2 * cosh(z / 2)^2)
    laplace <- function(t) cosh(z / 2) / cosh(sqrt(z^2 + 2 * t) / 2)
    expect_lt(abs(mean(draws) - mu) / sqrt(variance / n), 5)
    for (t in c(1, 10)) {
      se <- sqrt((laplace(2 * t) - laplace(t)^2) / n)
      expect_lt(abs(mean(exp(-t * draws)) - laplace(t)) / se, 5)
    }
  }
})

test_that("real PG shapes match analytic moments and Laplace transforms", {
  skip_on_cran()
  set.seed(220)
  n <- 12000L
  for (b in c(0.3, 1.7, 4.25)) for (z in c(0, 2.4, 8)) {
    draws <- tulpaRatio:::cpp_rpg_real(rep(b, n), rep(z, n))
    mu <- if (z == 0) b / 4 else b * tanh(z / 2) / (2 * z)
    variance <- if (z == 0) b / 24 else b *
      (tanh(z / 2) / (2 * z^3) - 1 / (4 * z^2 * cosh(z / 2)^2))
    laplace <- function(t) (cosh(z / 2) / cosh(sqrt(z^2 + 2 * t) / 2))^b
    expect_lt(abs(mean(draws) - mu) / sqrt(variance / n), 5)
    for (t in c(1, 10)) {
      se <- sqrt((laplace(2 * t) - laplace(t)^2) / n)
      expect_lt(abs(mean(exp(-t * draws)) - laplace(t)) / se, 5)
    }
  }
})
