# A factor predictor expands to one column per non-reference level. New data
# holding only some of the levels must still map onto the training columns.

factor_data <- function(n = 60, seed = 11) {
  set.seed(seed)
  season <- factor(rep(c("spring", "summer", "winter"), length.out = n))
  data.frame(
    count = rpois(n, 8 * c(spring = 1, summer = 1.6, winter = 0.6)[as.character(season)]),
    effort = rgamma(n, shape = 4, rate = 1),
    depth = rnorm(n),
    season = season
  )
}

test_that("predict() rebuilds factor columns from the training levels", {
  skip_on_cran()
  df <- factor_data()
  fit <- tratio(count | effort ~ depth + season, data = df,
                family = ratiod_poisson_gamma(),
                control = list(iter = 200, chains = 1, seed = 1))
  f <- fitted(fit, component = "ratio", summary = FALSE)$ratio

  p_all <- predict(fit, newdata = df, component = "ratio", summary = FALSE)$ratio
  expect_equal(unname(p_all), unname(f))

  rows <- which(df$season == "winter")
  p_one <- predict(fit, newdata = df[rows, ], component = "ratio",
                   summary = FALSE)$ratio
  expect_equal(unname(p_one), unname(f[, rows, drop = FALSE]))

  p_chr <- predict(fit, newdata = data.frame(depth = 0, season = "summer"),
                   component = "ratio", summary = FALSE)$ratio
  expect_equal(ncol(p_chr), 1L)

  expect_error(
    predict(fit, newdata = data.frame(depth = 0, season = "autumn")),
    "autumn"
  )
})
