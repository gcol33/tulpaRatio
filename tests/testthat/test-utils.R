test_that("inverse logit preserves values, tails, logs and array dimensions", {
  x <- c(-1000, -40, -10, -1, 0, 1, 10, 40, 1000)
  for (value in list(x, matrix(x, 3), array(rep(x, 2), c(3, 3, 2)))) {
    for (lower in c(TRUE, FALSE)) for (log in c(TRUE, FALSE)) {
      expect_identical(tulpaRatio:::inv_logit(value, lower.tail = lower, log.p = log),
                       stats::plogis(value, lower.tail = lower, log.p = log))
    }
  }
  expect_equal(tulpaRatio:::inv_logit(x), 1 / (1 + exp(-x)), tolerance = 1e-15)
})
