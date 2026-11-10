test_that("supported structures use the engine by default and honor the switch", {
  old <- options(tulpaRatio.use_specs = NULL)
  on.exit(options(old))
  calls <- 0L
  trace("cpp_tulpaRatio_run_nuts_specs", where = asNamespace("tulpaRatio"),
    tracer = function() calls <<- calls + 1L, print = FALSE)
  on.exit(untrace("cpp_tulpaRatio_run_nuts_specs", where = asNamespace("tulpaRatio")), add = TRUE)
  d <- data.frame(time = rep(1:4, 8), n = 12L, y = rep(c(3L, 4L, 6L, 7L), 8))
  fit <- function() suppressWarnings(tratio(y | n ~ 1, d, family = ratiod_binomial(),
    temporal = temporal_iid("time"), mode = "hmc",
    control = list(iter = 40L, warmup = 20L, chains = 1L, cores = 1L, seed = 19L,
      gradient_mode = "A_r", verbose = FALSE)))
  engine <- fit()
  expect_equal(calls, 1L)
  options(tulpaRatio.use_specs = FALSE)
  legacy <- fit()
  expect_equal(calls, 1L)
  expect_identical(dim(engine$draws), dim(legacy$draws))
})