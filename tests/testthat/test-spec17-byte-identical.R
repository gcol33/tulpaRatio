test_that("the archived likelihood and gradient reductions are byte identical", {
  reference <- readRDS(test_path("fixtures", "spec17-gradient-before-fused.rds"))
  for (key in names(reference)) {
    parts <- strsplit(key," ",fixed=TRUE)[[1]]
    actual <- tulpaRatio:::cpp_test_spec17(parts[1], parts[2])
    expected <- reference[[key]]
    expect_identical(writeBin(actual$analytic,raw(),size=8L),
                     writeBin(expected$analytic,raw(),size=8L),info=key)
    expect_identical(writeBin(actual$log_post,raw(),size=8L),
                     writeBin(expected$log_post,raw(),size=8L),info=key)
  }
})

test_that("the frozen margin cases have correct canonical engine gradients", {
  keys <- names(readRDS(test_path("fixtures", "spec17-gradient-before-fused.rds")))
  for (key in keys) {
    parts <- strsplit(key, " ", fixed = TRUE)[[1]]
    actual <- tulpaRatio:::cpp_gradient_check(parts[2], family = parts[1],
      n_obs = 80L, n_units = 5L, n_times = 5L, seed = 17L)
    deviation <- abs(actual$analytic - actual$finite_diff) /
      pmax(1, abs(actual$finite_diff))
    expect_lt(max(deviation), 1e-4, label = key)
    expect_equal(actual$log_post, actual$log_post_mode, tolerance = 1e-8,
                 info = key)
  }
})
