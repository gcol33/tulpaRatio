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
