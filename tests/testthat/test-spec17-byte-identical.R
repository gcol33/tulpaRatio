test_that("the archived likelihood and gradient reductions are byte identical", {
  # The fixture's data and parameter vector are drawn with the standard
  # library's distributions from a seeded mt19937. The engine's output is
  # specified, the distributions' algorithms are not: libstdc++ (Linux,
  # Windows) and libc++ (macOS) draw different values from the same seed, so
  # the archive, recorded under libstdc++, describes a different model there.
  # The next test checks the gradients against the engine on any platform.
  skip_if(identical(Sys.info()[["sysname"]], "Darwin"),
          "the archived fixture inputs are drawn by libstdc++")
  reference <- readRDS(test_path("fixtures", "spec17-gradient-before-fused.rds"))
  # Byte identity holds on the platform the archive was written on, Windows
  # x86-64. A floating-point reduction is not bit-reproducible across
  # compilers and instruction sets, so on any other platform the values are
  # compared to a relative 1e-12.
  archive_platform <- identical(.Platform$OS.type, "windows") &&
    identical(R.version$arch, "x86_64")
  for (key in names(reference)) {
    parts <- strsplit(key," ",fixed=TRUE)[[1]]
    actual <- tulpaRatio:::cpp_test_spec17(parts[1], parts[2])
    expected <- reference[[key]]
    if (archive_platform) {
      expect_identical(writeBin(actual$analytic,raw(),size=8L),
                       writeBin(expected$analytic,raw(),size=8L),info=key)
      expect_identical(writeBin(actual$log_post,raw(),size=8L),
                       writeBin(expected$log_post,raw(),size=8L),info=key)
    } else {
      expect_equal(actual$analytic, expected$analytic, tolerance = 1e-12,
                   info = key)
      expect_equal(actual$log_post, expected$log_post, tolerance = 1e-12,
                   info = key)
    }
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
