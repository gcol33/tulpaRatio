# test-benchmark.R
# Tests for benchmark utilities

test_that("ratiod_threads returns positive integer", {
  n_threads <- ratiod_threads()

  expect_true(is.numeric(n_threads))
  expect_true(n_threads >= 1)
  expect_true(n_threads == as.integer(n_threads))
})

test_that("print.ratiod_benchmark works", {
  bench <- list(
    config = list(
      N = 100,
      p = 3,
      n_groups = 5,
      n_params = 10,
      n_iter = 500,
      n_warmup = 250,
      n_chains = 1,
      n_threads = 1
    ),
    timing = list(
      elapsed_seconds = 10.5,
      samples_per_second = 23.8,
      iterations_per_second = 47.6
    ),
    diagnostics = list(
      n_divergent = 2,
      divergent_pct = 0.8
    )
  )
  class(bench) <- c("ratiod_benchmark", "list")

  output <- capture.output(print(bench))

  expect_true(any(grepl("tulpaRatio Benchmark Results", output)))
  expect_true(any(grepl("100", output)))  # N
  expect_true(any(grepl("10.5", output)))  # elapsed
})

test_that("print.ratiod_benchmark handles NA diagnostics", {
  bench <- list(
    config = list(
      N = 50,
      p = 2,
      n_groups = 0,
      n_params = 5,
      n_iter = 200,
      n_warmup = 100,
      n_chains = 1,
      n_threads = 1
    ),
    timing = list(
      elapsed_seconds = 5.2,
      samples_per_second = 19.2,
      iterations_per_second = 38.5
    ),
    diagnostics = list(
      n_divergent = NA,
      divergent_pct = NA
    )
  )
  class(bench) <- c("ratiod_benchmark", "list")

  output <- capture.output(print(bench))

  expect_true(any(grepl("tulpaRatio Benchmark Results", output)))
  expect_true(any(grepl("Configuration", output)))
  expect_true(any(grepl("Timing", output)))
  # Should NOT print diagnostics when NA
  expect_false(any(grepl("Divergent:", output)))
})

test_that("print.ratiod_benchmark returns invisibly", {
  bench <- list(
    config = list(
      N = 50,
      p = 2,
      n_groups = 0,
      n_params = 4,
      n_iter = 100,
      n_warmup = 50,
      n_chains = 1,
      n_threads = 1
    ),
    timing = list(
      elapsed_seconds = 2.0,
      samples_per_second = 25.0,
      iterations_per_second = 50.0
    ),
    diagnostics = list(
      n_divergent = 0,
      divergent_pct = 0
    )
  )
  class(bench) <- c("ratiod_benchmark", "list")

  result <- capture.output(x <- print(bench))
  expect_identical(x, bench)
})

test_that("print.ratiod_benchmark shows all configuration fields", {
  bench <- list(
    config = list(
      N = 200,
      p = 5,
      n_groups = 10,
      n_params = 20,
      n_iter = 1000,
      n_warmup = 500,
      n_chains = 4,
      n_threads = 2
    ),
    timing = list(
      elapsed_seconds = 30.0,
      samples_per_second = 66.7,
      iterations_per_second = 133.3
    ),
    diagnostics = list(
      n_divergent = 5,
      divergent_pct = 0.25
    )
  )
  class(bench) <- c("ratiod_benchmark", "list")

  output <- capture.output(print(bench))

  expect_true(any(grepl("Observations:", output)))
  expect_true(any(grepl("Predictors:", output)))
  expect_true(any(grepl("RE groups:", output)))
  expect_true(any(grepl("Parameters:", output)))
  expect_true(any(grepl("Iterations:", output)))
  expect_true(any(grepl("Chains:", output)))
  expect_true(any(grepl("Threads:", output)))
  expect_true(any(grepl("Total time:", output)))
  expect_true(any(grepl("Samples/sec:", output)))
  expect_true(any(grepl("Iter/sec:", output)))
})

# Integration tests for ratiod_benchmark
test_that("ratiod_benchmark runs with minimal settings", {
  skip_on_cran()

  bench <- ratiod_benchmark(
    N = 20,
    p = 2,
    n_groups = 0,
    n_iter = 50,
    n_warmup = 25,
    n_chains = 1,
    verbose = FALSE
  )

  expect_s3_class(bench, "ratiod_benchmark")
  expect_true(bench$timing$elapsed_seconds > 0)
  expect_true(bench$config$N == 20)
  expect_true(bench$config$p == 2)
})

test_that("ratiod_benchmark works with random effects", {
  skip_on_cran()

  bench <- ratiod_benchmark(
    N = 30,
    p = 2,
    n_groups = 3,
    n_iter = 50,
    n_warmup = 25,
    n_chains = 1,
    verbose = FALSE
  )

  expect_s3_class(bench, "ratiod_benchmark")
  expect_true(bench$config$n_groups == 3)
})

# Test ratiod_benchmark_compare exists and validates input
test_that("ratiod_benchmark_compare function exists", {
  expect_true(exists("ratiod_benchmark_compare"))
  expect_true(is.function(ratiod_benchmark_compare))
})

test_that("ratiod_threads returns integer", {
  threads <- ratiod_threads()
  expect_true(is.numeric(threads))
  expect_true(threads >= 1)
})

test_that("print.ratiod_benchmark formats output correctly", {
  # Create minimal benchmark object
  bench <- structure(
    list(
      config = list(
        N = 100L, p = 3L, n_groups = 0L, n_params = 5L,
        n_iter = 100L, n_warmup = 50L, n_chains = 1L, n_threads = 1L
      ),
      timing = list(
        elapsed_seconds = 1.5,
        samples_per_second = 33.3,
        iterations_per_second = 66.7
      ),
      diagnostics = list(
        n_divergent = 0,
        divergent_pct = 0.0
      )
    ),
    class = c("ratiod_benchmark", "list")
  )

  output <- capture.output(print(bench))
  expect_true(any(grepl("Benchmark Results", output)))
  expect_true(any(grepl("Configuration:", output)))
  expect_true(any(grepl("Timing:", output)))
})

test_that("print.ratiod_benchmark handles NA diagnostics", {
  bench <- structure(
    list(
      config = list(
        N = 100L, p = 3L, n_groups = 0L, n_params = 5L,
        n_iter = 100L, n_warmup = 50L, n_chains = 1L, n_threads = 1L
      ),
      timing = list(
        elapsed_seconds = 1.5,
        samples_per_second = 33.3,
        iterations_per_second = 66.7
      ),
      diagnostics = list(
        n_divergent = NA,
        divergent_pct = NA
      )
    ),
    class = c("ratiod_benchmark", "list")
  )

  output <- capture.output(print(bench))
  # NA diagnostics should be skipped
  expect_false(any(grepl("Divergent:", output)))
})

test_that("ratiod_benchmark_compare runs with configs", {
  skip_on_cran()

  configs <- list(
    list(N = 20, p = 2, n_groups = 0)
  )

  result <- ratiod_benchmark_compare(
    configs,
    n_iter = 50,
    n_warmup = 25,
    n_threads = 1
  )

  expect_true(is.data.frame(result))
  expect_true("N" %in% names(result))
  expect_true("elapsed_sec" %in% names(result))
  expect_equal(nrow(result), 1)
})

test_that("ratiod_benchmark_compare handles n_groups default", {
  skip_on_cran()

  # Config without explicit n_groups should default to 0
  configs <- list(
    list(N = 20, p = 2)
  )

  result <- ratiod_benchmark_compare(
    configs,
    n_iter = 50,
    n_warmup = 25
  )

  expect_equal(result$n_groups, 0)
})
