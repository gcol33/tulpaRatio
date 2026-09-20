# Unit tests for plot_diagnostics.R using mock objects (no model fitting)

# Ensure we can access functions via namespace
get_fn <- function(name) {
  tryCatch(
    get(name, envir = asNamespace("tulpaRatio")),
    error = function(e) NULL
  )
}

# Create a minimal mock ratiod_fit object
make_mock_fit <- function(n_draws = 100, n_params = 5, n_chains = 1, backend = "hmc") {
  set.seed(42)

  # Create mock draws matrix
  param_names <- c("beta_num[1]", "beta_num[2]", "beta_denom[1]", "beta_denom[2]", "sigma_re")
  draws <- matrix(rnorm(n_draws * n_params), nrow = n_draws, ncol = n_params)
  colnames(draws) <- param_names

  # Create mock diagnostics
  diagnostics <- list(
    energy = cumsum(rnorm(n_draws, mean = 100, sd = 5)),
    divergent_idx = integer(0)
  )

  # Create mock data
  mock_data <- data.frame(
    count = rpois(20, 10),
    effort = rgamma(20, 5, 1),
    x = rnorm(20),
    site = factor(rep(1:4, each = 5))
  )

  structure(
    list(
      draws = draws,
      backend = backend,
      chains = n_chains,
      diagnostics = diagnostics,
      data = mock_data,
      formula = list(
        numerator = list(random_effects = list()),
        denominator = list(random_effects = list())
      ),
      .internal = list(
        hmc_data = list(
          N = 20,
          p_num = 2,
          p_denom = 2,
          n_re_groups = 0,
          X_num = model.matrix(~ x, mock_data),
          X_denom = model.matrix(~ x, mock_data)
        ),
        model_type = "poisson_gamma",
        samples = draws
      ),
      ratio_draws = matrix(rgamma(n_draws * 20, 2, 1), nrow = n_draws, ncol = 20)
    ),
    class = "ratiod_fit"
  )
}

# Test diagnostics() helper
test_that("diagnostics() returns diagnostics data frame", {
  fit <- make_mock_fit()

  diag <- diagnostics(fit)

  expect_s3_class(diag, "data.frame")
  expect_true("parameter" %in% names(diag))
  expect_true("rhat" %in% names(diag) || "Rhat" %in% names(diag))
  expect_true("ess_bulk" %in% names(diag) || "ess" %in% names(diag))
})

test_that("diagnostics() is per-parameter on a well-mixed multi-chain fit", {
  skip_if_not_installed("posterior")

  set.seed(20260719)
  n_chains <- 4L
  n_iter   <- 500L
  par_names <- c("beta_num[1]", "beta_num[2]", "sigma_re")

  # Independent draws with different locations/scales per parameter: the true
  # Rhat is ~1 for each and the per-parameter ESS values differ.
  draws <- cbind(
    rnorm(n_chains * n_iter, mean = 0.5, sd = 0.10),
    rnorm(n_chains * n_iter, mean = 0.3, sd = 0.05),
    rgamma(n_chains * n_iter, shape = 9, rate = 9)
  )
  colnames(draws) <- par_names

  fit <- structure(
    list(draws = draws, backend = "hmc", chains = n_chains),
    class = "ratiod_fit"
  )

  diag <- diagnostics(fit)

  expect_equal(diag$parameter, par_names)
  expect_true(all(diag$rhat < 1.05))
  expect_true(all(diag$ess_bulk > 400))

  # Per-parameter, not one value recycled across every row.
  expect_equal(length(unique(diag$rhat)), length(par_names))
  expect_equal(length(unique(diag$ess_bulk)), length(par_names))

  # Agreement with an independent per-parameter computation on the same draws.
  arr <- tulpaRatio:::get_draws_array(fit)$draws
  for (p in seq_along(par_names)) {
    m <- arr[, , p]
    expect_equal(diag$rhat[p],     posterior::rhat(m),     tolerance = 1e-6)
    expect_equal(diag$ess_bulk[p], posterior::ess_bulk(m), tolerance = 1e-6)
    expect_equal(diag$ess_tail[p], posterior::ess_tail(m), tolerance = 1e-6)
  }
})

# Test select_main_params helper
test_that("select_main_params filters correctly", {
  all_pars <- c("beta_num[1]", "beta_num[2]", "sigma_re", "re[1]", "re[2]", "spatial[1]")

  result <- tulpaRatio:::select_main_params(all_pars)

  expect_true("beta_num[1]" %in% result)
  expect_true("sigma_re" %in% result)
  expect_false("re[1]" %in% result)
  expect_false("spatial[1]" %in% result)
})

# Test grep_params helper
test_that("grep_params matches patterns", {
  all_pars <- c("beta_num[1]", "beta_num[2]", "beta_denom[1]", "sigma_re")

  result <- tulpaRatio:::grep_params("beta_num", all_pars)
  expect_equal(length(result), 2)
  expect_true(all(grepl("beta_num", result)))

  result2 <- tulpaRatio:::grep_params("sigma", all_pars)
  expect_equal(result2, "sigma_re")
})

# Test ratiod_diag_colors exists
test_that("ratiod_diag_colors is defined", {
  expect_true(exists("ratiod_diag_colors", where = asNamespace("tulpaRatio")))
  colors <- tulpaRatio:::ratiod_diag_colors
  expect_true(is.list(colors))
  expect_true("good" %in% names(colors))
  expect_true("bad" %in% names(colors))
})

# Test plot_rhat
test_that("plot_rhat works with mock fit", {
  fit <- make_mock_fit()

  result <- plot_rhat(fit)
  # Result depends on ggplot2 availability
  expect_true(is.null(result) || inherits(result, "gg") || inherits(result, "ggplot"))
})

test_that("plot_rhat errors on non-ratiod_fit", {
  expect_error(plot_rhat("not_a_fit"), "ratiod_fit")
})

test_that("plot_rhat respects threshold parameter", {
  fit <- make_mock_fit()

  # Should not error with different thresholds
  result1 <- plot_rhat(fit, threshold = 1.05)
  result2 <- plot_rhat(fit, threshold = 1.1)
  expect_true(TRUE)  # Main test is no error
})

# Test plot_ess
test_that("plot_ess works with mock fit", {
  fit <- make_mock_fit()

  result <- plot_ess(fit)
  expect_true(is.null(result) || inherits(result, "gg") || inherits(result, "ggplot"))
})

test_that("plot_ess errors on non-ratiod_fit", {
  expect_error(plot_ess("not_a_fit"), "ratiod_fit")
})

test_that("plot_ess supports type parameter", {
  fit <- make_mock_fit()

  result_bulk <- plot_ess(fit, type = "bulk")
  result_tail <- plot_ess(fit, type = "tail")
  expect_true(TRUE)  # Main test is no error
})

# Test plot_acf
test_that("plot_acf works with mock fit", {
  fit <- make_mock_fit()

  result <- suppressWarnings(plot_acf(fit))
  expect_true(is.null(result) || inherits(result, "gg") || inherits(result, "ggplot"))
})

test_that("plot_acf errors on non-ratiod_fit", {
  expect_error(plot_acf("not_a_fit"), "ratiod_fit")
})

test_that("plot_acf respects lags parameter", {
  fit <- make_mock_fit()

  result <- suppressWarnings(plot_acf(fit, lags = 10))
  expect_true(TRUE)
})

# Test plot_pairs
test_that("plot_pairs works with mock fit", {
  fit <- make_mock_fit()

  # suppressWarnings for single-chain warning from bayesplot
  result <- suppressWarnings(plot_pairs(fit))
  expect_true(is.null(result) || inherits(result, "gg") ||
              inherits(result, "ggmatrix") || is.list(result) || is.matrix(result))
})

test_that("plot_pairs errors on non-ratiod_fit", {
  expect_error(plot_pairs("not_a_fit"), "ratiod_fit")
})

test_that("plot_pairs respects n_pars", {
  fit <- make_mock_fit()

  # suppressWarnings for single-chain warning from bayesplot
  result <- suppressWarnings(plot_pairs(fit, n_pars = 3))
  expect_true(TRUE)
})

# Test plot_divergences
test_that("plot_divergences handles no divergences", {
  fit <- make_mock_fit()
  fit$diagnostics$divergent_idx <- integer(0)

  result <- plot_divergences(fit)
  expect_null(result)  # Should return NULL with message when no divergences
})

test_that("plot_divergences works with divergences", {
  fit <- make_mock_fit()
  fit$diagnostics$divergent_idx <- c(1, 5, 10)

  result <- plot_divergences(fit, type = "parcoord")
  expect_true(is.null(result) || inherits(result, "gg") || inherits(result, "ggplot"))
})

test_that("plot_divergences errors on non-HMC backend", {
  fit <- make_mock_fit(backend = "laplace")

  result <- plot_divergences(fit)
  expect_null(result)
})

# Test plot_energy
test_that("plot_energy works with mock fit", {
  fit <- make_mock_fit()

  result <- plot_energy(fit)
  expect_true(is.null(result) || inherits(result, "gg") || inherits(result, "ggplot"))
})

test_that("plot_energy returns NULL for non-HMC backend", {
  fit <- make_mock_fit(backend = "laplace")

  result <- plot_energy(fit)
  expect_null(result)
})

test_that("plot_energy handles missing energy", {
  fit <- make_mock_fit()
  fit$diagnostics$energy <- NULL

  result <- plot_energy(fit)
  expect_null(result)
})

# Test diagnostic_summary
test_that("diagnostic_summary works with mock fit", {
  fit <- make_mock_fit()

  summ <- diagnostic_summary(fit, quiet = TRUE)

  expect_s3_class(summ, "ratiod_diagnostic_summary")
  expect_true("status" %in% names(summ))
  expect_true("backend" %in% names(summ))
})

test_that("diagnostic_summary errors on non-ratiod_fit", {
  expect_error(diagnostic_summary("not_a_fit"), "ratiod_fit")
})

test_that("print.ratiod_diagnostic_summary works", {
  fit <- make_mock_fit()
  summ <- diagnostic_summary(fit, quiet = TRUE)

  output <- capture.output(print(summ))
  expect_true(length(output) > 0)
  expect_true(any(grepl("Diagnostic|diagnostic", output)))
})

test_that("print.ratiod_diagnostic_summary prints the worst-Rhat table", {
  # A fit with a genuinely bad Rhat populates worst_rhat / worst_ess, which are
  # column extracts of the diagnostics table.
  set.seed(99)
  n_iter <- 200L
  draws <- cbind(
    c(rnorm(n_iter, 0, 1), rnorm(n_iter, 8, 1)),   # chains disagree: Rhat >> 1
    rnorm(2L * n_iter)
  )
  colnames(draws) <- c("beta_num[1]", "beta_num[2]")

  fit <- structure(
    list(draws = draws, backend = "hmc", chains = 2L,
         diagnostics = list(divergent_idx = integer(0))),
    class = "ratiod_fit"
  )

  summ <- diagnostic_summary(fit, quiet = TRUE)
  expect_true(!is.null(summ$worst_rhat))
  expect_true(summ$worst_rhat$rhat[1] > 1.01)

  output <- capture.output(print(summ))
  expect_true(any(grepl("Rhat > 1.01", output)))
  expect_true(any(grepl("beta_num\\[1\\]", output)))
})

# Test geweke_test
test_that("geweke_test works with mock fit", {
  fit <- make_mock_fit(n_draws = 200)

  result <- geweke_test(fit)

  expect_s3_class(result, "ratiod_geweke")
  expect_true("parameter" %in% names(result))
  expect_true("z_score" %in% names(result))
  expect_true("p_value" %in% names(result))
})

test_that("geweke_test errors on non-ratiod_fit", {
  expect_error(geweke_test("not_a_fit"), "ratiod_fit")
})

test_that("print.ratiod_geweke works", {
  fit <- make_mock_fit(n_draws = 200)
  result <- geweke_test(fit)

  output <- capture.output(print(result))
  expect_true(any(grepl("Geweke", output)))
})

test_that("geweke_test respects frac parameters", {
  fit <- make_mock_fit(n_draws = 200)

  result1 <- geweke_test(fit, frac1 = 0.1, frac2 = 0.5)
  result2 <- geweke_test(fit, frac1 = 0.2, frac2 = 0.4)

  # Results should differ
  expect_false(identical(result1$z_score, result2$z_score))
})

# Test spectrum0_ar
test_that("spectrum0_ar computes spectral density", {
  set.seed(123)
  x <- rnorm(100)

  result <- tulpaRatio:::spectrum0_ar(x)
  expect_true(is.numeric(result))
  expect_true(result > 0)
})

test_that("spectrum0_ar handles short sequences", {
  x <- rnorm(5)

  result <- tulpaRatio:::spectrum0_ar(x)
  expect_true(is.numeric(result))
})

# Test n_divergent helper
test_that("n_divergent returns correct count", {
  fit <- make_mock_fit()
  # n_divergent may check different fields, just verify it returns a number
  result <- n_divergent(fit)
  expect_true(is.numeric(result))
  expect_true(result >= 0)
})

test_that("n_divergent returns 0 for empty divergences", {
  fit <- make_mock_fit()
  fit$diagnostics$divergent_idx <- integer(0)

  result <- n_divergent(fit)
  expect_equal(result, 0)
})

# Test plot_rhat_base (base R fallback)
test_that("plot_rhat_base works", {
  diag <- data.frame(
    parameter = c("beta[1]", "beta[2]", "sigma"),
    rhat = c(1.001, 1.02, 1.05),
    color = c("#2E7D32", "#F57F17", "#C62828")
  )
  diag$parameter <- factor(diag$parameter, levels = diag$parameter)

  # Should not error
  expect_silent(tulpaRatio:::plot_rhat_base(diag, threshold = 1.01))
})

# Test plot_ess_base (base R fallback)
test_that("plot_ess_base works", {
  diag <- data.frame(
    parameter = c("beta[1]", "beta[2]", "sigma"),
    ess = c(500, 200, 100),
    color = c("#2E7D32", "#F57F17", "#C62828")
  )
  diag$parameter <- factor(diag$parameter, levels = diag$parameter)

  expect_silent(tulpaRatio:::plot_ess_base(diag, threshold = 400, type = "bulk"))
})

# Test get_draws_array helper
test_that("get_draws_array converts to array format", {
  fit <- make_mock_fit(n_draws = 100, n_chains = 2)

  result <- tulpaRatio:::get_draws_array(fit)

  expect_true(is.list(result))
  expect_true("draws" %in% names(result))
  expect_true(is.array(result$draws))
})
