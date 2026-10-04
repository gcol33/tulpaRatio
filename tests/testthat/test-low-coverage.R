# test-low-coverage.R
# Comprehensive tests for low-coverage files

# -----------------------------------------------------------------------------
# benchmark.R coverage
# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
# validate.R coverage
# -----------------------------------------------------------------------------

test_that("prior_predict refuses a family it cannot simulate from", {
  expect_error(prior_predict(NULL, NULL, NULL), "ratiod_family")
})

test_that("ratiod_compare requires at least two models", {
  expect_error(ratiod_compare(), "At least two models")
})

test_that("ratiod_compare requires ratiod_fit objects", {
  mock_fit <- list(stan_data = list(N = 10L))
  class(mock_fit) <- "not_ratiod"

  expect_error(ratiod_compare(mock_fit, mock_fit), "ratiod_fit")
})

test_that("ratiod_average requires at least two models", {
  expect_error(ratiod_average(), "At least two models")
})

test_that("ratiod_average requires ratiod_fit objects", {
  mock_fit <- list(stan_data = list(N = 10L))
  class(mock_fit) <- "not_ratiod"

  expect_error(ratiod_average(mock_fit, mock_fit), "ratiod_fit")
})

test_that("print.ratiod_average works", {
  avg_obj <- structure(
    list(
      weights = c(0.6, 0.4),
      predictions = data.frame(
        mean = c(1.0, 2.0, 3.0),
        sd = c(0.1, 0.2, 0.3),
        q2.5 = c(0.8, 1.6, 2.4),
        q50 = c(1.0, 2.0, 3.0),
        q97.5 = c(1.2, 2.4, 3.6)
      ),
      models = c("model1", "model2"),
      type = "ratio",
      weights_method = "loo",
      n_models = 2
    ),
    class = "ratiod_average"
  )

  output <- capture.output(print(avg_obj))
  expect_true(any(grepl("model averaging", output)))
  expect_true(any(grepl("Model weights:", output)))
})

test_that("fitted.ratiod_average extracts predictions", {
  avg_obj <- structure(
    list(
      predictions = data.frame(mean = 1:5)
    ),
    class = "ratiod_average"
  )

  preds <- fitted(avg_obj)
  expect_equal(nrow(preds), 5)
})

test_that("weights.ratiod_average extracts weights", {
  avg_obj <- structure(
    list(
      weights = c(0.6, 0.4)
    ),
    class = "ratiod_average"
  )

  w <- weights(avg_obj)
  expect_equal(w, c(0.6, 0.4))
})

test_that("get_predictions handles different types", {
  mock_model <- list(
    draws = list(
      ratio = matrix(1:20, nrow = 5, ncol = 4),
      eta_num = matrix(0.5, nrow = 5, ncol = 4),
      eta_denom = matrix(0.3, nrow = 5, ncol = 4),
      mu_num = matrix(2, nrow = 5, ncol = 4),
      mu_denom = matrix(3, nrow = 5, ncol = 4)
    )
  )
  class(mock_model) <- "ratiod_fit"

  # Type ratio with ratio available
  pred <- tulpaRatio:::get_predictions(mock_model, NULL, "ratio", FALSE)
  expect_equal(dim(pred), c(5, 4))

  # Type numerator
  pred_num <- tulpaRatio:::get_predictions(mock_model, NULL, "numerator", FALSE)
  expect_equal(dim(pred_num), c(5, 4))

  # Type denominator
  pred_denom <- tulpaRatio:::get_predictions(mock_model, NULL, "denominator", FALSE)
  expect_equal(dim(pred_denom), c(5, 4))
})

test_that("get_predictions falls back to eta when mu is NULL", {
  mock_model <- list(
    draws = list(
      eta_num = matrix(0.5, nrow = 5, ncol = 4),
      eta_denom = matrix(0.3, nrow = 5, ncol = 4)
    )
  )
  class(mock_model) <- "ratiod_fit"

  # Type numerator - should use exp(eta_num)
  pred_num <- tulpaRatio:::get_predictions(mock_model, NULL, "numerator", FALSE)
  expect_equal(dim(pred_num), c(5, 4))
  expect_equal(pred_num[1, 1], exp(0.5))

  # Type denominator - should use exp(eta_denom)
  pred_denom <- tulpaRatio:::get_predictions(mock_model, NULL, "denominator", FALSE)
  expect_equal(dim(pred_denom), c(5, 4))
  expect_equal(pred_denom[1, 1], exp(0.3))
})

test_that("get_predictions errors when unable to extract", {
  mock_model <- list(draws = list())
  class(mock_model) <- "ratiod_fit"

  expect_error(
    tulpaRatio:::get_predictions(mock_model, NULL, "numerator", FALSE),
    "Cannot extract"
  )
})

test_that("average_predictions works for ratio type", {
  pred1 <- matrix(2, nrow = 10, ncol = 5)
  pred2 <- matrix(4, nrow = 10, ncol = 5)
  predictions <- list(pred1, pred2)
  weights <- c(0.5, 0.5)

  # Ratio averaging on log scale
  avg <- tulpaRatio:::average_predictions(predictions, weights, "ratio", summary = FALSE)
  # exp(0.5*log(2) + 0.5*log(4)) = exp(0.5*0.693 + 0.5*1.386) = exp(1.0397) ~ 2.83
  expect_true(all(abs(avg - sqrt(2 * 4)) < 0.01))  # Geometric mean
})

test_that("average_predictions works for numerator type with summary", {
  pred1 <- matrix(rnorm(50, 2, 0.1), nrow = 10, ncol = 5)
  pred2 <- matrix(rnorm(50, 4, 0.1), nrow = 10, ncol = 5)
  predictions <- list(pred1, pred2)
  weights <- c(0.5, 0.5)

  avg <- tulpaRatio:::average_predictions(predictions, weights, "numerator", summary = TRUE)
  expect_true(is.data.frame(avg))
  expect_true("mean" %in% names(avg))
  expect_true("sd" %in% names(avg))
  expect_true("q2.5" %in% names(avg))
  expect_true("q97.5" %in% names(avg))
})

# -----------------------------------------------------------------------------
# backend_laplace.R coverage
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# priors.R coverage
# -----------------------------------------------------------------------------

test_that("prior_normal creates valid prior", {
  p <- prior_normal(0, 2.5)
  expect_s3_class(p, "ratiod_prior")
  expect_equal(p$dist, "normal")
  expect_equal(p$mean, 0)
  expect_equal(p$sd, 2.5)
})

test_that("prior_half_normal creates valid prior", {
  p <- prior_half_normal(2.5)
  expect_s3_class(p, "ratiod_prior")
  expect_equal(p$dist, "half_normal")
  expect_equal(p$sd, 2.5)
})

test_that("prior_half_cauchy creates valid prior", {
  p <- prior_half_cauchy(2.5)
  expect_s3_class(p, "ratiod_prior")
  expect_equal(p$dist, "half_cauchy")
  expect_equal(p$scale, 2.5)
})

test_that("prior_gamma creates valid prior", {
  p <- prior_gamma(2, 0.5)
  expect_s3_class(p, "ratiod_prior")
  expect_equal(p$dist, "gamma")
  expect_equal(p$shape, 2)
  expect_equal(p$rate, 0.5)
})

test_that("prior_beta creates valid prior", {
  p <- prior_beta(2, 2)
  expect_s3_class(p, "ratiod_prior")
  expect_equal(p$dist, "beta")
  expect_equal(p$alpha, 2)
  expect_equal(p$beta, 2)
})

test_that("prior_exponential creates valid prior", {
  p <- prior_exponential(1)
  expect_s3_class(p, "ratiod_prior")
  expect_equal(p$dist, "exponential")
  expect_equal(p$rate, 1)
})

test_that("prior_pc creates valid prior", {
  p <- prior_pc(U = 1.0, alpha = 0.01)
  expect_s3_class(p, "ratiod_prior")
  expect_equal(p$dist, "pc")
  expect_equal(p$U, 1.0)
  expect_equal(p$alpha, 0.01)
})

test_that("validate_prior rejects non-prior objects", {
  expect_error(
    tulpaRatio:::validate_prior(list(x = 1), "test"),
    "must be a prior object"
  )
})

test_that("ratiod_priors validates all priors", {
  expect_error(
    ratiod_priors(beta = list(x = 1)),
    "must be a prior object"
  )
})

# -----------------------------------------------------------------------------
# spatial.R coverage
# -----------------------------------------------------------------------------

test_that("spatial_car creates valid structure", {
  adj <- matrix(c(0, 1, 0, 1, 0, 1, 0, 1, 0), nrow = 3)
  spatial <- spatial_car(group_var = "site", adjacency = adj)

  expect_s3_class(spatial, "ratiod_spatial")
  expect_equal(spatial$type, "car")
  expect_equal(spatial$group_var, "site")
})

test_that("spatial_bym2 creates valid structure", {
  adj <- matrix(c(0, 1, 0, 1, 0, 1, 0, 1, 0), nrow = 3)
  spatial <- spatial_bym2(group_var = "site", adjacency = adj)

  expect_s3_class(spatial, "ratiod_spatial")
  expect_equal(spatial$type, "bym2")
})

test_that("spatial_gp creates valid structure", {
  gp <- spatial_gp(coords = c("lon", "lat"))

  expect_s3_class(gp, "ratiod_gp")
  # coords may be stored as coord_vars or similar
  expect_true("coord_vars" %in% names(gp) || "coords" %in% names(gp))
})

test_that("print.ratiod_spatial works", {
  adj <- matrix(c(0, 1, 0, 1, 0, 1, 0, 1, 0), nrow = 3)
  spatial <- spatial_car(group_var = "site", adjacency = adj)

  output <- capture.output(print(spatial))
  expect_true(length(output) > 0)
})

# -----------------------------------------------------------------------------
# temporal.R additional coverage
# -----------------------------------------------------------------------------

test_that("temporal_rw1 creates valid object", {
  temp <- temporal_rw1(time_var = "year")
  expect_s3_class(temp, "ratiod_temporal")
  expect_equal(temp$type, "rw1")
})

test_that("temporal_rw2 creates valid object", {
  temp <- temporal_rw2(time_var = "year")
  expect_s3_class(temp, "ratiod_temporal")
  expect_equal(temp$type, "rw2")
})

test_that("temporal_ar1 creates valid object with a correlation prior", {
  temp <- temporal_ar1(time_var = "year", rho_prior = prior_beta(9, 1))
  expect_s3_class(temp$rho_prior, "ratiod_prior")
  expect_equal(temp$rho_prior$alpha, 9)
})

# -----------------------------------------------------------------------------
# formula.R coverage
# -----------------------------------------------------------------------------

test_that("ratiod_formula parses basic formula", {
  df <- data.frame(
    count = rpois(10, 10),
    total = rpois(10, 100) + 10,
    x = rnorm(10)
  )

  formula <- ratiod_formula(count | total ~ x, data = df)
  expect_s3_class(formula, "ratiod_formula")
  expect_true("numerator" %in% names(formula))
  expect_true("denominator" %in% names(formula))
})

test_that("ratiod_formula rejects offset", {
  df <- data.frame(
    count = rpois(10, 10),
    total = rpois(10, 100) + 10,
    x = rnorm(10),
    effort = runif(10, 1, 5)
  )

  expect_error(
    ratiod_formula(count | total ~ x + offset(log(effort)), data = df),
    "offset"
  )
})

test_that("ratiod_formula parses random effects", {
  df <- data.frame(
    count = rpois(20, 10),
    total = rpois(20, 100) + 10,
    x = rnorm(20),
    site = factor(rep(1:5, each = 4))
  )

  formula <- ratiod_formula(count | total ~ x + (1 | site), data = df)
  expect_true(length(formula$numerator$random_effects) > 0)
})

# -----------------------------------------------------------------------------
# family.R coverage
# -----------------------------------------------------------------------------

test_that("all main families create valid objects", {
  fam1 <- ratiod_negbin_negbin()
  expect_s3_class(fam1, "ratiod_family")

  fam2 <- ratiod_binomial()
  expect_s3_class(fam2, "ratiod_family")

  fam3 <- ratiod_poisson_gamma()
  expect_s3_class(fam3, "ratiod_family")
})

test_that("print.ratiod_family works", {
  fam <- ratiod_negbin_negbin()
  output <- capture.output(print(fam))
  expect_true(length(output) > 0)
})

# -----------------------------------------------------------------------------
# ratio.R coverage
# -----------------------------------------------------------------------------

test_that("ratio_contrast validates inputs", {
  # ratio_contrast requires a contrast formula
  expect_error(ratio_contrast(NULL, NULL))
})
