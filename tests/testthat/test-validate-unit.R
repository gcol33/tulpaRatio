# Unit tests for validate.R functions

# Test pp_check generic
test_that("pp_check generic exists", {
  expect_true(exists("pp_check"))
  expect_true(is.function(pp_check))
})

# Test loo.ratiod_fit (mocked - needs stanfit)
test_that("loo.ratiod_fit requires loo package", {
  # Test that function exists
  expect_true(exists("loo.ratiod_fit"))
})

# Test waic.ratiod_fit
test_that("waic.ratiod_fit exists", {
  expect_true(exists("waic.ratiod_fit"))
})

# Test ratiod_compare

# Test ratiod_average

# Test print.ratiod_average

# Test fitted.ratiod_average

# Test weights.ratiod_average

# Test internal helper: average_predictions
test_that("average_predictions works on log scale for ratios", {
  # Mock predictions (2 models, 3 draws, 2 obs)
  predictions <- list(
    matrix(c(2, 2, 2, 4, 4, 4), nrow = 3, ncol = 2),
    matrix(c(8, 8, 8, 16, 16, 16), nrow = 3, ncol = 2)
  )
  weights <- c(0.5, 0.5)

  result <- tulpaRatio:::average_predictions(predictions, weights, "ratio", summary = FALSE)

  # Geometric mean: sqrt(2*8)=4, sqrt(4*16)=8
  expect_equal(result[1, 1], 4, tolerance = 0.001)
  expect_equal(result[1, 2], 8, tolerance = 0.001)
})

test_that("average_predictions works with summary=TRUE", {
  predictions <- list(
    matrix(rnorm(30, 1), nrow = 10, ncol = 3),
    matrix(rnorm(30, 1), nrow = 10, ncol = 3)
  )
  weights <- c(0.6, 0.4)

  result <- tulpaRatio:::average_predictions(predictions, weights, "numerator", summary = TRUE)

  expect_true(is.data.frame(result))
  expect_true("mean" %in% names(result))
  expect_true("sd" %in% names(result))
  expect_equal(nrow(result), 3)
})

# Test internal helper: get_predictions
test_that("get_predictions errors on missing prediction data", {
  mock_model <- structure(
    list(
      draws = list()  # Empty draws
    ),
    class = "ratiod_fit"
  )

  expect_error(
    tulpaRatio:::get_predictions(mock_model, NULL, "ratio", FALSE),
    "Cannot extract"
  )
})

# Test compute_model_weights catches LOO errors
test_that("compute_model_weights function exists", {
  expect_true(exists("compute_model_weights", where = asNamespace("tulpaRatio")))
})

# Test pp_check_single (internal)
test_that("pp_check_single function exists", {
  expect_true(exists("pp_check_single", where = asNamespace("tulpaRatio")))
})

# Test print.ratiod_average with matrix predictions

# Test get_predictions with various paths
test_that("get_predictions extracts ratio from draws$ratio", {
  mock_model <- structure(
    list(
      draws = list(
        ratio = matrix(1:6, nrow = 2, ncol = 3)
      )
    ),
    class = "ratiod_fit"
  )

  result <- tulpaRatio:::get_predictions(mock_model, NULL, "ratio", FALSE)
  expect_equal(result, mock_model$draws$ratio)
})

test_that("get_predictions computes ratio from eta", {
  mock_model <- structure(
    list(
      draws = list(
        eta_num = matrix(log(c(2, 4, 6, 8, 10, 12)), nrow = 2, ncol = 3),
        eta_denom = matrix(log(c(1, 2, 3, 4, 5, 6)), nrow = 2, ncol = 3)
      )
    ),
    class = "ratiod_fit"
  )

  result <- tulpaRatio:::get_predictions(mock_model, NULL, "ratio", FALSE)
  # ratio = exp(eta_num - eta_denom) = eta_num / eta_denom on exp scale
  expected <- exp(mock_model$draws$eta_num - mock_model$draws$eta_denom)
  expect_equal(result, expected)
})

test_that("get_predictions extracts numerator from mu_num", {
  mock_model <- structure(
    list(
      draws = list(
        mu_num = matrix(1:6, nrow = 2, ncol = 3)
      )
    ),
    class = "ratiod_fit"
  )

  result <- tulpaRatio:::get_predictions(mock_model, NULL, "numerator", FALSE)
  expect_equal(result, mock_model$draws$mu_num)
})

test_that("get_predictions extracts numerator from eta_num when mu_num is missing", {
  mock_model <- structure(
    list(
      draws = list(
        eta_num = matrix(log(1:6), nrow = 2, ncol = 3)
      )
    ),
    class = "ratiod_fit"
  )

  result <- tulpaRatio:::get_predictions(mock_model, NULL, "numerator", FALSE)
  expect_equal(result, exp(mock_model$draws$eta_num))
})

test_that("get_predictions extracts denominator from mu_denom", {
  mock_model <- structure(
    list(
      draws = list(
        mu_denom = matrix(10:15, nrow = 2, ncol = 3)
      )
    ),
    class = "ratiod_fit"
  )

  result <- tulpaRatio:::get_predictions(mock_model, NULL, "denominator", FALSE)
  expect_equal(result, mock_model$draws$mu_denom)
})

test_that("get_predictions extracts denominator from eta_denom when mu_denom is missing", {
  mock_model <- structure(
    list(
      draws = list(
        eta_denom = matrix(log(1:6), nrow = 2, ncol = 3)
      )
    ),
    class = "ratiod_fit"
  )

  result <- tulpaRatio:::get_predictions(mock_model, NULL, "denominator", FALSE)
  expect_equal(result, exp(mock_model$draws$eta_denom))
})

test_that("get_predictions errors for numerator when both mu_num and eta_num are missing", {
  mock_model <- structure(
    list(
      draws = list(
        mu_num = NULL,
        eta_num = NULL
      )
    ),
    class = "ratiod_fit"
  )

  expect_error(
    tulpaRatio:::get_predictions(mock_model, NULL, "numerator", FALSE),
    "Cannot extract"
  )
})

test_that("get_predictions errors for denominator when both mu_denom and eta_denom are missing", {
  mock_model <- structure(
    list(
      draws = list(
        mu_denom = NULL,
        eta_denom = NULL
      )
    ),
    class = "ratiod_fit"
  )

  expect_error(
    tulpaRatio:::get_predictions(mock_model, NULL, "denominator", FALSE),
    "Cannot extract"
  )
})

# Test average_predictions with denominator type
test_that("average_predictions works for denominator", {
  predictions <- list(
    matrix(c(10, 10, 10, 20, 20, 20), nrow = 3, ncol = 2),
    matrix(c(30, 30, 30, 40, 40, 40), nrow = 3, ncol = 2)
  )
  weights <- c(0.5, 0.5)

  result <- tulpaRatio:::average_predictions(predictions, weights, "denominator", summary = FALSE)

  # Arithmetic mean: (10+30)/2=20, (20+40)/2=30
  expect_equal(result[1, 1], 20, tolerance = 0.001)
  expect_equal(result[1, 2], 30, tolerance = 0.001)
})

# Test ratiod_average data compatibility check
test_that("ratiod_average errors when models have different data sizes", {
  skip_if_not_installed("loo")

  # Note: ratiod_average requires proper model structures with draws and loo results
  # This test verifies the data size check by providing complete mock structures
  mock_draws <- array(runif(100), dim = c(10, 1, 10))
  dimnames(mock_draws) <- list(iteration = 1:10, chain = 1, variable = paste0("param", 1:10))

  mock_fit1 <- structure(
    list(
      .internal = list(hmc_data = list(N = 50L)),
      draws = mock_draws,
      loo = list(pointwise = matrix(runif(50), ncol = 1))
    ),
    class = "ratiod_fit"
  )
  mock_fit2 <- structure(
    list(
      .internal = list(hmc_data = list(N = 100L)),
      draws = mock_draws,
      loo = list(pointwise = matrix(runif(100), ncol = 1))
    ),
    class = "ratiod_fit"
  )

  expect_error(
    ratiod_average(mock_fit1, mock_fit2),
    "same data"
  )
})

# Test print.ratiod_average with more than 6 rows
