







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
test_that("model comparison and averaging validate model lists", {
  fit <- structure(list(), class = "ratiod_fit")
  for (fn in list(ratiod_compare, ratiod_average)) {
    expect_error(fn(), "At least two models required")
    expect_error(fn(fit), "At least two models required")
    for (invalid in list("a", list(a = 1), structure(list(), class = "not_ratiod"))) {
      expect_error(fn(invalid, invalid), "must be ratiod_fit models")
    }
  }
})

test_that("averaged model methods preserve predictions and weights", {
  for (predictions in list(data.frame(mean = 1:3, sd = 0.1),
                           matrix(1:20, nrow = 10, ncol = 2))) {
    fit <- fixture_average(predictions)
    expect_identical(fitted(fit), predictions)
    expect_identical(weights(fit), fit$weights)
    output <- capture.output(print(fit))
    expect_true(any(grepl("ratio model averaging", output)))
    expect_true(any(grepl("Method:", output)))
    expect_true(any(grepl("Model weights", output)))
    expect_true(any(grepl("model1", output)))
    if (is.matrix(predictions)) expect_true(any(grepl("draws x", output)))
  }
})

test_that("averaged model printing reports table truncation", {
  fit <- fixture_average(data.frame(mean = seq_len(20), sd = 0.1))
  output <- capture.output(print(fit))
  expect_true(any(grepl("more", output)))
})
