expect_draw_accessors <- function(fit) {
  wide <- draws_wide(fit, beta_num)
  expect_s3_class(wide, "ratiod_draws")
  expect_true(all(c(".chain", ".iteration", ".draw") %in% names(wide)))
  expect_true(any(grepl("beta_num", names(wide))))
  expect_equal(nrow(draws_wide(fit, beta_num, ndraws = 50)), 50)
  expect_error(draws_wide(fit, nonexistent_param), "No parameters matched")
  long <- draws_long(fit, beta_num)
  expect_s3_class(long, "ratiod_draws_long")
  expect_true(all(c(".variable", ".value") %in% names(long)))
  expect_true(all(grepl("beta_num", long$.variable)))
  interval <- draws_interval(fit, beta_num, beta_denom)
  expect_s3_class(interval, "ratiod_draws_interval")
  expect_true(all(c(".variable", ".value", ".lower", ".upper", "probs") %in% names(interval)))
  widths <- unique(draws_interval(fit, beta_num, probs = c(0.5, 0.9))$probs)
  expect_equal(sort(widths), c(0.5, 0.9))
  median <- draws_interval(fit, beta_num, .point = "median")
  mean <- draws_interval(fit, beta_num, .point = "mean")
  expect_equal(unique(median$.point), "median")
  expect_equal(unique(mean$.point), "mean")
  expect_false(identical(median$.value[1], mean$.value[1]))
  expect_equal(unique(draws_interval(fit, beta_num, .interval = "qi")$.interval), "qi")
  expect_equal(unique(draws_interval(fit, beta_num, .interval = "hdi")$.interval), "hdi")
}

fixture_average <- function(predictions = data.frame(mean = 1:3, sd = 0.1),
                            weights = c(model1 = 0.6, model2 = 0.4),
                            models = names(weights), type = "ratio", weights_method = "loo") {
  structure(list(predictions = predictions, weights = weights, models = models,
                 type = type, weights_method = weights_method, n_models = length(weights)),
            class = "ratiod_average")
}
