test_that("combined formula syntax works: num | denom ~ x", {
  df <- data.frame(
    y_num = rpois(100, 10),
    y_denom = rpois(100, 20),
    x = rnorm(100),
    site = rep(1:10, each = 10)
  )

  f <- ratiod_formula(
    formula = y_num | y_denom ~ x + (1 | site),
    data = df
  )

  expect_s3_class(f, "ratiod_formula")
  expect_equal(f$numerator$response_var, "y_num")
  expect_equal(f$denominator$response_var, "y_denom")
  expect_equal(ncol(f$numerator$X), 2)  # intercept + x
  expect_equal(ncol(f$denominator$X), 2)  # same predictors
  expect_length(f$numerator$random_effects, 1)
  expect_length(f$denominator$random_effects, 1)
})

test_that("combined formula with additional process-specific terms", {
  df <- data.frame(
    y_num = rpois(100, 10),
    y_denom = rpois(100, 20),
    x = rnorm(100),
    z = rnorm(100),
    site = rep(1:10, each = 10)
  )

  f <- ratiod_formula(
    formula = y_num | y_denom ~ (1 | site),
    formula_num = ~ x,
    formula_denom = ~ z,
    data = df
  )

  # Numerator: intercept + x
  expect_equal(ncol(f$numerator$X), 2)
  expect_true("x" %in% colnames(f$numerator$X))

  # Denominator: intercept + z
  expect_equal(ncol(f$denominator$X), 2)
  expect_true("z" %in% colnames(f$denominator$X))
})

test_that("separate formula syntax works", {
  df <- data.frame(
    y_num = rpois(100, 10),
    y_denom = rpois(100, 20),
    x = rnorm(100),
    site = rep(1:10, each = 10)
  )

  f <- ratiod_formula(
    formula = y_num ~ x + (1 | site),
    formula_denom = y_denom ~ (1 | site),
    data = df
  )

  expect_equal(f$numerator$response_var, "y_num")
  expect_equal(f$denominator$response_var, "y_denom")
  expect_equal(ncol(f$numerator$X), 2)  # intercept + x
  expect_equal(ncol(f$denominator$X), 1)  # intercept only
})

test_that("separate formula requires formula_denom", {
  df <- data.frame(
    y_num = rpois(100, 10),
    x = rnorm(100)
  )

  expect_error(
    ratiod_formula(
      formula = y_num ~ x,
      data = df
    ),
    "formula_denom.*required"
  )
})

test_that("offset() is rejected", {
  df <- data.frame(
    y_num = rpois(100, 10),
    y_denom = rpois(100, 20),
    x = rnorm(100),
    effort = runif(100, 1, 10)
  )

  expect_error(
    ratiod_formula(
      formula = y_num | y_denom ~ x + offset(log(effort)),
      data = df
    ),
    "offset\\(\\) is not allowed"
  )
})

test_that("shared = ~ 0 triggers warning", {
  df <- data.frame(
    y_num = rpois(100, 10),
    y_denom = rpois(100, 20),
    x = rnorm(100),
    site = rep(1:10, each = 10)
  )

  expect_warning(
    ratiod_formula(
      formula = y_num | y_denom ~ x + (1 | site),
      shared = ~ 0,
      data = df
    ),
    "independence"
  )
})

test_that("random effects are correctly inferred as shared", {
  df <- data.frame(
    y_num = rpois(100, 10),
    y_denom = rpois(100, 20),
    x = rnorm(100),
    site = rep(1:10, each = 10)
  )

  f <- ratiod_formula(
    formula = y_num | y_denom ~ x + (1 | site),
    data = df
  )

  # Site should be inferred as shared (appears in both)
  expect_equal(f$shared$type, "inferred")
  expect_length(f$shared$random_effects, 1)
})

test_that("different RE in each process are handled", {
  df <- data.frame(
    y_num = rpois(100, 10),
    y_denom = rpois(100, 20),
    x = rnorm(100),
    site = rep(1:10, each = 10),
    observer = rep(1:5, 20)
  )

  f <- ratiod_formula(
    formula = y_num ~ x + (1 | site) + (1 | observer),
    formula_denom = y_denom ~ (1 | site),
    data = df
  )

  # Numerator has 2 RE, denominator has 1
  expect_length(f$numerator$random_effects, 2)
  expect_length(f$denominator$random_effects, 1)

  # Only site is shared
  expect_equal(f$shared$type, "inferred")
  expect_length(f$shared$random_effects, 1)
  expect_equal(f$shared$random_effects[[1]]$group_var, "site")
})

test_that("model matrix is built correctly with factors", {
  df <- data.frame(
    y = rpois(100, 10),
    n = rpois(100, 20),
    x1 = rnorm(100),
    x2 = rnorm(100),
    cat = factor(rep(c("A", "B"), 50))
  )

  f <- ratiod_formula(
    formula = y | n ~ x1 + x2 + cat,
    data = df
  )

  # Intercept + x1 + x2 + catB
  expect_equal(ncol(f$numerator$X), 4)
  expect_true("(Intercept)" %in% colnames(f$numerator$X))
  expect_true("catB" %in% colnames(f$numerator$X))
})

test_that("missing variables trigger errors", {
  df <- data.frame(
    y_num = rpois(100, 10),
    x = rnorm(100)
  )

  expect_error(
    ratiod_formula(
      formula = y_num | missing_var ~ x,
      data = df
    ),
    "not found in data"
  )
})

test_that("(1 | a:b) groups by the interaction, as the second term of (1 | a/b)", {
  df <- data.frame(site = factor(rep(1:3, each = 4)), plot = factor(rep(1:2, 6)))
  inter <- tulpaRatio:::extract_random_effects("(1 | site:plot)", df)
  nested <- tulpaRatio:::extract_random_effects("(1 | site/plot)", df)
  expect_length(inter, 1L)
  expect_identical(inter[[1]]$group_var, "site:plot")
  expect_identical(inter[[1]]$group, nested[[2]]$group)
  expect_identical(inter[[1]]$n_groups, 6L)
  expect_error(tulpaRatio:::extract_random_effects("(1 | site:block)", df),
               "'block' not found")
})

test_that("random-effect terms are removed without touching inline transforms", {
  rre <- function(f) deparse(tulpaRatio:::remove_random_effects(f))
  expect_identical(rre(y ~ x + I(x^2) + (1 | site)), "y ~ x + I(x^2)")
  expect_identical(rre(y ~ log(x) + (1 + x || site) + poly(z, 2)), "y ~ log(x) + poly(z, 2)")
  expect_identical(rre(y ~ (1 | site)), "y ~ 1")
  expect_identical(rre(y ~ 0 + x + (1 | a/b)), "y ~ 0 + x")
})

test_that("a formula with transforms and a random effect fits", {
  skip_on_cran()
  set.seed(2)
  df <- data.frame(count = rpois(60, 20), effort = rgamma(60, 10, 1),
                   x = runif(60, 1, 3), site = factor(rep(1:6, 10)))
  fit <- tratio(count | effort ~ log(x) + I(x^2) + (1 | site), data = df,
                family = ratiod_poisson_gamma(),
                control = list(iter = 200, chains = 1, seed = 1))
  expect_named(coef(fit), c("num:(Intercept)", "num:log(x)", "num:I(x^2)",
                            "denom:(Intercept)", "denom:log(x)", "denom:I(x^2)"))
})
