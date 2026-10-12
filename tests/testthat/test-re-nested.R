# Tests for nested random effects: (1 | site/plot)

test_that("formula parsing expands nested RE syntax", {
  df <- data.frame(
    y = rpois(100, 10),
    n = rpois(100, 20),
    x = rnorm(100),
    site = factor(rep(1:5, each = 20)),
    plot = factor(rep(1:4, 25))
  )

  f <- ratiod_formula(y | n ~ x + (1 | site/plot), data = df)

  # Should expand to 2 RE terms
  expect_equal(length(f$numerator$random_effects), 2)
  expect_equal(f$numerator$random_effects[[1]]$group_var, "site")
  expect_equal(f$numerator$random_effects[[2]]$group_var, "site:plot")
  expect_equal(f$numerator$random_effects[[1]]$n_groups, 5)
  expect_equal(f$numerator$random_effects[[2]]$n_groups, 20)
})


test_that("formula parsing handles deeply nested RE", {
  df <- data.frame(
    y = rpois(200, 10),
    n = rpois(200, 20),
    x = rnorm(200),
    region = factor(rep(1:2, each = 100)),
    site = factor(rep(1:5, each = 40)),
    plot = factor(rep(1:10, 20))
  )

  f <- ratiod_formula(y | n ~ x + (1 | region/site/plot), data = df)

  # Should expand to 3 RE terms
  expect_equal(length(f$numerator$random_effects), 3)
  expect_equal(f$numerator$random_effects[[1]]$group_var, "region")
  expect_equal(f$numerator$random_effects[[2]]$group_var, "region:site")
  expect_equal(f$numerator$random_effects[[3]]$group_var, "region:site:plot")
})


test_that("nested RE produces correct number of groups", {
  # In a proper nested design, site:plot should have n_site * n_plots_per_site groups
  # With 5 sites and 4 plots per site = 20 unique site:plot combinations
  set.seed(123)
  df <- data.frame(
    y = rpois(100, 10),
    n = rpois(100, 20),
    site = factor(rep(1:5, each = 20)),
    plot = factor(rep(1:4, 25))  # 4 plots cycling across data
  )

  f <- ratiod_formula(y | n ~ (1 | site/plot), data = df)
  re_info <- tulpaRatio:::extract_re_for_hmc(f)

  expect_equal(re_info$n_re_terms, 2)
  expect_equal(re_info$re_terms[[1]]$n_groups, 5)
  # The interaction creates 20 unique levels (5 sites * 4 plots per site)
  expect_equal(re_info$re_terms[[2]]$n_groups, 20)
})


test_that("nested RE metadata is correct", {
  df <- data.frame(
    y = rpois(100, 10),
    n = rpois(100, 20),
    site = factor(rep(1:5, each = 20)),
    plot = factor(rep(1:4, 25))
  )

  f <- ratiod_formula(y | n ~ (1 | site/plot), data = df)

  expect_equal(f$numerator$random_effects[[1]]$nested_vars, "site")
  expect_equal(f$numerator$random_effects[[1]]$group_var, "site")
  expect_equal(f$numerator$random_effects[[2]]$nested_vars, c("site", "plot"))
  expect_equal(f$numerator$random_effects[[2]]$group_var, "site:plot")
})


test_that("mixed nested and crossed RE works", {
  df <- data.frame(
    y = rpois(100, 10),
    n = rpois(100, 20),
    x = rnorm(100),
    site = factor(rep(1:5, each = 20)),
    plot = factor(rep(1:4, 25)),
    year = factor(rep(1:5, 20))
  )

  f <- ratiod_formula(y | n ~ x + (1 | site/plot) + (1 | year), data = df)

  # Should have 3 RE terms: site, site:plot, year
  expect_equal(length(f$numerator$random_effects), 3)
  expect_equal(f$numerator$random_effects[[1]]$group_var, "site")
  expect_equal(f$numerator$random_effects[[2]]$group_var, "site:plot")
  expect_equal(f$numerator$random_effects[[3]]$group_var, "year")
})


test_that("extract_re_for_hmc handles nested RE", {
  df <- data.frame(
    y = rpois(100, 10),
    n = rpois(100, 20),
    site = factor(rep(1:5, each = 20)),
    plot = factor(rep(1:4, 25))
  )

  f <- ratiod_formula(y | n ~ (1 | site/plot), data = df)
  re_info <- tulpaRatio:::extract_re_for_hmc(f)

  expect_equal(re_info$n_re_terms, 2)
  expect_equal(re_info$total_groups, 25)  # 5 + 20

  # Check offsets (field renamed to re_offset for slopes support)
  expect_equal(re_info$re_terms[[1]]$re_offset, 0)
  expect_equal(re_info$re_terms[[2]]$re_offset, 5)
})


test_that("prepare_hmc_data includes nested RE info", {
  df <- data.frame(
    y = rpois(100, 10),
    n = rpois(100, 20),
    site = factor(rep(1:5, each = 20)),
    plot = factor(rep(1:4, 25))
  )

  f <- ratiod_formula(y | n ~ (1 | site/plot), data = df)
  family <- ratiod_binomial()

  hmc_data <- tulpaRatio:::prepare_hmc_data(f, df, family, "binomial")

  expect_equal(hmc_data$n_re_terms, 2)
  expect_equal(hmc_data$total_re_groups, 25)
  expect_equal(length(hmc_data$re_terms), 2)
})


# Integration tests

test_that("HMC fits model with nested RE", {
  skip_on_cran()

  set.seed(12345)
  df <- data.frame(
    y = rpois(100, 10),
    n = rpois(100, 20) + 10,
    x = rnorm(100),
    site = factor(rep(1:5, each = 20)),
    plot = factor(rep(1:4, 25))
  )

  fit <- tratio(
    y | n ~ x + (1 | site/plot),
    data = df,
    family = ratiod_binomial(),
    mode = "hmc",
    control = list(iter = 100, warmup = 50, chains = 1, verbose = FALSE)
  )

  expect_s3_class(fit, "ratiod_fit")
  expect_true("samples" %in% names(fit))

  param_names <- colnames(fit$samples[[1]])

  # Should have 2 sigma_re parameters
  sigma_re_params <- grep("^sigma_re", param_names, value = TRUE)
  expect_equal(length(sigma_re_params), 2)

  # Should have 25 RE effects (5 sites + 20 site:plot combinations)
  re_params <- grep("^re\\[", param_names, value = TRUE)
  expect_equal(length(re_params), 25)

  # Check structure: re[1,] for site, re[2,] for site:plot
  re1_params <- grep("^re\\[1,", param_names, value = TRUE)
  re2_params <- grep("^re\\[2,", param_names, value = TRUE)
  expect_equal(length(re1_params), 5)
  expect_equal(length(re2_params), 20)
})


test_that("HMC fits deeply nested model", {
  skip_on_cran()

  set.seed(54321)
  df <- data.frame(
    y = rpois(200, 10),
    n = rpois(200, 20) + 10,
    x = rnorm(200),
    region = factor(rep(1:2, each = 100)),
    site = factor(rep(1:5, each = 40)),
    plot = factor(rep(1:10, 20))
  )

  fit <- tratio(
    y | n ~ x + (1 | region/site/plot),
    data = df,
    family = ratiod_binomial(),
    mode = "hmc",
    control = list(iter = 100, warmup = 50, chains = 1, verbose = FALSE)
  )

  expect_s3_class(fit, "ratiod_fit")

  param_names <- colnames(fit$samples[[1]])

  # Should have 3 sigma_re parameters
  sigma_re_params <- grep("^sigma_re", param_names, value = TRUE)
  expect_equal(length(sigma_re_params), 3)
})


test_that("mixed nested and crossed RE fits", {
  skip_on_cran()

  set.seed(98765)
  df <- data.frame(
    y = rpois(100, 10),
    n = rpois(100, 20) + 10,
    x = rnorm(100),
    site = factor(rep(1:5, each = 20)),
    plot = factor(rep(1:4, 25)),
    year = factor(rep(1:5, 20))
  )

  fit <- tratio(
    y | n ~ x + (1 | site/plot) + (1 | year),
    data = df,
    family = ratiod_binomial(),
    mode = "hmc",
    control = list(iter = 100, warmup = 50, chains = 1, verbose = FALSE)
  )

  expect_s3_class(fit, "ratiod_fit")

  param_names <- colnames(fit$samples[[1]])

  # Should have 3 sigma_re parameters (site, site:plot, year)
  sigma_re_params <- grep("^sigma_re", param_names, value = TRUE)
  expect_equal(length(sigma_re_params), 3)

  # Should have 30 RE effects (5 + 20 + 5)
  re_params <- grep("^re\\[", param_names, value = TRUE)
  expect_equal(length(re_params), 30)
})
