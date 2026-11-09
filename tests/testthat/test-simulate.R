# Tests for simulation functions (R/simulate.R)

test_that("sim_ratiod generates valid data for negbin_negbin family", {
  sim <- sim_ratiod(
    n = 50,
    family = ratiod_negbin_negbin(),
    beta_num = c(2, 0.5),
    beta_denom = c(1.5, 0.3),
    sigma_re = 0.4,
    phi_num = 5,
    phi_denom = 4,
    n_groups = 5,
    seed = 123
  )

  expect_s3_class(sim, "ratiod_simdata")
  expect_equal(nrow(sim$data), 50)
  expect_true(all(c("y_num", "y_denom", "x1", "x2", "group") %in% names(sim$data)))

  # Check responses are valid counts

  expect_true(all(sim$data$y_num >= 0))
  expect_true(all(sim$data$y_num == floor(sim$data$y_num)))
  expect_true(all(sim$data$y_denom >= 1))
  expect_true(all(sim$data$y_denom == floor(sim$data$y_denom)))

  # Check true params stored
  expect_equal(sim$true_params$beta_num, c(2, 0.5))
  expect_equal(sim$true_params$beta_denom, c(1.5, 0.3))
  expect_equal(sim$true_params$sigma_re, 0.4)
  expect_equal(sim$true_params$phi_num, 5)
  expect_equal(sim$true_params$phi_denom, 4)
  expect_equal(length(sim$true_params$re), 5)
})

test_that("sim_ratiod generates valid data for binomial family", {
  sim <- sim_ratiod(
    n = 40,
    family = ratiod_binomial(),
    beta_num = c(-0.5, 1),
    sigma_re = 0.3,
    n_groups = 8,
    seed = 456
  )

  expect_s3_class(sim, "ratiod_simdata")
  expect_equal(nrow(sim$data), 40)

  # Check binomial responses are valid
  expect_true(all(sim$data$y_num >= 0))
  expect_true(all(sim$data$y_num <= sim$data$y_denom))
  expect_true(all(sim$data$y_denom > 0))
})

test_that("sim_ratiod generates valid data for poisson_gamma family", {
  sim <- sim_ratiod(
    n = 60,
    family = ratiod_poisson_gamma(),
    beta_num = c(2, 0.3),
    beta_denom = c(1, 0.2),
    sigma_re = 0.5,
    phi_denom = 3,
    n_groups = 6,
    seed = 789
  )

  expect_s3_class(sim, "ratiod_simdata")
  expect_equal(nrow(sim$data), 60)

  # Numerator: Poisson counts
  expect_true(all(sim$data$y_num >= 0))
  expect_true(all(sim$data$y_num == floor(sim$data$y_num)))

  # Denominator: positive continuous (Gamma)
  expect_true(all(sim$data$y_denom > 0))
})

test_that("sim_ratiod generates valid data for negbin_gamma family", {
  sim <- sim_ratiod(
    n = 60,
    family = ratiod_negbin_gamma(),
    beta_num = c(2, 0.3),
    beta_denom = c(1, 0.2),
    sigma_re = 0.5,
    phi_num = 5,
    phi_denom = 3,
    n_groups = 6,
    seed = 321
  )

  expect_s3_class(sim, "ratiod_simdata")
  expect_equal(nrow(sim$data), 60)

  # Numerator: integer counts (NegBin)
  expect_true(all(sim$data$y_num >= 0))
  expect_true(all(sim$data$y_num == floor(sim$data$y_num)))

  # Denominator: positive continuous (Gamma)
  expect_true(all(sim$data$y_denom > 0))

  # True params stored
  expect_equal(sim$true_params$phi_num, 5)
  expect_equal(sim$true_params$phi_denom, 3)
})

test_that("sim_ratiod uses default coefficients when not specified", {
  sim <- sim_ratiod(
    n = 30,
    family = ratiod_negbin_negbin(),
    n_groups = 5,
    seed = 111
  )

  # Should use defaults
  expect_equal(sim$true_params$beta_num, c(2, 0.3))
  expect_equal(sim$true_params$beta_denom, c(1.5, 0.2))
  expect_equal(sim$true_params$sigma_re, 0.5)
})

test_that("sim_ratiod handles balanced groups with n_per_group", {
  sim <- sim_ratiod(
    n = 100,  # will be adjusted to 50
    family = ratiod_negbin_negbin(),
    n_groups = 10,
    n_per_group = 5,
    seed = 222
  )

  # Check balanced groups
  expect_equal(nrow(sim$data), 50)
  group_counts <- table(sim$data$group)
  expect_true(all(group_counts == 5))
})

test_that("sim_ratiod print method works", {
  sim <- sim_ratiod(
    n = 20,
    family = ratiod_poisson_gamma(),
    n_groups = 4,
    seed = 333
  )

  # Capture print output
  output <- capture.output(print(sim))
  expect_true(any(grepl("Simulated tulpaRatio data", output)))
  expect_true(any(grepl("Family:", output)))
  expect_true(any(grepl("Observations:", output)))
  expect_true(any(grepl("Groups:", output)))
  expect_true(any(grepl("True parameters:", output)))
})

test_that("sim_ratiod reproducibility with seed",
 {
  sim1 <- sim_ratiod(n = 30, family = ratiod_negbin_negbin(), seed = 12345)
  sim2 <- sim_ratiod(n = 30, family = ratiod_negbin_negbin(), seed = 12345)

  expect_equal(sim1$data$y_num, sim2$data$y_num)
  expect_equal(sim1$data$y_denom, sim2$data$y_denom)
  expect_equal(sim1$true_params$re, sim2$true_params$re)
})

test_that("sim_ratiod_sbc generates multiple datasets", {
  sims <- sim_ratiod_sbc(
    n_sims = 5,
    n = 30,
    family = ratiod_negbin_negbin(),
    n_groups = 4,
    seed = 444
  )

  expect_s3_class(sims, "ratiod_sbc_sims")
  expect_equal(length(sims), 5)
  expect_equal(attr(sims, "n_sims"), 5)

  # Each element should be a ratiod_simdata
  for (sim in sims) {
    expect_s3_class(sim, "ratiod_simdata")
    expect_equal(nrow(sim$data), 30)
    expect_equal(sim$n_groups, 4)
  }

  # Each simulation should have different true params (drawn from prior)
  beta1 <- sims[[1]]$true_params$beta_num
  beta2 <- sims[[2]]$true_params$beta_num
  expect_false(all(beta1 == beta2))
})

test_that("sim_ratiod_sbc print method works", {
  sims <- sim_ratiod_sbc(
    n_sims = 3,
    n = 20,
    family = ratiod_binomial(),
    n_groups = 3,
    seed = 555
  )

  output <- capture.output(print(sims))
  expect_true(any(grepl("tulpaRatio SBC simulations", output)))
  expect_true(any(grepl("Simulations:", output)))
  expect_true(any(grepl("Family:", output)))
})

test_that("sim_ratiod generates valid data", {
  skip_on_cran()

  # Test with default parameters
  data <- sim_ratiod(
    n = 50,
    family = ratiod_negbin_negbin()
  )

  expect_s3_class(data, "ratiod_simdata")
  expect_true("data" %in% names(data))
  expect_equal(nrow(data$data), 50)
})

test_that("sim_ratiod works with random effects", {
  skip_on_cran()

  data <- sim_ratiod(
    n = 60,
    n_groups = 6,
    family = ratiod_negbin_negbin()
  )

  expect_s3_class(data, "ratiod_simdata")
  expect_true("group" %in% names(data$data) || "site" %in% names(data$data))
})
