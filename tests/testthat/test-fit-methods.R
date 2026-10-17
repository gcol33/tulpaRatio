methods_fit <- function() {
  sim <- sim_ratiod(n = 60, family = ratiod_poisson_gamma(), seed = 2)
  tratio(y_num | y_denom ~ x1 + (1 | group), data = sim$data,
         family = ratiod_poisson_gamma(),
         control = list(iter = 300, chains = 2, seed = 1))
}

test_that("an HMC fit records the Hamiltonian of every saved iteration", {
  skip_on_cran()
  fit <- methods_fit()
  energy <- fit$diagnostics$energy
  expect_length(energy, nrow(fit$draws))
  expect_true(all(is.finite(energy)))
  expect_equal(as.vector(table(fit$diagnostics$energy_chain)), c(150L, 150L))
  bfmi <- tulpa::energy_bfmi(energy, fit$diagnostics$energy_chain)$value
  expect_equal(diagnostic_summary(fit)$e_bfmi, bfmi)
  expect_s3_class(plot_energy(fit), "ggplot")
})

test_that("as.data.frame(), coef(), vcov() and formula() read a fit", {
  skip_on_cran()
  fit <- methods_fit()
  draws <- as.data.frame(fit)
  expect_s3_class(draws, "draws_df")
  expect_equal(nrow(draws), nrow(fit$draws))
  b <- coef(fit)
  expect_named(b, c("num:(Intercept)", "num:x1", "denom:(Intercept)", "denom:x1"))
  expect_equal(unname(b[["num:x1"]]), mean(fit$draws[, "beta_num[2]"]))
  expect_equal(dim(vcov(fit)), c(4L, 4L))
  expect_identical(deparse(formula(fit)), "y_num | y_denom ~ x1 + (1 | group)")

  ci <- confint(fit)
  expect_identical(rownames(ci), names(b))
  expect_identical(colnames(ci), c("2.5 %", "97.5 %"))
  expect_equal(ci["num:x1", ],
               stats::quantile(fit$draws[, "beta_num[2]"], c(0.025, 0.975), names = FALSE),
               ignore_attr = TRUE)
  expect_identical(colnames(confint(fit, "num:x1", level = 0.9)), c("5 %", "95 %"))
  expect_identical(rownames(confint(fit, 2)), "num:x1")

  # plot() takes a fixed effect by the name coef() gives it
  pdf(NULL)
  on.exit(grDevices::dev.off(), add = TRUE)
  expect_no_error(plot(fit, pars = c("num:x1", "beta_denom[1]")))
  expect_error(plot(fit, pars = "num:nothing"), "No parameters selected")
})

test_that("validate_mode() accepts the backend a fit ran on", {
  skip_on_cran()
  fit <- methods_fit()
  expect_true(validate_mode(fit, "hmc"))
  expect_true(validate_mode(fit, "exact"))
  expect_false(validate_mode(fit, "laplace", error = FALSE))
})

test_that("ratio_contrast() compares factor levels and refuses a numeric covariate", {
  skip_on_cran()
  set.seed(3)
  df <- data.frame(count = rpois(40, 20), effort = rgamma(40, 10, 1),
                   season = factor(rep(c("winter", "summer"), 20),
                                   levels = c("winter", "summer")),
                   x = rnorm(40))
  fit <- tratio(count | effort ~ season + x, data = df,
                family = ratiod_poisson_gamma(),
                control = list(iter = 300, chains = 1, seed = 1))
  ct <- ratio_contrast(fit, ~ season)
  expect_identical(as.character(ct$contrast), "summer vs winter")
  expect_error(ratio_contrast(fit, ~ x), "factor()", fixed = TRUE)
})

test_that("summary() reports every fixed effect", {
  skip_on_cran()
  set.seed(4)
  df <- as.data.frame(matrix(rnorm(60 * 8), 60, 8))
  df$count <- rpois(60, 20)
  df$effort <- rgamma(60, 10, 1)
  fit <- tratio(count | effort ~ V1 + V2 + V3 + V4 + V5 + V6 + V7 + V8, data = df,
                family = ratiod_poisson_gamma(),
                control = list(iter = 200, chains = 1, seed = 1))
  params <- coef(summary(fit))$parameter
  expect_true(all(sprintf("beta_num[%d]", 1:9) %in% params))
  expect_true(all(sprintf("beta_denom[%d]", 1:9) %in% params))
})

test_that("coef(), vcov() and summary() cover a zero-inflated binomial fit", {
  skip_on_cran()
  set.seed(123)
  n <- 60
  detect <- data.frame(detected = ifelse(runif(n) < 0.3, 0, rbinom(n, 10, 0.4)),
                       trials = 10,
                       habitat = factor(rep(c("forest", "grassland"), each = n / 2)))
  fit <- tratio(detected | trials ~ habitat, zi = zi_binomial(~ habitat),
                data = detect, family = ratiod_zibinomial(),
                control = list(iter = 200, chains = 1, seed = 1))
  expect_named(coef(fit), c("num:(Intercept)", "num:habitatgrassland"))
  expect_equal(dim(vcov(fit)), c(2L, 2L))
  expect_output(print(summary(fit)), "Zero/one inflation")
})

test_that("draws_wide(), draws_long() and draws_interval() keep parameter names", {
  skip_on_cran()
  sim <- sim_ratiod(n = 60, family = ratiod_poisson_gamma(), seed = 2)
  fit <- tratio(y_num | y_denom ~ x1 + (1 | group), data = sim$data,
                family = ratiod_poisson_gamma(),
                control = list(iter = 200, chains = 1, seed = 1))
  expect_true("beta_num[2]" %in% names(draws_wide(fit, beta_num)))
  expect_true("beta_num[2]" %in% draws_long(fit, beta_num)$.variable)
  expect_true("beta_num[2]" %in% draws_interval(fit, beta_num)$.variable)
})

test_that("print() names the random effects whichever backend fitted them", {
  skip_on_cran()
  set.seed(4)
  d <- data.frame(site = factor(rep(1:8, each = 10)), x = rnorm(80))
  d$effort <- rgamma(80, shape = 6, rate = 1)
  d$catch <- rpois(80, d$effort * exp(0.2 + 0.3 * d$x))
  for (mode in c("hmc", "laplace")) {
    fit <- tratio(catch | effort ~ x + (1 | site), data = d,
                  family = ratiod_poisson_gamma(), mode = mode,
                  control = list(iter = 200, chains = 1, seed = 1))
    expect_true(any(grepl("random effects: site (8)", capture.output(print(fit)),
                          fixed = TRUE)), label = mode)
  }
})
