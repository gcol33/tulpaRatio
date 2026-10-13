# Tests for latent factor specification

test_that("latent_factor creates valid object", {
  lf <- latent_factor()
  expect_s3_class(lf, "ratiod_latent")
  expect_equal(lf$n_factors, 1L)
  expect_true(lf$shared)
  expect_equal(lf$constraint, "sum_to_zero")
})

test_that("latent_factor validates n_factors", {
  expect_error(latent_factor(n_factors = 0), "positive integer")
  expect_error(latent_factor(n_factors = -1), "positive integer")
  expect_error(latent_factor(n_factors = 1.5), "positive integer")

  lf <- latent_factor(n_factors = 2)
  expect_equal(lf$n_factors, 2L)
})

# Two processes have a 2 x 2 residual covariance, which two factors with
# lower-triangular loadings fill; a third has nothing left to identify, and a
# numerator-only factor has one process (gcol33/tulpaRatio#134).
test_that("latent_factor refuses factors the processes cannot identify", {
  expect_error(latent_factor(n_factors = 3), "at most 2 are identified")
  expect_error(latent_factor(n_factors = 2, shared = FALSE),
               "at most 1 are identified")
})

test_that("latent_factor validates prior", {
  lf <- latent_factor(prior = prior_pc(U = 0.5, alpha = 0.05))
  expect_equal(lf$prior$U, 0.5)

  expect_error(latent_factor(prior = "invalid"), "prior object")
})

test_that("latent_factor validates shared",
{
  lf_shared <- latent_factor(shared = TRUE)
  expect_true(lf_shared$shared)

  lf_not_shared <- latent_factor(shared = FALSE)
  expect_false(lf_not_shared$shared)

  expect_error(latent_factor(shared = "yes"), "TRUE or FALSE")
})

test_that("latent_factor validates constraint", {
  lf1 <- latent_factor(constraint = "sum_to_zero")
  expect_equal(lf1$constraint, "sum_to_zero")

  lf2 <- latent_factor(constraint = "first_zero")
  expect_equal(lf2$constraint, "first_zero")

  expect_error(latent_factor(constraint = "invalid"), "should be one of")
})

test_that("validate_latent checks n_factors vs observations", {
  lf <- latent_factor(n_factors = 2)

  validated <- validate_latent(lf, N = 100)
  expect_equal(validated$n_obs, 100)

  expect_error(validate_latent(lf, N = 2), "less than number of observations")
})

test_that("prepare_latent_for_hmc handles NULL", {
  info <- prepare_latent_for_hmc(NULL, N = 50)

  expect_equal(info$type, "none")
  expect_equal(info$n_factors, 0L)
})

test_that("prepare_latent_for_hmc prepares valid info", {
  lf <- latent_factor(n_factors = 2, prior = prior_pc(U = 1, alpha = 0.01))
  info <- prepare_latent_for_hmc(lf, N = 50)

  expect_equal(info$type, "latent")
  expect_equal(info$n_factors, 2L)
  expect_equal(info$n_obs, 50L)
  expect_true(info$shared)
  expect_equal(info$constraint, "sum_to_zero")
  expect_true(info$sigma_prior_rate > 0)
  # |off-diagonal| shares the diagonal prior's 99% quantile
  q99 <- -log(0.01) / info$sigma_prior_rate
  expect_equal(2 * stats::pnorm(-q99 / info$offdiag_sd), 0.01)
})

test_that("print.ratiod_latent produces output", {
  lf <- latent_factor(n_factors = 2)

  output <- capture.output(print(lf))
  expect_true(any(grepl("Latent factor", output)))
  expect_true(any(grepl("Number of factors: 2", output)))
  expect_true(any(grepl("Shared: Yes", output)))
})

test_that("latent_factors() reads a fitted model's factor scores", {
  skip_on_cran()
  set.seed(1)
  n <- 40
  df <- data.frame(
    y = rnbinom(n, size = 5, mu = 15),
    total = rnbinom(n, size = 8, mu = 100),
    x = rnorm(n)
  )
  fit <- tratio(y | total ~ x, data = df, family = ratiod_negbin_negbin(),
                latent = latent_factor(n_factors = 1),
                control = list(iter = 200, chains = 1, seed = 1))
  scores <- fit$.internal$latent_draws
  s <- latent_factors(fit)
  expect_equal(nrow(s), n)
  expect_equal(s$observation, seq_len(n))
  expect_equal(s$mean, colMeans(scores[, , 1]))
  d <- latent_factors(fit, summary = FALSE)
  expect_equal(dim(d), c(dim(scores)[1], n))
  expect_equal(unname(d[, 3]), scores[, 3, 1])
  expect_error(latent_factors(tratio(y | total ~ x, data = df,
                                     family = ratiod_negbin_negbin(),
                                     mode = "laplace")),
               "latent factors")
})
