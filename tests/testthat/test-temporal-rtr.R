# temporal_rtr() reaches the sampler: the temporal effect enters the linear
# predictor projected onto the complement of the restricted covariates, as a
# spatial_rsr() field does.

rtr_data <- function(n_times = 12L, reps = 6L, seed = 5) {
  set.seed(seed)
  time <- rep(seq_len(n_times), each = reps)
  temperature <- time / n_times + rnorm(length(time), 0, 0.2)
  n <- sample(15:30, length(time), replace = TRUE)
  y <- rbinom(length(time), n, plogis(-0.2 + 0.8 * temperature + sin(time / 2) * 0.5))
  data.frame(y = y, n = n, time = time, temperature = temperature)
}

test_that("temporal_rtr() restricts a temporal effect, not a coefficient", {
  expect_s3_class(temporal_rtr(temporal_rw1("time"), ~ temperature), "ratiod_rtr")
  expect_error(temporal_rtr(temporal_rw1("time"), "temperature"), "formula")
  expect_error(temporal_rtr(suppressWarnings(temporal_tvc("time", ~ x)), ~ x),
               "temporal effect")
})

test_that("the restricted basis has one column per unit of rank", {
  d <- rtr_data()
  d$temp2 <- 2 * d$temperature
  Q <- tulpaRatio:::rtr_basis(temporal_rtr(temporal_rw1("time"), ~ temperature + temp2), d)
  expect_equal(ncol(Q), 2L)
  expect_equal(crossprod(Q), diag(2), tolerance = 1e-10)
})

test_that("only HMC carries a restricted temporal effect", {
  spec <- temporal_rtr(temporal_ar1("time"), ~ temperature)
  for (b in c("ess", "sghmc", "sgld", "vi", "pg", "laplace")) {
    expect_match(tulpaRatio:::unsupported_structures(b, list(temporal = spec))[["temporal"]],
                 "temporal_rtr", label = b)
  }
  expect_length(tulpaRatio:::unsupported_structures("hmc", list(temporal = spec)), 0L)
})

test_that("the restriction changes the density the sampler evaluates", {
  for (f in c("rw1", "temporal_gp", "ms_temporal")) {
    plain <- tulpaRatio:::cpp_gradient_check(f, mode = "arena")
    restricted <- tulpaRatio:::cpp_gradient_check(paste0(f, "_rtr"), mode = "arena")
    expect_false(isTRUE(all.equal(plain$log_post, restricted$log_post)), label = f)
  }
})

test_that("the fitted temporal contribution is orthogonal to the restricted covariates", {
  skip_on_cran()
  d <- rtr_data()
  for (spec in list(temporal_rw1("time"), temporal_gp("time"))) {
    fit <- suppressWarnings(tratio(
      y | n ~ temperature, data = d, family = ratiod_binomial(),
      temporal = temporal_rtr(spec, ~ temperature),
      control = list(iter = 200, warmup = 100, chains = 1, verbose = FALSE, seed = 3)))
    contribs <- tulpaRatio:::hmc_structure_contributions(
      tulpaRatio:::hmc_fit_unpack(fit), tulpaRatio:::hmc_fit_design(fit))
    eta_t <- Filter(function(x) x$name == "temporal", contribs)[[1]]$eta
    X <- cbind(1, d$temperature)
    expect_lt(max(abs(eta_t %*% X)), 1e-8)
  }
})
