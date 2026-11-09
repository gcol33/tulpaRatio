test_that("PG NB support follows the family and structure tables", {
  expect_true("pg" %in% tulpaRatio:::MODEL_TYPES$negbin_negbin$samplers)
  expect_true(tulpaRatio:::can_use_pg_backend(ratiod_negbin_negbin()))
  for (family in list(ratiod_negbin(), ratiod_negbin_gamma(),
      ratiod_zi_negbin(), ratiod_hurdle_negbin(), ratiod_binomial("probit")))
    expect_false(tulpaRatio:::can_use_pg_backend(family))
  for (family in list(ratiod_binomial(), ratiod_negbin_negbin())) {
    fields <- list(spatial = spatial_car(build_chain_adjacency(4), group_var = "g"),
      temporal = temporal_multiscale("time", trend = "rw2", short_term = "none"))
    for (name in names(fields)) {
      structures <- c(list(family = family), fields[name])
      expect_identical(tulpaRatio:::backend_fits_structures("pg", structures),
                       identical(family$name, "binomial_fixed"))
    }
  }
})

test_that("NB-NB PG front door forwards priors, sharing, designs and chain seeds", {
  skip_on_cran()
  set.seed(184)
  d <- data.frame(x = rnorm(40), z = rnorm(40), g = factor(rep(1:5, 8)))
  d$y <- rnbinom(40, mu = exp(0.2 + 0.3 * d$x), size = 1.2)
  d$den <- rnbinom(40, mu = exp(0.4 - 0.2 * d$x), size = 2)
  fit <- tratio(y | den ~ x + (1 | g), formula_num = ~ z,
    formula_denom = ~ I(x^2), data = d, family = ratiod_negbin_negbin(),
    mode = "pg", priors = ratiod_priors(beta = prior_normal(0, 0.7),
      sigma = prior_half_cauchy(1.3), phi = prior_gamma(2.4, 0.7)),
    control = list(iter = 40L, warmup = 15L, thin = 3L, chains = 2L,
                    cores = 1L, seed = 184L, verbose = FALSE))
  expect_identical(fit$backend, "pg")
  expect_identical(fit$inference_tier, 1L)
  expect_identical(dim(fit$draws), c(9L, 2L, 9L))
  expect_true(all(c("phi_num", "phi_denom", "sigma_re") %in%
                    dimnames(fit$draws)$variable))
  for (chain in 1:2) {
    set.seed(184L + chain - 1L)
    raw <- tulpaRatio:::cpp_pg_negbin_negbin_gibbs(
      as.integer(d$y), as.integer(d$den), fit$formula$numerator$X,
      fit$formula$denominator$X, as.integer(d$g), 5L,
      n_iter = 40L, n_warmup = 15L, thin = 3L, prior_beta_sd = 0.7,
      prior_sigma_scale = 1.3, prior_r_shape = 2.4, prior_r_rate = 0.7,
      r_num_init = 1, r_denom_init = 1, shared = TRUE, store_eta = TRUE,
      verbose = FALSE, n_threads = 1L)
    expect_identical(fit$.internal$chain_results[[chain]], raw)
  }
  for (arm in c("num", "denom")) {
    expected <- fit$.internal[[paste0("beta_", arm)]] %*%
      t(fit$formula[[if (arm == "num") "numerator" else "denominator"]]$X) +
      fit$.internal$re[, as.integer(d$g)]
    expect_equal(unname(fit$.internal[[paste0("eta_", arm)]]),
                 unname(expected), tolerance = 1e-12)
  }
  fitted <- fitted(fit, summary = FALSE)
  expect_equal(fitted$numerator, exp(fit$.internal$eta_num), tolerance = 1e-12)
  expect_equal(fitted$denominator, exp(fit$.internal$eta_denom), tolerance = 1e-12)
  prediction <- predict(fit, d, component = "all", summary = FALSE)
  for (arm in c("numerator", "denominator", "ratio"))
    expect_equal(unname(prediction[[arm]]), unname(fitted[[arm]]), tolerance = 1e-12)
  population <- predict(fit, d, component = "all", summary = FALSE, re_formula = NA)
  expect_equal(unname(population$numerator),
    unname(exp(fit$.internal$beta_num %*% t(fit$formula$numerator$X))), tolerance = 1e-12)
  expect_equal(unname(population$ratio), unname(fitted$ratio), tolerance = 1e-12)
  expect_equal(unname(ratio(fit)$draws), unname(fitted$ratio), tolerance = 1e-12)
  expect_length(coef(fit), 6L)
  draws <- tulpaRatio:::draw_matrix(fit$draws)
  set.seed(184)
  ids <- sort(sample.int(nrow(draws), 4L))
  expected_num <- expected_denom <- matrix(0, 4, nrow(d))
  for (i in seq_along(ids)) {
    expected_num[i, ] <- rnbinom(nrow(d), size = draws[ids[i], "phi_num"],
                                 mu = fitted$numerator[ids[i], ])
    expected_denom[i, ] <- rnbinom(nrow(d), size = draws[ids[i], "phi_denom"],
                                   mu = fitted$denominator[ids[i], ])
  }
  set.seed(184)
  replicated <- posterior_predict(fit, ndraws = 4L)
  expect_identical(replicated$draw_ids, ids)
  expect_identical(replicated$y_num_rep, expected_num)
  expect_identical(replicated$y_denom_rep, expected_denom)
})

test_that("an NB-NB PG fit carries an explicit independence specification", {
  skip_on_cran()
  d <- data.frame(y = c(0L, 1L, 2L, 1L), den = c(2L, 4L, 1L, 3L), x = -1:2)
  expect_warning(fit <- tratio(y | den ~ x, shared = ~ 0, data = d,
    family = ratiod_negbin_negbin(), mode = "pg",
    control = list(iter = 12, warmup = 4, chains = 1, seed = 184,
                    cores = 1, verbose = FALSE)), "assuming independence")
  expect_equal(ncol(fit$.internal$re), 0L)
  expect_false("sigma_re" %in% dimnames(fit$draws)$variable)
  expect_identical(fit$formula$shared$type, "independent")
})

test_that("NB-NB PG front-door coefficients match normalized dense NB2 references", {
  skip_on_cran()
  skip_unless_slow()
  d <- data.frame(y = c(0L, 1L, 2L, 1L, 1L, 0L, 1L, 2L),
    den = c(2L, 3L, 0L, 4L, 1L, 0L, 2L, 0L), x = rep(c(-1, 1), each = 4))
  for (prior_sd in c(0.1, 2)) {
    fit <- tratio(y | den ~ x, data = d, family = ratiod_negbin_negbin(), mode = "pg",
      priors = ratiod_priors(beta = prior_normal(0, prior_sd), phi = prior_gamma(100, 50)),
      control = list(iter = 18000L, warmup = 3000L, chains = 1L,
                      seed = 184L, cores = 1L, verbose = FALSE))
    for (arm in c("num", "denom")) {
      y <- if (arm == "num") d$y else d$den
      reference <- pg_nb_dense(y, model.matrix(~x, d), prior_sd)
      beta <- fit$.internal[[paste0("beta_", arm)]]
      for (j in 1:2) expect_true(
        max(abs(quantile(beta[, j], c(0.025, 0.5, 0.975)) -
          reference[[j]]$quantile)) / reference[[j]]$sd < 0.15,
        info = paste(arm, j, "prior SD", prior_sd))
    }
  }
})

test_that("an NB-NB PG model with fixed unit means retains its dispersion draws", {
  d <- data.frame(y = c(0L, 1L, 2L, 1L), den = c(2L, 0L, 1L, 1L))
  fit <- tratio(y | den ~ 0, data = d, family = ratiod_negbin_negbin(), mode = "pg",
    control = list(iter = 12L, warmup = 4L, chains = 1L,
                    seed = 184L, cores = 1L, verbose = FALSE))
  expect_identical(dim(fit$draws), c(8L, 1L, 2L))
  expect_identical(dimnames(fit$draws)$variable, c("phi_num", "phi_denom"))
  expect_equal(ncol(fit$.internal$beta_num), 0L)
  expect_equal(ncol(fit$.internal$beta_denom), 0L)
  expect_equal(as.vector(fitted(fit, component = "numerator", summary = FALSE)$numerator),
               rep(1, 8L * nrow(d)))
})
