precision_ratio_reference <- function(name) {
  refs <- read.csv(testthat::test_path("fixtures", paste0("ratio-", name, "-precision.csv")))
  stopifnot(identical(refs$phi, c(.3, 2, 100, 1e8, 1e12, 1e15)))
  refs
}

test_that("gamma densities and precision scores match 100-digit references on every path", {
  refs <- precision_ratio_reference("gamma")
  for (i in seq_len(nrow(refs))) {
    r <- refs[i, ]
    for (name in c("legacy", "spec")) {
      fun <- get(paste0("cpp_test_precision_", name), asNamespace("tulpaRatio"))
      v <- fun(r$y, 1, r$phi)
      expect_lt(max(abs(v[, 1] - r$expected)), 2e-12, label = paste(name, r$phi))
      expect_lt(max(abs(v[-1, 2] / r$dphi - 1)), 2e-12, label = paste(name, r$phi))
    }
    v <- tulpaRatio:::cpp_test_precision_gibbs(r$y, 1, r$phi)
    expect_lt(max(abs(v - r$expected)), 2e-12, label = paste("Gibbs", r$phi))
    v <- tulpaRatio:::cpp_test_precision_spec_h(r$y, 1, r$phi)
    expect_lt(max(abs(v[, 1] - c(1, 1, 2) * r$expected)), 2e-12)
    expect_lt(max(abs(v[, 2] / r$dphi - 1)), 2e-12)
    for (family in 0:2) {
      h <- tulpaRatio:::cpp_test_precision_h(r$y, 1, r$phi, family)
      expect_lt(max(abs(h[, 1] - (if (family == 2) 2 else 1) * r$expected), na.rm = TRUE),
        2e-12, label = paste("H and collapsed", family, r$phi))
      expect_lt(max(abs(h[, 2] / r$dphi - 1), na.rm = TRUE),
        2e-12, label = paste("H precision", family, r$phi))
    }
    jet <- tulpaRatio:::cpp_test_precision_jet(r$y, r$phi, FALSE)
    expect_lt(abs(jet[1] - r$expected), 2e-12)
    expect_lt(abs(jet[2] / r$dphi - 1), 2e-12)
    expect_lt(abs(jet[3] / r$dphi2 - 1), 2e-12)
  }
})

test_that("beta-binomial densities and precision scores match 100-digit references", {
  refs <- precision_ratio_reference("beta-binomial")
  for (i in seq_len(nrow(refs))) {
    r <- refs[i, ]
    for (name in c("legacy", "spec")) {
      fun <- get(paste0("cpp_test_precision_beta_", name), asNamespace("tulpaRatio"))
      v <- fun(r$y, r$n, r$phi)
      expect_lt(max(abs(v[, 1] - r$expected)), 2e-12, label = paste(name, r$phi))
      expect_lt(max(abs(v[-1, 2] / r$dphi - 1)), 2e-11, label = paste(name, r$phi))
    }
    h <- tulpaRatio:::cpp_test_precision_h(r$y, .5, r$phi, 3L)
    expect_lt(max(abs(h[, 1] - r$expected), na.rm = TRUE), 2e-12)
    expect_lt(max(abs(h[, 2] / r$dphi - 1), na.rm = TRUE), 2e-11)
    h <- tulpaRatio:::cpp_test_precision_beta_spec_h(r$y, r$n, r$phi)
    expect_lt(abs(h[1] - r$expected), 2e-12)
    expect_lt(abs(h[2] / r$dphi - 1), 2e-11)
    jet <- tulpaRatio:::cpp_test_precision_jet(r$y, r$phi, TRUE)
    expect_lt(abs(jet[1] - r$expected), 2e-12)
    expect_lt(abs(jet[2] / r$dphi - 1), 2e-11)
    expect_lt(abs(jet[3] / r$dphi2 - 1), 2e-11)
  }
})

test_that("pointwise draw likelihoods use the shared precision densities", {
  gamma <- precision_ratio_reference("gamma")
  families <- list(poisson_gamma = tulpaRatio::ratiod_poisson_gamma(),
    negbin_gamma = tulpaRatio::ratiod_negbin_gamma(),
    gamma_gamma = tulpaRatio::ratiod_gamma_gamma())
  for (name in names(families)) {
    draws <- cbind(shape = gamma$phi, shape_num = gamma$phi,
      shape_denom = gamma$phi, phi_num = 2, phi_denom = gamma$phi)
    fit <- list(backend = "gibbs", model_type = name, family = families[[name]],
      draws = draws, .internal = list(
        eta_num = matrix(0, 6, 6), eta_denom = matrix(0, 6, 6),
        hmc_data = list(y_num = rep(0L, 6), y_denom = rep(0L, 6),
          y_num_cont = gamma$y, y_denom_cont = gamma$y)))
    actual <- tulpaRatio:::.ratiod_pointwise_loglik(fit)
    expected <- switch(name, poisson_gamma = gamma$expected - 1,
      negbin_gamma = gamma$expected - 2 * log1p(.5),
      gamma_gamma = 2 * gamma$expected)
    expect_equal(dim(actual), c(6L, 6L))
    expect_lt(max(abs(diag(actual) - expected)), 2e-12)
  }
  beta <- precision_ratio_reference("beta-binomial")
  fit <- list(backend = "gibbs", model_type = "beta_binomial",
    family = tulpaRatio::ratiod_beta_binomial(), draws = cbind(phi = beta$phi),
    .internal = list(eta_num = matrix(0, 6, 6),
      hmc_data = list(y_num = beta$y, y_denom = beta$n)))
  expect_lt(max(abs(diag(tulpaRatio:::.ratiod_pointwise_loglik(fit)) - beta$expected)), 2e-12)
})
