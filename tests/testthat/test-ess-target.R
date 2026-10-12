# Elliptical slice sampling draws each ellipse from a reference distribution
# and slices on the posterior divided by it, so it targets the same
# posterior as HMC whatever the reference.

test_that("ESS and HMC agree on a random-intercept model", {
  skip_on_cran()
  sim <- sim_ratiod(n = 120, family = ratiod_poisson_gamma(), sigma_re = 0.6,
                    seed = 9)
  fit_one <- function(mode, iter) {
    tratio(y_num | y_denom ~ x1 + (1 | group), data = sim$data,
           family = ratiod_poisson_gamma(), mode = mode,
           control = list(iter = iter, chains = 2, seed = 1))
  }
  expect_warning(ess <- fit_one("ess", 2000), NA)
  hmc <- fit_one("hmc", 2000)
  pars <- c("beta_num[1]", "beta_num[2]", "sigma_re")
  m_ess <- colMeans(as.matrix(ess$draws)[, pars])
  m_hmc <- colMeans(as.matrix(hmc$draws)[, pars])
  s_hmc <- apply(as.matrix(hmc$draws)[, pars], 2, sd)
  expect_true(all(abs(m_ess - m_hmc) / s_hmc < 0.3),
              info = paste(round((m_ess - m_hmc) / s_hmc, 2), collapse = ", "))

  # ESS samples the HMC parameter vector, so its fit is unpacked, predicted
  # from and scored the same way.
  expect_equal(colnames(ess$draws), colnames(hmc$draws))
  expect_equal(ess$chains, 2)
  expect_equal(nrow(predict(ess, newdata = sim$data[1:3, ])), 3L)
  l <- suppressWarnings(loo(ess))
  expect_true(is.finite(l$estimates["elpd_loo", "Estimate"]))
})

test_that("ESS refuses random-effect structures it does not sample", {
  sim <- sim_ratiod(n = 60, family = ratiod_poisson_gamma(), seed = 2)
  expect_error(
    tratio(y_num | y_denom ~ x1 + (1 + x1 | group), data = sim$data,
           family = ratiod_poisson_gamma(), mode = "ess",
           control = list(iter = 100, chains = 1, seed = 1)),
    "one random intercept"
  )
})
