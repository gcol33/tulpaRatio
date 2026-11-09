fixed_contract_data <- function() {
  set.seed(216)
  d <- data.frame(x = rnorm(60), habitat = factor(rep(c("a", "b"), 30)),
                  site = factor(rep(1:6, each = 10)), trials = 15L)
  d$y <- rbinom(nrow(d), d$trials, plogis(-0.3 + 0.4 * d$x))
  adj <- outer(1:6, 1:6, function(a, b) as.integer(abs(a - b) == 1L))
  dimnames(adj) <- list(levels(d$site), levels(d$site))
  list(data = d, spatial = spatial_car(adj, spatial_level = "group", group_var = "site"))
}

check_fixed_contract <- function(fit) {
  variables <- tulpaRatio:::fixed_effect_variables(fit$formula$numerator$X)
  m <- tulpaRatio:::draw_matrix(fit$draws)[, variables, drop = FALSE]
  terms <- paste0("num:", colnames(fit$formula$numerator$X))
  expected <- stats::setNames(colMeans(m), terms)
  expect_identical(names(coef(fit)), terms)
  expect_equal(coef(fit), expected, tolerance = 0)
  expect_equal(tulpaRatio::fixef(fit), expected, tolerance = 0)
  alpha <- (1 - 0.95) / 2
  ci <- t(apply(m, 2L, quantile, c(alpha, 1 - alpha), names = FALSE))
  expect_identical(rownames(confint(fit)), terms)
  expect_equal(unname(confint(fit)), unname(ci), tolerance = 0)
  expect_equal(unname(vcov(fit)), unname(cov(m)), tolerance = 0)
  tab <- tulpaRatio::tidy(fit, conf.int = TRUE)
  expect_identical(tab$term, terms)
  expect_equal(tab$estimate, unname(expected), tolerance = 0)
  expect_equal(tab$conf.low, unname(ci[, 1]), tolerance = 0)
  summ <- summary(fit)$coefficients
  rows <- match(variables, summ$parameter)
  expect_false(anyNA(rows))
  expect_identical(summ$term[rows], terms)
  if (fit$backend == "laplace") {
    expect_equal(summ$mean[rows], unname(coef(fit$engine)), tolerance = 0)
    expect_lt(max(abs(summ$mean[rows] - colMeans(m)) / apply(m, 2L, sd)), 0.3)
  } else {
    expect_equal(summ$mean[rows], unname(expected), tolerance = 0)
    expect_equal(summ$q_lower[rows], unname(ci[, 1]), tolerance = 0)
    expect_equal(summ$q_upper[rows], unname(ci[, 2]), tolerance = 0)
  }
}

for (backend in c("hmc", "pg", "gibbs", "laplace", "ess", "vi", "sghmc", "sgld")) {
  local({
    mode <- backend
    test_that(paste("fixed-effect accessors read the", mode, "fit contract"), {
      skip_on_cran()
      fx <- fixed_contract_data()
      fit <- suppressWarnings(tratio(y | trials ~ x + habitat,
        data = fx$data, family = ratiod_binomial(), mode = mode,
        spatial = if (mode == "gibbs") fx$spatial,
        control = list(iter = 600, warmup = 300, chains = 1, seed = 216,
                        cores = 1, verbose = FALSE)))
      expect_identical(fit$backend, mode)
      check_fixed_contract(fit)
    })
  })
}

test_that("PG variable names and accessors cover spatial and temporal dispatch", {
  skip_on_cran()
  fx <- fixed_contract_data()
  d <- fx$data
  d$time <- rep(1:10, 6)
  d$lon <- as.integer(d$site)
  d$lat <- 0
  car <- fx$spatial
  bym2 <- spatial_bym2(car$adjacency, spatial_level = "group", group_var = "site")
  configs <- list(list(), list(spatial = car), list(spatial = bym2),
    list(spatial = spatial_gp(~lon + lat, nn = 3, scale_coords = FALSE)),
    list(spatial = spatial_rsr(car, ~x)),
    list(temporal = temporal_multiscale("time", trend = "rw1", seasonal = 4,
                                       short_term = "ar1")))
  for (cfg in configs) {
    fit <- do.call(tratio, c(list(formula = y | trials ~ x + habitat,
      data = d, family = ratiod_binomial(), mode = "pg",
      control = list(iter = 120, warmup = 60, chains = 2, seed = 216,
                      cores = 1, verbose = FALSE)), cfg))
    expect_named(dimnames(fit$draws), c("iteration", "chain", "variable"))
    check_fixed_contract(fit)
    raw <- fit$.internal$chain_results[[1L]]
    for (v in intersect(c("sigma_trend", "sigma_seasonal", "sigma_short", "rho_short"),
                        dimnames(fit$draws)[[3L]]))
      expect_equal(unname(fit$draws[, 1, v]), raw[[v]], tolerance = 0)
  }
})
