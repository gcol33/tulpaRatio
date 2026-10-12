# One walk of the sampler's parameter vector feeds fitted(), predict(),
# ratio(), the flat draws matrix and the per-structure extractors. These tests
# hold those readings to each other and to the sampler's own layout.

ctl <- list(iter = 80, warmup = 40, chains = 1)

unpack_data <- function(n = 60, seed = 11) {
  set.seed(seed)
  data.frame(
    count = rpois(n, 12),
    total = rpois(n, 90),
    x = rnorm(n),
    z = rnorm(n),
    site = factor(rep(seq_len(n / 6), each = 6)),
    region = factor(rep(seq_len(n / 12), each = 12)),
    year = rep(seq_len(6), length.out = n),
    lon = runif(n),
    lat = runif(n)
  )
}

quiet_fit <- function(...) {
  suppressWarnings(suppressMessages(utils::capture.output(fit <- tratio(...))))
  fit
}


test_that("the layout accounts for every column the sampler returns", {
  skip_on_cran()
  df <- unpack_data()

  fits <- list(
    re = quiet_fit(count | total ~ x + (1 | site), data = df,
                   family = ratiod_poisson_gamma(), control = ctl),
    crossed = quiet_fit(count | total ~ x + (1 | site) + (1 | region), data = df,
                        family = ratiod_poisson_gamma(), control = ctl),
    slopes = quiet_fit(count | total ~ x + (1 + x | site), data = df,
                       family = ratiod_poisson_gamma(), control = ctl),
    temporal = quiet_fit(count | total ~ x, data = df,
                         family = ratiod_poisson_gamma(),
                         temporal = temporal_rw1("year"), control = ctl),
    tvc = quiet_fit(count | total ~ x, data = df,
                    family = ratiod_poisson_gamma(),
                    temporal = temporal_tvc("year", terms = c(1, 2)),
                    control = ctl),
    svc = quiet_fit(count | total ~ x, data = df,
                    family = ratiod_poisson_gamma(),
                    spatial = spatial_svc(~ lon + lat, terms = 1),
                    control = ctl)
  )

  for (nm in names(fits)) {
    fit <- fits[[nm]]
    expect_equal(fit$.internal$layout$total,
                 ncol(fit$.internal$samples),
                 info = nm)
  }
})


test_that("fitted(), ratio() and predict() read the same linear predictor", {
  skip_on_cran()
  df <- unpack_data()

  configs <- list(
    re = list(formula = count | total ~ x + (1 | site)),
    crossed = list(formula = count | total ~ x + (1 | site) + (1 | region)),
    slopes = list(formula = count | total ~ x + (1 + x | site)),
    uncorrelated = list(formula = count | total ~ x + (1 + x || site))
  )

  for (nm in names(configs)) {
    fit <- quiet_fit(configs[[nm]]$formula, data = df,
                     family = ratiod_poisson_gamma(), control = ctl)

    r <- ratio(fit, summary = FALSE)$draws
    f <- fitted(fit, component = "ratio", summary = FALSE)$ratio
    expect_equal(unname(f), unname(r), info = nm)

    p <- suppressWarnings(predict(fit, newdata = df, component = "ratio",
                                  summary = FALSE))$ratio
    expect_equal(unname(p), unname(r), info = nm)
  }
})


test_that("the reported random effects are the effects the fit used", {
  skip_on_cran()
  # The sampler stores z under the default non-centred parameterization; a fit
  # that reported z as the effect would fail this by a factor of sigma_re.
  df <- unpack_data()
  fit <- quiet_fit(count | total ~ x + (1 | site), data = df,
                   family = ratiod_poisson_gamma(), control = ctl)

  draws <- fit$draws
  hmc_data <- fit$.internal$hmc_data
  beta_num <- draws[, grep("^beta_num", colnames(draws)), drop = FALSE]
  beta_denom <- draws[, grep("^beta_denom", colnames(draws)), drop = FALSE]
  re <- draws[, grep("^re\\[", colnames(draws)), drop = FALSE]

  eta_num <- beta_num %*% t(hmc_data$X_num) + re[, hmc_data$re_group]
  eta_denom <- beta_denom %*% t(hmc_data$X_denom) + re[, hmc_data$re_group]

  expected <- exp(eta_num - eta_denom)
  expect_equal(unname(ratio(fit, summary = FALSE)$draws), unname(expected))
})


test_that("fitted values are on the response scale of the family", {
  skip_on_cran()
  df <- unpack_data()

  # A count ratio is a rate, not a probability, so it is free to exceed 1
  fit <- quiet_fit(count | total ~ x + (1 | site), data = df,
                   family = ratiod_negbin_negbin(), control = ctl)
  vals <- fitted(fit, component = "numerator", summary = FALSE)$numerator
  expect_true(all(vals > 0))
  expect_gt(max(vals), 1)

  # A binomial fit reports a probability
  df$hits <- rbinom(nrow(df), size = 20, prob = 0.3)
  df$trials <- 20L
  bfit <- quiet_fit(hits | trials ~ x + (1 | site), data = df,
                    family = ratiod_binomial(), control = ctl)
  p <- fitted(bfit, component = "ratio", summary = FALSE)$ratio
  expect_true(all(p > 0 & p < 1))
})


test_that("a Laplace count fit is not squashed into the unit interval", {
  skip_on_cran()
  df <- unpack_data()
  fit <- quiet_fit(count | total ~ x + (1 | site), data = df,
                   family = ratiod_poisson_gamma(),
                   mode = "laplace", control = ctl)
  skip_if(fit$backend != "laplace", "model did not route to the Laplace backend")

  vals <- fitted(fit, component = "numerator", summary = FALSE)$numerator
  expect_true(all(vals > 0))
  expect_gt(max(vals), 1)
})


test_that("the per-structure extractors return their documented classes", {
  skip_on_cran()
  df <- unpack_data()

  tfit <- quiet_fit(count | total ~ x, data = df,
                    family = ratiod_poisson_gamma(),
                    temporal = temporal_rw1("year"), control = ctl)
  tp <- temporal(tfit)
  expect_s3_class(tp, "ratiod_temporal_posterior")
  expect_equal(dim(tp$draws)[2], tfit$temporal$n_temporal_params)

  vfit <- quiet_fit(count | total ~ x, data = df,
                    family = ratiod_poisson_gamma(),
                    temporal = temporal_tvc("year", terms = c(1, 2)),
                    control = ctl)
  tv <- tvc(vfit)
  expect_s3_class(tv, "ratiod_tvc_posterior")
  expect_equal(dim(tv$draws)[3], 2L)

  sfit <- quiet_fit(count | total ~ x, data = df,
                    family = ratiod_poisson_gamma(),
                    spatial = spatial_svc(~ lon + lat, terms = 1),
                    control = ctl)
  sv <- svc(sfit)
  expect_s3_class(sv, "ratiod_svc_posterior")
  expect_equal(dim(sv$draws)[2], nrow(df))
})


test_that("the extractors name the argument that routes them", {
  skip_on_cran()
  df <- unpack_data()
  fit <- quiet_fit(count | total ~ x + (1 | site), data = df,
                   family = ratiod_poisson_gamma(), control = ctl)

  expect_error(svc(fit), "spatial = spatial_svc")
  expect_error(tvc(fit), "temporal = temporal_tvc")
})


test_that("a spatiotemporal fit reports its interaction", {
  skip_on_cran()
  set.seed(3)
  S <- 5; T_times <- 4; reps <- 3
  df <- data.frame(
    region = factor(rep(seq_len(S), each = T_times * reps)),
    year = rep(rep(seq_len(T_times), each = reps), S)
  )
  df$x <- rnorm(nrow(df))
  df$count <- rpois(nrow(df), 10)
  df$total <- rpois(nrow(df), 70)

  adj <- matrix(0, S, S)
  for (i in seq_len(S - 1)) adj[i, i + 1] <- adj[i + 1, i] <- 1

  fit <- quiet_fit(
    count | total ~ x, data = df, family = ratiod_poisson_gamma(),
    spatiotemporal = spatiotemporal(
      spatial = spatial_car(adj, level = "group", group_var = "region"),
      temporal = temporal_rw1("year"), type = "IV"
    ),
    control = list(iter = 60, warmup = 30, chains = 1)
  )

  expect_equal(fit$.internal$layout$total, ncol(fit$.internal$samples))

  eff <- spatiotemporal_effects(fit)
  expect_true(is.array(eff))
  expect_equal(dim(eff)[1:2], c(S, T_times))

  summ <- spatiotemporal_effects(fit, format = "summary")
  expect_s3_class(summ, "tulpa_st_summary")
  for (i in seq_len(S)) for (j in seq_len(T_times)) {
    expect_equal(summ$mean[summ$s == i & summ$t == j], mean(eff[i, j, ]))
  }

  expect_equal(
    unname(fitted(fit, component = "ratio", summary = FALSE)$ratio),
    unname(ratio(fit, summary = FALSE)$draws)
  )
})
