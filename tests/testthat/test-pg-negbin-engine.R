test_that("the dense NB grid likelihood retains the normalized R density", {
  y <- c(0L, 1L, 7L, 2L)
  eta <- matrix(seq(-3, 3, length.out = 20), nrow = 5)
  loglik <- pg_nb_grid_loglik(y, eta)
  for (r in c(0.3, 2, 100)) {
    reference <- apply(eta, 1, function(value)
      sum(dnbinom(y, size = r, mu = exp(value), log = TRUE)))
    expect_equal(loglik(r), reference, tolerance = 1e-11)
  }
})

test_that("NB engine hooks match dense NB2 coefficient priors", {
  skip_on_cran()
  skip_unless_slow()
  records <- list()
  for (prior_sd in c(0.1, 2)) {
    fixture <- pg_nb_engine_fixture(prior_sd, FALSE)
    reference <- pg_nb_dense(fixture$y, fixture$X, prior_sd)
    refined <- pg_nb_dense(fixture$y, fixture$X, prior_sd, 241L)
    for (j in seq_along(reference)) {
      expect_lt(max(abs(reference[[j]]$quantile - refined[[j]]$quantile)) /
                  refined[[j]]$sd, 0.025)
    }
    for (spatial in c(FALSE, TRUE)) {
      fixture <- pg_nb_engine_fixture(prior_sd, spatial)
      set.seed(191)
      timing <- system.time(fit <- do.call(fixture$entry, fixture$args))
      error <- vapply(seq_along(reference), function(j) {
        max(abs(quantile(fit$beta[, j], c(0.025, 0.5, 0.975)) -
                  refined[[j]]$quantile)) / refined[[j]]$sd
      }, 0)
      expect_true(max(error) < 0.15,
        info = paste("prior SD", prior_sd, "spatial", spatial))
      expect_equal(fit$eta, fit$beta %*% t(fixture$X), tolerance = 1e-12)
      if (spatial) expect_equal(fit$spatial, fit$spatial * 0, tolerance = 1e-12)
      records[[length(records) + 1L]] <- data.frame(prior_sd, spatial,
        intercept_error_sd = error[1], slope_error_sd = error[2],
        seconds = timing[["elapsed"]])
    }
  }
  target <- getOption("tulpa.pg.engine.record")
  if (!is.null(target)) write.csv(do.call(rbind, records), target, row.names = FALSE)
})

test_that("the spatial NB engine hook matches a dense two-unit ICAR posterior", {
  skip_on_cran()
  skip_unless_slow()
  records <- list()
  for (prior_sd in c(0.1, 2)) {
    fixture <- pg_nb_engine_fixture(prior_sd, TRUE)
    group <- rep(1:2, 4)
    fixture$args$spatial_group <- group
    fixture$args$n_spatial_units <- 2L
    fixture$args$adj_list <- list(2L, 1L)
    fixture$args$n_neighbors <- c(1L, 1L)
    fixture$args$prior_sigma_spatial_U <- 0.5
    fixture$args$prior_sigma_spatial_alpha <- 0.1
    reference <- pg_nb_spatial_dense(fixture$y, fixture$X, group, prior_sd)
    refined <- pg_nb_spatial_dense(fixture$y, fixture$X, group, prior_sd,
                                   points = 101L, cells = 80L, r_points = 61L)
    for (j in 1:3) expect_lt(
      max(abs(reference[[j]]$quantile - refined[[j]]$quantile)) / refined[[j]]$sd,
      0.04)
    set.seed(191)
    timing <- system.time(fit <- do.call(fixture$entry, fixture$args))
    draws <- cbind(fit$beta, fit$spatial[, 1])
    error <- vapply(1:3, function(j)
      max(abs(quantile(draws[, j], c(0.025, 0.5, 0.975)) -
        refined[[j]]$quantile)) / refined[[j]]$sd, 0)
    expect_true(max(error) < 0.15, info = paste("prior SD", prior_sd))
    expect_equal(rowSums(fit$spatial), rep(0, nrow(draws)), tolerance = 1e-12)
    expect_equal(fit$eta, fit$beta %*% t(fixture$X) + fit$spatial[, group],
                 tolerance = 1e-12)
    records[[length(records) + 1L]] <- data.frame(prior_sd,
      intercept_error_sd = error[1], slope_error_sd = error[2],
      field_error_sd = error[3], seconds = timing[["elapsed"]])
  }
  target <- getOption("tulpa.pg.engine.record")
  if (!is.null(target)) write.csv(do.call(rbind, records),
    sub("[.]csv$", "-spatial.csv", target), row.names = FALSE)
})
