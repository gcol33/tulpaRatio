pg_nb_dense <- function(y, X, prior_sd, points = 181L) {
  bound <- if (prior_sd < 0.5) 6 * prior_sd else 3
  axis <- seq(-bound, bound, length.out = points)
  beta <- expand.grid(intercept = axis, slope = axis)
  eta <- as.matrix(beta) %*% t(X)
  log_prior <- rowSums(vapply(beta, dnorm, numeric(nrow(beta)),
                            mean = 0, sd = prior_sd, log = TRUE))
  log_r <- seq(log(0.5), log(4), length.out = 121L)
  mass <- numeric(nrow(beta))
  for (lr in log_r) {
    r <- exp(lr)
    ll <- rowSums(matrix(dnbinom(rep(y, each = nrow(beta)),
      size = r, mu = exp(eta), log = TRUE), nrow(beta)))
    mass <- mass + exp(ll + log_prior + dgamma(r, 100, 50, log = TRUE) + lr)
  }
  mass <- mass / sum(mass)
  summarize <- function(value) {
    w <- as.numeric(rowsum(mass, value, reorder = TRUE))
    at <- sort(unique(value))
    m <- sum(at * w)
    list(mean = m, sd = sqrt(sum(w * (at - m)^2)),
      quantile = approx(cumsum(w) - w / 2, at, c(0.025, 0.5, 0.975))$y)
  }
  lapply(beta, summarize)
}

pg_nb_engine_fixture <- function(prior_sd, spatial, iterations = 24000L) {
  y <- c(0L, 1L, 2L, 1L, 1L, 0L, 1L, 2L)
  X <- cbind(1, rep(c(-1, 1), each = 4))
  args <- list(y = y, X = X, n_iter = iterations, n_warmup = 4000L,
    thin = 1L, prior_beta_sd = prior_sd, prior_r_shape = 100,
    prior_r_rate = 50, r_init = 2, store_eta = TRUE,
    verbose = FALSE, n_threads = 1L)
  if (spatial) {
    args <- c(args, list(re_group = rep(1L, length(y)), n_re_groups = 0L,
      spatial_group = rep(1L, length(y)), n_spatial_units = 1L,
      adj_list = list(integer()), n_neighbors = 0L,
      prior_sigma_re_scale = 1, prior_sigma_spatial_U = 1,
      prior_sigma_spatial_alpha = 0.01))
    entry <- tulpaRatio:::cpp_pg_negbin_gibbs_spatial
  } else {
    args <- c(args, list(group = rep(1L, length(y)), n_groups = 0L,
                        prior_sigma_scale = 1))
    entry <- tulpaRatio:::cpp_pg_negbin_gibbs
  }
  list(args = args, entry = entry, y = y, X = X)
}

test_that("NB engine hooks match dense NB2 coefficient priors", {
  skip_on_cran()
  skip_if(Sys.getenv("TULPA_SLOW_TESTS") != "true")
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
