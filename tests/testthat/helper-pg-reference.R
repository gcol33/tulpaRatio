pg_grid_summary <- function(value, mass) {
  w <- as.numeric(rowsum(mass, value, reorder = TRUE))
  at <- sort(unique(value))
  m <- sum(at * w)
  list(mean = m, sd = sqrt(sum(w * (at - m)^2)),
    quantile = approx(cumsum(w) - w / 2, at, c(0.025, 0.5, 0.975))$y)
}

pg_nb_grid_loglik <- function(y, eta) {
  mu <- exp(eta)
  linear <- drop(eta %*% y)
  function(r) {
    constant <- sum(dnbinom(y, size = r, mu = 1, log = TRUE)) +
      sum(y + r) * log1p(r)
    constant + linear - drop(log(r + mu) %*% (y + r))
  }
}

pg_nb_dense <- function(y, X, prior_sd, points = 181L) {
  bound <- if (prior_sd < 0.5) 6 * prior_sd else 3
  axis <- seq(-bound, bound, length.out = points)
  beta <- expand.grid(intercept = axis, slope = axis)
  eta <- as.matrix(beta) %*% t(X)
  loglik <- pg_nb_grid_loglik(y, eta)
  log_prior <- rowSums(vapply(beta, dnorm, numeric(nrow(beta)),
                            mean = 0, sd = prior_sd, log = TRUE))
  log_r <- seq(log(0.5), log(4), length.out = 121L)
  mass <- numeric(nrow(beta))
  for (lr in log_r) {
    r <- exp(lr)
    ll <- loglik(r)
    mass <- mass + exp(ll + log_prior + dgamma(r, 100, 50, log = TRUE) + lr)
  }
  mass <- mass / sum(mass)
  lapply(beta, pg_grid_summary, mass = mass)
}

pg_nb_spatial_dense <- function(y, X, group, prior_sd, points = 81L,
                                cells = 60L, r_points = 31L) {
  bound <- if (prior_sd < 0.5) 6 * prior_sd else 3
  axis <- seq(-bound, bound, length.out = points)
  width <- 3 / cells
  u <- seq(-1.5 + width / 2, 1.5 - width / 2, length.out = cells)
  lambda <- -log(0.1) / 0.5
  u_mass <- vapply(abs(u), function(mid) {
    lower <- max(0, mid - width / 2)
    upper <- mid + width / 2
    integrate(function(sigma) (pnorm(-2 * lower / sigma) -
      pnorm(-2 * upper / sigma)) * dexp(sigma, lambda),
      0, Inf, rel.tol = 1e-9)$value
  }, 0)
  grid <- expand.grid(intercept = axis, slope = axis, u = u)
  eta <- as.matrix(grid[, 1:2]) %*% t(X) + outer(grid$u, c(1, -1)[group])
  loglik <- pg_nb_grid_loglik(y, eta)
  prior <- dnorm(grid$intercept, sd = prior_sd) * dnorm(grid$slope, sd = prior_sd) *
    rep(u_mass, each = points^2)
  mass <- numeric(nrow(grid))
  for (r in qgamma((seq_len(r_points) - 0.5) / r_points, 100, 50)) {
    ll <- loglik(r)
    mass <- mass + exp(ll) * prior
  }
  mass <- mass / sum(mass)
  lapply(grid, pg_grid_summary, mass = mass)
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
