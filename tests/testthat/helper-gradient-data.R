
generate_msgp_data_family <- function(n, seed, family = "poisson_gamma") {
  set.seed(seed)

  # Generate coordinates on unit square
  coords <- data.frame(x = runif(n), y = runif(n))

  # Generate true spatial effects
  dist_mat <- as.matrix(dist(coords))

  # Local effect (short range)
  sigma_local <- 0.4
  phi_local <- 0.1
  K_local <- sigma_local^2 * exp(-dist_mat / phi_local)
  diag(K_local) <- diag(K_local) + 1e-6
  L_local <- chol(K_local)
  w_local <- as.numeric(t(L_local) %*% rnorm(n))

  # Regional effect (long range)
  sigma_regional <- 0.3
  phi_regional <- 0.4
  K_regional <- sigma_regional^2 * exp(-dist_mat / phi_regional)
  diag(K_regional) <- diag(K_regional) + 1e-6
  L_regional <- chol(K_regional)
  w_regional <- as.numeric(t(L_regional) %*% rnorm(n))

  w_total <- w_local + w_regional

  if (family == "negbin_negbin") {
    mu_num <- exp(2 + w_total)
    mu_denom <- exp(2.5 + 0.5 * w_total)
    y_num <- rnbinom(n, size = 5, mu = mu_num)
    y_denom <- rnbinom(n, size = 8, mu = mu_denom)
    y_denom_cont <- rep(0.0, n)
  } else if (family == "binomial") {
    p <- plogis(w_total)
    trials <- rpois(n, 30) + 10
    y_num <- rbinom(n, trials, p)
    y_denom <- trials
    y_denom_cont <- rep(0.0, n)
  } else {  # poisson_gamma
    y_num <- rpois(n, exp(2 + w_total))
    y_denom <- rep(0L, n)
    y_denom_cont <- rgamma(n, shape = 5, rate = 5 / exp(2 + 0.5 * w_total))
  }

  list(
    x = coords$x,
    y = coords$y,
    y_num = y_num,
    y_denom = y_denom,
    y_denom_cont = y_denom_cont
  )
}

generate_gp_temporal_data <- function(n_spatial, n_times, seed, family = "poisson_gamma") {
  set.seed(seed)

  n <- n_spatial * n_times

  # Spatial coordinates
  coords <- data.frame(
    x = rep(runif(n_spatial), n_times),
    y = rep(runif(n_spatial), n_times)
  )

  # GP spatial effect
  dist_mat <- as.matrix(dist(coords[1:n_spatial, ]))
  sigma_gp <- 0.5
  phi_gp <- 0.2
  K <- sigma_gp^2 * exp(-dist_mat / phi_gp)
  diag(K) <- diag(K) + 1e-6
  L <- chol(K)
  w_spatial <- as.numeric(t(L) %*% rnorm(n_spatial))
  w_spatial_full <- rep(w_spatial, n_times)

  # Temporal effect (RW1)
  temporal_effect <- cumsum(rnorm(n_times, 0, 0.2))
  temporal_effect <- temporal_effect - mean(temporal_effect)
  temporal_full <- rep(temporal_effect, each = n_spatial)

  time_idx <- rep(1:n_times, each = n_spatial)

  if (family == "negbin_negbin") {
    mu_num <- exp(2 + w_spatial_full + temporal_full)
    mu_denom <- exp(2.5 + 0.5 * w_spatial_full + 0.5 * temporal_full)
    y_num <- rnbinom(n, size = 5, mu = mu_num)
    y_denom <- rnbinom(n, size = 8, mu = mu_denom)
    y_denom_cont <- rep(0.0, n)
  } else if (family == "binomial") {
    p <- plogis(w_spatial_full + temporal_full)
    trials <- rpois(n, 25) + 10
    y_num <- rbinom(n, trials, p)
    y_denom <- trials
    y_denom_cont <- rep(0.0, n)
  } else {  # poisson_gamma
    y_num <- rpois(n, exp(2 + w_spatial_full + temporal_full))
    y_denom <- rep(0L, n)
    y_denom_cont <- rgamma(n, shape = 5, rate = 5 / exp(2 + 0.5 * w_spatial_full))
  }

  list(
    x = coords$x,
    y = coords$y,
    y_num = y_num,
    y_denom = y_denom,
    y_denom_cont = y_denom_cont,
    time = time_idx,
    n_times = n_times
  )
}

generate_spatial_zi_data <- function(n_sites, seed, spatial_type = "icar", family = "poisson_gamma") {
  set.seed(seed)

  # Generate adjacency for ICAR/BYM2 (grid layout)
  side <- ceiling(sqrt(n_sites))
  coords <- expand.grid(x = 1:side, y = 1:side)[1:n_sites, ]

  # Build adjacency list
  adj_list <- list()
  for (i in 1:n_sites) {
    neighbors <- c()
    for (j in 1:n_sites) {
      if (i != j) {
        dist_ij <- abs(coords$x[i] - coords$x[j]) + abs(coords$y[i] - coords$y[j])
        if (dist_ij == 1) neighbors <- c(neighbors, j)
      }
    }
    adj_list[[i]] <- neighbors
  }

  # Spatial effect (simplified)
  w_spatial <- rnorm(n_sites, 0, 0.5)

  # Zero-inflation probability
  pi_zi <- 0.2

  if (family == "poisson_gamma") {
    mu <- exp(2 + w_spatial)
    y_num <- rpois(n_sites, mu)
    # Apply ZI
    zi_mask <- rbinom(n_sites, 1, pi_zi)
    y_num[zi_mask == 1] <- 0

    y_denom <- rep(0L, n_sites)
    y_denom_cont <- rgamma(n_sites, shape = 5, rate = 5 / exp(2))
  } else if (family == "negbin_negbin") {
    mu_num <- exp(2 + w_spatial)
    mu_denom <- exp(2.5 + 0.5 * w_spatial)
    y_num <- rnbinom(n_sites, size = 5, mu = mu_num)
    y_denom <- rnbinom(n_sites, size = 8, mu = mu_denom)

    # Apply ZI to numerator
    zi_mask <- rbinom(n_sites, 1, pi_zi)
    y_num[zi_mask == 1] <- 0

    y_denom_cont <- rep(0.0, n_sites)
  } else {  # binomial
    p <- plogis(w_spatial)
    trials <- rpois(n_sites, 30) + 10
    y_num <- rbinom(n_sites, trials, p)

    # Apply ZI
    zi_mask <- rbinom(n_sites, 1, pi_zi)
    y_num[zi_mask == 1] <- 0

    y_denom <- trials
    y_denom_cont <- rep(0.0, n_sites)
  }

  data.frame(
    x = coords$x / side,
    y = coords$y / side,
    y_num = y_num,
    y_denom = y_denom,
    y_denom_cont = y_denom_cont,
    site = 1:n_sites,
    adj_list = I(adj_list)
  )
}
