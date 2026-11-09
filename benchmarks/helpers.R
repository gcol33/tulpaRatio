# Benchmark helper functions
# Used by run_all.R to validate gradient_methods.md speedups

library(tulpaRatio)

# Check if brms is available for Stan comparison
has_brms <- requireNamespace("brms", quietly = TRUE)
if (has_brms) {
  library(brms)
}

# ============================================================================
# Data generation
# ============================================================================

generate_data <- function(family, n = 150, n_groups = 15, n_times = 10,
                          spatial = "none", temporal = "none") {
  set.seed(42)

  # Base structure
  if (temporal != "none") {
    data <- expand.grid(group = 1:n_groups, time = 1:n_times)
    n <- nrow(data)
  } else {
    data <- data.frame(group = rep(1:n_groups, length.out = n))
  }

  data$x <- rnorm(n)
  data$z <- rnorm(n)  # For slopes
  data$group <- factor(data$group)

  # Spatial coordinates
  if (spatial != "none") {
    data$lon <- runif(n, 0, 10)
    data$lat <- runif(n, 0, 10)
  }

  # Time index for temporal models
  if (temporal != "none") {
    data$time_f <- factor(data$time)
  }

  # Generate response based on family
  if (family == "poisson_gamma") {
    mu_num <- exp(2 + 0.3 * data$x)
    data$y_num <- rpois(n, mu_num)
    data$y_denom <- rgamma(n, 5, 1)
  } else if (family == "negbin_negbin") {
    mu_num <- exp(2 + 0.3 * data$x)
    data$y_num <- rnbinom(n, size = 5, mu = mu_num)
    data$y_denom <- rnbinom(n, size = 5, mu = 5) + 1
  } else if (family == "binomial") {
    p <- plogis(0.5 + 0.3 * data$x)
    trials <- sample(10:50, n, replace = TRUE)
    data$y_num <- rbinom(n, trials, p)
    data$y_denom <- trials
  }

  data
}

# Create adjacency matrix for spatial models
create_adjacency <- function(n_groups) {
  adj <- matrix(0, n_groups, n_groups)
  for (i in 1:(n_groups - 1)) {
    adj[i, i + 1] <- adj[i + 1, i] <- 1
  }
  adj
}

# ============================================================================
# Benchmark single configuration
# ============================================================================

benchmark_config <- function(row, family, re, spatial, temporal, zi, grad,
                             n_iter = 400, n_warmup = 200) {
  cat(sprintf("Row %2d: %s + %s", row, family, re))
  if (spatial != "none") cat(sprintf(" + %s", spatial))
  if (temporal != "none") cat(sprintf(" + %s", temporal))
  if (zi != "none") cat(sprintf(" + %s", zi))
  cat("... ")

  # Generate data
  data <- generate_data(family, spatial = spatial, temporal = temporal)
  n_groups <- 15

  # Build tulpaRatio formula
  if (re == "none") {
    nd_formula <- y_num | y_denom ~ x
  } else if (re == "intercept") {
    nd_formula <- y_num | y_denom ~ x + (1 | group)
  } else if (re == "slopes") {
    nd_formula <- y_num | y_denom ~ x + (1 + z | group)
  } else if (re == "crossed") {
    data$group2 <- factor(sample(1:5, nrow(data), replace = TRUE))
    nd_formula <- y_num | y_denom ~ x + (1 | group) + (1 | group2)
  }

  # Family
  nd_family <- switch(family,
    "poisson_gamma" = ratiod_poisson_gamma(),
    "negbin_negbin" = ratiod_negbin_negbin(),
    "binomial" = ratiod_binomial()
  )

  # Spatial
  nd_spatial <- NULL
  if (spatial == "ICAR") {
    adj <- create_adjacency(n_groups)
    nd_spatial <- spatial_car(adj, level = "group", group_var = "group")
  } else if (spatial == "BYM2") {
    adj <- create_adjacency(n_groups)
    nd_spatial <- spatial_bym2(adj, level = "group", group_var = "group")
  }

 # Temporal
  nd_temporal <- NULL
  if (temporal == "RW1") {
    nd_temporal <- temporal_rw1("time")
  } else if (temporal == "RW2") {
    nd_temporal <- temporal_rw2("time")
  } else if (temporal == "AR1") {
    nd_temporal <- temporal_ar1("time")
  }

  # ZI
  nd_zi <- NULL
  if (zi == "ZI") {
    # Add zeros to data
    data$y_num <- ifelse(runif(nrow(data)) < 0.3, 0, data$y_num)
    if (family == "poisson_gamma") {
      nd_zi <- zi_poisson()
    } else {
      nd_zi <- zi_negbin()
    }
  } else if (zi == "Hurdle") {
    data$y_num <- ifelse(runif(nrow(data)) < 0.3, 0, data$y_num)
    if (family == "poisson_gamma") {
      nd_zi <- tulpaRatio::hurdle_poisson()
    } else {
      nd_zi <- tulpaRatio::hurdle_negbin()
    }
  }

  # Benchmark tulpaRatio
  nd_time <- tryCatch({
    t <- system.time({
      fit <- tratio(nd_formula, data = data,
                    family = nd_family,
                    spatial = nd_spatial,
                    temporal = nd_temporal,
                    zi = nd_zi,
                    control = list(iter = n_iter, warmup = n_warmup, chains = 1))
    })
    t["elapsed"]
  }, error = function(e) {
    cat(sprintf("ERROR: %s\n", e$message))
    NA
  })

  # Benchmark brms/Stan (if available)
  stan_time <- NA
  if (has_brms && !is.na(nd_time)) {
    stan_time <- tryCatch({
      # Build brms formula (simplified - just numerator)
      if (re == "none") {
        brms_formula <- y_num ~ x
      } else if (re %in% c("intercept", "slopes", "crossed")) {
        brms_formula <- y_num ~ x + (1 | group)
      }

      # brms family
      brms_family <- switch(family,
        "poisson_gamma" = poisson(),
        "negbin_negbin" = negbinomial(),
        "binomial" = binomial()
      )

      t <- system.time({
        if (family == "binomial") {
          fit_stan <- brm(y_num | trials(y_denom) ~ x + (1 | group),
                          data = data, family = brms_family,
                          iter = n_iter, warmup = n_warmup,
                          chains = 1, refresh = 0, silent = 2)
        } else if (zi == "ZI") {
          brms_family <- switch(family,
            "poisson_gamma" = zero_inflated_poisson(),
            "negbin_negbin" = zero_inflated_negbinomial()
          )
          fit_stan <- brm(brms_formula, data = data, family = brms_family,
                          iter = n_iter, warmup = n_warmup,
                          chains = 1, refresh = 0, silent = 2)
        } else if (zi == "Hurdle") {
          brms_family <- switch(family,
            "poisson_gamma" = brms::hurdle_poisson(),
            "negbin_negbin" = brms::hurdle_negbinomial()
          )
          fit_stan <- brm(brms_formula, data = data, family = brms_family,
                          iter = n_iter, warmup = n_warmup,
                          chains = 1, refresh = 0, silent = 2)
        } else {
          fit_stan <- brm(brms_formula, data = data, family = brms_family,
                          iter = n_iter, warmup = n_warmup,
                          chains = 1, refresh = 0, silent = 2)
        }
      })
      t["elapsed"]
    }, error = function(e) {
      NA
    })
  }

  # Calculate speedup
  speedup <- NA
  if (!is.na(nd_time) && !is.na(stan_time) && stan_time > 0) {
    speedup <- stan_time / nd_time
  }

  if (!is.na(speedup)) {
    cat(sprintf("%.1fx\n", speedup))
  } else if (!is.na(nd_time)) {
    cat(sprintf("OK (%.1fs, no Stan comparison)\n", nd_time))
    speedup <- NA
  }

  data.frame(
    row = row,
    family = family,
    re = re,
    spatial = spatial,
    temporal = temporal,
    zi = zi,
    grad = grad,
    nd_time = nd_time,
    stan_time = stan_time,
    speedup = speedup,
    stringsAsFactors = FALSE
  )
}
