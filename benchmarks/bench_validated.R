# Integrated Benchmark + Stan Validation
#
# IMPORTANT: For poisson_gamma and negbin_negbin families, brms comparison is INVALID
# because tulpaRatio models BOTH numerator and denominator as random variables,
# while brms with offset() treats the denominator as FIXED.
#
# Use bench_joint_validation.R for pg/nb families (custom Stan models)
# This script validates BINOMIAL family only (trials are fixed, so brms is valid)
#
# Standard parameters: N=500, iter=1000, warmup=500, chains=2

library(tulpaRatio)
library(brms)
set.seed(123)

# Setup data
N <- 500
N_SITES <- 50
N_TIMES <- 20
x <- rnorm(N)
site <- factor(rep(1:N_SITES, length.out = N))
time <- factor(rep(1:N_TIMES, length.out = N))

# Spatial grid
n_side <- ceiling(sqrt(N_SITES))
grid <- expand.grid(lon = 1:n_side, lat = 1:n_side)[1:N_SITES, ]
site_int <- as.integer(site)
lon <- grid$lon[site_int]
lat <- grid$lat[site_int]

# Adjacency matrix
adj_mat <- matrix(0, N_SITES, N_SITES)
for (i in 1:N_SITES) {
  for (j in 1:N_SITES) {
    if (i != j) {
      dist <- sqrt((grid$lon[i] - grid$lon[j])^2 + (grid$lat[i] - grid$lat[j])^2)
      if (dist <= 1.5) adj_mat[i, j] <- 1
    }
  }
}
spatial_site <- factor(rep(1:N_SITES, length.out = N))

# Generate BINOMIAL data (trials are fixed - valid for brms comparison)
trials <- sample(10:50, N, replace = TRUE)
y_bin <- rbinom(N, trials, plogis(0.5 + 0.3*x))
df_bin <- data.frame(y = y_bin, trials = trials, x = x, site = site, time = time,
                     lon = lon, lat = lat, spatial_site = spatial_site)

# ZI binomial data
df_bin_zi <- df_bin
df_bin_zi$y <- ifelse(runif(N) < 0.3, 0, df_bin_zi$y)

# Validation function
validate_against_stan <- function(nd_fit, brms_fit, param_name = "x") {
  # Extract tulpaRatio draws
  nd_draws <- as.matrix(nd_fit$draws)
  nd_col <- grep(paste0("beta_num\\[2\\]"), colnames(nd_draws), value = TRUE)[1]
  if (is.na(nd_col)) nd_col <- grep(param_name, colnames(nd_draws), value = TRUE)[1]
  nd_x <- mean(nd_draws[, nd_col])
  nd_x_sd <- sd(nd_draws[, nd_col])

  # Extract brms fixed effects
  brms_sum <- fixef(brms_fit)
  brms_x <- brms_sum[param_name, "Estimate"]
  brms_x_se <- brms_sum[param_name, "Est.Error"]

  diff <- abs(nd_x - brms_x)
  threshold <- 2 * max(nd_x_sd, brms_x_se)
  pass <- diff < threshold

  list(
    nd_x = nd_x,
    nd_sd = nd_x_sd,
    brms_x = brms_x,
    brms_se = brms_x_se,
    diff = diff,
    pass = pass
  )
}

# Main benchmark function
bench_validated <- function(name, row, nd_call, brms_call, true_x = 0.3,
                           run_4mode = FALSE, skip_stan = FALSE) {
  cat(sprintf("\n\n========== Row %d: %s ==========\n", row, name))
  cat(sprintf("True x = %.2f\n\n", true_x))

  result <- list(row = row, name = name, true_x = true_x)

  # Run tulpaRatio
  cat("Running tulpaRatio... ")
  flush.console()
  tryCatch({
    time_nd <- system.time({
      fit_nd <- eval(nd_call)
    })["elapsed"]
    result$time_nd <- time_nd
    result$divergent <- fit_nd$diagnostics$divergent
    cat(sprintf("%.1fs (divergent: %d)\n", time_nd, result$divergent))
  }, error = function(e) {
    cat(sprintf("ERROR: %s\n", conditionMessage(e)))
    result$error_nd <<- conditionMessage(e)
  })

  # Run brms/Stan
  if (!skip_stan) {
    cat("Running brms/Stan... ")
    flush.console()
    tryCatch({
      time_brms <- system.time({
        fit_brms <- eval(brms_call)
      })["elapsed"]
      result$time_brms <- time_brms
      cat(sprintf("%.1fs\n", time_brms))
    }, error = function(e) {
      cat(sprintf("ERROR: %s\n", conditionMessage(e)))
      result$error_brms <<- conditionMessage(e)
    })
  }

  # Validate
  if (!is.null(result$time_nd) && !is.null(result$time_brms)) {
    val <- validate_against_stan(fit_nd, fit_brms)
    result$nd_x <- val$nd_x
    result$brms_x <- val$brms_x
    result$diff <- val$diff
    result$pass <- val$pass
    result$speedup <- result$time_brms / result$time_nd

    cat(sprintf("\n=== VALIDATION ===\n"))
    cat(sprintf("tulpaRatio: x = %.4f (SD=%.4f)\n", val$nd_x, val$nd_sd))
    cat(sprintf("brms:     x = %.4f (SE=%.4f)\n", val$brms_x, val$brms_se))
    cat(sprintf("Diff: %.4f | Status: %s | Speedup: %.1fx\n",
                val$diff, if(val$pass) "PASS" else "FAIL", result$speedup))
  }

  result
}

results <- list()

cat("=======================================================\n")
cat("BINOMIAL FAMILY VALIDATION (brms comparison is valid)\n")
cat("=======================================================\n")
cat("For poisson_gamma and negbin_negbin, see bench_joint_validation.R\n")

# ============================================================
# BINOMIAL FAMILY (trials fixed - brms comparison IS valid)
# ============================================================

# Row 61: binomial base
results[["61"]] <- bench_validated("bin_base", 61,
  nd_call = quote(tratio(y | trials ~ x, data = df_bin, family = ratiod_binomial(),
                         control = list(iter = 1000, warmup = 500, chains = 2, verbose = FALSE))),
  brms_call = quote(brm(y | trials(trials) ~ x, data = df_bin, family = binomial(),
                        iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# Row 62: binomial + RE
results[["62"]] <- bench_validated("bin_re", 62,
  nd_call = quote(tratio(y | trials ~ x + (1|site), data = df_bin, family = ratiod_binomial(),
                         control = list(iter = 1000, warmup = 500, chains = 2, verbose = FALSE))),
  brms_call = quote(brm(y | trials(trials) ~ x + (1|site), data = df_bin, family = binomial(),
                        iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# Row 65: binomial + ICAR
results[["65"]] <- bench_validated("bin_icar", 65,
  nd_call = quote(tratio(y | trials ~ x + (1|site), data = df_bin, family = ratiod_binomial(),
                         spatial = spatial_car(adj_mat, spatial_level = "group", group_var = "spatial_site"),
                         control = list(iter = 1000, warmup = 500, chains = 2, verbose = FALSE))),
  brms_call = quote(brm(y | trials(trials) ~ x + (1|site) + (1|spatial_site),
                        data = df_bin, family = binomial(),
                        iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# Row 66: binomial + BYM2
results[["66"]] <- bench_validated("bin_bym2", 66,
  nd_call = quote(tratio(y | trials ~ x + (1|site), data = df_bin, family = ratiod_binomial(),
                         spatial = spatial_bym2(adj_mat, spatial_level = "group", group_var = "spatial_site"),
                         control = list(iter = 1000, warmup = 500, chains = 2, verbose = FALSE))),
  brms_call = quote(brm(y | trials(trials) ~ x + (1|site) + (1|spatial_site),
                        data = df_bin, family = binomial(),
                        iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# Row 71: binomial + RW1 temporal
results[["71"]] <- bench_validated("bin_rw1", 71,
  nd_call = quote(tratio(y | trials ~ x + (1|site), data = df_bin, family = ratiod_binomial(),
                         temporal = temporal_rw1("time"),
                         control = list(iter = 1000, warmup = 500, chains = 2, verbose = FALSE))),
  brms_call = quote(brm(y | trials(trials) ~ x + (1|site) + (1|time),
                        data = df_bin, family = binomial(),
                        iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# Row 76: binomial + ZI
results[["76"]] <- bench_validated("bin_zi", 76,
  nd_call = quote(tratio(y | trials ~ x + (1|site), data = df_bin_zi, family = ratiod_zi_binomial(),
                         control = list(iter = 1000, warmup = 500, chains = 2, verbose = FALSE))),
  brms_call = quote(brm(y | trials(trials) ~ x + (1|site),
                        data = df_bin_zi, family = zero_inflated_binomial(),
                        iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# Row 80: binomial + ICAR + RW1
results[["80"]] <- bench_validated("bin_icar_rw1", 80,
  nd_call = quote(tratio(y | trials ~ x + (1|site), data = df_bin, family = ratiod_binomial(),
                         spatial = spatial_car(adj_mat, spatial_level = "group", group_var = "spatial_site"),
                         temporal = temporal_rw1("time"),
                         control = list(iter = 1000, warmup = 500, chains = 2, verbose = FALSE))),
  brms_call = quote(brm(y | trials(trials) ~ x + (1|site) + (1|spatial_site) + (1|time),
                        data = df_bin, family = binomial(),
                        iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# ============================================================
# Summary
# ============================================================
cat("\n\n")
cat(paste(rep("=", 80), collapse = ""), "\n")
cat("BINOMIAL FAMILY VALIDATION SUMMARY\n")
cat(paste(rep("=", 80), collapse = ""), "\n")
cat(sprintf("%-8s %-15s %8s %8s %8s %6s %s\n",
            "Row", "Model", "tulpaRatio", "Stan", "Speedup", "Diff", "Status"))
cat(paste(rep("-", 80), collapse = ""), "\n")

n_pass <- 0
n_total <- 0

for (r in results) {
  if (!is.null(r$pass)) {
    n_total <- n_total + 1
    if (r$pass) n_pass <- n_pass + 1
    status <- if(r$pass) "PASS" else "FAIL"
    cat(sprintf("%-8s %-15s %8.1fs %8.1fs %8.1fx %6.4f %s\n",
                r$row, r$name, r$time_nd, r$time_brms, r$speedup, r$diff, status))
  } else if (!is.null(r$error_nd) || !is.null(r$error_brms)) {
    cat(sprintf("%-8s %-15s %s\n", r$row, r$name, "ERROR"))
  }
}

cat(paste(rep("-", 80), collapse = ""), "\n")
cat(sprintf("Validated: %d/%d passed\n", n_pass, n_total))

if (n_pass < n_total) {
  cat("\n*** WARNING: Some models failed validation! ***\n")
}

cat("\n")
cat("NOTE: For poisson_gamma and negbin_negbin validation,\n")
cat("run bench_joint_validation.R (uses custom Stan models).\n")

saveRDS(results, "benchmarks/results_validated.rds")
cat("\nResults saved to benchmarks/results_validated.rds\n")
