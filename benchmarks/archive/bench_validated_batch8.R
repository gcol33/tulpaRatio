# Validated Benchmark Batch 8: Remaining binomial validations
# Focus: Binomial rows that CAN be validated against brms but haven't been yet
# Standard parameters: N=500, iter=500, warmup=250, chains=1 (for timing)
#
# Rows targeted:
#   - Row 64: binomial + crossed RE
#   - Row 66: binomial + BYM2
#   - Row 78: binomial + OI (one-inflated)
#   - Row 79: binomial + ZOIB (zero-one inflated)
#   - Row 89: binomial + TVC (time-varying coefficients)

library(tulpaRatio)
library(brms)
library(cmdstanr)
set.seed(42)

# ============================================================
# DATA SETUP
# ============================================================

N <- 500
N_SITES <- 50
N_TIMES <- 20
x <- rnorm(N)
x2 <- rnorm(N)
site <- factor(rep(1:N_SITES, length.out = N))
site2 <- factor(sample(1:20, N, replace = TRUE))
time <- rep(1:N_TIMES, length.out = N)
time_factor <- factor(time)

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

# Binomial data (standard)
trials <- sample(10:50, N, replace = TRUE)
y_bin <- rbinom(N, trials, plogis(0.5 + 0.3*x + 0.2*x2))
df_bin <- data.frame(y = y_bin, trials = trials, x = x, x2 = x2, site = site,
                     site2 = site2, time = time, time_factor = time_factor,
                     lon = lon, lat = lat, spatial_site = spatial_site)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

time_H_only <- function(nd_call, timeout = 300) {
  times <- setNames(rep(NA_real_, 4), c("H", "A", "A_t", "N"))

  timing_call <- nd_call
  timing_call$gradient_mode <- "H"
  timing_call$iter <- 500
  timing_call$warmup <- 250
  timing_call$chains <- 1
  timing_call$verbose <- FALSE

  tryCatch({
    t <- system.time({ eval(timing_call) })["elapsed"]
    times["H"] <- round(t, 1)
    cat(sprintf("  H: %.1fs\n", t))
  }, error = function(e) {
    cat(sprintf("  H: ERR(%s)\n", substr(conditionMessage(e), 1, 40)))
  })

  times
}

validate_stan <- function(nd_fit, brms_fit, param = "x") {
  if (is.null(nd_fit) || is.null(nd_fit$draws)) {
    return(list(nd = NA, brms = NA, diff = NA, pass = NA, error = "numdenom fit NULL"))
  }

  nd_draws <- tryCatch(as.matrix(nd_fit$draws), error = function(e) NULL)
  if (is.null(nd_draws)) {
    return(list(nd = NA, brms = NA, diff = NA, pass = NA, error = "draws conversion failed"))
  }

  cols <- colnames(nd_draws)
  nd_col <- grep("^beta_num\\[2\\]$|^beta\\[2\\]$", cols, value = TRUE)[1]

  if (is.null(nd_col) || is.na(nd_col)) {
    return(list(nd = NA, brms = NA, diff = NA, pass = NA,
                error = paste("param not found in:", paste(head(cols, 5), collapse=", "))))
  }

  nd_est <- mean(nd_draws[, nd_col])
  nd_sd <- sd(nd_draws[, nd_col])

  brms_sum <- fixef(brms_fit)
  brms_est <- brms_sum[param, "Estimate"]
  brms_se <- brms_sum[param, "Est.Error"]

  diff <- abs(nd_est - brms_est)
  threshold <- 2 * max(nd_sd, brms_se)

  list(nd = nd_est, brms = brms_est, diff = diff, pass = diff < threshold)
}

run_bench <- function(row, name, nd_call, brms_call = NULL, skip_stan = FALSE, val_chains = 2) {
  cat(sprintf("\n========== Row %d: %s ==========\n", row, name))
  result <- list(row = row, name = name)

  # H-only timing
  cat("Gradient timing:\n")
  result$times <- time_H_only(nd_call)

  # Validation
  if (!skip_stan && !is.null(brms_call)) {
    cat(sprintf("Validation (numdenom, chains=%d)... ", val_chains))
    flush.console()
    tryCatch({
      nd_call_full <- nd_call
      nd_call_full$iter <- 1000
      nd_call_full$warmup <- 500
      nd_call_full$chains <- val_chains
      nd_call_full$verbose <- FALSE

      t_nd <- system.time({ fit_nd <- eval(nd_call_full) })["elapsed"]
      result$time_nd <- round(t_nd, 1)
      cat(sprintf("%.1fs\n", t_nd))

      cat("Validation (brms)... ")
      flush.console()
      t_brms <- system.time({ fit_brms <- eval(brms_call) })["elapsed"]
      result$time_brms <- round(t_brms, 1)
      cat(sprintf("%.1fs\n", t_brms))

      val <- validate_stan(fit_nd, fit_brms)

      if (!is.null(val$error)) {
        cat(sprintf("  ERROR: %s\n", val$error))
        result$error <- val$error
      } else {
        result$diff <- round(val$diff, 4)
        result$pass <- val$pass
        result$speedup <- round(result$time_brms / result$time_nd, 1)
        cat(sprintf("  nd=%.3f, brms=%.3f, diff=%.4f => %s (%.1fx)\n",
                    val$nd, val$brms, val$diff, if(val$pass) "PASS" else "FAIL", result$speedup))
      }
    }, error = function(e) {
      cat(sprintf("ERROR: %s\n", conditionMessage(e)))
      result$error <<- conditionMessage(e)
    })
  } else {
    result$note <- "Stan validation skipped (no brms equivalent)"
  }

  result
}

# ============================================================
# BATCH 8: REMAINING BINOMIAL VALIDATIONS
# ============================================================

results <- list()

# ----- Row 64: binomial + crossed RE -----
results[["64"]] <- run_bench(64, "bin_crossed",
  nd_call = quote(tratio(y | trials ~ x + (1|site) + (1|site2), data = df_bin,
                         family = ratiod_binomial())),
  brms_call = quote(brm(y | trials(trials) ~ x + (1|site) + (1|site2), data = df_bin,
                        family = binomial(), iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# ----- Row 66: binomial + BYM2 -----
results[["66"]] <- run_bench(66, "bin_bym2",
  nd_call = quote(tratio(y | trials ~ x + (1|site), data = df_bin,
                         family = ratiod_binomial(),
                         spatial = spatial_bym2(adj_mat, spatial_level = "group", group_var = "spatial_site"))),
  brms_call = quote(brm(y | trials(trials) ~ x + (1|site) + (1|spatial_site), data = df_bin,
                        family = binomial(), iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# ----- Row 70: binomial + pCAR -----
results[["70"]] <- run_bench(70, "bin_pcar",
  nd_call = quote(tratio(y | trials ~ x + (1|site), data = df_bin,
                         family = ratiod_binomial(),
                         spatial = spatial_car(adj_mat, proper = TRUE,
                                               spatial_level = "group", group_var = "spatial_site"))),
  brms_call = quote(brm(y | trials(trials) ~ x + (1|site) + (1|spatial_site), data = df_bin,
                        family = binomial(), iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# ----- Row 78: binomial + OI (one-inflated) -----
# OI doesn't have brms equivalent - skip Stan validation
results[["78"]] <- run_bench(78, "bin_oi",
  nd_call = quote(tratio(y | trials ~ x + (1|site), data = df_bin,
                         family = ratiod_oi_binomial())),
  skip_stan = TRUE
)

# ----- Row 79: binomial + ZOIB (zero-one inflated) -----
# ZOIB doesn't have brms equivalent - skip Stan validation
results[["79"]] <- run_bench(79, "bin_zoib",
  nd_call = quote(tratio(y | trials ~ x + (1|site), data = df_bin,
                         family = ratiod_zoi_binomial())),
  skip_stan = TRUE
)

# ----- Row 89: binomial + TVC -----
# TVC validation uses brms with time random intercepts as proxy
# (brms doesn't support true TVC, so we validate fixed effect only)
results[["89"]] <- run_bench(89, "bin_tvc",
  nd_call = quote(tratio(y | trials ~ x + (1|site), data = df_bin,
                         family = ratiod_binomial(),
                         temporal = temporal_tvc("time_factor", ~ x))),
  brms_call = quote(brm(y | trials(trials) ~ x + (1|site) + (x|time_factor), data = df_bin,
                        family = binomial(), iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# ============================================================
# SUMMARY
# ============================================================

cat("\n\n")
cat(paste(rep("=", 90), collapse = ""), "\n")
cat("BATCH 8 RESULTS SUMMARY\n")
cat(paste(rep("=", 90), collapse = ""), "\n")

# Gradient timing table
cat("\nGRADIENT TIMING (seconds):\n")
cat(sprintf("%-6s %-20s %8s\n", "Row", "Model", "H"))
cat(paste(rep("-", 40), collapse = ""), "\n")
for (r in results) {
  if (!is.null(r$times)) {
    cat(sprintf("%-6s %-20s %8.1f\n",
                r$row, r$name,
                ifelse(is.na(r$times["H"]), NA, r$times["H"])))
  }
}

# Validation table
cat("\n\nSTAN VALIDATION:\n")
cat(sprintf("%-6s %-20s %10s %10s %8s %8s %s\n",
            "Row", "Model", "numdenom", "brms/Stan", "Speedup", "Diff", "Status"))
cat(paste(rep("-", 80), collapse = ""), "\n")

n_pass <- 0
n_total <- 0
n_skip <- 0

for (r in results) {
  if (!is.null(r$pass)) {
    n_total <- n_total + 1
    if (r$pass) n_pass <- n_pass + 1
    time_other <- if (!is.null(r$time_brms)) r$time_brms else r$time_stan
    cat(sprintf("%-6s %-20s %10.1fs %10.1fs %8.1fx %8.4f %s\n",
                r$row, r$name, r$time_nd, time_other, r$speedup, r$diff,
                if(r$pass) "PASS" else "FAIL"))
  } else if (!is.null(r$note)) {
    n_skip <- n_skip + 1
    cat(sprintf("%-6s %-20s SKIPPED (%s)\n", r$row, r$name, r$note))
  } else if (!is.null(r$error)) {
    cat(sprintf("%-6s %-20s ERROR: %s\n", r$row, r$name, r$error))
  }
}

cat(paste(rep("-", 80), collapse = ""), "\n")
cat(sprintf("Validated: %d/%d passed, %d skipped\n", n_pass, n_total, n_skip))

# Save results
saveRDS(results, "benchmarks/results_batch8.rds")
cat("\nResults saved to benchmarks/results_batch8.rds\n")

# Print rows to update in gradient_methods.md
cat("\n\nUPDATE gradient_methods.md:\n")
cat("Add ✓Stan to Notes column for these rows:\n")
for (r in results) {
  if (!is.null(r$pass) && r$pass) {
    cat(sprintf("  Row %s: %s - PASS\n", r$row, r$name))
  }
}
