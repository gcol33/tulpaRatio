# Hand-coded and arena gradient timings
#
# Standard parameters: N=500, iter=500, warmup=250, chains=1

library(tulpaRatio)
set.seed(42)

# ============================================================
# DATA SETUP
# ============================================================

N <- 500
N_SITES <- 50
N_TIMES <- 20
x <- rnorm(N)
site <- factor(rep(1:N_SITES, length.out = N))
time <- rep(1:N_TIMES, length.out = N)
time_factor <- factor(time)

# Spatial grid and adjacency
n_side <- ceiling(sqrt(N_SITES))
grid <- expand.grid(lon = 1:n_side, lat = 1:n_side)[1:N_SITES, ]
site_int <- as.integer(site)
lon <- grid$lon[site_int]
lat <- grid$lat[site_int]

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

# Data frames for different families
eta_num <- 2 + 0.3 * x
eta_denom <- 4 + 0.2 * x

# negbin_negbin
y_nb <- rnbinom(N, mu = exp(eta_num), size = 5)
denom_nb <- rnbinom(N, mu = exp(eta_denom), size = 10)
denom_nb[denom_nb == 0] <- 1

df_nb <- data.frame(
  y = y_nb, denom = denom_nb, x = x, site = site, time = time,
  time_factor = time_factor, lon = lon, lat = lat, spatial_site = spatial_site
)

# binomial
trials <- sample(10:50, N, replace = TRUE)
y_bin <- rbinom(N, trials, plogis(0.5 + 0.3*x))
df_bin <- data.frame(
  y = y_bin, trials = trials, x = x, site = site, time = time,
  time_factor = time_factor, lon = lon, lat = lat, spatial_site = spatial_site
)

# poisson_gamma
y_pg <- rpois(N, exp(eta_num))
denom_pg <- rgamma(N, shape = 5, rate = 5 / exp(eta_denom))
denom_pg[denom_pg < 0.01] <- 0.01

df_pg <- data.frame(
  y = y_pg, denom = denom_pg, x = x, site = site, time = time,
  time_factor = time_factor, lon = lon, lat = lat, spatial_site = spatial_site
)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

time_gradient_modes <- function(nd_call, modes = c("H", "A_r"), timeout = 600) {
  times <- setNames(rep(NA_real_, length(modes)), modes)

  for (mode in modes) {
    timing_call <- nd_call
    timing_call$gradient_mode <- mode
    timing_call$iter <- 500
    timing_call$warmup <- 250
    timing_call$chains <- 1
    timing_call$verbose <- FALSE

    tryCatch({
      t <- system.time({ fit <- eval(timing_call) })["elapsed"]
      if (t > timeout) {
        times[mode] <- NA
        cat(sprintf("  %s: TIMEOUT (>%ds)\n", mode, timeout))
      } else {
        times[mode] <- round(t, 1)
        # Check for divergences
        if (!is.null(fit$diagnostics)) {
          div <- sum(fit$diagnostics$divergent)
          if (div > 0) {
            cat(sprintf("  %s: %.1fs (%d div)\n", mode, t, div))
          } else {
            cat(sprintf("  %s: %.1fs\n", mode, t))
          }
        } else {
          cat(sprintf("  %s: %.1fs\n", mode, t))
        }
      }
    }, error = function(e) {
      cat(sprintf("  %s: ERR(%s)\n", mode, substr(conditionMessage(e), 1, 50)))
    })
  }

  times
}

run_bench <- function(row, name, nd_call, modes = c("H", "A_r"), timeout = 600) {
  cat(sprintf("\n========== Row %d: %s ==========\n", row, name))
  result <- list(row = row, name = name)

  cat("Gradient timing:\n")
  result$times <- time_gradient_modes(nd_call, modes, timeout)

  result
}

# ============================================================
# BATCH 9: MISSING TIMINGS
# ============================================================

results <- list()

# ----- Row 45: negbin + MS_t (multi-scale temporal) -----
# This row has no timing at all
results[["45"]] <- run_bench(45, "nb_ms_t",
  nd_call = quote(tratio(y | denom ~ x + (1|site), data = df_nb,
                         family = ratiod_negbin_negbin(),
                         temporal = temporal_multiscale("time_factor"))),
  modes = c("H"),
  timeout = 600
)

# ----- Row 75: binomial + MS_t -----
# This row has no timing at all
results[["75"]] <- run_bench(75, "bin_ms_t",
  nd_call = quote(tratio(y | trials ~ x + (1|site), data = df_bin,
                         family = ratiod_binomial(),
                         temporal = temporal_multiscale("time_factor"))),
  modes = c("H"),
  timeout = 600
)

# ----- Row 8: poisson_gamma + HSGP -----
results[["8"]] <- run_bench(8, "pg_hsgp",
  nd_call = quote(tratio(y | denom ~ x + (1|site), data = df_pg,
                         family = ratiod_poisson_gamma(),
                         spatial = spatial_hsgp(coords = c("lon", "lat"), m = 8))),
  modes = c("H", "A_r")
)

# ----- Row 38: negbin + HSGP -----
results[["38"]] <- run_bench(38, "nb_hsgp",
  nd_call = quote(tratio(y | denom ~ x + (1|site), data = df_nb,
                         family = ratiod_negbin_negbin(),
                         spatial = spatial_hsgp(coords = c("lon", "lat"), m = 8))),
  modes = c("H", "A_r")
)

# ----- Row 68: binomial + HSGP -----
results[["68"]] <- run_bench(68, "bin_hsgp",
  nd_call = quote(tratio(y | trials ~ x + (1|site), data = df_bin,
                         family = ratiod_binomial(),
                         spatial = spatial_hsgp(coords = c("lon", "lat"), m = 8))),
  modes = c("H", "A_r")
)

# ----- Row 22: pg + HSGP + RW1 -----
results[["22"]] <- run_bench(22, "pg_hsgp_rw1",
  nd_call = quote(tratio(y | denom ~ x + (1|site), data = df_pg,
                         family = ratiod_poisson_gamma(),
                         spatial = spatial_hsgp(coords = c("lon", "lat"), m = 8),
                         temporal = temporal_rw1("time_factor"))),
  modes = c("H", "A_r")
)

# ----- Row 52: nb + HSGP + RW1 -----
results[["52"]] <- run_bench(52, "nb_hsgp_rw1",
  nd_call = quote(tratio(y | denom ~ x + (1|site), data = df_nb,
                         family = ratiod_negbin_negbin(),
                         spatial = spatial_hsgp(coords = c("lon", "lat"), m = 8),
                         temporal = temporal_rw1("time_factor"))),
  modes = c("H", "A_r")
)

# ----- Row 84: bin + HSGP + RW1 -----
results[["84"]] <- run_bench(84, "bin_hsgp_rw1",
  nd_call = quote(tratio(y | trials ~ x + (1|site), data = df_bin,
                         family = ratiod_binomial(),
                         spatial = spatial_hsgp(coords = c("lon", "lat"), m = 8),
                         temporal = temporal_rw1("time_factor"))),
  modes = c("H", "A_r")
)

# ============================================================
# SUMMARY
# ============================================================

cat("\n\n")
cat(paste(rep("=", 90), collapse = ""), "\n")
cat("BATCH 9 RESULTS SUMMARY\n")
cat(paste(rep("=", 90), collapse = ""), "\n")

cat("\nGRADIENT TIMING (seconds):\n")
cat(sprintf("%-6s %-20s %10s %10s %10s\n", "Row", "Model", "H", "A_r", "Speedup"))
cat(paste(rep("-", 60), collapse = ""), "\n")

for (r in results) {
  if (!is.null(r$times)) {
    h <- r$times["H"]
    at <- if ("A_r" %in% names(r$times)) r$times["A_r"] else NA
    speedup <- if (!is.na(h) && !is.na(at) && at > 0) round(at / h, 1) else NA

    cat(sprintf("%-6s %-20s %10s %10s %10s\n",
                r$row, r$name,
                ifelse(is.na(h), "-", sprintf("%.1f", h)),
                ifelse(is.na(at), "-", sprintf("%.1f", at)),
                ifelse(is.na(speedup), "-", sprintf("%.1fx", speedup))))
  }
}

# Save results
saveRDS(results, "benchmarks/results_timing_batch9.rds")
cat("\nResults saved to benchmarks/results_timing_batch9.rds\n")

# Print update instructions
cat("\n\nUPDATE gradient_methods.md with new timings:\n")
for (r in results) {
  if (!is.null(r$times)) {
    updates <- c()
    if (!is.na(r$times["H"])) updates <- c(updates, sprintf("H=%.1f", r$times["H"]))
    if ("A_r" %in% names(r$times) && !is.na(r$times["A_r"])) {
      updates <- c(updates, sprintf("A_r=%.1f", r$times["A_r"]))
    }
    if (length(updates) > 0) {
      cat(sprintf("  Row %s: %s\n", r$row, paste(updates, collapse = ", ")))
    }
  }
}
