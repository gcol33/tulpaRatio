#!/usr/bin/env Rscript
# Benchmark hot-path optimizations (updated 2026-03-06)
# Baselines: v1.3.1 (before pointer views), Current: v1.3.2 (pointer views + unrolling)

library(tulpaRatio)

set.seed(42)
N <- 500
x <- rnorm(N)
site_id <- rep(1:50, each = 10)
time_id <- rep(1:20, times = 25)

# --- PG base ---
cat("=== PG base ===\n")
df <- data.frame(y = rpois(N, exp(0.5 + 0.3*x)), effort = rgamma(N, 10, 10), x = x)
t0 <- proc.time()[3]
fit <- tratio(y | effort ~ x, data = df, family = ratiod_poisson_gamma(),
              mode = "hmc",
              control = list(iter = 500, warmup = 250, chains = 1, verbose = FALSE))
cat(sprintf("  Time: %.2fs (baseline: 0.05s)\n", proc.time()[3] - t0))

# --- PG+RE ---
cat("=== PG+RE ===\n")
df2 <- data.frame(y = rpois(N, exp(0.5 + 0.3*x)), effort = rgamma(N, 10, 10),
                  x = x, site = site_id)
t0 <- proc.time()[3]
fit <- tratio(y | effort ~ x + (1 | site), data = df2,
              family = ratiod_poisson_gamma(),
              mode = "hmc",
              control = list(iter = 500, warmup = 250, chains = 1, verbose = FALSE))
cat(sprintf("  Time: %.2fs (baseline: 0.04s)\n", proc.time()[3] - t0))

# --- NB base ---
cat("=== NB base ===\n")
df3 <- data.frame(y_num = rnbinom(N, mu=10, size=5), y_den = rnbinom(N, mu=10, size=5), x = x)
t0 <- proc.time()[3]
fit <- tratio(y_num | y_den ~ x, data = df3, family = ratiod_negbin_negbin(),
              mode = "hmc",
              control = list(iter = 500, warmup = 250, chains = 1, verbose = FALSE))
cat(sprintf("  Time: %.2fs (baseline: 0.06s)\n", proc.time()[3] - t0))

# --- NB+RE ---
cat("=== NB+RE ===\n")
df4 <- data.frame(y_num = rnbinom(N, mu=10, size=5), y_den = rnbinom(N, mu=10, size=5),
                  x = x, site = site_id)
t0 <- proc.time()[3]
fit <- tratio(y_num | y_den ~ x + (1 | site), data = df4,
              family = ratiod_negbin_negbin(),
              mode = "hmc",
              control = list(iter = 500, warmup = 250, chains = 1, verbose = FALSE))
cat(sprintf("  Time: %.2fs (baseline: 0.05s)\n", proc.time()[3] - t0))

# --- Bin base ---
cat("=== Bin base ===\n")
df5 <- data.frame(y = rbinom(N, 20, 0.5), trials = 20L, x = x)
t0 <- proc.time()[3]
fit <- tratio(y | trials ~ x, data = df5, family = ratiod_binomial(),
              mode = "hmc",
              control = list(iter = 500, warmup = 250, chains = 1, verbose = FALSE))
cat(sprintf("  Time: %.2fs (baseline: 0.06s)\n", proc.time()[3] - t0))

# --- NB+ICAR ---
cat("=== NB+ICAR ===\n")
n_sites <- 50
adj <- matrix(0L, n_sites, n_sites)
for (i in 1:(n_sites-1)) { adj[i, i+1] <- 1L; adj[i+1, i] <- 1L }
df6 <- data.frame(y_num = rnbinom(N, mu=10, size=5), y_den = rnbinom(N, mu=10, size=5),
                  x = x, site = site_id)
t0 <- proc.time()[3]
fit <- tratio(y_num | y_den ~ x, data = df6, family = ratiod_negbin_negbin(),
              spatial = spatial_car(adj, group_var = "site"),
              mode = "hmc",
              control = list(iter = 500, warmup = 250, chains = 1, verbose = FALSE))
cat(sprintf("  Time: %.2fs (baseline: 0.33s)\n", proc.time()[3] - t0))

# --- NB+AR1 ---
cat("=== NB+AR1 ===\n")
df7 <- data.frame(y_num = rnbinom(N, mu=10, size=5), y_den = rnbinom(N, mu=10, size=5),
                  x = x, site = site_id, time = time_id)
t0 <- proc.time()[3]
fit <- tratio(y_num | y_den ~ x, data = df7, family = ratiod_negbin_negbin(),
              temporal = temporal_ar1(time_var = "time", group_var = "site"),
              mode = "hmc",
              control = list(iter = 500, warmup = 250, chains = 1, verbose = FALSE))
cat(sprintf("  Time: %.2fs (baseline: 0.85s)\n", proc.time()[3] - t0))

# --- PG+ICAR+AR1 ---
cat("=== PG+ICAR+AR1 ===\n")
df8 <- data.frame(y = rpois(N, exp(0.5 + 0.3*x)), effort = rgamma(N, 10, 10),
                  x = x, site = site_id, time = time_id)
t0 <- proc.time()[3]
fit <- tratio(y | effort ~ x, data = df8, family = ratiod_poisson_gamma(),
              spatial = spatial_car(adj, group_var = "site"),
              temporal = temporal_ar1(time_var = "time", group_var = "site"),
              mode = "hmc",
              control = list(iter = 500, warmup = 250, chains = 1, verbose = FALSE))
cat(sprintf("  Time: %.2fs (baseline: 1.05s)\n", proc.time()[3] - t0))

cat("\nDone.\n")
