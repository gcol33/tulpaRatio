#!/usr/bin/env Rscript
# BYM2 benchmark to verify OAS shrinkage fixes divergences
# Before fix: 5 divergences (dense mass never set, fell back to diagonal)
# After fix: expect 0 divergences (OAS enables dense mass with n < p)

library(tulpaRatio)
set.seed(42)
N_OBS <- 500
N_SITES <- 50
N_ITER <- 500
N_WARMUP <- 250

# Create spatial adjacency (grid)
# Build simple chain adjacency for grid
adj <- Matrix::sparseMatrix(
  i = c(1:(N_SITES-1), 2:N_SITES),
  j = c(2:N_SITES, 1:(N_SITES-1)),
  x = rep(1, 2*(N_SITES-1)),
  dims = c(N_SITES, N_SITES)
)
adj <- as(adj, "dgCMatrix")

# Simulate data
site_ids <- rep(1:N_SITES, length.out = N_OBS)
x <- rnorm(N_OBS)
eta_num <- 1.0 + 0.3 * x + rnorm(N_SITES, 0, 0.3)[site_ids]
eta_denom <- 0.5 + 0.1 * x + rnorm(N_SITES, 0, 0.3)[site_ids]
y_num <- MASS::rnegbin(N_OBS, mu = exp(eta_num), theta = 5)
y_denom <- MASS::rnegbin(N_OBS, mu = exp(eta_denom), theta = 5)

df <- data.frame(y_num = y_num, y_denom = y_denom, x = x, site = site_ids)

cat("=== BYM2 + OAS Shrinkage Benchmark ===\n")
cat(sprintf("N=%d, S=%d, iter=%d, warmup=%d\n\n", N_OBS, N_SITES, N_ITER, N_WARMUP))

# Fit BYM2 model with verbose to see dense mass + shrinkage info
t0 <- proc.time()["elapsed"]
fit <- tratio(
  y_num | y_denom ~ x + (1 | site),
  data = df,
  family = ratiod_negbin_negbin(),
  spatial = spatial_bym2(adj = adj, group_var = "site"),
  mode = "hmc",
  control = list(iter = N_ITER, warmup = N_WARMUP, chains = 1, seed = 123, verbose = TRUE, metric = "dense", gradient_mode = "H")
)
elapsed <- proc.time()["elapsed"] - t0

cat(sprintf("\n=== RESULTS ===\n"))
cat(sprintf("Time: %.1fs\n", elapsed))
cat(sprintf("Divergences: %d\n", fit$diagnostics$n_divergent))
cat(sprintf("Max treedepth hits: %d\n", sum(fit$diagnostics$treedepth >= fit$diagnostics$max_treedepth)))
cat(sprintf("Avg acceptance: %.3f\n", mean(fit$diagnostics$acceptance_rate)))
cat(sprintf("Avg treedepth: %.1f\n", mean(fit$diagnostics$treedepth)))

# Key test
if (fit$diagnostics$n_divergent == 0) {
  cat("\n*** PASS: 0 divergences ***\n")
} else {
  cat(sprintf("\n*** FAIL: %d divergences ***\n", fit$diagnostics$n_divergent))
}
