# Benchmark SVC and TVC models (rows 26, 27, 56, 57, 88, 89)
library(tulpaRatio)
# Standard benchmark parameters
N_OBS <- 500
N_ITER <- 500
N_WARMUP <- 250
N_CHAINS <- 1
N_SITES <- 50
N_TIMES <- 20

set.seed(42)

# Generate base data with coordinates
df <- data.frame(
  lon = runif(N_OBS, 0, 10),
  lat = runif(N_OBS, 0, 10),
  x1 = rnorm(N_OBS),
  time = sample(1:N_TIMES, N_OBS, replace = TRUE)
)

# Generate responses
df$y_num <- rpois(N_OBS, lambda = 10)
df$y_denom <- rpois(N_OBS, lambda = 20)

# For binomial
df$y_denom_bin <- rpois(N_OBS, lambda = 50) + 10
df$y_num_bin <- rbinom(N_OBS, size = df$y_denom_bin, prob = 0.3)

results <- list()

# Helper function to run benchmark
run_bench <- function(name, expr) {
  cat("\n=== Benchmarking:", name, "===\n")
  tryCatch({
    t <- system.time(eval(expr))["elapsed"]
    cat("  Time:", round(t, 1), "s\n")
    t
  }, error = function(e) {
    cat("  ERROR:", e$message, "\n")
    NA
  })
}

# ============================================================================
# Row 26: poisson_gamma + SVC
# ============================================================================
cat("\n### Row 26: poisson_gamma + SVC ###\n")
results$pg_svc <- run_bench("poisson_gamma + SVC", quote({
  fit <- tratio(
    y_num | y_denom ~ x1,
    data = df,
    family = ratiod_poisson_gamma(),
    spatial = spatial_svc(coords = ~ lon + lat, terms = 1, nn = 15),
    control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, gradient_mode = "H", verbose = FALSE)
  )
}))

# ============================================================================
# Row 27: poisson_gamma + TVC
# ============================================================================
cat("\n### Row 27: poisson_gamma + TVC ###\n")
results$pg_tvc <- run_bench("poisson_gamma + TVC", quote({
  fit <- tratio(
    y_num | y_denom ~ x1,
    data = df,
    family = ratiod_poisson_gamma(),
    temporal = temporal_tvc(time_var = "time", terms = 1),
    control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, gradient_mode = "H", verbose = FALSE)
  )
}))

# ============================================================================
# Row 56: negbin_negbin + SVC
# ============================================================================
cat("\n### Row 56: negbin_negbin + SVC ###\n")
results$nb_svc <- run_bench("negbin_negbin + SVC", quote({
  fit <- tratio(
    y_num | y_denom ~ x1,
    data = df,
    family = ratiod_negbin_negbin(),
    spatial = spatial_svc(coords = ~ lon + lat, terms = 1, nn = 15),
    control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, gradient_mode = "H", verbose = FALSE)
  )
}))

# ============================================================================
# Row 57: negbin_negbin + TVC
# ============================================================================
cat("\n### Row 57: negbin_negbin + TVC ###\n")
results$nb_tvc <- run_bench("negbin_negbin + TVC", quote({
  fit <- tratio(
    y_num | y_denom ~ x1,
    data = df,
    family = ratiod_negbin_negbin(),
    temporal = temporal_tvc(time_var = "time", terms = 1),
    control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, gradient_mode = "H", verbose = FALSE)
  )
}))

# ============================================================================
# Row 88: binomial + SVC
# ============================================================================
cat("\n### Row 88: binomial + SVC ###\n")
results$bin_svc <- run_bench("binomial + SVC", quote({
  fit <- tratio(
    y_num_bin | y_denom_bin ~ x1,
    data = df,
    family = ratiod_binomial(),
    spatial = spatial_svc(coords = ~ lon + lat, terms = 1, nn = 15),
    control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, gradient_mode = "H", verbose = FALSE)
  )
}))

# ============================================================================
# Row 89: binomial + TVC
# ============================================================================
cat("\n### Row 89: binomial + TVC ###\n")
results$bin_tvc <- run_bench("binomial + TVC", quote({
  fit <- tratio(
    y_num_bin | y_denom_bin ~ x1,
    data = df,
    family = ratiod_binomial(),
    temporal = temporal_tvc(time_var = "time", terms = 1),
    control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, gradient_mode = "H", verbose = FALSE)
  )
}))

# ============================================================================
# Summary
# ============================================================================
cat("\n\n========================================\n")
cat("BENCHMARK SUMMARY (SVC/TVC)\n")
cat("========================================\n\n")

cat(sprintf("| %-4s | %-15s | %-8s | %8s |\n", "Row", "Model", "Feature", "H(s)"))
cat("|------|-----------------|----------|----------|\n")
cat(sprintf("| %-4d | %-15s | %-8s | %8.1f |\n", 26, "poisson_gamma", "SVC", results$pg_svc))
cat(sprintf("| %-4d | %-15s | %-8s | %8.1f |\n", 27, "poisson_gamma", "TVC", results$pg_tvc))
cat(sprintf("| %-4d | %-15s | %-8s | %8.1f |\n", 56, "negbin_negbin", "SVC", results$nb_svc))
cat(sprintf("| %-4d | %-15s | %-8s | %8.1f |\n", 57, "negbin_negbin", "TVC", results$nb_tvc))
cat(sprintf("| %-4d | %-15s | %-8s | %8.1f |\n", 88, "binomial", "SVC", results$bin_svc))
cat(sprintf("| %-4d | %-15s | %-8s | %8.1f |\n", 89, "binomial", "TVC", results$bin_tvc))

# Save results
saveRDS(results, "benchmarks/results_svc_tvc.rds")
cat("\nResults saved to benchmarks/results_svc_tvc.rds\n")
