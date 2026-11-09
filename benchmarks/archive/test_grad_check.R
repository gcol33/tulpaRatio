# Check gradient correctness for the suspicious results
library(tulpaRatio)
set.seed(42)

check_grad <- function(label, fit_call) {
  cat(sprintf("\n=== %s ===\n", label))
  fit <- tryCatch(eval(fit_call), error = function(e) {
    cat(sprintf("  FIT ERROR: %s\n", e$message))
    NULL
  })
  if (is.null(fit)) return()

  # Check basic diagnostics
  draws <- fit$draws
  if (is.null(draws)) {
    cat("  No draws\n")
    return()
  }
  # Handle different draw formats
  if (is.list(draws)) {
    all_draws <- do.call(rbind, draws)
    n_chains <- length(draws)
  } else if (is.matrix(draws)) {
    all_draws <- draws
    n_chains <- 1
  } else {
    cat(sprintf("  Unknown draws format: %s\n", class(draws)))
    return()
  }
  n_params <- ncol(all_draws)
  cat(sprintf("  Chains: %d, Samples: %d, Params: %d\n",
              n_chains, nrow(all_draws), n_params))

  # Check for NaN/Inf in draws
  n_nan <- sum(is.nan(all_draws))
  n_inf <- sum(is.infinite(all_draws))
  cat(sprintf("  NaN: %d, Inf: %d\n", n_nan, n_inf))

  # Check posterior means are reasonable
  means <- colMeans(all_draws)
  sds <- apply(all_draws, 2, sd)
  cat(sprintf("  Mean range: [%.3f, %.3f]\n", min(means), max(means)))
  cat(sprintf("  SD range: [%.3f, %.3f]\n", min(sds), max(sds)))

  # Check for constant chains (gradient=0 symptom)
  const_cols <- sum(sds < 1e-10)
  cat(sprintf("  Constant params: %d / %d\n", const_cols, n_params))
}

# 1. PG+HSGP (was 75.8s, should be ~19s)
N <- 500
df_pg <- data.frame(
  y = rpois(N, 5),
  denom = rgamma(N, 10, 1),
  x = rnorm(N),
  lon = runif(N, 0, 10),
  lat = runif(N, 0, 10)
)
check_grad("PG+HSGP H", quote(
  tratio(y | denom ~ x, data = df_pg,
         family = ratiod_poisson_gamma(),
         spatial = spatial_hsgp(~ lon + lat),
         control = list(iter = 100, warmup = 50, chains = 1, gradient_mode = "H", verbose = FALSE))
))

# 2. PG+temporal_gp (was 0.4s, should be ~40s)
N2 <- 80
df_gpt <- data.frame(
  y = rpois(N2, 5),
  denom = rgamma(N2, 10, 1),
  x = rnorm(N2),
  site = rep(1:20, each = 4),
  time = rep(1:4, 20),
  lon = rep(runif(20, 0, 10), each = 4),
  lat = rep(runif(20, 0, 10), each = 4)
)
check_grad("PG+GP_t H", quote(
  tratio(y | denom ~ x + (1 | site), data = df_gpt,
         family = ratiod_poisson_gamma(),
         temporal = temporal_gp(time_var = "time"),
         control = list(iter = 100, warmup = 50, chains = 1, gradient_mode = "H", verbose = FALSE))
))

# 3. Compare PG+GP_t N mode vs H mode
cat("\n=== PG+GP_t: Compare H vs N mode draws ===\n")
set.seed(123)
fit_n <- tryCatch(
  tratio(y | denom ~ x + (1 | site), data = df_gpt,
         family = ratiod_poisson_gamma(),
         temporal = temporal_gp(time_var = "time"),
         control = list(iter = 100, warmup = 50, chains = 1, gradient_mode = "N", verbose = FALSE)),
  error = function(e) { cat(sprintf("N ERROR: %s\n", e$message)); NULL }
)
set.seed(123)
fit_h <- tryCatch(
  tratio(y | denom ~ x + (1 | site), data = df_gpt,
         family = ratiod_poisson_gamma(),
         temporal = temporal_gp(time_var = "time"),
         control = list(iter = 100, warmup = 50, chains = 1, gradient_mode = "H", verbose = FALSE)),
  error = function(e) { cat(sprintf("H ERROR: %s\n", e$message)); NULL }
)

if (!is.null(fit_n) && !is.null(fit_h)) {
  dn <- if (is.list(fit_n$draws)) do.call(rbind, fit_n$draws) else fit_n$draws
  dh <- if (is.list(fit_h$draws)) do.call(rbind, fit_h$draws) else fit_h$draws
  mn <- colMeans(dn)
  mh <- colMeans(dh)
  sdn <- apply(dn, 2, sd)
  cat(sprintf("  Max |mean_H - mean_N|: %.4f\n", max(abs(mh - mn))))
  cat(sprintf("  Max |mean_H - mean_N| / sd_N: %.4f\n", max(abs(mh - mn) / (sdn + 1e-10))))
}

cat("\nDone.\n")
