# Diagnose temporal model NUTS behavior
library(tulpaRatio)
N <- 500; S <- 50; T_times <- 10
set.seed(42)
x <- rnorm(N)
site <- factor(sample(1:S, N, replace = TRUE))
time <- factor(sample(1:T_times, N, replace = TRUE))

y_num <- rnbinom(N, mu = exp(1.0 + 0.5 * x), size = 3)
y_denom <- rnbinom(N, mu = exp(0.8 + 0.2 * x), size = 5)
df <- data.frame(y_num = y_num, y_denom = y_denom, x = x, site = site, time = time)

cat("=== Temporal Model NUTS Diagnostics ===\n\n")

run_diag <- function(desc, expr) {
  cat(sprintf("--- %s ---\n", desc))
  gc(FALSE)
  t0 <- proc.time()["elapsed"]
  fit <- eval(expr)
  t1 <- proc.time()["elapsed"]
  d <- fit$diagnostics
  cat(sprintf("  Time: %.1fs\n", t1 - t0))
  cat(sprintf("  Divergences: %d\n", d$n_divergent))
  cat(sprintf("  Sampler: %s\n", d$sampler))
  if (!is.null(d$treedepth_mean)) cat(sprintf("  Avg treedepth: %.1f\n", d$treedepth_mean))
  if (!is.null(d$max_treedepth_pct)) cat(sprintf("  Max treedepth %%: %.1f%%\n", d$max_treedepth_pct * 100))
  cat("\n")
  invisible(fit)
}

# Reference: base models
run_diag("NB base (reference)", quote(
  tratio(y_num | y_denom ~ x, data = df, family = ratiod_negbin_negbin(),
         control = list(iter = 500, warmup = 250, chains = 1, gradient_mode = "H", verbose = TRUE))
))

# NB + RW1 (the slow one: ~45s)
run_diag("NB + RW1", quote(
  tratio(y_num | y_denom ~ x, data = df, family = ratiod_negbin_negbin(),
         temporal = temporal_rw1(time_var = "time"),
         control = list(iter = 500, warmup = 250, chains = 1, gradient_mode = "H", verbose = TRUE))
))

# NB + RE + RW1 (paradoxically faster: ~16s)
run_diag("NB + RE + RW1", quote(
  tratio(y_num | y_denom ~ x + (1 | site), data = df, family = ratiod_negbin_negbin(),
         temporal = temporal_rw1(time_var = "time"),
         control = list(iter = 500, warmup = 250, chains = 1, gradient_mode = "H", verbose = TRUE))
))

# NB + AR1 (also slow: ~48s)
run_diag("NB + AR1", quote(
  tratio(y_num | y_denom ~ x, data = df, family = ratiod_negbin_negbin(),
         temporal = temporal_ar1(time_var = "time"),
         control = list(iter = 500, warmup = 250, chains = 1, gradient_mode = "H", verbose = TRUE))
))

# PG + RW1
run_diag("PG + RW1", quote(
  tratio(y_num | y_denom ~ x, data = df[, c("y_num", "y_denom", "x", "time")],
         family = ratiod_poisson_gamma(),
         temporal = temporal_rw1(time_var = "time"),
         control = list(iter = 500, warmup = 250, chains = 1, gradient_mode = "H", verbose = TRUE))
))

cat("Done.\n")
