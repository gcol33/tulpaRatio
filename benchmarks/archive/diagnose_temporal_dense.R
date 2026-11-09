# Diagnose temporal models with DENSE mass matrix
library(tulpaRatio)
N <- 500; S <- 50; T_times <- 10
set.seed(42)
x <- rnorm(N)
site <- factor(sample(1:S, N, replace = TRUE))
time <- factor(sample(1:T_times, N, replace = TRUE))

y_num <- rnbinom(N, mu = exp(1.0 + 0.5 * x), size = 3)
y_denom <- rnbinom(N, mu = exp(0.8 + 0.2 * x), size = 5)
df <- data.frame(y_num = y_num, y_denom = y_denom, x = x, site = site, time = time)

y_num_pg <- rpois(N, exp(0.5 + 0.3 * x))
y_denom_pg <- rgamma(N, shape = 5, rate = 5 / exp(0.2 + 0.1 * x))
df_pg <- data.frame(y_num = y_num_pg, y_denom = y_denom_pg, x = x, time = time)

cat("=== Temporal Models: DIAG vs DENSE Mass ===\n\n")

run_fit <- function(desc, expr) {
  cat(sprintf("--- %s ---\n", desc))
  gc(FALSE)
  t0 <- proc.time()["elapsed"]
  fit <- eval(expr)
  t1 <- proc.time()["elapsed"]
  d <- fit$diagnostics
  td <- d$treedepth
  cat(sprintf("  Time: %.1fs  Div: %d  Avg td: %.1f  Max td%%: %.0f%%\n",
      t1 - t0, d$n_divergent,
      mean(td), 100 * mean(td >= 10)))
  cat(sprintf("  Sampler: %s\n\n", d$algorithm))
  invisible(fit)
}

# NB + RW1: diag vs dense
run_fit("NB + RW1 (diag)", quote(
  tratio(y_num | y_denom ~ x, data = df, family = ratiod_negbin_negbin(),
         temporal = temporal_rw1(time_var = "time"),
         control = list(iter = 500, warmup = 250, chains = 1, gradient_mode = "H", metric = "diag", verbose = TRUE))
))

run_fit("NB + RW1 (dense)", quote(
  tratio(y_num | y_denom ~ x, data = df, family = ratiod_negbin_negbin(),
         temporal = temporal_rw1(time_var = "time"),
         control = list(iter = 500, warmup = 250, chains = 1, gradient_mode = "H", metric = "dense", verbose = TRUE))
))

# NB + AR1: diag vs dense
run_fit("NB + AR1 (diag)", quote(
  tratio(y_num | y_denom ~ x, data = df, family = ratiod_negbin_negbin(),
         temporal = temporal_ar1(time_var = "time"),
         control = list(iter = 500, warmup = 250, chains = 1, gradient_mode = "H", metric = "diag", verbose = TRUE))
))

run_fit("NB + AR1 (dense)", quote(
  tratio(y_num | y_denom ~ x, data = df, family = ratiod_negbin_negbin(),
         temporal = temporal_ar1(time_var = "time"),
         control = list(iter = 500, warmup = 250, chains = 1, gradient_mode = "H", metric = "dense", verbose = TRUE))
))

# PG + RW1: diag vs dense
run_fit("PG + RW1 (diag)", quote(
  tratio(y_num | y_denom ~ x, data = df_pg, family = ratiod_poisson_gamma(),
         temporal = temporal_rw1(time_var = "time"),
         control = list(iter = 500, warmup = 250, chains = 1, gradient_mode = "H", metric = "diag", verbose = TRUE))
))

run_fit("PG + RW1 (dense)", quote(
  tratio(y_num | y_denom ~ x, data = df_pg, family = ratiod_poisson_gamma(),
         temporal = temporal_rw1(time_var = "time"),
         control = list(iter = 500, warmup = 250, chains = 1, gradient_mode = "H", metric = "dense", verbose = TRUE))
))

# NB + RE + RW1: diag vs dense
run_fit("NB + RE + RW1 (diag)", quote(
  tratio(y_num | y_denom ~ x + (1 | site), data = df, family = ratiod_negbin_negbin(),
         temporal = temporal_rw1(time_var = "time"),
         control = list(iter = 500, warmup = 250, chains = 1, gradient_mode = "H", metric = "diag", verbose = TRUE))
))

run_fit("NB + RE + RW1 (dense)", quote(
  tratio(y_num | y_denom ~ x + (1 | site), data = df, family = ratiod_negbin_negbin(),
         temporal = temporal_rw1(time_var = "time"),
         control = list(iter = 500, warmup = 250, chains = 1, gradient_mode = "H", metric = "dense", verbose = TRUE))
))

cat("Done.\n")
