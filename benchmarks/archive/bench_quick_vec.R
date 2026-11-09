# Quick benchmark: vectorized gradient paths
# N=500, 500 iter, 1 chain, H mode
library(tulpaRatio)
cat("=== Quick Vectorized Gradient Benchmark ===\n")
cat("Date:", format(Sys.time()), "\n\n")

N <- 500
set.seed(42)
x <- rnorm(N)

# --- PG data ---
y_num_pg <- rpois(N, exp(0.5 + 0.3 * x))
y_denom_pg <- rgamma(N, shape = 5, rate = 5 / exp(0.2 + 0.1 * x))
df_pg <- data.frame(y_num = y_num_pg, y_denom = y_denom_pg, x = x)

# --- NB data ---
y_num_nb <- rnbinom(N, mu = exp(1.0 + 0.5 * x), size = 3)
y_denom_nb <- rnbinom(N, mu = exp(0.8 + 0.2 * x), size = 5)
df_nb <- data.frame(y_num = y_num_nb, y_denom = y_denom_nb, x = x)

# --- Binomial data ---
n_trials <- sample(10:50, N, replace = TRUE)
y_binom <- rbinom(N, n_trials, plogis(0.3 + 0.5 * x))
df_bin <- data.frame(y = y_binom, n = n_trials, x = x)

timed <- function(desc, expr) {
  cat(sprintf("  %-25s", desc))
  gc(FALSE)
  t <- system.time(eval(expr))["elapsed"]
  cat(sprintf(" %6.2fs\n", t))
  t
}

cat("Base models (fused single-pass, p<=4):\n")
t1 <- timed("NB base", quote(
  tratio(y_num | y_denom ~ x, data = df_nb, family = ratiod_negbin_negbin(),
         control = list(iter = 500, warmup = 250, chains = 1, gradient_mode = "H", verbose = FALSE))
))
t2 <- timed("PG base", quote(
  tratio(y_num | y_denom ~ x, data = df_pg, family = ratiod_poisson_gamma(),
         control = list(iter = 500, warmup = 250, chains = 1, gradient_mode = "H", verbose = FALSE))
))
t3 <- timed("Binomial base", quote(
  tratio(y | n ~ x, data = df_bin, family = ratiod_binomial(),
         control = list(iter = 500, warmup = 250, chains = 1, gradient_mode = "H", verbose = FALSE))
))

cat("\nReference (bench_3fixes.R, 5-rep median):\n")
cat("  NB base:   1.36s (Stan 1.5s)\n")
cat("  PG base:   1.64s (Stan 1.2s)\n")
cat("  Bin base:  0.5s  (Stan ~0.3s)\n")

cat("\nDone.\n")
