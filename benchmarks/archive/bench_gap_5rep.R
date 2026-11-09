# bench_gap_5rep.R — 5-rep benchmark of the 1.8x gap models
# Compare current performance against Stan baselines
library(tulpaRatio)
set.seed(42)
N <- 500
S <- 50
df <- data.frame(x = rnorm(N), site = factor(sample(1:S, N, replace = TRUE)))
beta_num <- c(1.5, 0.3)
beta_denom <- c(2.0, -0.2)
eta_num <- beta_num[1] + beta_num[2] * df$x
eta_denom <- beta_denom[1] + beta_denom[2] * df$x
df$y_num <- rnbinom(N, mu = exp(eta_num), size = 3)
df$y_denom <- rnbinom(N, mu = exp(eta_denom), size = 5)
df$count <- rpois(N, lambda = exp(eta_num))
df$effort <- rgamma(N, shape = 5, rate = 5 / exp(eta_denom))

n_trials <- sample(10:50, N, replace = TRUE)
df$y_bin <- rbinom(N, n_trials, plogis(0.3 + 0.5 * df$x))
df$n_trials <- n_trials

run5 <- function(label, ...) {
  times <- numeric(5)
  for (i in 1:5) {
    gc(FALSE)
    times[i] <- system.time(tratio(...,
                                    control = list(iter = 500, warmup = 250, chains = 1, gradient_mode = "H", verbose = FALSE)))["elapsed"]
  }
  cat(sprintf("%-20s  runs: %s  median: %.2fs\n", label,
      paste(sprintf("%.2f", times), collapse = ", "), median(times)))
  invisible(median(times))
}

cat("=== 5-rep benchmark (N=500, iter=500, 1 chain, H mode) ===\n\n")

t1 <- run5("NB base", y_num | y_denom ~ x, data = df, family = ratiod_negbin_negbin())
t2 <- run5("NB + RE", y_num | y_denom ~ x + (1 | site), data = df, family = ratiod_negbin_negbin())
t3 <- run5("PG base", count | effort ~ x, data = df, family = ratiod_poisson_gamma())
t4 <- run5("PG + RE", count | effort ~ x + (1 | site), data = df, family = ratiod_poisson_gamma())
t5 <- run5("Bin base", y_bin | n_trials ~ x, data = df, family = ratiod_binomial())
t6 <- run5("Bin + RE", y_bin | n_trials ~ x + (1 | site), data = df, family = ratiod_binomial())

cat("\n=== vs Stan baselines ===\n")
cat(sprintf("NB base:  %.2fs vs Stan 1.5s  = %.2fx %s\n", t1, t1 / 1.5, ifelse(t1 < 1.5, "WIN", "")))
cat(sprintf("NB + RE:  %.2fs vs Stan 2.9s  = %.2fx %s\n", t2, t2 / 2.9, ifelse(t2 < 2.9, "WIN", "")))
cat(sprintf("PG base:  %.2fs vs Stan 1.2s  = %.2fx %s\n", t3, t3 / 1.2, ifelse(t3 < 1.2, "WIN", "")))
cat(sprintf("PG + RE:  %.2fs vs Stan 2.5s  = %.2fx %s\n", t4, t4 / 2.5, ifelse(t4 < 2.5, "WIN", "")))
cat(sprintf("Bin base: %.2fs vs Stan ~9.4s = %.2fx %s\n", t5, t5 / 9.4, ifelse(t5 < 9.4, "WIN", "")))

cat("\n=== vs Previous (gradient_methods.md values) ===\n")
cat(sprintf("NB base:  %.2fs  was 2.7s  (%.1fx improvement)\n", t1, 2.7 / t1))
cat(sprintf("NB + RE:  %.2fs  was 5.2s  (%.1fx improvement)\n", t2, 5.2 / t2))
cat(sprintf("PG base:  %.2fs  was 1.1s  (%.1fx improvement)\n", t3, 1.1 / t3))
cat(sprintf("PG + RE:  %.2fs  was 4.8s  (%.1fx improvement)\n", t4, 4.8 / t4))
cat(sprintf("Bin base: %.2fs  was 1.1s  (%.1fx improvement)\n", t5, 1.1 / t5))
