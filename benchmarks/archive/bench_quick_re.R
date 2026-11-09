# Quick RE benchmark: vectorized gradient paths
library(tulpaRatio)
N <- 500; S <- 50
set.seed(42)
x <- rnorm(N)
site <- factor(sample(1:S, N, replace = TRUE))

y_num_nb <- rnbinom(N, mu = exp(1.0 + 0.5 * x), size = 3)
y_denom_nb <- rnbinom(N, mu = exp(0.8 + 0.2 * x), size = 5)
df <- data.frame(y_num = y_num_nb, y_denom = y_denom_nb, x = x, site = site)

y_num_pg <- rpois(N, exp(0.5 + 0.3 * x))
y_denom_pg <- rgamma(N, shape = 5, rate = 5 / exp(0.2 + 0.1 * x))
df_pg <- data.frame(y_num = y_num_pg, y_denom = y_denom_pg, x = x, site = site)

timed <- function(desc, expr) {
  cat(sprintf("  %-25s", desc))
  gc(FALSE)
  t <- system.time(eval(expr))["elapsed"]
  cat(sprintf(" %6.2fs\n", t))
  t
}

cat("+RE models (fused with RE scatter):\n")
timed("NB + RE", quote(
  tratio(y_num | y_denom ~ x + (1 | site), data = df, family = ratiod_negbin_negbin(),
         control = list(iter = 500, warmup = 250, chains = 1, gradient_mode = "H", verbose = FALSE))
))
timed("PG + RE", quote(
  tratio(y_num | y_denom ~ x + (1 | site), data = df_pg, family = ratiod_poisson_gamma(),
         control = list(iter = 500, warmup = 250, chains = 1, gradient_mode = "H", verbose = FALSE))
))

cat("\nReference: NB+RE 5.18s (Stan 2.9s), PG+RE 3.56s (Stan 2.5s)\n")
cat("Done.\n")
