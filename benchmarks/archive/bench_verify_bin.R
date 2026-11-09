# bench_verify_bin.R — verify binomial is sampling correctly (not stuck at depth 0)
library(tulpaRatio)
set.seed(42)
N <- 500
n_trials <- sample(10:50, N, replace = TRUE)
x <- rnorm(N)
p <- plogis(0.3 + 0.5 * x)
y <- rbinom(N, n_trials, p)
df <- data.frame(y = y, n = n_trials, x = x)

cat("=== Binomial base: checking NUTS tree depth ===\n")
fit <- tratio(y | n ~ x, data = df, family = ratiod_binomial(),
              control = list(iter = 500, warmup = 250, chains = 1, gradient_mode = "H", verbose = FALSE))

# Check tree depths
td <- fit$diagnostics$treedepth
if (is.list(td)) td <- td[[1]]
cat(sprintf("Tree depth: min=%d, median=%d, max=%d, mean=%.2f\n",
    min(td), median(td), max(td), mean(td)))
cat("Tree depth table:\n")
print(table(td))

# Check actual posterior
draws <- fit$draws
if (is.list(draws)) draws <- draws[[1]]
cat(sprintf("\nIntercept: mean=%.3f  (true=0.3)\n", mean(draws[, 1])))
cat(sprintf("Slope:     mean=%.3f  (true=0.5)\n", mean(draws[, 2])))

# Check n_leapfrog
n_lf <- fit$diagnostics$n_leapfrog
if (is.list(n_lf)) n_lf <- n_lf[[1]]
cat(sprintf("\nLeapfrog: total=%d, mean=%.1f per iter\n", sum(n_lf), mean(n_lf)))
