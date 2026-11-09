# Quick smoke test
library(tulpaRatio)
cat("LOAD OK\n")

set.seed(42)
N <- 50
x <- rnorm(N)
y_num <- rpois(N, exp(0.5 + 0.3 * x))
y_denom <- rgamma(N, shape = 5, rate = 5 / exp(0.2))
df <- data.frame(y_num = y_num, y_denom = y_denom, x = x)

cat("Fitting PG base (H mode)...\n")
fit <- tratio(y_num | y_denom ~ x, data = df,
              family = ratiod_poisson_gamma(),
              control = list(iter = 50, warmup = 25, chains = 1, gradient_mode = "H", verbose = FALSE))
cat("Done! Divergences:", fit$diagnostics$n_divergent, "\n")

# NB base
cat("Fitting NB base...\n")
y_num_nb <- rnbinom(N, mu = exp(1 + 0.5 * x), size = 3)
y_denom_nb <- rnbinom(N, mu = exp(0.8), size = 5)
df2 <- data.frame(y_num = y_num_nb, y_denom = y_denom_nb, x = x)
fit2 <- tratio(y_num | y_denom ~ x, data = df2,
               family = ratiod_negbin_negbin(),
               control = list(iter = 50, warmup = 25, chains = 1, gradient_mode = "H", verbose = FALSE))
cat("Done! Divergences:", fit2$diagnostics$n_divergent, "\n")

# Binomial base
cat("Fitting binomial base...\n")
n_trials <- sample(10:50, N, replace = TRUE)
y_binom <- rbinom(N, n_trials, plogis(0.3 + 0.5 * x))
df3 <- data.frame(y = y_binom, n = n_trials, x = x)
fit3 <- tratio(y | n ~ x, data = df3,
               family = ratiod_binomial(),
               control = list(iter = 50, warmup = 25, chains = 1, gradient_mode = "H", verbose = FALSE))
cat("Done! Divergences:", fit3$diagnostics$n_divergent, "\n")

# NB + RE (use NB instead of PG, more iterations for warmup)
cat("Fitting NB + RE...\n")
site <- factor(sample(1:10, N, replace = TRUE))
df4 <- data.frame(y_num = y_num_nb, y_denom = y_denom_nb, x = x, site = site)
fit4 <- tratio(y_num | y_denom ~ x + (1 | site), data = df4,
               family = ratiod_negbin_negbin(),
               control = list(iter = 100, warmup = 50, chains = 1, gradient_mode = "H", verbose = FALSE))
cat("Done! Divergences:", fit4$diagnostics$n_divergent, "\n")

cat("\nAll smoke tests passed!\n")
