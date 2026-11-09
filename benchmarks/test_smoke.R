library(tulpaRatio)
cat("loaded OK\n")

# Binomial smoke test
set.seed(1)
N <- 30
df <- data.frame(y = rbinom(N, 10, 0.5), trials = rep(10L, N), x = rnorm(N))
fit <- tratio(y | trials ~ x, data = df, family = ratiod_binomial(),
              control = list(iter = 20, warmup = 10, chains = 1, verbose = TRUE))
cat("binomial OK\n")

# NB smoke test
set.seed(1)
N <- 50
df2 <- data.frame(
  y = rnbinom(N, mu = exp(2), size = 5),
  denom = pmax(rnbinom(N, mu = 100, size = 10), 1L),
  x = rnorm(N)
)
fit2 <- tratio(y | denom ~ x, data = df2, family = ratiod_negbin_negbin(),
               control = list(iter = 50, warmup = 25, chains = 1, verbose = TRUE))
cat("NB OK\n")
