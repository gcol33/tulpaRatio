set.seed(42)
library(tulpaRatio)
# Test PG+HSGP at N=500 (the size that segfaulted)
N <- 500
df <- data.frame(
  y = rpois(N, 5),
  denom = rgamma(N, 10, 1),
  x = rnorm(N),
  lon = runif(N, 0, 10),
  lat = runif(N, 0, 10)
)

cat("=== Test 1: PG+HSGP N=500 gradient_mode=N ===\n")
fit_n <- tryCatch({
  tratio(y | denom ~ x, data = df,
         family = ratiod_poisson_gamma(),
         spatial = spatial_hsgp(~ lon + lat),
         control = list(iter = 50, warmup = 25, chains = 1, gradient_mode = "N", verbose = FALSE))
}, error = function(e) paste0("ERROR: ", e$message))
if (is.character(fit_n)) cat(fit_n, "\n") else cat("N mode: SUCCESS\n")

cat("=== Test 2: PG+HSGP N=500 gradient_mode=H ===\n")
fit_h <- tryCatch({
  tratio(y | denom ~ x, data = df,
         family = ratiod_poisson_gamma(),
         spatial = spatial_hsgp(~ lon + lat),
         control = list(iter = 50, warmup = 25, chains = 1, gradient_mode = "H", verbose = FALSE))
}, error = function(e) paste0("ERROR: ", e$message))
if (is.character(fit_h)) cat(fit_h, "\n") else cat("H mode: SUCCESS\n")

cat("=== Test 3: NB+HSGP N=500 gradient_mode=H ===\n")
df2 <- data.frame(
  y_num = rnbinom(N, mu = 5, size = 3),
  y_denom = rnbinom(N, mu = 10, size = 5),
  x = rnorm(N),
  lon = runif(N, 0, 10),
  lat = runif(N, 0, 10)
)
fit_nb <- tryCatch({
  tratio(y_num | y_denom ~ x, data = df2,
         family = ratiod_negbin_negbin(),
         spatial = spatial_hsgp(~ lon + lat),
         control = list(iter = 50, warmup = 25, chains = 1, gradient_mode = "H", verbose = FALSE))
}, error = function(e) paste0("ERROR: ", e$message))
if (is.character(fit_nb)) cat(fit_nb, "\n") else cat("NB+HSGP H mode: SUCCESS\n")

cat("\nAll tests completed.\n")
