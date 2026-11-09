set.seed(42)
library(tulpaRatio)
N <- 500
df <- data.frame(
  y = rpois(N, 5),
  denom = rgamma(N, 10, 1),
  x = rnorm(N),
  lon = runif(N, 0, 10),
  lat = runif(N, 0, 10)
)

cat("=== PG+HSGP N=500, 500 iter, H mode ===\n")
t0 <- proc.time()
fit <- tryCatch({
  tratio(y | denom ~ x, data = df,
         family = ratiod_poisson_gamma(),
         spatial = spatial_hsgp(~ lon + lat),
         control = list(iter = 500, warmup = 250, chains = 1, gradient_mode = "H", verbose = FALSE))
}, error = function(e) paste0("ERROR: ", e$message))
elapsed <- (proc.time() - t0)[3]
if (is.character(fit)) cat(fit, "\n") else cat(sprintf("SUCCESS in %.1fs\n", elapsed))
