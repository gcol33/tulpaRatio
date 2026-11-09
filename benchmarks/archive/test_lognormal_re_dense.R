# test_lognormal_re_dense.R — Test lognormal+RE mixing with dense mass matrix
library(tulpaRatio)
cat("=== Lognormal + RE Mixing Test (Dense Mass Matrix) ===\n\n")

# Setup data
set.seed(42)
N <- 200
K <- 20
df <- data.frame(
  x = rnorm(N),
  site = factor(rep(1:K, length.out = N))
)

# Simulate lognormal data with RE
true_sigma_re <- 0.5
re_true <- rnorm(K, 0, true_sigma_re)
eta <- 1.0 + 0.3 * df$x + re_true[as.integer(df$site)]
df$y_num <- exp(rnorm(N, eta, 0.5))
df$y_denom <- exp(rnorm(N, eta + 0.3, 0.5))

cat("--- Test 1: Lognormal + RE (2000 iter, 4 chains) ---\n")
tryCatch({
  t1 <- system.time({
    set.seed(123)
    fit_ln <- tratio(y_num | y_denom ~ x + (1|site), data = df,
      family = ratiod_lognormal(),
      control = list(iter = 2000, warmup = 1000, chains = 4, verbose = FALSE))
  })["elapsed"]

  draws <- fit_ln$draws
  summ <- posterior::summarise_draws(draws, "mean", "sd", "ess_bulk", "rhat")
  key_params <- summ[grepl("beta_num|beta_denom|sigma_num|sigma_denom|sigma_re", summ$variable), ]
  cat(sprintf("  Time: %.1fs\n", t1))
  cat("  Key parameter ESS/Rhat:\n")
  for (i in seq_len(nrow(key_params))) {
    cat(sprintf("    %-20s mean=%.4f  ESS=%.0f  Rhat=%.4f\n",
      key_params$variable[i], key_params$mean[i],
      key_params$ess_bulk[i], key_params$rhat[i]))
  }

  # Check minimum ESS
  min_ess <- min(key_params$ess_bulk, na.rm = TRUE)
  max_rhat <- max(key_params$rhat, na.rm = TRUE)
  cat(sprintf("\n  Min ESS: %.0f (target: >100)\n", min_ess))
  cat(sprintf("  Max Rhat: %.4f (target: <1.01)\n", max_rhat))
  if (min_ess > 100 && max_rhat < 1.02) {
    cat("  PASS: Adequate mixing\n")
  } else {
    cat("  WARN: Mixing still needs improvement\n")
  }
}, error = function(e) cat(sprintf("  ERROR: %s\n", e$message)))

cat("\n--- Test 2: Negbin + RE (sanity check — should still work) ---\n")
tryCatch({
  set.seed(42)
  df2 <- data.frame(
    x = rnorm(N),
    site = factor(rep(1:K, length.out = N))
  )
  re2 <- rnorm(K, 0, 0.5)
  eta2 <- 1.5 + 0.3 * df2$x + re2[as.integer(df2$site)]
  df2$y_num <- rnbinom(N, size = 5, mu = exp(eta2))
  df2$y_denom <- rnbinom(N, size = 5, mu = exp(eta2 + 0.3))

  t2 <- system.time({
    set.seed(123)
    fit_nb <- tratio(y_num | y_denom ~ x + (1|site), data = df2,
      family = ratiod_negbin_negbin(),
      control = list(iter = 2000, warmup = 1000, chains = 4, verbose = FALSE))
  })["elapsed"]

  draws2 <- fit_nb$draws
  summ2 <- posterior::summarise_draws(draws2, "mean", "sd", "ess_bulk", "rhat")
  key2 <- summ2[grepl("beta_num|beta_denom|phi_num|phi_denom|sigma_re", summ2$variable), ]
  cat(sprintf("  Time: %.1fs\n", t2))
  cat("  Key parameter ESS/Rhat:\n")
  for (i in seq_len(nrow(key2))) {
    cat(sprintf("    %-20s mean=%.4f  ESS=%.0f  Rhat=%.4f\n",
      key2$variable[i], key2$mean[i],
      key2$ess_bulk[i], key2$rhat[i]))
  }

  min_ess2 <- min(key2$ess_bulk, na.rm = TRUE)
  max_rhat2 <- max(key2$rhat, na.rm = TRUE)
  cat(sprintf("\n  Min ESS: %.0f\n", min_ess2))
  cat(sprintf("  Max Rhat: %.4f\n", max_rhat2))
  if (min_ess2 > 200 && max_rhat2 < 1.01) {
    cat("  PASS: Good mixing (negbin+RE unaffected by dense change)\n")
  } else {
    cat("  WARN: Unexpected mixing degradation\n")
  }
}, error = function(e) cat(sprintf("  ERROR: %s\n", e$message)))

cat("\n=== Done ===\n")
