# Direct Stan validation for hurdle_binomial (Row 77)
library(tulpaRatio)
library(cmdstanr)
set.seed(123)

N <- 500
N_SITES <- 50
x <- rnorm(N)
site <- factor(rep(1:N_SITES, length.out = N))

# Binomial data with zeros (hurdle structure)
trials <- sample(10:50, N, replace = TRUE)
y_bin <- rbinom(N, trials, plogis(0.5 + 0.3*x))
y_bin <- ifelse(runif(N) < 0.3, 0, y_bin)
df_bin_zi <- data.frame(y = y_bin, trials = trials, x = x, site = site)

# Prepare Stan data
stan_data <- list(
  N = N,
  y = y_bin,
  trials = trials,
  X = cbind(1, x),
  N_sites = N_SITES,
  site = as.integer(site)
)

cat("\n========== Row 77: bin_hurdle (Stan validation) ==========\n")

# Run numdenom
cat("Running numdenom... ")
t_nd <- system.time({
  fit_nd <- tratio(y | trials ~ x + (1|site), data = df_bin_zi,
                   family = ratiod_hurdle_binomial(),
                   control = list(iter = 1000, warmup = 500, chains = 2, verbose = FALSE))
})["elapsed"]
cat(sprintf("%.1fs (div: %d)\n", t_nd, fit_nd$diagnostics$divergent))

# Run Stan
cat("Compiling Stan model... ")
mod <- cmdstan_model("benchmarks/hurdle_binomial.stan")
cat("done\n")

cat("Running Stan... ")
t_stan <- system.time({
  fit_stan <- mod$sample(
    data = stan_data,
    chains = 2,
    parallel_chains = 2,
    iter_warmup = 500,
    iter_sampling = 500,
    refresh = 0,
    show_messages = FALSE
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_stan))

# Extract and compare
nd_draws <- as.matrix(fit_nd$draws)
nd_col <- grep("beta_num\\[2\\]", colnames(nd_draws), value = TRUE)[1]
nd_x <- mean(nd_draws[, nd_col])
nd_sd <- sd(nd_draws[, nd_col])

stan_sum <- fit_stan$summary("beta[2]")
stan_x <- stan_sum$mean
stan_sd <- stan_sum$sd

diff <- abs(nd_x - stan_x)
threshold <- 2 * max(nd_sd, stan_sd)
pass <- diff < threshold

cat(sprintf("\n=== VALIDATION Row 77 ===\n"))
cat(sprintf("numdenom: x = %.4f (SD=%.4f)\n", nd_x, nd_sd))
cat(sprintf("Stan:     x = %.4f (SD=%.4f)\n", stan_x, stan_sd))
cat(sprintf("Diff: %.4f | Threshold: %.4f | Status: %s\n", diff, threshold, if(pass) "PASS" else "FAIL"))
cat(sprintf("Speedup: %.1fx\n", t_stan / t_nd))

# Also compare hurdle parameter
nd_hu_col <- grep("zi_intercept", colnames(nd_draws), value = TRUE)[1]
if (!is.na(nd_hu_col)) {
  nd_hu <- mean(nd_draws[, nd_hu_col])
  nd_hu_sd <- sd(nd_draws[, nd_hu_col])

  stan_hu_sum <- fit_stan$summary("beta_hu")
  stan_hu <- stan_hu_sum$mean
  stan_hu_sd <- stan_hu_sum$sd

  hu_diff <- abs(nd_hu - stan_hu)
  hu_threshold <- 2 * max(nd_hu_sd, stan_hu_sd)
  hu_pass <- hu_diff < hu_threshold

  cat(sprintf("\nHurdle intercept:\n"))
  cat(sprintf("numdenom: hu = %.4f (SD=%.4f)\n", nd_hu, nd_hu_sd))
  cat(sprintf("Stan:     hu = %.4f (SD=%.4f)\n", stan_hu, stan_hu_sd))
  cat(sprintf("Diff: %.4f | Status: %s\n", hu_diff, if(hu_pass) "PASS" else "FAIL"))
}

cat(sprintf("\n========== SUMMARY ==========\n"))
cat(sprintf("Row 77 (bin_hurdle): %s\n", if(pass) "PASS" else "FAIL"))
