# NUTS vs Custom Joint Stan: Head-to-head timing comparison
# Re-benchmarks rows where Stan was previously faster (pre-NUTS, fixed L=20)
#
# Rows tested: 1, 2, 4, 31, 32, 34, 77
# NO brms. Custom joint Stan models only.

library(tulpaRatio)
library(cmdstanr)
library(posterior)

set.seed(42)

N_OBS <- 200
N_ITER <- 1000
N_WARMUP <- 500
N_CHAINS <- 2
N_SITES <- 20
N_SITES2 <- 10

cat("=======================================================\n")
cat("NUTS vs Custom Joint Stan: Speed Comparison\n")
cat("=======================================================\n")
cat(sprintf("N=%d, iter=%d, warmup=%d, chains=%d\n\n", N_OBS, N_ITER, N_WARMUP, N_CHAINS))

# Shared data
site <- factor(rep(1:N_SITES, length.out = N_OBS))
site2 <- factor(sample(1:N_SITES2, N_OBS, replace = TRUE))
x <- rnorm(N_OBS)

# Storage for results
timing_results <- list()

# =============================================================================
# Row 1: poisson_gamma (no RE)
# =============================================================================
cat("\n========== Row 1: poisson_gamma (no RE) ==========\n")

df1 <- data.frame(
  y = rpois(N_OBS, lambda = 30),
  effort = rgamma(N_OBS, shape = 5, rate = 0.1),
  x = x
)
df1$effort[df1$effort < 0.01] <- 0.01

cat("numdenom NUTS... ")
t_nd <- system.time({
  fit_nd <- tratio(y | effort ~ x, data = df1,
                   family = ratiod_poisson_gamma(),
                   control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE))
})["elapsed"]
diag <- fit_nd$diagnostics
cat(sprintf("%.2fs (treedepth=%.1f, leapfrog=%.1f)\n", t_nd,
            mean(diag$treedepth), mean(diag$n_leapfrog)))

stan_mod <- cmdstan_model("stan/joint_pg_base.stan")
cat("Stan NUTS... ")
t_stan <- system.time({
  fit_stan <- stan_mod$sample(
    data = list(N = N_OBS, y_num = df1$y, y_denom = df1$effort,
                p = 2, X = cbind(1, df1$x)),
    iter_sampling = N_ITER - N_WARMUP, iter_warmup = N_WARMUP,
    chains = N_CHAINS, parallel_chains = N_CHAINS,
    refresh = 0, show_messages = FALSE)
})["elapsed"]
cat(sprintf("%.2fs\n", t_stan))

ratio <- t_nd / t_stan
cat(sprintf("  => numdenom/Stan = %.2fx %s\n", ratio,
            if(ratio < 1) "(numdenom FASTER)" else "(Stan faster)"))
timing_results$row_1 <- c(nd = t_nd, stan = t_stan, ratio = ratio)

# =============================================================================
# Row 2: poisson_gamma + RE
# =============================================================================
cat("\n========== Row 2: poisson_gamma + RE ==========\n")

site_eff <- rnorm(N_SITES, 0, 0.5)
df2 <- data.frame(
  y = rpois(N_OBS, lambda = 30 * exp(site_eff[as.numeric(site)])),
  effort = rgamma(N_OBS, shape = 5, rate = 0.1) * exp(site_eff[as.numeric(site)]),
  x = x, site = site
)
df2$effort[df2$effort < 0.01] <- 0.01

cat("numdenom NUTS... ")
t_nd <- system.time({
  fit_nd <- tratio(y | effort ~ x + (1|site), data = df2,
                   family = ratiod_poisson_gamma(),
                   control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE))
})["elapsed"]
diag <- fit_nd$diagnostics
cat(sprintf("%.2fs (treedepth=%.1f, leapfrog=%.1f)\n", t_nd,
            mean(diag$treedepth), mean(diag$n_leapfrog)))

stan_mod_re <- cmdstan_model("stan/joint_pg_re.stan")
cat("Stan NUTS... ")
t_stan <- system.time({
  fit_stan <- stan_mod_re$sample(
    data = list(N = N_OBS, y_num = df2$y, y_denom = df2$effort,
                p = 2, X = cbind(1, df2$x),
                n_groups = N_SITES, group_idx = as.numeric(df2$site)),
    iter_sampling = N_ITER - N_WARMUP, iter_warmup = N_WARMUP,
    chains = N_CHAINS, parallel_chains = N_CHAINS,
    refresh = 0, show_messages = FALSE)
})["elapsed"]
cat(sprintf("%.2fs\n", t_stan))

ratio <- t_nd / t_stan
cat(sprintf("  => numdenom/Stan = %.2fx %s\n", ratio,
            if(ratio < 1) "(numdenom FASTER)" else "(Stan faster)"))
timing_results$row_2 <- c(nd = t_nd, stan = t_stan, ratio = ratio)

# =============================================================================
# Row 4: poisson_gamma + crossed RE
# =============================================================================
cat("\n========== Row 4: poisson_gamma + crossed RE ==========\n")

site_eff1 <- rnorm(N_SITES, 0, 0.3)
site_eff2 <- rnorm(N_SITES2, 0, 0.3)
df4 <- data.frame(
  y = rpois(N_OBS, lambda = 30 * exp(site_eff1[as.numeric(site)] + site_eff2[as.numeric(site2)])),
  effort = rgamma(N_OBS, shape = 5, rate = 0.1) *
    exp(site_eff1[as.numeric(site)] + site_eff2[as.numeric(site2)]),
  x = x, site = site, site2 = site2
)
df4$effort[df4$effort < 0.01] <- 0.01

cat("numdenom NUTS... ")
t_nd <- system.time({
  fit_nd <- tratio(y | effort ~ x + (1|site) + (1|site2), data = df4,
                   family = ratiod_poisson_gamma(),
                   control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE))
})["elapsed"]
diag <- fit_nd$diagnostics
cat(sprintf("%.2fs (treedepth=%.1f, leapfrog=%.1f)\n", t_nd,
            mean(diag$treedepth), mean(diag$n_leapfrog)))

stan_mod_crossed <- cmdstan_model("stan/joint_pg_crossed.stan")
cat("Stan NUTS... ")
t_stan <- system.time({
  fit_stan <- stan_mod_crossed$sample(
    data = list(N = N_OBS, y_num = df4$y, y_denom = df4$effort,
                p = 2, X = cbind(1, df4$x),
                n_groups1 = N_SITES, group_idx1 = as.numeric(df4$site),
                n_groups2 = N_SITES2, group_idx2 = as.numeric(df4$site2)),
    iter_sampling = N_ITER - N_WARMUP, iter_warmup = N_WARMUP,
    chains = N_CHAINS, parallel_chains = N_CHAINS,
    refresh = 0, show_messages = FALSE)
})["elapsed"]
cat(sprintf("%.2fs\n", t_stan))

ratio <- t_nd / t_stan
cat(sprintf("  => numdenom/Stan = %.2fx %s\n", ratio,
            if(ratio < 1) "(numdenom FASTER)" else "(Stan faster)"))
timing_results$row_4 <- c(nd = t_nd, stan = t_stan, ratio = ratio)

# =============================================================================
# Row 31: negbin_negbin (no RE)
# =============================================================================
cat("\n========== Row 31: negbin_negbin (no RE) ==========\n")

df31 <- data.frame(
  y = rnbinom(N_OBS, size = 5, mu = 30),
  denom = rnbinom(N_OBS, size = 8, mu = 100),
  x = x
)

cat("numdenom NUTS... ")
t_nd <- system.time({
  fit_nd <- tratio(y | denom ~ x, data = df31,
                   family = ratiod_negbin_negbin(),
                   control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE))
})["elapsed"]
diag <- fit_nd$diagnostics
cat(sprintf("%.2fs (treedepth=%.1f, leapfrog=%.1f)\n", t_nd,
            mean(diag$treedepth), mean(diag$n_leapfrog)))

stan_mod_nb <- cmdstan_model("stan/joint_nb_base.stan")
cat("Stan NUTS... ")
t_stan <- system.time({
  fit_stan <- stan_mod_nb$sample(
    data = list(N = N_OBS, y_num = df31$y, y_denom = df31$denom,
                p = 2, X = cbind(1, df31$x)),
    iter_sampling = N_ITER - N_WARMUP, iter_warmup = N_WARMUP,
    chains = N_CHAINS, parallel_chains = N_CHAINS,
    refresh = 0, show_messages = FALSE)
})["elapsed"]
cat(sprintf("%.2fs\n", t_stan))

ratio <- t_nd / t_stan
cat(sprintf("  => numdenom/Stan = %.2fx %s\n", ratio,
            if(ratio < 1) "(numdenom FASTER)" else "(Stan faster)"))
timing_results$row_31 <- c(nd = t_nd, stan = t_stan, ratio = ratio)

# =============================================================================
# Row 32: negbin_negbin + RE
# =============================================================================
cat("\n========== Row 32: negbin_negbin + RE ==========\n")

site_eff_nb <- rnorm(N_SITES, 0, 0.5)
df32 <- data.frame(
  y = rnbinom(N_OBS, size = 5, mu = 30 * exp(site_eff_nb[as.numeric(site)])),
  denom = rnbinom(N_OBS, size = 8, mu = 100 * exp(site_eff_nb[as.numeric(site)])),
  x = x, site = site
)

cat("numdenom NUTS... ")
t_nd <- system.time({
  fit_nd <- tratio(y | denom ~ x + (1|site), data = df32,
                   family = ratiod_negbin_negbin(),
                   control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE))
})["elapsed"]
diag <- fit_nd$diagnostics
cat(sprintf("%.2fs (treedepth=%.1f, leapfrog=%.1f)\n", t_nd,
            mean(diag$treedepth), mean(diag$n_leapfrog)))

stan_mod_nb_re <- cmdstan_model("stan/joint_nb_re.stan")
cat("Stan NUTS... ")
t_stan <- system.time({
  fit_stan <- stan_mod_nb_re$sample(
    data = list(N = N_OBS, y_num = df32$y, y_denom = df32$denom,
                p = 2, X = cbind(1, df32$x),
                n_groups = N_SITES, group_idx = as.numeric(df32$site)),
    iter_sampling = N_ITER - N_WARMUP, iter_warmup = N_WARMUP,
    chains = N_CHAINS, parallel_chains = N_CHAINS,
    refresh = 0, show_messages = FALSE)
})["elapsed"]
cat(sprintf("%.2fs\n", t_stan))

ratio <- t_nd / t_stan
cat(sprintf("  => numdenom/Stan = %.2fx %s\n", ratio,
            if(ratio < 1) "(numdenom FASTER)" else "(Stan faster)"))
timing_results$row_32 <- c(nd = t_nd, stan = t_stan, ratio = ratio)

# =============================================================================
# Row 34: negbin_negbin + crossed RE
# =============================================================================
cat("\n========== Row 34: negbin_negbin + crossed RE ==========\n")

df34 <- data.frame(
  y = rnbinom(N_OBS, size = 5, mu = 30 * exp(site_eff1[as.numeric(site)] + site_eff2[as.numeric(site2)])),
  denom = rnbinom(N_OBS, size = 8, mu = 100 *
    exp(site_eff1[as.numeric(site)] + site_eff2[as.numeric(site2)])),
  x = x, site = site, site2 = site2
)

cat("numdenom NUTS... ")
t_nd <- system.time({
  fit_nd <- tratio(y | denom ~ x + (1|site) + (1|site2), data = df34,
                   family = ratiod_negbin_negbin(),
                   control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE))
})["elapsed"]
diag <- fit_nd$diagnostics
cat(sprintf("%.2fs (treedepth=%.1f, leapfrog=%.1f)\n", t_nd,
            mean(diag$treedepth), mean(diag$n_leapfrog)))

stan_mod_nb_crossed <- cmdstan_model("stan/joint_nb_crossed.stan")
cat("Stan NUTS... ")
t_stan <- system.time({
  fit_stan <- stan_mod_nb_crossed$sample(
    data = list(N = N_OBS, y_num = df34$y, y_denom = df34$denom,
                p = 2, X = cbind(1, df34$x),
                n_groups1 = N_SITES, group_idx1 = as.numeric(df34$site),
                n_groups2 = N_SITES2, group_idx2 = as.numeric(df34$site2)),
    iter_sampling = N_ITER - N_WARMUP, iter_warmup = N_WARMUP,
    chains = N_CHAINS, parallel_chains = N_CHAINS,
    refresh = 0, show_messages = FALSE)
})["elapsed"]
cat(sprintf("%.2fs\n", t_stan))

ratio <- t_nd / t_stan
cat(sprintf("  => numdenom/Stan = %.2fx %s\n", ratio,
            if(ratio < 1) "(numdenom FASTER)" else "(Stan faster)"))
timing_results$row_34 <- c(nd = t_nd, stan = t_stan, ratio = ratio)

# =============================================================================
# Row 77: binomial + hurdle + RE
# =============================================================================
cat("\n========== Row 77: binomial + hurdle + RE ==========\n")

theta <- 0.8  # P(Y > 0)
p_success <- 0.4
trials <- sample(10:30, N_OBS, replace = TRUE)
site_eff_bin <- rnorm(N_SITES, 0, 0.3)

# Generate hurdle data
y77 <- integer(N_OBS)
for (i in 1:N_OBS) {
  if (runif(1) < theta) {
    eta <- qlogis(p_success) + 0.3 * x[i] + site_eff_bin[as.numeric(site[i])]
    p_i <- plogis(eta)
    repeat {
      draw <- rbinom(1, trials[i], p_i)
      if (draw > 0) { y77[i] <- draw; break }
    }
  } else {
    y77[i] <- 0
  }
}

df77 <- data.frame(y = y77, trials = trials, x = x, site = site)

cat("numdenom NUTS... ")
t_nd <- system.time({
  fit_nd <- tratio(y | trials ~ x + (1|site), data = df77,
                   family = ratiod_hurdle_binomial(),
                   control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE))
})["elapsed"]
diag <- fit_nd$diagnostics
cat(sprintf("%.2fs (treedepth=%.1f, leapfrog=%.1f)\n", t_nd,
            mean(diag$treedepth), mean(diag$n_leapfrog)))

stan_mod_hurdle <- cmdstan_model("hurdle_binomial.stan")
cat("Stan NUTS... ")
t_stan <- system.time({
  fit_stan <- stan_mod_hurdle$sample(
    data = list(N = N_OBS, y = df77$y, trials = df77$trials,
                X = cbind(1, df77$x),
                N_sites = N_SITES, site = as.numeric(df77$site)),
    iter_sampling = N_ITER - N_WARMUP, iter_warmup = N_WARMUP,
    chains = N_CHAINS, parallel_chains = N_CHAINS,
    refresh = 0, show_messages = FALSE)
})["elapsed"]
cat(sprintf("%.2fs\n", t_stan))

ratio <- t_nd / t_stan
cat(sprintf("  => numdenom/Stan = %.2fx %s\n", ratio,
            if(ratio < 1) "(numdenom FASTER)" else "(Stan faster)"))
timing_results$row_77 <- c(nd = t_nd, stan = t_stan, ratio = ratio)

# =============================================================================
# SUMMARY
# =============================================================================
cat("\n=======================================================\n")
cat("SUMMARY: NUTS vs Custom Joint Stan\n")
cat("=======================================================\n")
cat(sprintf("%-8s %10s %10s %10s %s\n", "Row", "numdenom", "Stan", "Ratio", "Winner"))
cat(sprintf("%-8s %10s %10s %10s %s\n", "---", "--------", "----", "-----", "------"))

for (name in names(timing_results)) {
  r <- timing_results[[name]]
  winner <- if(r["ratio"] < 1) "numdenom" else "Stan"
  cat(sprintf("%-8s %9.2fs %9.2fs %9.2fx %s\n",
              name, r["nd"], r["stan"], r["ratio"], winner))
}

nd_wins <- sum(sapply(timing_results, function(r) r["ratio"] < 1))
stan_wins <- length(timing_results) - nd_wins
cat(sprintf("\nnumdenom wins: %d/%d\n", nd_wins, length(timing_results)))
cat(sprintf("Stan wins:     %d/%d\n", stan_wins, length(timing_results)))
