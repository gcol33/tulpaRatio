# Quick Batch 5b: Validate remaining easy binomial rows
# Focus: rows 61, 62, 63, 65, 66, 71, 72, 73, 76, 77, 80, 81, 82

library(tulpaRatio)
library(brms)
set.seed(42)

N <- 300  # Reduced for speed
N_SITES <- 30
N_TIMES <- 15
x <- rnorm(N)
site <- factor(rep(1:N_SITES, length.out = N))
time_factor <- factor(rep(1:N_TIMES, length.out = N))

# Adjacency
adj_mat <- matrix(0, N_SITES, N_SITES)
for (i in 1:(N_SITES-1)) {
  adj_mat[i, i+1] <- adj_mat[i+1, i] <- 1
}
spatial_site <- factor(rep(1:N_SITES, length.out = N))

# Binomial data
trials <- sample(10:50, N, replace = TRUE)
y_bin <- rbinom(N, trials, plogis(0.5 + 0.3*x))
df <- data.frame(y = y_bin, trials = trials, x = x, site = site,
                 time = time_factor, spatial_site = spatial_site)

# ZI data
df_zi <- df
df_zi$y <- ifelse(runif(N) < 0.2, 0, df_zi$y)

results <- list()

validate <- function(row, name, nd_call, brms_call) {
  cat(sprintf("\n=== Row %d: %s ===\n", row, name))

  t_nd <- tryCatch({
    system.time({
      fit_nd <- eval(nd_call)
    })["elapsed"]
  }, error = function(e) { cat("ND error:", e$message, "\n"); NA })

  if (is.na(t_nd)) return(list(row = row, name = name, pass = NA))
  cat(sprintf("  numdenom: %.1fs\n", t_nd))

  t_brms <- tryCatch({
    system.time({
      fit_brms <- eval(brms_call)
    })["elapsed"]
  }, error = function(e) { cat("brms error:", e$message, "\n"); NA })

  if (is.na(t_brms)) return(list(row = row, name = name, nd_time = t_nd, pass = NA))
  cat(sprintf("  brms: %.1fs\n", t_brms))

  # Compare
  nd_draws <- as.matrix(fit_nd$draws)
  beta_col <- grep("beta_num\\[2\\]|beta\\[2\\]", colnames(nd_draws), value = TRUE)[1]
  if (is.na(beta_col)) {
    cat("  Could not find beta param\n")
    return(list(row = row, name = name, nd_time = t_nd, brms_time = t_brms, pass = NA))
  }

  nd_mean <- mean(nd_draws[, beta_col])
  nd_sd <- sd(nd_draws[, beta_col])
  brms_fix <- fixef(fit_brms)
  brms_mean <- brms_fix["x", "Estimate"]
  brms_se <- brms_fix["x", "Est.Error"]

  diff <- abs(nd_mean - brms_mean)
  threshold <- 2 * max(nd_sd, brms_se)
  pass <- diff < threshold

  cat(sprintf("  nd=%.3f, brms=%.3f, diff=%.4f => %s\n",
              nd_mean, brms_mean, diff, if(pass) "PASS" else "FAIL"))

  list(row = row, name = name, nd_time = t_nd, brms_time = t_brms,
       speedup = round(t_brms / t_nd, 1), diff = diff, pass = pass)
}

# Row 61: basic
results[[61]] <- validate(61, "bin_basic",
  quote(tratio(y | trials ~ x, data = df, family = ratiod_binomial(),
               control = list(iter = 500, warmup = 250, chains = 1, verbose = FALSE))),
  quote(brm(y | trials(trials) ~ x, data = df, family = binomial(),
            iter = 500, warmup = 250, chains = 1, backend = "cmdstanr", silent = 2, refresh = 0))
)

# Row 62: RE
results[[62]] <- validate(62, "bin_re",
  quote(tratio(y | trials ~ x + (1|site), data = df, family = ratiod_binomial(),
               control = list(iter = 500, warmup = 250, chains = 1, verbose = FALSE))),
  quote(brm(y | trials(trials) ~ x + (1|site), data = df, family = binomial(),
            iter = 500, warmup = 250, chains = 1, backend = "cmdstanr", silent = 2, refresh = 0))
)

# Row 63: slopes
results[[63]] <- validate(63, "bin_slopes",
  quote(tratio(y | trials ~ x + (x|site), data = df, family = ratiod_binomial(),
               control = list(iter = 500, warmup = 250, chains = 1, verbose = FALSE))),
  quote(brm(y | trials(trials) ~ x + (x|site), data = df, family = binomial(),
            iter = 500, warmup = 250, chains = 1, backend = "cmdstanr", silent = 2, refresh = 0))
)

# Row 65: ICAR
results[[65]] <- validate(65, "bin_icar",
  quote(tratio(y | trials ~ x + (1|site), data = df, family = ratiod_binomial(),
               spatial = spatial_car(adj_mat, level = "group", group_var = "spatial_site"),
               control = list(iter = 500, warmup = 250, chains = 1, verbose = FALSE))),
  quote(brm(y | trials(trials) ~ x + (1|site) + (1|spatial_site), data = df, family = binomial(),
            iter = 500, warmup = 250, chains = 1, backend = "cmdstanr", silent = 2, refresh = 0))
)

# Row 66: BYM2
results[[66]] <- validate(66, "bin_bym2",
  quote(tratio(y | trials ~ x + (1|site), data = df, family = ratiod_binomial(),
               spatial = spatial_bym2(adj_mat, level = "group", group_var = "spatial_site"),
               control = list(iter = 500, warmup = 250, chains = 1, verbose = FALSE))),
  quote(brm(y | trials(trials) ~ x + (1|site) + (1|spatial_site), data = df, family = binomial(),
            iter = 500, warmup = 250, chains = 1, backend = "cmdstanr", silent = 2, refresh = 0))
)

# Row 71: RW1
results[[71]] <- validate(71, "bin_rw1",
  quote(tratio(y | trials ~ x + (1|site), data = df, family = ratiod_binomial(),
               temporal = temporal_rw1("time"),
               control = list(iter = 500, warmup = 250, chains = 1, verbose = FALSE))),
  quote(brm(y | trials(trials) ~ x + (1|site) + (1|time), data = df, family = binomial(),
            iter = 500, warmup = 250, chains = 1, backend = "cmdstanr", silent = 2, refresh = 0))
)

# Row 72: RW2
results[[72]] <- validate(72, "bin_rw2",
  quote(tratio(y | trials ~ x + (1|site), data = df, family = ratiod_binomial(),
               temporal = temporal_rw2("time"),
               control = list(iter = 500, warmup = 250, chains = 1, verbose = FALSE))),
  quote(brm(y | trials(trials) ~ x + (1|site) + (1|time), data = df, family = binomial(),
            iter = 500, warmup = 250, chains = 1, backend = "cmdstanr", silent = 2, refresh = 0))
)

# Row 73: AR1
results[[73]] <- validate(73, "bin_ar1",
  quote(tratio(y | trials ~ x + (1|site), data = df, family = ratiod_binomial(),
               temporal = temporal_ar1("time"),
               control = list(iter = 500, warmup = 250, chains = 1, verbose = FALSE))),
  quote(brm(y | trials(trials) ~ x + (1|site) + (1|time), data = df, family = binomial(),
            iter = 500, warmup = 250, chains = 1, backend = "cmdstanr", silent = 2, refresh = 0))
)

# Row 76: ZI
results[[76]] <- validate(76, "bin_zi",
  quote(tratio(y | trials ~ x + (1|site), data = df_zi, family = ratiod_zibinomial(),
               control = list(iter = 500, warmup = 250, chains = 1, verbose = FALSE))),
  quote(brm(y | trials(trials) ~ x + (1|site), data = df_zi, family = zero_inflated_binomial(),
            iter = 500, warmup = 250, chains = 1, backend = "cmdstanr", silent = 2, refresh = 0))
)

# Row 77: Hurdle
results[[77]] <- validate(77, "bin_hurdle",
  quote(tratio(y | trials ~ x + (1|site), data = df_zi, family = ratiod_hurdle_binomial(),
               control = list(iter = 500, warmup = 250, chains = 1, verbose = FALSE))),
  quote(brm(y | trials(trials) ~ x + (1|site), data = df_zi, family = hurdle_cumulative(),
            iter = 500, warmup = 250, chains = 1, backend = "cmdstanr", silent = 2, refresh = 0))
)

# Row 80: ICAR + RW1
results[[80]] <- validate(80, "bin_icar_rw1",
  quote(tratio(y | trials ~ x + (1|site), data = df, family = ratiod_binomial(),
               spatial = spatial_car(adj_mat, level = "group", group_var = "spatial_site"),
               temporal = temporal_rw1("time"),
               control = list(iter = 500, warmup = 250, chains = 1, verbose = FALSE))),
  quote(brm(y | trials(trials) ~ x + (1|site) + (1|spatial_site) + (1|time), data = df, family = binomial(),
            iter = 500, warmup = 250, chains = 1, backend = "cmdstanr", silent = 2, refresh = 0))
)

# Row 81: BYM2 + RW1
results[[81]] <- validate(81, "bin_bym2_rw1",
  quote(tratio(y | trials ~ x + (1|site), data = df, family = ratiod_binomial(),
               spatial = spatial_bym2(adj_mat, level = "group", group_var = "spatial_site"),
               temporal = temporal_rw1("time"),
               control = list(iter = 500, warmup = 250, chains = 1, verbose = FALSE))),
  quote(brm(y | trials(trials) ~ x + (1|site) + (1|spatial_site) + (1|time), data = df, family = binomial(),
            iter = 500, warmup = 250, chains = 1, backend = "cmdstanr", silent = 2, refresh = 0))
)

# Row 82: ICAR + AR1
results[[82]] <- validate(82, "bin_icar_ar1",
  quote(tratio(y | trials ~ x + (1|site), data = df, family = ratiod_binomial(),
               spatial = spatial_car(adj_mat, level = "group", group_var = "spatial_site"),
               temporal = temporal_ar1("time"),
               control = list(iter = 500, warmup = 250, chains = 1, verbose = FALSE))),
  quote(brm(y | trials(trials) ~ x + (1|site) + (1|spatial_site) + (1|time), data = df, family = binomial(),
            iter = 500, warmup = 250, chains = 1, backend = "cmdstanr", silent = 2, refresh = 0))
)

# Summary
cat("\n\n=== SUMMARY ===\n")
n_pass <- 0
for (r in results) {
  if (!is.null(r$pass) && !is.na(r$pass) && r$pass) n_pass <- n_pass + 1
  status <- if (is.null(r$pass) || is.na(r$pass)) "SKIP" else if (r$pass) "PASS" else "FAIL"
  cat(sprintf("Row %2d: %-15s %s\n", r$row, r$name, status))
}
cat(sprintf("\nPassed: %d/%d\n", n_pass, length(results)))

saveRDS(results, "benchmarks/results_batch5b.rds")
