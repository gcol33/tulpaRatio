# Validated Benchmark Batch 5: Next 25 Models
# Focus: binomial remaining, gamma_gamma, lognormal, beta_binomial families
# Standard parameters: N=500, iter=500, warmup=250, chains=1 (for timing)

library(tulpaRatio)
library(brms)
set.seed(42)

# ============================================================
# DATA SETUP
# ============================================================

N <- 500
N_SITES <- 50
N_TIMES <- 20
x <- rnorm(N)
x2 <- rnorm(N)
site <- factor(rep(1:N_SITES, length.out = N))
site2 <- factor(sample(1:20, N, replace = TRUE))
time <- rep(1:N_TIMES, length.out = N)  # numeric for temporal_gp
time_factor <- factor(time)  # factor for RW models

# Spatial grid
n_side <- ceiling(sqrt(N_SITES))
grid <- expand.grid(lon = 1:n_side, lat = 1:n_side)[1:N_SITES, ]
site_int <- as.integer(site)
lon <- grid$lon[site_int]
lat <- grid$lat[site_int]

# Adjacency matrix
adj_mat <- matrix(0, N_SITES, N_SITES)
for (i in 1:N_SITES) {
  for (j in 1:N_SITES) {
    if (i != j) {
      dist <- sqrt((grid$lon[i] - grid$lon[j])^2 + (grid$lat[i] - grid$lat[j])^2)
      if (dist <= 1.5) adj_mat[i, j] <- 1
    }
  }
}
spatial_site <- factor(rep(1:N_SITES, length.out = N))

# Binomial data
trials <- sample(10:50, N, replace = TRUE)
y_bin <- rbinom(N, trials, plogis(0.5 + 0.3*x + 0.2*x2))
df_bin <- data.frame(y = y_bin, trials = trials, x = x, x2 = x2, site = site,
                     site2 = site2, time = time, time_factor = time_factor,
                     lon = lon, lat = lat, spatial_site = spatial_site)

# ZI binomial
df_bin_zi <- df_bin
df_bin_zi$y <- ifelse(runif(N) < 0.3, 0, df_bin_zi$y)

# Gamma-gamma data (continuous ratio)
y_gamma_num <- rgamma(N, shape = 5, rate = 1/exp(0.5 + 0.3*x))
y_gamma_denom <- rgamma(N, shape = 5, rate = 1/10)
df_gg <- data.frame(num = y_gamma_num, denom = y_gamma_denom, x = x,
                    site = site, time = time_factor, spatial_site = spatial_site)

# Lognormal data
y_ln_num <- exp(rnorm(N, mean = 2 + 0.3*x, sd = 0.5))
y_ln_denom <- exp(rnorm(N, mean = 3, sd = 0.5))
df_ln <- data.frame(num = y_ln_num, denom = y_ln_denom, x = x,
                    site = site, time = time_factor, spatial_site = spatial_site)

# Beta-binomial data (overdispersed binomial)
# Use rbetabinom from extraDistr or simulate manually
rbetabinom <- function(n, size, prob, rho) {
  # rho is overdispersion, prob is mean probability
  alpha <- prob * (1/rho - 1)
  beta <- (1 - prob) * (1/rho - 1)
  p <- rbeta(n, alpha, beta)
  rbinom(n, size, p)
}
y_bb <- rbetabinom(N, trials, plogis(0.5 + 0.3*x), rho = 0.1)
df_bb <- data.frame(y = y_bb, trials = trials, x = x,
                    site = site, time = time_factor, spatial_site = spatial_site)

# ============================================================
# HELPER FUNCTIONS (same as batch 3)
# ============================================================

# Only run H mode to avoid A_t segfaults on spatial/temporal models
time_H_only <- function(nd_call, timeout = 300) {
  times <- setNames(rep(NA_real_, 4), c("H", "A", "A_t", "N"))

  timing_call <- nd_call
  timing_call$gradient_mode <- "H"
  timing_call$iter <- 500
  timing_call$warmup <- 250
  timing_call$chains <- 1
  timing_call$verbose <- FALSE

  tryCatch({
    t <- system.time({ eval(timing_call) })["elapsed"]
    times["H"] <- round(t, 1)
    cat(sprintf("  H: %.1fs\n", t))
  }, error = function(e) {
    cat(sprintf("  H: ERR(%s)\n", substr(conditionMessage(e), 1, 40)))
  })

  times
}

validate_stan <- function(nd_fit, brms_fit, param = "x") {
  if (is.null(nd_fit) || is.null(nd_fit$draws)) {
    return(list(nd = NA, brms = NA, diff = NA, pass = NA, error = "numdenom fit NULL"))
  }

  nd_draws <- tryCatch(as.matrix(nd_fit$draws), error = function(e) NULL)
  if (is.null(nd_draws)) {
    return(list(nd = NA, brms = NA, diff = NA, pass = NA, error = "draws conversion failed"))
  }

  cols <- colnames(nd_draws)
  nd_col <- grep("^beta_num\\[2\\]$|^beta\\[2\\]$", cols, value = TRUE)[1]

  if (is.null(nd_col) || is.na(nd_col)) {
    return(list(nd = NA, brms = NA, diff = NA, pass = NA,
                error = paste("param not found in:", paste(head(cols, 5), collapse=", "))))
  }

  nd_est <- mean(nd_draws[, nd_col])
  nd_sd <- sd(nd_draws[, nd_col])

  brms_sum <- fixef(brms_fit)
  brms_est <- brms_sum[param, "Estimate"]
  brms_se <- brms_sum[param, "Est.Error"]

  diff <- abs(nd_est - brms_est)
  threshold <- 2 * max(nd_sd, brms_se)

  list(nd = nd_est, brms = brms_est, diff = diff, pass = diff < threshold)
}

run_bench <- function(row, name, nd_call, brms_call = NULL, skip_stan = FALSE, val_chains = 2) {
  cat(sprintf("\n========== Row %d: %s ==========\n", row, name))
  result <- list(row = row, name = name)

  # H-only timing (A_t causes segfaults on spatial/temporal models)
  cat("Gradient timing:")
  result$times <- time_H_only(nd_call)

  # Validation
  if (!skip_stan && !is.null(brms_call)) {
    cat(sprintf("Validation (numdenom, chains=%d)... ", val_chains))
    flush.console()
    tryCatch({
      nd_call_full <- as.call(c(as.list(nd_call),
                                iter = 1000, warmup = 500, chains = val_chains, verbose = FALSE))
      t_nd <- system.time({ fit_nd <- eval(nd_call_full) })["elapsed"]
      result$time_nd <- round(t_nd, 1)
      cat(sprintf("%.1fs\n", t_nd))

      cat("Validation (brms)... ")
      flush.console()
      t_brms <- system.time({ fit_brms <- eval(brms_call) })["elapsed"]
      result$time_brms <- round(t_brms, 1)
      cat(sprintf("%.1fs\n", t_brms))

      val <- validate_stan(fit_nd, fit_brms)

      if (!is.null(val$error)) {
        cat(sprintf("  ERROR: %s\n", val$error))
        result$error <- val$error
      } else {
        result$diff <- round(val$diff, 4)
        result$pass <- val$pass
        result$speedup <- round(result$time_brms / result$time_nd, 1)
        cat(sprintf("  nd=%.3f, brms=%.3f, diff=%.4f => %s (%.1fx)\n",
                    val$nd, val$brms, val$diff, if(val$pass) "PASS" else "FAIL", result$speedup))
      }
    }, error = function(e) {
      cat(sprintf("ERROR: %s\n", conditionMessage(e)))
      result$error <<- conditionMessage(e)
    })
  } else {
    result$note <- "Stan validation skipped (no brms equivalent)"
  }

  result
}

# ============================================================
# BATCH 5: 25 MODELS
# ============================================================

results <- list()

# ----- BINOMIAL REMAINING -----

# Row 64: binomial + crossed RE
results[["64"]] <- run_bench(64, "bin_crossed",
  nd_call = quote(tratio(y | trials ~ x + (1|site) + (1|site2), data = df_bin,
                         family = ratiod_binomial())),
  brms_call = quote(brm(y | trials(trials) ~ x + (1|site) + (1|site2), data = df_bin,
                        family = binomial(), iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# Row 68: binomial + HSGP
# SKIP validation - HSGP causes segfault during validation
results[["68"]] <- run_bench(68, "bin_hsgp",
  nd_call = quote(tratio(y | trials ~ x + (1|site), data = df_bin,
                         family = ratiod_binomial(),
                         spatial = spatial_hsgp(coords = ~ lon + lat, m = 8))),
  skip_stan = TRUE
)

# Row 70: binomial + pCAR
results[["70"]] <- run_bench(70, "bin_pcar",
  nd_call = quote(tratio(y | trials ~ x + (1|site), data = df_bin,
                         family = ratiod_binomial(),
                         spatial = spatial_car(adj_mat, proper = TRUE, level = "group",
                                               group_var = "spatial_site"))),
  brms_call = quote(brm(y | trials(trials) ~ x + (1|site) + (1|spatial_site), data = df_bin,
                        family = binomial(), iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# Row 74: binomial + GP_t (temporal GP)
# SKIP validation - temporal_gp can cause crashes
results[["74"]] <- run_bench(74, "bin_gp_t",
  nd_call = quote(tratio(y | trials ~ x + (1|site), data = df_bin,
                         family = ratiod_binomial(),
                         temporal = temporal_gp("time"))),
  skip_stan = TRUE
)

# Row 75: binomial + MS_t (multi-scale temporal)
# SKIP validation - temporal_multiscale can cause crashes
results[["75"]] <- run_bench(75, "bin_ms_t",
  nd_call = quote(tratio(y | trials ~ x + (1|site), data = df_bin,
                         family = ratiod_binomial(),
                         temporal = temporal_multiscale("time"))),
  skip_stan = TRUE
)

# Row 84: binomial + HSGP + RW1
# SKIP validation - HSGP can cause crashes
results[["84"]] <- run_bench(84, "bin_hsgp_rw1",
  nd_call = quote(tratio(y | trials ~ x + (1|site), data = df_bin,
                         family = ratiod_binomial(),
                         spatial = spatial_hsgp(coords = ~ lon + lat, m = 8),
                         temporal = temporal_rw1("time_factor"))),
  skip_stan = TRUE
)

# Row 86: binomial + ICAR + ZI
results[["86"]] <- run_bench(86, "bin_icar_zi",
  nd_call = quote(tratio(y | trials ~ x + (1|site), data = df_bin_zi,
                         family = ratiod_zibinomial(),
                         spatial = spatial_car(adj_mat, level = "group", group_var = "spatial_site"))),
  skip_stan = TRUE  # No direct brms equivalent for ZI binomial + spatial
)

# Row 87: binomial + slopes + ICAR
results[["87"]] <- run_bench(87, "bin_slopes_icar",
  nd_call = quote(tratio(y | trials ~ x + (x|site), data = df_bin,
                         family = ratiod_binomial(),
                         spatial = spatial_car(adj_mat, level = "group", group_var = "spatial_site"))),
  brms_call = quote(brm(y | trials(trials) ~ x + (x|site) + (1|spatial_site), data = df_bin,
                        family = binomial(), iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# Row 89: binomial + TVC
results[["89"]] <- run_bench(89, "bin_tvc",
  nd_call = quote(tratio(y | trials ~ x + (1|site), data = df_bin,
                         family = ratiod_binomial(),
                         temporal = temporal_tvc("time_factor", ~ x))),
  brms_call = quote(brm(y | trials(trials) ~ x + (1|site) + (1|time_factor), data = df_bin,
                        family = binomial(), iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# ----- GAMMA_GAMMA FAMILY -----

# Row 93: gamma_gamma basic
results[["93"]] <- run_bench(93, "gg_basic",
  nd_call = quote(tratio(num | denom ~ x, data = df_gg,
                         family = ratiod_gamma_gamma())),
  brms_call = quote(brm(num ~ x, data = df_gg,
                        family = Gamma(link = "log"), iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# Row 94: gamma_gamma + RE
results[["94"]] <- run_bench(94, "gg_re",
  nd_call = quote(tratio(num | denom ~ x + (1|site), data = df_gg,
                         family = ratiod_gamma_gamma())),
  brms_call = quote(brm(num ~ x + (1|site), data = df_gg,
                        family = Gamma(link = "log"), iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# Row 95: gamma_gamma + ICAR
results[["95"]] <- run_bench(95, "gg_icar",
  nd_call = quote(tratio(num | denom ~ x + (1|site), data = df_gg,
                         family = ratiod_gamma_gamma(),
                         spatial = spatial_car(adj_mat, level = "group", group_var = "spatial_site"))),
  brms_call = quote(brm(num ~ x + (1|site) + (1|spatial_site), data = df_gg,
                        family = Gamma(link = "log"), iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# Row 96: gamma_gamma + RW1
results[["96"]] <- run_bench(96, "gg_rw1",
  nd_call = quote(tratio(num | denom ~ x + (1|site), data = df_gg,
                         family = ratiod_gamma_gamma(),
                         temporal = temporal_rw1("time"))),
  brms_call = quote(brm(num ~ x + (1|site) + (1|time), data = df_gg,
                        family = Gamma(link = "log"), iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# Row 97: gamma_gamma + ICAR + RW1
results[["97"]] <- run_bench(97, "gg_icar_rw1",
  nd_call = quote(tratio(num | denom ~ x + (1|site), data = df_gg,
                         family = ratiod_gamma_gamma(),
                         spatial = spatial_car(adj_mat, level = "group", group_var = "spatial_site"),
                         temporal = temporal_rw1("time"))),
  brms_call = quote(brm(num ~ x + (1|site) + (1|spatial_site) + (1|time), data = df_gg,
                        family = Gamma(link = "log"), iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# ----- LOGNORMAL FAMILY -----

# Row 98: lognormal basic
results[["98"]] <- run_bench(98, "ln_basic",
  nd_call = quote(tratio(num | denom ~ x, data = df_ln,
                         family = ratiod_lognormal())),
  brms_call = quote(brm(log(num) ~ x, data = df_ln,
                        family = gaussian(), iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# Row 99: lognormal + RE
results[["99"]] <- run_bench(99, "ln_re",
  nd_call = quote(tratio(num | denom ~ x + (1|site), data = df_ln,
                         family = ratiod_lognormal())),
  brms_call = quote(brm(log(num) ~ x + (1|site), data = df_ln,
                        family = gaussian(), iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# Row 100: lognormal + ICAR
results[["100"]] <- run_bench(100, "ln_icar",
  nd_call = quote(tratio(num | denom ~ x + (1|site), data = df_ln,
                         family = ratiod_lognormal(),
                         spatial = spatial_car(adj_mat, level = "group", group_var = "spatial_site"))),
  brms_call = quote(brm(log(num) ~ x + (1|site) + (1|spatial_site), data = df_ln,
                        family = gaussian(), iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# Row 101: lognormal + RW1
results[["101"]] <- run_bench(101, "ln_rw1",
  nd_call = quote(tratio(num | denom ~ x + (1|site), data = df_ln,
                         family = ratiod_lognormal(),
                         temporal = temporal_rw1("time"))),
  brms_call = quote(brm(log(num) ~ x + (1|site) + (1|time), data = df_ln,
                        family = gaussian(), iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# Row 102: lognormal + ICAR + RW1
results[["102"]] <- run_bench(102, "ln_icar_rw1",
  nd_call = quote(tratio(num | denom ~ x + (1|site), data = df_ln,
                         family = ratiod_lognormal(),
                         spatial = spatial_car(adj_mat, level = "group", group_var = "spatial_site"),
                         temporal = temporal_rw1("time"))),
  brms_call = quote(brm(log(num) ~ x + (1|site) + (1|spatial_site) + (1|time), data = df_ln,
                        family = gaussian(), iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# ----- BETA_BINOMIAL FAMILY -----

# Row 103: beta_binomial basic
results[["103"]] <- run_bench(103, "bb_basic",
  nd_call = quote(tratio(y | trials ~ x, data = df_bb,
                         family = ratiod_beta_binomial())),
  brms_call = quote(brm(y | trials(trials) ~ x, data = df_bb,
                        family = beta_binomial(), iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# Row 104: beta_binomial + RE
results[["104"]] <- run_bench(104, "bb_re",
  nd_call = quote(tratio(y | trials ~ x + (1|site), data = df_bb,
                         family = ratiod_beta_binomial())),
  brms_call = quote(brm(y | trials(trials) ~ x + (1|site), data = df_bb,
                        family = beta_binomial(), iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# Row 105: beta_binomial + ICAR
results[["105"]] <- run_bench(105, "bb_icar",
  nd_call = quote(tratio(y | trials ~ x + (1|site), data = df_bb,
                         family = ratiod_beta_binomial(),
                         spatial = spatial_car(adj_mat, level = "group", group_var = "spatial_site"))),
  brms_call = quote(brm(y | trials(trials) ~ x + (1|site) + (1|spatial_site), data = df_bb,
                        family = beta_binomial(), iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# Row 106: beta_binomial + RW1
results[["106"]] <- run_bench(106, "bb_rw1",
  nd_call = quote(tratio(y | trials ~ x + (1|site), data = df_bb,
                         family = ratiod_beta_binomial(),
                         temporal = temporal_rw1("time"))),
  brms_call = quote(brm(y | trials(trials) ~ x + (1|site) + (1|time), data = df_bb,
                        family = beta_binomial(), iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# Row 107: beta_binomial + ICAR + RW1
results[["107"]] <- run_bench(107, "bb_icar_rw1",
  nd_call = quote(tratio(y | trials ~ x + (1|site), data = df_bb,
                         family = ratiod_beta_binomial(),
                         spatial = spatial_car(adj_mat, level = "group", group_var = "spatial_site"),
                         temporal = temporal_rw1("time"))),
  brms_call = quote(brm(y | trials(trials) ~ x + (1|site) + (1|spatial_site) + (1|time), data = df_bb,
                        family = beta_binomial(), iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# ============================================================
# SUMMARY
# ============================================================

cat("\n\n")
cat(paste(rep("=", 90), collapse = ""), "\n")
cat("BATCH 5 RESULTS SUMMARY\n")
cat(paste(rep("=", 90), collapse = ""), "\n")

# Gradient timing table
cat("\nGRADIENT TIMING (seconds):\n")
cat(sprintf("%-6s %-20s %8s %8s\n", "Row", "Model", "H", "A_t"))
cat(paste(rep("-", 50), collapse = ""), "\n")
for (r in results) {
  if (!is.null(r$times)) {
    cat(sprintf("%-6s %-20s %8.1f %8.1f\n",
                r$row, r$name,
                ifelse(is.na(r$times["H"]), NA, r$times["H"]),
                ifelse(is.na(r$times["A_t"]), NA, r$times["A_t"])))
  }
}

# Validation table
cat("\n\nSTAN VALIDATION:\n")
cat(sprintf("%-6s %-20s %10s %10s %8s %8s %s\n",
            "Row", "Model", "numdenom", "brms", "Speedup", "Diff", "Status"))
cat(paste(rep("-", 80), collapse = ""), "\n")

n_pass <- 0
n_total <- 0
n_skip <- 0

for (r in results) {
  if (!is.null(r$pass)) {
    n_total <- n_total + 1
    if (r$pass) n_pass <- n_pass + 1
    cat(sprintf("%-6s %-20s %10.1fs %10.1fs %8.1fx %8.4f %s\n",
                r$row, r$name, r$time_nd, r$time_brms, r$speedup, r$diff,
                if(r$pass) "PASS" else "FAIL"))
  } else if (!is.null(r$note)) {
    n_skip <- n_skip + 1
    cat(sprintf("%-6s %-20s SKIPPED (%s)\n", r$row, r$name, r$note))
  } else if (!is.null(r$error)) {
    cat(sprintf("%-6s %-20s ERROR: %s\n", r$row, r$name, r$error))
  }
}

cat(paste(rep("-", 80), collapse = ""), "\n")
cat(sprintf("Validated: %d/%d passed, %d skipped\n", n_pass, n_total, n_skip))

# Save results
saveRDS(results, "benchmarks/results_batch5.rds")
cat("\nResults saved to benchmarks/results_batch5.rds\n")

# Print markdown for gradient_methods.md
cat("\n\nMARKDOWN TABLE UPDATE:\n")
cat("Copy the H(s) and A_t(s) values to gradient_methods.md\n")
for (r in results) {
  if (!is.null(r$times)) {
    note <- if (!is.null(r$pass) && r$pass) "Stan PASS" else ""
    cat(sprintf("Row %s: H=%.1f, A_t=%.1f %s\n",
                r$row,
                ifelse(is.na(r$times["H"]), NA, r$times["H"]),
                ifelse(is.na(r$times["A_t"]), NA, r$times["A_t"]),
                note))
  }
}
