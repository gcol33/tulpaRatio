# =============================================================================
# tulpaRatio vs Stan (brms/cmdstanr), binomial family, live head-to-head.
#
# The binomial family is the fair brms comparison: trials are fixed, so brms
# with y | trials(trials) fits the same model tulpaRatio does. (For
# poisson_gamma / negbin_negbin the denominator is a random variable in
# tulpaRatio but a fixed offset in brms, so those need custom Stan models and
# are out of scope here.)
#
# Protocol (both engines, exact HMC):
#   - 4 chains, 1000 iter, 500 warmup, N = 500 observations
#   - Both engines run live on identical data in one R session.
#   - Stan model compilation is a one-time cost: each model is compiled once
#     (untimed warm-up), then the timed fit reuses the cached binary, so the
#     reported Stan time is sampling wall-clock, not compilation.
#   - Each tulpaRatio fit is checked against what was requested: same number of
#     chains, named draws. A backend that accepts an argument and discards it
#     (tulpaRatio#5) yields a fit that did less sampling than the Stan side, so
#     the row is refused as not comparable instead of being timed.
#   - Convergence is verified independently for BOTH engines (posterior::rhat
#     on the reshaped chains) -- tulpaRatio's own summary rhat is not trusted.
#     Both sides are restricted to FIXED EFFECTS, since a structured field is
#     weakly identified by construction and would otherwise be read as a
#     tulpaRatio convergence failure against Stan's fixed-only summary.
#   - A row is published only if both engines converge (rhat < 1.05) and their
#     x coefficients agree within 2 SE. Failing rows are printed with the
#     reason they failed.
#   - The headline metric is ESS/second, not wall-clock. A sampler that fails
#     to mix finishes quickly, so wall-clock alone rewards non-convergence;
#     ESS/s prices in mixing and cannot be gamed that way.
#
#   - Every config is run REPS times and the median is reported with its
#     observed range. A single timing of a sub-second fit is noise, and quoting
#     one to three significant figures overstates what was measured.
#   - Both engines are given an explicit seed, so a published row can be
#     reproduced rather than re-rolled.
#   - Time-varying rows use an iid random intercept (1|time) on BOTH sides.
#     temporal_rw1() fits an intrinsic RW1 GMRF, which brms has no equivalent
#     for -- pairing it against (1|time) would compare a rank-deficient
#     structured field against a full-rank exchangeable one and read the
#     geometry difference as a tulpaRatio result. Benchmarking the RW1 path
#     itself is blocked on tulpaRatio#6.
#
# Usage:
#   Rscript benchmarks/bench_stan_binomial.R                     # 7 configs
#   Rscript benchmarks/bench_stan_binomial.R --n=5000 --reps=3
#   Rscript benchmarks/bench_stan_binomial.R --smoke             # 1 config
# =============================================================================

args  <- commandArgs(trailingOnly = TRUE)
SMOKE <- "--smoke" %in% args
argval <- function(flag, default) {
  hit <- grep(paste0("^", flag, "="), args, value = TRUE)
  if (!length(hit)) return(default)
  as.integer(sub(paste0("^", flag, "="), "", hit[1]))
}

suppressMessages({
  library(tulpaRatio)
  library(brms)
  stopifnot(requireNamespace("posterior", quietly = TRUE))
})

ITER   <- if (SMOKE) 1000 else 1000
WARMUP <- 500
CHAINS <- if (SMOKE) 2 else 4

REPS <- if (SMOKE) 1L else argval("--reps", 3L)
SEED <- 20260720L

set.seed(123)
N <- if (SMOKE) 500L else argval("--n", 500L)
N_SITES <- 50; N_TIMES <- 20
x    <- rnorm(N)
site <- factor(rep(1:N_SITES, length.out = N))
time <- factor(rep(1:N_TIMES, length.out = N))

n_side <- ceiling(sqrt(N_SITES))
grid   <- expand.grid(lon = 1:n_side, lat = 1:n_side)[1:N_SITES, ]
si     <- as.integer(site)
lon <- grid$lon[si]; lat <- grid$lat[si]

adj_mat <- matrix(0, N_SITES, N_SITES)
for (i in 1:N_SITES) for (j in 1:N_SITES) {
  if (i != j &&
      sqrt((grid$lon[i]-grid$lon[j])^2 + (grid$lat[i]-grid$lat[j])^2) <= 1.5)
    adj_mat[i, j] <- 1
}
spatial_site <- factor(rep(1:N_SITES, length.out = N))
# brms car() matches the adjacency to the grouping factor by rownames
dimnames(adj_mat) <- list(levels(spatial_site), levels(spatial_site))

trials <- sample(10:50, N, replace = TRUE)
y_bin  <- rbinom(N, trials, plogis(0.5 + 0.3 * x))     # true x = 0.3
df     <- data.frame(y = y_bin, trials = trials, x = x, site = site,
                     time = time, lon = lon, lat = lat,
                     spatial_site = spatial_site)
df_zi  <- df; df_zi$y <- ifelse(runif(N) < 0.3, 0, df_zi$y)

# ---- the fit must be the fit that was requested ------------------------------
# tratio() routes some structures to backends that do not accept every argument
# (the Gibbs spatial path drops `chains` and returns unnamed draws, tulpaRatio#5).
# Such a fit is not comparable to a 4-chain NUTS run, so the row is refused
# rather than timed: a benchmark that silently accepts less sampling than it
# asked for reports a speedup it did not measure.
nd_contract <- function(fit) {
  if (is.null(fit$draws))                    return("fit has no draws")
  dr <- tryCatch(as.matrix(fit$draws), error = function(e) NULL)
  if (is.null(dr) || !ncol(dr))              return("draws matrix is empty")
  if (is.null(colnames(dr)))                 return("draws carry no parameter names")
  ch <- fit$chains %||% 1
  if (!identical(as.integer(ch), as.integer(CHAINS)))
    return(sprintf("ran %d chain(s), requested %d (backend '%s')",
                   ch, CHAINS, fit$backend %||% "?"))
  if (nrow(dr) %% ch != 0)                   return("draws not divisible by chains")
  NA_character_
}

# Fixed effects only, to match what summary(brmsfit)$fixed reports on the Stan
# side. A structured field (ICAR, RW1) is weakly identified by construction, so
# folding it into the same max() would compare all-parameters against
# fixed-only and read the difference as a tulpaRatio failure.
FIXED_RE <- "^beta_num\\["
nd_conv <- function(fit) {
  dr <- as.matrix(fit$draws); ch <- fit$chains
  cols <- grep(FIXED_RE, colnames(dr))
  if (!length(cols)) return(c(rhat = NA, ess = NA))
  per <- nrow(dr) / ch
  rr <- ee <- numeric(length(cols))
  for (k in seq_along(cols)) {
    m <- matrix(dr[, cols[k]], nrow = per, ncol = ch)   # contiguous chain blocks
    rr[k] <- posterior::rhat(m)
    ee[k] <- posterior::ess_bulk(m)
  }
  c(rhat = max(rr, na.rm = TRUE), ess = min(ee, na.rm = TRUE))
}
nd_x_summary <- function(fit) {
  dr <- as.matrix(fit$draws); nm <- colnames(dr)
  col <- grep("beta_num\\[2\\]", nm, value = TRUE)
  if (!length(col)) col <- grep("(^|[^a-z])x($|[^a-z])", nm, value = TRUE)
  if (!length(col)) return(c(mean = NA, sd = NA))
  v <- dr[, col[1]]; c(mean = mean(v), sd = sd(v))
}
# ESS/s is driven by the worst fixed effect, so agreement has to cover the same
# set. Checking only the slope lets an unidentified intercept destroy ESS/s
# while the row still reports agree=TRUE, which reads as a sampler being slow
# rather than a parameter being unidentified.
nd_intercept_summary <- function(fit) {
  dr <- as.matrix(fit$draws); nm <- colnames(dr)
  col <- grep("beta_num\\[1\\]", nm, value = TRUE)
  if (!length(col)) return(c(mean = NA, sd = NA))
  v <- dr[, col[1]]; c(mean = mean(v), sd = sd(v))
}

safe <- function(expr) tryCatch(eval(expr, envir = globalenv()),
                                error = function(e) structure(list(), err = conditionMessage(e)))
is_err <- function(x) !is.null(attr(x, "err"))

run_row <- function(name, nd_call, brm_call) {
  cat(sprintf("\n===== %s =====\n", name)); flush.console()
  blank <- data.frame(model = name, tulpaRatio_s = NA, stan_s = NA, stan_cold_s = NA,
                      speedup = NA, nd_x = NA, stan_x = NA, agree = NA,
                      nd_rhat = NA, stan_rhat = NA, nd_ess = NA, stan_ess = NA,
                      nd_ess_s = NA, stan_ess_s = NA, eff_ratio = NA,
                      contract = NA_character_)

  # --- tulpaRatio (in-process; no per-fit compilation) ---
  cat("  tulpaRatio ... "); flush.console()
  t_nd <- system.time(fit_nd <- safe(nd_call))[["elapsed"]]
  if (is_err(fit_nd)) { cat("ERROR:", attr(fit_nd, "err"), "\n"); return(blank) }

  bad <- nd_contract(fit_nd)
  if (!is.na(bad)) {
    cat(sprintf("%.2fs -- NOT COMPARABLE: %s\n", t_nd, bad))
    blank$tulpaRatio_s <- round(t_nd, 2); blank$contract <- bad
    return(blank)
  }
  cv  <- nd_conv(fit_nd); ndx <- nd_x_summary(fit_nd)
  ndi <- nd_intercept_summary(fit_nd)
  cat(sprintf("%.2fs (rhat=%.3f ess=%.0f)\n", t_nd, cv["rhat"], cv["ess"]))

  # --- Stan: compile once (untimed), then time the warm fit ---
  cat("  Stan compile ... "); flush.console()
  t_cold <- system.time(fit_warm <- safe(brm_call))[["elapsed"]]
  if (is_err(fit_warm)) { cat("ERROR:", attr(fit_warm, "err"), "\n")
    blank$tulpaRatio_s <- round(t_nd, 2); return(blank) }
  cat(sprintf("%.1fs (cold)\n", t_cold))
  cat("  Stan sample  ... "); flush.console()
  t_st <- system.time(fit_st <- safe(brm_call))[["elapsed"]]     # reuses cached binary
  if (is_err(fit_st)) fit_st <- fit_warm                          # fall back to warm fit
  cat(sprintf("%.2fs (warm)\n", t_st))

  fe   <- tryCatch(brms::fixef(fit_st), error = function(e) NULL)
  st_x <- if (!is.null(fe) && "x" %in% rownames(fe)) fe["x", "Estimate"] else NA
  st_se<- if (!is.null(fe) && "x" %in% rownames(fe)) fe["x", "Est.Error"] else NA
  st_sum  <- tryCatch(summary(fit_st)$fixed, error = function(e) NULL)
  st_rhat <- if (!is.null(st_sum)) max(st_sum[, "Rhat"], na.rm = TRUE) else NA
  st_ess  <- if (!is.null(st_sum)) min(st_sum[, "Bulk_ESS"], na.rm = TRUE) else NA
  st_i    <- if (!is.null(fe) && "Intercept" %in% rownames(fe)) fe["Intercept", "Estimate"] else NA
  st_i_se <- if (!is.null(fe) && "Intercept" %in% rownames(fe)) fe["Intercept", "Est.Error"] else NA
  agree_x <- isTRUE(abs(ndx["mean"] - st_x) < 2 * max(ndx["sd"], st_se, na.rm = TRUE))
  agree_i <- if (is.na(st_i) || is.na(ndi["mean"])) NA else
    isTRUE(abs(ndi["mean"] - st_i) < 2 * max(ndi["sd"], st_i_se, na.rm = TRUE))
  agree <- if (isTRUE(is.na(agree_i))) agree_x else (agree_x && agree_i)
  if (isTRUE(agree_x) && isTRUE(!agree_i))
    cat(sprintf("  intercept DISAGREES: tulpaRatio %.3f (sd %.3f) | Stan %.3f (se %.3f)\n",
                ndi["mean"], ndi["sd"], st_i, st_i_se))

  # Efficiency, not wall-clock, is the honest MCMC metric: a stuck sampler is
  # always fast. ESS/s prices in mixing, so a non-converged fit cannot win.
  nd_eff <- unname(cv["ess"]) / t_nd
  st_eff <- unname(st_ess) / t_st

  cat(sprintf("  x: tulpaRatio %.3f | Stan %.3f | agree=%s | %.1fx wall | %.1fx ESS/s\n",
              ndx["mean"], st_x, agree, t_st / t_nd, nd_eff / st_eff))

  data.frame(model = name, tulpaRatio_s = round(t_nd, 2), stan_s = round(t_st, 2),
             stan_cold_s = round(t_cold, 1), speedup = round(t_st / t_nd, 1),
             nd_x = round(unname(ndx["mean"]), 3), stan_x = round(unname(st_x), 3),
             agree = agree, nd_rhat = round(unname(cv["rhat"]), 3),
             stan_rhat = round(unname(st_rhat), 3),
             nd_ess = round(unname(cv["ess"])), stan_ess = round(unname(st_ess)),
             nd_ess_s = round(nd_eff, 1), stan_ess_s = round(st_eff, 1),
             eff_ratio = round(nd_eff / st_eff, 1), contract = NA_character_)
}

brmf <- function(f, family, data = quote(df))
  bquote(brm(.(f), data = .(data), family = .(family), iter = .(ITER),
             warmup = .(WARMUP), chains = .(CHAINS), backend = "cmdstanr",
             seed = .(SEED), data2 = list(adj_mat = adj_mat),
             silent = 2, refresh = 0))
ndf <- function(extra)
  bquote(tratio(.(extra$formula), data = .(extra$data %||% quote(df)),
                family = .(extra$family), spatial = .(extra$spatial),
                temporal = .(extra$temporal),
                control = list(iter = .(ITER), warmup = .(WARMUP), chains = .(CHAINS), seed = .(SEED), verbose = FALSE)))
`%||%` <- function(a, b) if (is.null(a)) b else a

rows <- list(
  list("Bin_base",
       ndf(list(formula = quote(y | trials ~ x), family = quote(ratiod_binomial()),
                spatial = NULL, temporal = NULL)),
       brmf(quote(y | trials(trials) ~ x), quote(binomial()))),
  list("Bin_RE",
       ndf(list(formula = quote(y | trials ~ x + (1|site)), family = quote(ratiod_binomial()),
                spatial = NULL, temporal = NULL)),
       brmf(quote(y | trials(trials) ~ x + (1|site)), quote(binomial()))),
  list("Bin_ICAR",
       ndf(list(formula = quote(y | trials ~ x + (1|site)), family = quote(ratiod_binomial()),
                spatial = quote(spatial_car(adj_mat, spatial_level="group", group_var="spatial_site")),
                temporal = NULL)),
       brmf(quote(y | trials(trials) ~ x + (1|site) +
                    car(adj_mat, gr = spatial_site, type = "icar")), quote(binomial()))),
  list("Bin_BYM2",
       ndf(list(formula = quote(y | trials ~ x + (1|site)), family = quote(ratiod_binomial()),
                spatial = quote(spatial_bym2(adj_mat, spatial_level="group", group_var="spatial_site")),
                temporal = NULL)),
       brmf(quote(y | trials(trials) ~ x + (1|site) +
                    car(adj_mat, gr = spatial_site, type = "bym2")), quote(binomial()))),
  list("Bin_TIME_RE",
       ndf(list(formula = quote(y | trials ~ x + (1|site) + (1|time)),
                family = quote(ratiod_binomial()),
                spatial = NULL, temporal = NULL)),
       brmf(quote(y | trials(trials) ~ x + (1|site) + (1|time)), quote(binomial()))),
  list("Bin_ZI",
       ndf(list(formula = quote(y | trials ~ x + (1|site)), family = quote(ratiod_zi_binomial()),
                spatial = NULL, temporal = NULL, data = quote(df_zi))),
       brmf(quote(y | trials(trials) ~ x + (1|site)), quote(zero_inflated_binomial()), quote(df_zi))),
  list("Bin_ICAR_TIME",
       ndf(list(formula = quote(y | trials ~ x + (1|site) + (1|time)),
                family = quote(ratiod_binomial()),
                spatial = quote(spatial_car(adj_mat, spatial_level="group", group_var="spatial_site")),
                temporal = NULL)),
       brmf(quote(y | trials(trials) ~ x + (1|site) +
                    car(adj_mat, gr = spatial_site, type = "icar") + (1|time)), quote(binomial())))
)
if (SMOKE) rows <- rows[1]

cat(sprintf("tulpaRatio %s vs brms %s / cmdstan %s | %d chains x %d iter (warmup %d), N=%d | %d configs%s\n",
            packageVersion("tulpaRatio"), packageVersion("brms"),
            cmdstanr::cmdstan_version(), CHAINS, ITER, WARMUP, N, length(rows),
            if (SMOKE) " [SMOKE]" else ""))

raw <- do.call(rbind, lapply(seq_len(REPS), function(rep) {
  cat(sprintf("\n############ replicate %d/%d ############\n", rep, REPS))
  cbind(rep = rep, do.call(rbind, lapply(rows, function(r) run_row(r[[1]], r[[2]], r[[3]]))))
}))

cat("\n\n================ ALL REPLICATES ================\n")
print(raw, row.names = FALSE)

# Aggregate across replicates: report the median and the observed spread rather
# than a single draw from a noisy timer. A model is disqualified if it failed in
# ANY replicate -- the worst case is the honest one.
agg1 <- function(d) {
  med <- function(v) if (all(is.na(v))) NA else stats::median(v, na.rm = TRUE)
  ctr <- d$contract[!is.na(d$contract)]
  data.frame(
    model        = d$model[1],
    n_rep        = nrow(d),
    tulpaRatio_s = round(med(d$tulpaRatio_s), 2),
    stan_s       = round(med(d$stan_s), 2),
    nd_ess_s     = round(med(d$nd_ess_s), 1),
    stan_ess_s   = round(med(d$stan_ess_s), 1),
    eff_ratio    = round(med(d$eff_ratio), 1),
    eff_lo       = if (all(is.na(d$eff_ratio))) NA else round(min(d$eff_ratio, na.rm = TRUE), 1),
    eff_hi       = if (all(is.na(d$eff_ratio))) NA else round(max(d$eff_ratio, na.rm = TRUE), 1),
    nd_rhat      = if (all(is.na(d$nd_rhat))) NA else round(max(d$nd_rhat, na.rm = TRUE), 3),
    stan_rhat    = if (all(is.na(d$stan_rhat))) NA else round(max(d$stan_rhat, na.rm = TRUE), 3),
    nd_ess       = if (all(is.na(d$nd_ess))) NA else round(min(d$nd_ess, na.rm = TRUE)),
    agree        = all(d$agree %in% TRUE),
    contract     = if (length(ctr)) ctr[1] else NA_character_,
    stringsAsFactors = FALSE)
}
res <- do.call(rbind, lapply(split(raw, factor(raw$model, levels = vapply(rows, `[[`, "", 1))), agg1))

cat("\n\n================ RESULTS (median of", REPS, "replicates) ================\n")
print(res, row.names = FALSE)

if (!SMOKE) {
  saveRDS(list(agg = res, raw = raw, n = N, reps = REPS, chains = CHAINS,
               iter = ITER, warmup = WARMUP),
          "benchmarks/results_stan_binomial.rds")

  # A row is publishable only if BOTH engines converged and agree on x. Rows
  # that fail are listed with the reason, never silently dropped: an excluded
  # row is a finding about the sampler, not a blemish to hide.
  why <- function(i) {
    if (!is.na(res$contract[i]))                     return(res$contract[i])
    if (is.na(res$stan_s[i]))                        return("Stan fit failed")
    if (is.na(res$nd_rhat[i]))                       return("no tulpaRatio draws")
    if (isTRUE(res$nd_rhat[i]   >= 1.05))            return(sprintf("tulpaRatio not converged (Rhat %.2f, ESS %d)", res$nd_rhat[i], res$nd_ess[i]))
    if (isTRUE(res$stan_rhat[i] >= 1.05))            return(sprintf("Stan not converged (Rhat %.2f)", res$stan_rhat[i]))
    if (!isTRUE(res$agree[i]))                       return("x coefficients disagree")
    NA_character_
  }
  reason <- vapply(seq_len(nrow(res)), why, character(1))
  ok     <- res[is.na(reason), ]

  md <- c(
    sprintf("N = %d, %d chains x %d iter (warmup %d), median of %d replicates.",
            N, CHAINS, ITER, WARMUP, REPS),
    "",
    "| Model | tulpaRatio (s) | Stan (s) | tulpaRatio ESS/s | Stan ESS/s | Efficiency | Range |",
    "|-------|---------------:|---------:|-----------------:|-----------:|-----------:|------:|")
  for (i in which(is.na(reason))) md <- c(md, sprintf(
    "| %s | %.2f | %.2f | %.1f | %.1f | %.1fx | %.1f-%.1f |",
    res$model[i], res$tulpaRatio_s[i], res$stan_s[i],
    res$nd_ess_s[i], res$stan_ess_s[i], res$eff_ratio[i],
    res$eff_lo[i], res$eff_hi[i]))

  excl <- which(!is.na(reason))
  if (length(excl)) {
    md <- c(md, "", "Excluded:", "")
    for (i in excl) md <- c(md, sprintf("- `%s` -- %s", res$model[i], reason[i]))
  }
  writeLines(md, "benchmarks/results_stan_binomial.md")

  cat(sprintf("\nSaved results_stan_binomial.{rds,md}  (%d/%d rows publishable)\n",
              nrow(ok), nrow(res)))
  for (i in excl) cat(sprintf("  excluded %-14s %s\n", res$model[i], reason[i]))
}
writeLines("DONE", "benchmarks/.bench_done")
cat("DONE\n")
