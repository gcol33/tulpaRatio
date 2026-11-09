# =============================================================================
# Single-row benchmark runner (called as subprocess)
# Usage: Rscript benchmarks/bench_single_row.R <row_number> [gradient_mode] [timeout] [seed]
# Output: RESULT:<row>:<mode>:<time_or_error>
# =============================================================================

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) stop("Usage: Rscript bench_single_row.R <row_number> [gradient_mode] [timeout] [seed]")
ROW_NUM <- as.integer(args[1])
GRAD_MODE <- if (length(args) >= 2) args[2] else "H"

suppressPackageStartupMessages(library(tulpaRatio))

# Standard parameters
N_OBS       <- 500L
N_ITER      <- 500L
N_WARMUP    <- 250L
N_CHAINS    <- 1L
N_SITES     <- 50L
N_TIMES     <- 20L
N_OBS_GP    <- 80L
N_OBS_LATENT <- 50L
N_SITES_GP  <- 20L
N_TIMES_GP  <- 10L
N_SITES_LATENT <- 10L
N_TIMES_LAT <- 10L
TIMEOUT_SEC <- if (length(args) >= 3) as.integer(args[3]) else 600L
SEED        <- if (length(args) >= 4) as.integer(args[4]) else 123L

set.seed(SEED)

# --- Data generation (same as bench_4modes_all.R) ---
generate_datasets <- function(n, n_sites, n_times, seed = 123) {
  set.seed(seed)
  site <- factor(rep(1:n_sites, length.out = n))
  time <- rep(1:n_times, length.out = n)
  time_factor <- factor(time)
  x <- rnorm(n)
  z <- rnorm(n)

  n_side <- ceiling(sqrt(n_sites))
  grid <- expand.grid(lon = 1:n_side, lat = 1:n_side)[1:n_sites, ]
  site_int <- as.integer(site)
  lon_site <- grid$lon[site_int]
  lat_site <- grid$lat[site_int]
  lon_obs <- runif(n, 0, 10)
  lat_obs <- runif(n, 0, 10)

  adj_mat <- matrix(0L, n_sites, n_sites)
  for (i in 1:n_sites) {
    for (j in 1:n_sites) {
      if (i != j) {
        d <- sqrt((grid$lon[i] - grid$lon[j])^2 + (grid$lat[i] - grid$lat[j])^2)
        if (d <= 1.5) adj_mat[i, j] <- 1L
      }
    }
  }

  y_pg_num   <- rpois(n, exp(2 + 0.5 * x))
  y_pg_denom <- rgamma(n, 10, 1)
  df_pg <- data.frame(y = y_pg_num, denom = y_pg_denom, x = x, z = z,
                      site = site, time = time_factor, time_num = time,
                      lon = lon_site, lat = lat_site,
                      lon_obs = lon_obs, lat_obs = lat_obs,
                      spatial_site = site)

  y_nb_num   <- rnbinom(n, mu = exp(2 + 0.3 * x), size = 5)
  y_nb_denom <- rnbinom(n, mu = 100, size = 10)
  y_nb_denom[y_nb_denom == 0] <- 1L
  df_nb <- data.frame(y = y_nb_num, denom = y_nb_denom, x = x, z = z,
                      site = site, time = time_factor, time_num = time,
                      lon = lon_site, lat = lat_site,
                      lon_obs = lon_obs, lat_obs = lat_obs,
                      spatial_site = site)

  trials <- sample(10:50, n, replace = TRUE)
  y_bin  <- rbinom(n, trials, plogis(0.5 + 0.3 * x))
  df_bin <- data.frame(y = y_bin, trials = trials, x = x, z = z,
                       site = site, time = time_factor, time_num = time,
                       lon = lon_site, lat = lat_site,
                       lon_obs = lon_obs, lat_obs = lat_obs,
                       spatial_site = site)

  y_gg_num   <- rgamma(n, shape = 5, rate = 5 / exp(2 + 0.3 * x))
  y_gg_denom <- rgamma(n, shape = 8, rate = 8 / exp(3 + 0.2 * x))
  y_gg_num[y_gg_num <= 0]     <- 0.01
  y_gg_denom[y_gg_denom <= 0] <- 0.01
  df_gg <- data.frame(y = y_gg_num, denom = y_gg_denom, x = x, z = z,
                      site = site, time = time_factor, time_num = time,
                      lon = lon_site, lat = lat_site,
                      spatial_site = site)

  y_ln_num   <- exp(rnorm(n, mean = 2 + 0.3 * x, sd = 0.5))
  y_ln_denom <- exp(rnorm(n, mean = 3 + 0.2 * x, sd = 0.5))
  df_ln <- data.frame(y = y_ln_num, denom = y_ln_denom, x = x, z = z,
                      site = site, time = time_factor, time_num = time,
                      lon = lon_site, lat = lat_site,
                      spatial_site = site)

  mu_bb  <- plogis(0.5 + 0.3 * x)
  phi_bb <- 10
  a_bb   <- mu_bb * phi_bb
  b_bb   <- (1 - mu_bb) * phi_bb
  y_bb   <- rbeta(n, a_bb, b_bb)
  y_bb   <- rbinom(n, trials, y_bb)
  df_bb <- data.frame(y = y_bb, trials = trials, x = x, z = z,
                      site = site, time = time_factor, time_num = time,
                      lon = lon_site, lat = lat_site,
                      spatial_site = site)

  # NegBin-Gamma data (NegBin numerator + Gamma denominator)
  y_nbg_num   <- rnbinom(n, mu = exp(2 + 0.3 * x), size = 5)
  y_nbg_denom <- rgamma(n, 10, 1)
  df_nbg <- data.frame(y = y_nbg_num, denom = y_nbg_denom, x = x, z = z,
                        site = site, time = time_factor, time_num = time,
                        lon = lon_site, lat = lat_site,
                        lon_obs = lon_obs, lat_obs = lat_obs,
                        spatial_site = site)

  list(pg = df_pg, nb = df_nb, bin = df_bin,
       gg = df_gg, ln = df_ln, bb = df_bb, nbg = df_nbg,
       adj_mat = adj_mat, grid = grid)
}

inject_zeros <- function(df, col = "y", prob = 0.3) {
  df[[col]] <- ifelse(runif(nrow(df)) < prob, 0L, df[[col]])
  df
}
inject_ones_binomial <- function(df, prob_oi = 0.15) {
  is_oi <- rbinom(nrow(df), 1, prob_oi)
  df$y <- ifelse(is_oi == 1, df$trials, df$y)
  df
}
inject_zoib <- function(df, prob_zi = 0.1, prob_oi = 0.1) {
  n <- nrow(df)
  is_zi <- rbinom(n, 1, prob_zi)
  is_oi <- rbinom(n, 1, prob_oi)
  df$y <- ifelse(is_zi == 1, 0L, ifelse(is_oi == 1, df$trials, df$y))
  df
}

# --- Row configurations ---
make_row <- function(row, fam, re = "int", sp = "none", temp = "none",
                     zi = "none", st = "none", latent = FALSE,
                     use_gp_data = FALSE) {
  list(row = row, fam = fam, re = re, sp = sp, temp = temp,
       zi = zi, st = st, latent = latent, use_gp_data = use_gp_data)
}

ROW_CONFIGS <- list(
  make_row( 1, "pg", re="none"),  make_row( 2, "pg", re="int"),
  make_row( 3, "pg", re="slopes"), make_row( 4, "pg", re="crossed"),
  make_row( 5, "pg", sp="icar"),  make_row( 6, "pg", sp="bym2"),
  make_row( 7, "pg", sp="gp", use_gp_data=TRUE),
  make_row( 8, "pg", sp="hsgp"),
  make_row( 9, "pg", sp="msgp", use_gp_data=TRUE),
  make_row(10, "pg", sp="pcar"),
  make_row(11, "pg", temp="rw1"), make_row(12, "pg", temp="rw2"),
  make_row(13, "pg", temp="ar1"), make_row(14, "pg", temp="gp_t"),
  make_row(15, "pg", temp="ms_t"),
  make_row(16, "pg", zi="zi"),    make_row(17, "pg", zi="hurdle"),
  make_row(18, "pg", sp="icar", temp="rw1"),
  make_row(19, "pg", sp="bym2", temp="rw1"),
  make_row(20, "pg", sp="icar", temp="ar1"),
  make_row(21, "pg", sp="gp", temp="rw1", use_gp_data=TRUE),
  make_row(22, "pg", sp="hsgp", temp="rw1"),
  make_row(23, "pg", sp="msgp", temp="rw1", use_gp_data=TRUE),
  make_row(24, "pg", sp="icar", zi="zi"),
  make_row(25, "pg", re="slopes", sp="icar"),
  make_row(26, "pg", sp="svc"),
  make_row(27, "pg", temp="tvc"),
  make_row(28, "pg", sp="icar", temp="rw1", st="I"),
  make_row(29, "pg", sp="icar", temp="rw1", st="IV"),
  make_row(30, "pg", latent=TRUE),
  make_row(31, "nb", re="none"),  make_row(32, "nb", re="int"),
  make_row(33, "nb", re="slopes"), make_row(34, "nb", re="crossed"),
  make_row(35, "nb", sp="icar"),  make_row(36, "nb", sp="bym2"),
  make_row(37, "nb", sp="gp", use_gp_data=TRUE),
  make_row(38, "nb", sp="hsgp"),
  make_row(39, "nb", sp="msgp", use_gp_data=TRUE),
  make_row(40, "nb", sp="pcar"),
  make_row(41, "nb", temp="rw1"), make_row(42, "nb", temp="rw2"),
  make_row(43, "nb", temp="ar1"), make_row(44, "nb", temp="gp_t"),
  make_row(45, "nb", temp="ms_t"),
  make_row(46, "nb", zi="zi"),    make_row(47, "nb", zi="hurdle"),
  make_row(48, "nb", sp="icar", temp="rw1"),
  make_row(49, "nb", sp="bym2", temp="rw1"),
  make_row(50, "nb", sp="icar", temp="ar1"),
  make_row(51, "nb", sp="gp", temp="rw1", use_gp_data=TRUE),
  make_row(52, "nb", sp="hsgp", temp="rw1"),
  make_row(53, "nb", sp="msgp", temp="rw1", use_gp_data=TRUE),
  make_row(54, "nb", sp="icar", zi="zi"),
  make_row(55, "nb", re="slopes", sp="icar"),
  make_row(56, "nb", sp="svc"),
  make_row(57, "nb", temp="tvc"),
  make_row(58, "nb", sp="icar", temp="rw1", st="I"),
  make_row(59, "nb", sp="icar", temp="rw1", st="IV"),
  make_row(60, "nb", latent=TRUE),
  make_row(61, "bin", re="none"), make_row(62, "bin", re="int"),
  make_row(63, "bin", re="slopes"), make_row(64, "bin", re="crossed"),
  make_row(65, "bin", sp="icar"), make_row(66, "bin", sp="bym2"),
  make_row(67, "bin", sp="gp", use_gp_data=TRUE),
  make_row(68, "bin", sp="hsgp"),
  make_row(69, "bin", sp="msgp", use_gp_data=TRUE),
  make_row(70, "bin", sp="pcar"),
  make_row(71, "bin", temp="rw1"), make_row(72, "bin", temp="rw2"),
  make_row(73, "bin", temp="ar1"), make_row(74, "bin", temp="gp_t"),
  make_row(75, "bin", temp="ms_t"),
  make_row(76, "bin", zi="zi"),   make_row(77, "bin", zi="hurdle"),
  make_row(78, "bin", zi="oi"),   make_row(79, "bin", zi="zoib"),
  make_row(80, "bin", sp="icar", temp="rw1"),
  make_row(81, "bin", sp="bym2", temp="rw1"),
  make_row(82, "bin", sp="icar", temp="ar1"),
  make_row(83, "bin", sp="gp", temp="rw1", use_gp_data=TRUE),
  make_row(84, "bin", sp="hsgp", temp="rw1"),
  make_row(85, "bin", sp="msgp", temp="rw1", use_gp_data=TRUE),
  make_row(86, "bin", sp="icar", zi="zi"),
  make_row(87, "bin", re="slopes", sp="icar"),
  make_row(88, "bin", sp="svc"),
  make_row(89, "bin", temp="tvc"),
  make_row(90, "bin", sp="icar", temp="rw1", st="I"),
  make_row(91, "bin", sp="icar", temp="rw1", st="IV"),
  make_row(92, "bin", latent=TRUE),
  make_row(93, "gg", re="none"),  make_row(94, "gg", re="int"),
  make_row(95, "gg", sp="icar"),  make_row(96, "gg", temp="rw1"),
  make_row(97, "gg", sp="icar", temp="rw1"),
  make_row( 98, "ln", re="none"), make_row( 99, "ln", re="int"),
  make_row(100, "ln", sp="icar"), make_row(101, "ln", temp="rw1"),
  make_row(102, "ln", sp="icar", temp="rw1"),
  make_row(103, "bb", re="none"), make_row(104, "bb", re="int"),
  make_row(105, "bb", sp="icar"), make_row(106, "bb", temp="rw1"),
  make_row(107, "bb", sp="icar", temp="rw1"),
  # Section 7: negbin_gamma (rows 108-137)
  make_row(108, "nbg", re="none"), make_row(109, "nbg", re="int"),
  make_row(110, "nbg", re="slopes"), make_row(111, "nbg", re="crossed"),
  make_row(112, "nbg", sp="icar"), make_row(113, "nbg", sp="bym2"),
  make_row(114, "nbg", sp="gp", use_gp_data=TRUE),
  make_row(115, "nbg", sp="hsgp"),
  make_row(116, "nbg", sp="msgp", use_gp_data=TRUE),
  make_row(117, "nbg", sp="pcar"),
  make_row(118, "nbg", temp="rw1"), make_row(119, "nbg", temp="rw2"),
  make_row(120, "nbg", temp="ar1"), make_row(121, "nbg", temp="gp_t"),
  make_row(122, "nbg", temp="ms_t"),
  make_row(123, "nbg", zi="zi"), make_row(124, "nbg", zi="hurdle"),
  make_row(125, "nbg", sp="icar", temp="rw1"),
  make_row(126, "nbg", sp="bym2", temp="rw1"),
  make_row(127, "nbg", sp="icar", temp="ar1"),
  make_row(128, "nbg", sp="gp", temp="rw1", use_gp_data=TRUE),
  make_row(129, "nbg", sp="hsgp", temp="rw1"),
  make_row(130, "nbg", sp="msgp", temp="rw1", use_gp_data=TRUE),
  make_row(131, "nbg", sp="icar", zi="zi"),
  make_row(132, "nbg", re="slopes", sp="icar"),
  make_row(133, "nbg", sp="svc"),
  make_row(134, "nbg", temp="tvc"),
  make_row(135, "nbg", sp="icar", temp="rw1", st="I"),
  make_row(136, "nbg", sp="icar", temp="rw1", st="IV"),
  make_row(137, "nbg", latent=TRUE),
  # Section 6: Collapsed parameterizations (rows 138-149)
  make_row(138, "pg", re="none", sp="cicar"),
  make_row(139, "pg", re="none", sp="cbym2"),
  make_row(140, "nb", re="none", sp="cicar"),
  make_row(141, "nb", re="none", sp="cbym2"),
  make_row(142, "bin", re="none", sp="cicar"),
  make_row(143, "bin", re="none", sp="cbym2"),
  make_row(144, "nbg", re="none", sp="cicar"),
  make_row(145, "nbg", re="none", sp="cbym2"),
  make_row(146, "pg", re="int", sp="cicar"),
  make_row(147, "pg", re="int", sp="cbym2"),
  make_row(148, "nb", re="int", sp="cicar"),
  make_row(149, "nb", re="int", sp="cbym2"),
  # Section 8: Exotic multi-feature combos (rows 150-170)
  # These lack H-mode gradients — fall back to N (via verify_gradient_runtime) or A_r
  # HSGP + exotic temporal/features (branch 2 catches HSGP, missing other feature)
  make_row(150, "pg", sp="hsgp", temp="gp_t"),
  make_row(151, "pg", sp="hsgp", temp="ms_t"),
  make_row(152, "pg", sp="hsgp", temp="tvc"),
  make_row(153, "pg", sp="hsgp", temp="rw1", st="IV"),
  make_row(154, "pg", sp="hsgp", latent=TRUE),
  make_row(155, "pg", re="slopes", sp="hsgp"),
  # SVC + temporal (branch 10/11 catches SVC, missing temporal)
  make_row(156, "pg", sp="svc", temp="rw1"),
  make_row(157, "pg", sp="svc", temp="ar1"),
  make_row(158, "pg", sp="svc", temp="gp_t"),
  make_row(159, "pg", sp="svc", temp="tvc"),
  # TVC + spatial (branch 12 catches TVC, missing spatial)
  make_row(160, "pg", sp="icar", temp="tvc"),
  make_row(161, "pg", sp="bym2", temp="tvc"),
  # Latent + spatial/temporal (branch 16 catches latent, missing spatial/temporal)
  make_row(163, "pg", sp="icar", latent=TRUE),
  make_row(164, "pg", temp="rw1", latent=TRUE),
  make_row(165, "pg", sp="icar", temp="rw1", latent=TRUE),
  # Temporal GP + spatial (branch 14 catches GP_t, missing spatial)
  make_row(166, "pg", sp="icar", temp="gp_t"),
  make_row(167, "pg", sp="bym2", temp="gp_t"),
  # MS_t + spatial (branch 15 catches MS_t, missing spatial)
  make_row(168, "pg", sp="icar", temp="ms_t"),
  # Crossed RE + slopes (no branch catches)
  make_row(170, "pg", re="crossed_slopes")
)

config_by_row <- list()
for (cfg in ROW_CONFIGS) config_by_row[[as.character(cfg$row)]] <- cfg

cfg <- config_by_row[[as.character(ROW_NUM)]]
if (is.null(cfg)) {
  cat(sprintf("RESULT:%d:ERROR:undefined row\n", ROW_NUM))
  quit(save = "no", status = 0)
}

# --- Generate appropriate dataset ---
if (cfg$latent) {
  ds <- generate_datasets(N_OBS_LATENT, N_SITES_LATENT, N_TIMES_LAT, seed = SEED)
} else if (cfg$use_gp_data) {
  ds <- generate_datasets(N_OBS_GP, N_SITES_GP, N_TIMES_GP, seed = SEED)
} else {
  ds <- generate_datasets(N_OBS, N_SITES, N_TIMES, seed = SEED)
}

# --- Build ratiod args (same logic as bench_4modes_all.R) ---
fam <- cfg$fam; re <- cfg$re; sp <- cfg$sp
temp <- cfg$temp; zi <- cfg$zi; st <- cfg$st

df <- ds[[fam]]
adj_mat <- ds$adj_mat

if (zi == "zi" && fam %in% c("pg", "nb", "nbg", "bin")) {
  df <- inject_zeros(df, "y", 0.3)
} else if (zi == "hurdle" && fam %in% c("pg", "nb", "nbg", "bin")) {
  df <- inject_zeros(df, "y", 0.3)
} else if (zi == "oi") {
  df <- inject_ones_binomial(df, 0.15)
} else if (zi == "zoib") {
  df <- inject_zoib(df, 0.1, 0.1)
}

lhs <- if (fam %in% c("bin", "bb")) "y | trials" else "y | denom"
rhs <- if (re == "none") {
  "x"
} else if (re == "int") {
  "x + (1 | site)"
} else if (re == "slopes") {
  "x + (1 + z | site)"
} else if (re == "crossed") {
  df$site2 <- factor(sample(1:5, nrow(df), replace = TRUE))
  "x + (1 | site) + (1 | site2)"
} else if (re == "crossed_slopes") {
  df$site2 <- factor(sample(1:5, nrow(df), replace = TRUE))
  "x + (1 + z | site) + (1 | site2)"
}
form <- as.formula(paste(lhs, "~", rhs))

nd_family <- switch(fam,
  "pg"  = ratiod_poisson_gamma(),
  "nb"  = ratiod_negbin_negbin(),
  "bin" = ratiod_binomial(),
  "gg"  = ratiod_gamma_gamma(),
  "ln"  = ratiod_lognormal(),
  "bb"  = ratiod_beta_binomial(),
  "nbg" = ratiod_negbin_gamma()
)

nd_spatial <- NULL
if (sp == "icar") {
  nd_spatial <- spatial_car(adj_mat, spatial_level = "group", group_var = "spatial_site")
} else if (sp == "bym2") {
  nd_spatial <- spatial_bym2(adj_mat, spatial_level = "group", group_var = "spatial_site")
} else if (sp == "pcar") {
  nd_spatial <- spatial_car(adj_mat, spatial_level = "group", group_var = "spatial_site", proper = TRUE)
} else if (sp == "gp") {
  nd_spatial <- spatial_gp(coords = ~ lon_obs + lat_obs)
} else if (sp == "hsgp") {
  nd_spatial <- spatial_hsgp(coords = ~ lon + lat)
} else if (sp == "msgp") {
  nd_spatial <- spatial_multiscale(coords = ~ lon_obs + lat_obs)
} else if (sp == "svc") {
  nd_spatial <- spatial_svc(coords = ~ lon_obs + lat_obs, terms = 1, approx = "hsgp")
} else if (sp == "cicar") {
  nd_spatial <- spatial_car(adj_mat, spatial_level = "group", group_var = "spatial_site",
                            parameterization = "collapsed")
} else if (sp == "cbym2") {
  nd_spatial <- spatial_bym2(adj_mat, spatial_level = "group", group_var = "spatial_site",
                             parameterization = "collapsed")
}

nd_temporal <- NULL
if (temp == "rw1") {
  nd_temporal <- temporal_rw1("time")
} else if (temp == "rw2") {
  nd_temporal <- temporal_rw2("time")
} else if (temp == "ar1") {
  nd_temporal <- temporal_ar1("time")
} else if (temp == "gp_t") {
  nd_temporal <- temporal_gp("time_num")
} else if (temp == "ms_t") {
  nd_temporal <- temporal_multiscale("time")
} else if (temp == "tvc") {
  nd_temporal <- temporal_tvc(time_var = "time", terms = 1)
}

denom_family <- switch(nd_family$denominator$distribution,
    gamma = "gamma", neg_binomial_2 = "negbin", fixed = "fixed")
if (zi == "zi" && fam %in% c("pg", "nb", "nbg")) {
  nd_family <- if (fam == "pg") ratiod_zi_poisson(denom_family = denom_family) else ratiod_zi_negbin(denom_family = denom_family)
} else if (zi == "zi" && fam == "bin") {
  nd_family <- ratiod_zi_binomial()
} else if (zi == "hurdle" && fam %in% c("pg", "nb", "nbg")) {
  nd_family <- if (fam == "pg") ratiod_hurdle_poisson(denom_family = denom_family) else ratiod_hurdle_negbin(denom_family = denom_family)
} else if (zi == "hurdle" && fam == "bin") {
  nd_family <- ratiod_hurdle_binomial()
} else if (zi == "oi") {
  nd_family <- ratiod_oi_binomial()
} else if (zi == "zoib") {
  nd_family <- ratiod_zoi_binomial()
}

nd_spatiotemporal <- NULL
if (st != "none") {
  nd_spatiotemporal <- spatiotemporal(
    spatial = nd_spatial, temporal = nd_temporal, type = st
  )
  # Keep nd_spatial and nd_temporal for main effects — tratio() will also

  # auto-extract them from spatiotemporal, but explicit is better
}

nd_latent <- NULL
if (cfg$latent) nd_latent <- latent_factor(n_factors = 2)

# --- Run benchmark ---
# Force HMC for collapsed spatial (auto mode routes ICAR/BYM2 to Gibbs)
nd_mode <- NULL
if (sp %in% c("cicar", "cbym2")) nd_mode <- "hmc"

ratiod_args <- list(
  formula          = form,
  data             = df,
  family           = nd_family,
  spatial          = nd_spatial,
  temporal         = nd_temporal,
  spatiotemporal   = nd_spatiotemporal,
  latent           = nd_latent,
  control          = list(iter = N_ITER,
  warmup           = N_WARMUP,
  chains           = N_CHAINS,
  verbose          = FALSE,
  gradient_mode    = GRAD_MODE)
)
if (!is.null(nd_mode)) ratiod_args$mode <- nd_mode

tryCatch({
  setTimeLimit(cpu = TIMEOUT_SEC, elapsed = TIMEOUT_SEC, transient = TRUE)
  on.exit(setTimeLimit(cpu = Inf, elapsed = Inf, transient = FALSE))
  elapsed <- system.time(do.call(tratio, ratiod_args))["elapsed"]
  cat(sprintf("RESULT:%d:%s:%.1f\n", ROW_NUM, GRAD_MODE, elapsed))
}, error = function(e) {
  msg <- conditionMessage(e)
  if (grepl("timeout|elapsed", msg, ignore.case = TRUE)) {
    cat(sprintf("RESULT:%d:%s:TIMEOUT\n", ROW_NUM, GRAD_MODE))
  } else {
    cat(sprintf("RESULT:%d:%s:ERROR:%s\n", ROW_NUM, GRAD_MODE, gsub("\n", " ", msg)))
  }
})
