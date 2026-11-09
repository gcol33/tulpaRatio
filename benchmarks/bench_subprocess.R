# =============================================================================
# Single-model subprocess worker for bench_fair.R
# Usage: Rscript benchmarks/bench_subprocess.R <model> <seed>
#
# Each model runs in its own R process to avoid thermal throttling.
# Standard params: N=500, N_s=50, N_t=20, iter=500, warmup=250, chains=1
# =============================================================================

library(tulpaRatio)
args <- commandArgs(trailingOnly = TRUE)
model <- args[1]
seed <- as.integer(args[2])
N <- 500L; N_s <- 50L; N_t <- 20L

set.seed(seed)
site <- factor(rep(1:N_s, length.out = N))
time_num <- rep(1:N_t, length.out = N)
x <- rnorm(N)

# Spatial grid and adjacency
n_side <- ceiling(sqrt(N_s))
grid <- expand.grid(lon = 1:n_side, lat = 1:n_side)[1:N_s, ]
adj_mat <- matrix(0L, N_s, N_s)
for (ii in 1:N_s) for (jj in 1:N_s) {
  if (ii != jj && sqrt((grid$lon[ii]-grid$lon[jj])^2+(grid$lat[ii]-grid$lat[jj])^2) <= 1.5)
    adj_mat[ii, jj] <- 1L
}

# Common data generators
make_pg_df <- function(...) {
  data.frame(y = rpois(N, exp(2 + 0.5*x)), denom = rgamma(N, 10, 1), x = x, ...)
}
make_nb_df <- function(...) {
  data.frame(y = rnbinom(N, mu = exp(2 + 0.3*x), size = 5),
             denom = pmax(rnbinom(N, mu = 100, size = 10), 1L), x = x, ...)
}
make_bin_df <- function(...) {
  trials <- sample(10:50, N, replace = TRUE)
  data.frame(y = rbinom(N, trials, plogis(0.5 + 0.3*x)), trials = trials, x = x, ...)
}
sp_car <- function() spatial_car(adj_mat, spatial_level = "group", group_var = "spatial_site")

COMMON <- list(control = list(iter = 500, warmup = 250, chains = 1, verbose = FALSE))

t <- system.time({
  fit <- switch(model,
    # === Poisson-Gamma ===
    PG_base = do.call(tratio, c(list(y | denom ~ x, data = make_pg_df(),
                     family = ratiod_poisson_gamma()), COMMON)),
    PG_RE = do.call(tratio, c(list(y | denom ~ x + (1|site),
                    data = make_pg_df(site = site),
                    family = ratiod_poisson_gamma()), COMMON)),
    PG_ICAR = do.call(tratio, c(list(y | denom ~ x + (1|site),
                      data = make_pg_df(site = site, spatial_site = site),
                      family = ratiod_poisson_gamma(), spatial = sp_car()), COMMON)),
    PG_BYM2 = {
      df <- make_pg_df(site = site, spatial_site = site)
      do.call(tratio, c(list(y | denom ~ x + (1|site), data = df,
              family = ratiod_poisson_gamma(),
              spatial = spatial_bym2(adj_mat, group_var = "spatial_site")), COMMON))
    },
    PG_GPt = do.call(tratio, c(list(y | denom ~ x,
                     data = make_pg_df(time_num = time_num),
                     family = ratiod_poisson_gamma(),
                     temporal = temporal_gp("time_num")), COMMON)),
    PG_ST_IV = {
      df <- make_pg_df(site = site, spatial_site = site, time_num = time_num)
      sp <- sp_car(); tp <- temporal_rw1("time_num")
      do.call(tratio, c(list(y | denom ~ x, data = df,
              family = ratiod_poisson_gamma(),
              spatial = sp, temporal = tp,
              spatiotemporal = spatiotemporal(spatial = sp, temporal = tp, type = "IV")), COMMON))
    },

    # === NegBin-NegBin ===
    NB_base = do.call(tratio, c(list(y | denom ~ x, data = make_nb_df(),
                     family = ratiod_negbin_negbin()), COMMON)),
    NB_RE = do.call(tratio, c(list(y | denom ~ x + (1|site),
                    data = make_nb_df(site = site),
                    family = ratiod_negbin_negbin()), COMMON)),
    NB_ICAR = do.call(tratio, c(list(y | denom ~ x + (1|site),
                      data = make_nb_df(site = site, spatial_site = site),
                      family = ratiod_negbin_negbin(), spatial = sp_car()), COMMON)),
    NB_HSGP = {
      coords <- grid[as.integer(site), ]
      df <- make_nb_df(site = site, lon = coords$lon, lat = coords$lat)
      do.call(tratio, c(list(y | denom ~ x + (1|site), data = df,
              family = ratiod_negbin_negbin(),
              spatial = spatial_hsgp(coords = c("lon", "lat"))), COMMON))
    },
    NB_BYM2 = {
      df <- make_nb_df(site = site, spatial_site = site)
      do.call(tratio, c(list(y | denom ~ x + (1|site), data = df,
              family = ratiod_negbin_negbin(),
              spatial = spatial_bym2(adj_mat, group_var = "spatial_site")), COMMON))
    },
    NB_GPt = do.call(tratio, c(list(y | denom ~ x,
                     data = make_nb_df(time_num = time_num),
                     family = ratiod_negbin_negbin(),
                     temporal = temporal_gp("time_num")), COMMON)),
    NB_ST_IV = {
      df <- make_nb_df(site = site, spatial_site = site, time_num = time_num)
      sp <- sp_car(); tp <- temporal_rw1("time_num")
      do.call(tratio, c(list(y | denom ~ x, data = df,
              family = ratiod_negbin_negbin(),
              spatial = sp, temporal = tp,
              spatiotemporal = spatiotemporal(spatial = sp, temporal = tp, type = "IV")), COMMON))
    },

    # === Binomial ===
    Bin_base = do.call(tratio, c(list(y | trials ~ x, data = make_bin_df(),
                      family = ratiod_binomial()), COMMON)),
    Bin_RE = do.call(tratio, c(list(y | trials ~ x + (1|site),
                     data = make_bin_df(site = site),
                     family = ratiod_binomial()), COMMON)),
    Bin_BYM2 = {
      df <- make_bin_df(site = site, spatial_site = site)
      do.call(tratio, c(list(y | trials ~ x + (1|site), data = df,
              family = ratiod_binomial(),
              spatial = spatial_bym2(adj_mat, group_var = "spatial_site")), COMMON))
    },
    Bin_GPt = do.call(tratio, c(list(y | trials ~ x,
                      data = make_bin_df(time_num = time_num),
                      family = ratiod_binomial(),
                      temporal = temporal_gp("time_num")), COMMON)),
    Bin_ST_IV = {
      df <- make_bin_df(site = site, spatial_site = site, time_num = time_num)
      sp <- sp_car(); tp <- temporal_rw1("time_num")
      do.call(tratio, c(list(y | trials ~ x, data = df,
              family = ratiod_binomial(),
              spatial = sp, temporal = tp,
              spatiotemporal = spatiotemporal(spatial = sp, temporal = tp, type = "IV")), COMMON))
    },

    stop("Unknown model: ", model)
  )
})[["elapsed"]]
cat(sprintf("%s seed=%d: %.1fs\n", model, seed, t))
