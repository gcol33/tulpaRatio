#' Parameter layout, unpacking and linear-predictor assembly for HMC fits
#'
#' @description
#' One walk of the sampler's parameter vector serves every consumer: the flat
#' posterior draws matrix, the per-structure draws the `svc()` / `tvc()` /
#' `temporal()` / `spatiotemporal_effects()` extractors read, the fitted values,
#' the predictions and the ratio draws.
#'
#' The block order mirrors `compute_param_layout()` in `src/hmc_sampler.cpp`,
#' which is the sampler's own layout, and `hmc_param_layout()` is the only place
#' an offset is computed. The eta assembly mirrors the observation loop of
#' `compute_log_post_impl()` in `src/log_post_impl.h`, including which
#' structures a `shared` flag lets reach the denominator.
#'
#' @name hmc_unpack
#' @keywords internal
NULL


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

#' Cursor over the parameter vector
#' @keywords internal
layout_cursor <- function() {
  pos <- 1L
  list(
    take = function(n) {
      n <- as.integer(n)
      if (is.na(n) || n <= 0L) return(integer(0))
      out <- seq.int(pos, length.out = n)
      pos <<- pos + n
      out
    },
    one = function() {
      out <- pos
      pos <<- pos + 1L
      out
    },
    total = function() pos - 1L
  )
}


#' Model types carrying a numerator dispersion parameter
#' @keywords internal
MODEL_TYPES_PHI_NUM <- c("negbin_negbin", "negbin_gamma", "gamma_gamma",
                         "lognormal", "beta_binomial")

#' Model types carrying a denominator dispersion parameter
#' @keywords internal
MODEL_TYPES_PHI_DENOM <- c("negbin_negbin", "poisson_gamma", "negbin_gamma",
                           "gamma_gamma", "lognormal")

#' Draw names for the dispersion parameters of each model type
#' @keywords internal
MODEL_DISPERSION_NAMES <- list(
  negbin_negbin = c(num = "phi_num", denom = "phi_denom"),
  negbin_gamma  = c(num = "phi_num", denom = "phi_denom"),
  poisson_gamma = c(denom = "shape"),
  gamma_gamma   = c(num = "shape_num", denom = "shape_denom"),
  lognormal     = c(num = "sigma_num", denom = "sigma_denom"),
  beta_binomial = c(num = "phi")
)

#' Temporal types the sampler allocates a temporal block for
#' @keywords internal
TEMPORAL_TYPES_SAMPLED <- c("rw1", "rw2", "ar1", "iid", "gp")


#' Single-component temporal types, the ones carrying one precision parameter
#'
#' @description
#' The structures `temporal_rw1()`, `temporal_rw2()`, `temporal_ar1()` and
#' `temporal_iid()` build. Each puts `log_tau` and one coefficient per time
#' point in the parameter vector, AR1 adding its correlation. The temporal GP
#' and the multi-scale temporal term have layouts of their own.
#'
#' @keywords internal
TEMPORAL_TYPES_SINGLE <- c("rw1", "rw2", "ar1", "iid")


#' Build the parameter layout of an HMC fit
#'
#' @param hmc_data Data list from `prepare_hmc_data()`
#' @param spatial_info Spatial info list
#' @param temporal_info Temporal info list
#' @param zi_info Zero-inflation info list
#' @param model_type Model type string
#' @param latent_info Latent factor info list
#' @param st_info Spatiotemporal info list
#' @param ms_temporal_info Multiscale temporal info list
#'
#' @return A list of index vectors, one entry per parameter block, plus `total`.
#' @keywords internal
hmc_param_layout <- function(hmc_data, spatial_info, temporal_info, zi_info,
                             model_type, latent_info = NULL, st_info = NULL,
                             ms_temporal_info = NULL) {
  cur <- layout_cursor()
  L <- list()

  # Fixed effects
  L$beta_num <- cur$take(hmc_data$p_num)
  L$beta_denom <- cur$take(hmc_data$p_denom)

  # Random effects
  L$re <- layout_re_block(cur, hmc_data)

  # Dispersion
  disp_names <- MODEL_DISPERSION_NAMES[[model_type]]
  L$phi_num <- if (model_type %in% MODEL_TYPES_PHI_NUM) cur$one() else NULL
  L$phi_denom <- if (model_type %in% MODEL_TYPES_PHI_DENOM) cur$one() else NULL
  L$dispersion_names <- disp_names

  # Areal spatial (ICAR / proper CAR / BYM2)
  L$spatial <- layout_areal_block(cur, spatial_info)

  # Temporal
  L$temporal <- layout_temporal_block(cur, temporal_info)

  # Zero- and one-inflation
  zi_type <- zi_info$type %||% "none"
  L$zi <- if (!identical(zi_type, "none")) cur$take(zi_info$p_zi %||% 0L) else integer(0)
  L$oi <- if (zi_type %in% c("oi_binomial", "zoib") && (zi_info$p_oi %||% 0L) > 0L) {
    cur$take(zi_info$p_oi)
  } else {
    integer(0)
  }

  # GP and multi-scale GP
  L$gp <- layout_gp_block(cur, spatial_info)
  L$msgp <- layout_msgp_block(cur, spatial_info)

  # Multi-scale temporal
  L$ms_temporal <- layout_ms_temporal_block(cur, ms_temporal_info)

  # Spatially-varying coefficients
  L$svc <- layout_svc_block(cur, spatial_info)

  # Latent factors
  L$latent <- layout_latent_block(cur, latent_info)

  # Spatiotemporal interaction
  L$st <- layout_st_block(cur, st_info)

  # HSGP
  L$hsgp <- layout_hsgp_block(cur, spatial_info)

  # Temporally-varying coefficients
  L$tvc <- layout_tvc_block(cur, temporal_info)

  L$total <- cur$total()
  L
}


#' @keywords internal
layout_re_block <- function(cur, hmc_data) {
  n_terms <- hmc_data$n_re_terms %||% 0L
  n_groups_single <- hmc_data$n_re_groups %||% 0L
  has_slopes <- isTRUE(hmc_data$has_slopes)

  if (n_terms > 0L && has_slopes) {
    terms <- vector("list", n_terms)
    for (t in seq_len(n_terms)) {
      term <- hmc_data$re_terms[[t]]
      terms[[t]] <- list(
        n_coefs = term$n_coefs,
        n_groups = term$n_groups,
        correlated = isTRUE(term$correlated),
        has_intercept = isTRUE(term$has_intercept),
        slope_names = term$slope_vars_clean %||% term$slope_vars,
        sigma = cur$take(term$n_coefs)
      )
    }
    for (t in seq_len(n_terms)) {
      n_chol <- hmc_data$re_terms[[t]]$n_chol %||% 0L
      terms[[t]]$chol <- if (n_chol > 0L && terms[[t]]$correlated) {
        cur$take(n_chol)
      } else {
        integer(0)
      }
    }
    for (t in seq_len(n_terms)) {
      terms[[t]]$effects <- cur$take(terms[[t]]$n_groups * terms[[t]]$n_coefs)
    }
    return(list(kind = "slopes", terms = terms))
  }

  if (n_terms > 1L) {
    terms <- vector("list", n_terms)
    for (t in seq_len(n_terms)) {
      terms[[t]] <- list(
        n_coefs = 1L,
        n_groups = hmc_data$re_terms[[t]]$n_groups,
        correlated = FALSE,
        has_intercept = TRUE,
        slope_names = character(0),
        sigma = cur$one(),
        chol = integer(0)
      )
    }
    for (t in seq_len(n_terms)) {
      terms[[t]]$effects <- cur$take(terms[[t]]$n_groups)
    }
    return(list(kind = "multi", terms = terms))
  }

  if (n_groups_single > 0L) {
    term <- list(
      n_coefs = 1L,
      n_groups = n_groups_single,
      correlated = FALSE,
      has_intercept = TRUE,
      slope_names = character(0),
      sigma = cur$one(),
      chol = integer(0)
    )
    term$effects <- cur$take(n_groups_single)
    return(list(kind = "single", terms = list(term)))
  }

  list(kind = "none", terms = list())
}


#' @keywords internal
layout_areal_block <- function(cur, spatial_info) {
  type <- spatial_info$type %||% "none"
  if (!type %in% c("icar", "car_proper", "bym2")) {
    return(list(type = "none"))
  }
  n_units <- spatial_info$n_units
  param <- spatial_info$parameterization %||% "standard"
  collapsed <- identical(param, "collapsed")

  if (identical(type, "bym2")) {
    out <- list(type = "bym2", collapsed = collapsed,
                log_sigma = cur$one(), logit_rho = cur$one())
    out$phi <- if (collapsed) integer(0) else cur$take(n_units)
    out$theta <- if (collapsed) integer(0) else cur$take(n_units)
    return(out)
  }

  out <- list(type = type, collapsed = collapsed, log_tau = cur$one())
  out$logit_rho <- if (identical(type, "car_proper")) cur$one() else integer(0)
  out$phi <- if (collapsed) integer(0) else cur$take(n_units)
  out
}


#' @keywords internal
layout_temporal_block <- function(cur, temporal_info) {
  type <- temporal_info$type %||% "none"
  if (!type %in% TEMPORAL_TYPES_SAMPLED) {
    return(list(type = "none"))
  }
  if (identical(type, "gp")) {
    out <- list(type = "gp", log_sigma2 = cur$one(), logit_phi = cur$one())
  } else {
    out <- list(type = type, log_tau = cur$one())
    out$logit_rho <- if (identical(type, "ar1")) cur$one() else integer(0)
  }
  out$effects <- cur$take(temporal_info$n_temporal_params %||% 0L)
  out
}


#' @keywords internal
layout_gp_block <- function(cur, spatial_info) {
  if (!identical(spatial_info$type %||% "none", "gp")) {
    return(list(type = "none"))
  }
  collapsed <- identical(spatial_info$parameterization %||% "centered", "collapsed")
  out <- list(type = "gp", collapsed = collapsed,
              log_sigma2 = cur$one(), log_phi = cur$one())
  out$w <- if (collapsed) integer(0) else cur$take(spatial_info$n_units)
  out
}


#' @keywords internal
layout_msgp_block <- function(cur, spatial_info) {
  if (!identical(spatial_info$type %||% "none", "multiscale_gp")) {
    return(list(type = "none"))
  }
  is_hsgp <- identical(spatial_info$msgp_approx %||% "nngp", "hsgp")
  n_per_scale <- if (is_hsgp) {
    (spatial_info$hsgp_m %||% 6L)^2
  } else {
    spatial_info$n_units
  }
  list(
    type = "multiscale_gp",
    is_hsgp = is_hsgp,
    local = list(log_sigma2 = cur$one(), log_phi = cur$one(),
                 w = cur$take(n_per_scale)),
    regional = list(log_sigma2 = cur$one(), log_phi = cur$one(),
                    w = cur$take(n_per_scale))
  )
}


#' @keywords internal
layout_ms_temporal_block <- function(cur, ms_temporal_info) {
  if (is.null(ms_temporal_info) ||
      !identical(ms_temporal_info$ms_temporal_type %||% "none", "multiscale")) {
    return(list(type = "none"))
  }
  n_times <- ms_temporal_info$ms_n_times %||% 0L
  out <- list(type = "multiscale")

  trend_type <- ms_temporal_info$trend_type %||% "none"
  if (!identical(trend_type, "none")) {
    out$trend <- list(log_sigma2 = cur$one(), effects = cur$take(n_times),
                      structure = trend_type)
  }

  period <- ms_temporal_info$seasonal_period %||% 0L
  if (period > 0L) {
    out$seasonal <- list(log_sigma2 = cur$one(), effects = cur$take(period))
  }

  short_type <- ms_temporal_info$short_term_type %||% "none"
  if (!identical(short_type, "none")) {
    short <- list(log_sigma2 = cur$one(), structure = short_type)
    short$logit_rho <- if (identical(short_type, "ar1")) cur$one() else integer(0)
    short$effects <- cur$take(n_times)
    out$short <- short
  }

  out
}


#' @keywords internal
layout_svc_block <- function(cur, spatial_info) {
  n_svc <- spatial_info$n_svc %||% 0L
  if (n_svc <= 0L) return(list(type = "none"))
  is_hsgp <- identical(spatial_info$svc_approx %||% "nngp", "hsgp")
  n_per_term <- if (is_hsgp) {
    as.integer(spatial_info$svc_hsgp_m)^2
  } else {
    spatial_info$n_units
  }
  list(
    type = "svc",
    is_hsgp = is_hsgp,
    n_svc = n_svc,
    n_per_term = n_per_term,
    names = spatial_info$svc_names,
    log_sigma2 = cur$take(n_svc),
    log_phi = cur$take(n_svc),
    w = cur$take(n_svc * n_per_term)
  )
}


#' @keywords internal
layout_latent_block <- function(cur, latent_info) {
  n_factors <- latent_info$n_factors %||% 0L
  if (is.null(latent_info) || identical(latent_info$type %||% "none", "none") ||
      n_factors <= 0L) {
    return(list(type = "none"))
  }
  n_obs <- latent_info$n_obs
  list(
    type = "latent",
    n_factors = n_factors,
    n_obs = n_obs,
    log_sigma = cur$take(n_factors),
    factors = cur$take(n_obs * n_factors)
  )
}


#' @keywords internal
layout_st_block <- function(cur, st_info) {
  if (is.null(st_info) || !isTRUE(st_info$has_spatiotemporal) ||
      identical(st_info$type %||% "none", "none")) {
    return(list(type = "none"))
  }
  is_gp <- st_info$type %in% c("separable", "nonsep_gp")
  is_hsgp <- isTRUE(st_info$spatial_is_hsgp)

  out <- list(type = st_info$type, is_gp = is_gp, is_hsgp = is_hsgp,
              log_tau = cur$one())
  out$logit_rho <- if (st_has_rho(st_info)) cur$one() else integer(0)
  if (is_gp) {
    out$log_phi_space <- cur$one()
    out$log_phi_time <- cur$one()
  }
  if (is_hsgp) {
    out$log_sigma2_hsgp <- cur$one()
    out$log_lengthscale_hsgp <- cur$one()
  }
  out$delta <- cur$take(st_info$n_params)
  out
}


#' @keywords internal
layout_hsgp_block <- function(cur, spatial_info) {
  if (!identical(spatial_info$type %||% "none", "hsgp")) {
    return(list(type = "none"))
  }
  m <- as.integer(spatial_info$hsgp_m %||% 8L)
  list(
    type = "hsgp",
    m = m,
    log_sigma2 = cur$one(),
    log_lengthscale = cur$one(),
    beta = cur$take(m * m)
  )
}


#' @keywords internal
layout_tvc_block <- function(cur, temporal_info) {
  if (!identical(temporal_info$type %||% "none", "tvc")) {
    return(list(type = "none"))
  }
  n_tvc <- temporal_info$n_tvc %||% 0L
  if (n_tvc <= 0L) return(list(type = "none"))
  n_times <- temporal_info$n_times %||% 0L
  n_groups <- temporal_info$n_groups %||% 1L
  structure <- temporal_info$structure %||% "rw1"

  out <- list(type = "tvc", n_tvc = n_tvc, n_times = n_times,
              n_groups = n_groups, structure = structure,
              names = temporal_info$tvc_names,
              log_tau = cur$take(n_tvc))
  out$logit_rho <- if (identical(structure, "ar1")) cur$take(n_tvc) else integer(0)
  out$w <- cur$take(n_groups * n_tvc * n_times)
  out
}


#' Layout of an HMC fit, from the fit object
#' @keywords internal
hmc_fit_layout <- function(object) {
  layout <- object$.internal$layout
  if (is.null(layout)) {
    stop("This fit carries no parameter layout. Refit with the current ",
         "version of the package.", call. = FALSE)
  }
  layout
}


# ---------------------------------------------------------------------------
# Unpacking: parameter blocks to natural-scale draws
# ---------------------------------------------------------------------------

#' Read a parameter block as a draws matrix
#' @keywords internal
block_draws <- function(samples, idx) {
  if (length(idx) == 0L) return(NULL)
  samples[, idx, drop = FALSE]
}

#' Inverse logit
#' @keywords internal
inv_logit <- function(x) 1 / (1 + exp(-x))


#' Unpack the sampler's parameter vector into per-structure draws
#'
#' @param samples Sample matrix, draws in rows
#' @param layout Layout from `hmc_param_layout()`
#' @param hmc_data,spatial_info,temporal_info Model info lists
#' @param model_type Model type string
#' @param re_param Random-effect parameterization
#' @param latent_info,st_info,ms_temporal_info Optional info lists
#' @param gp_w_star,icar_phi_star,bym2_theta_star Fields from the inner Laplace
#'   optimization of a collapsed parameterization
#'
#' @return A list with one entry per structure, each holding the draws on the
#'   natural scale.
#' @keywords internal
hmc_unpack_draws <- function(samples, layout, hmc_data, spatial_info,
                             temporal_info, model_type,
                             re_param = "noncentered",
                             latent_info = NULL, st_info = NULL,
                             ms_temporal_info = NULL,
                             gp_w_star = NULL, icar_phi_star = NULL,
                             bym2_theta_star = NULL) {
  n_samples <- nrow(samples)
  out <- list(n_samples = n_samples)

  out$beta_num <- block_draws(samples, layout$beta_num)
  out$beta_denom <- block_draws(samples, layout$beta_denom)

  out$re <- unpack_re(samples, layout$re, re_param)

  out$dispersion <- list()
  disp_names <- layout$dispersion_names
  if (!is.null(layout$phi_num)) {
    out$dispersion[[disp_names[["num"]]]] <- exp(samples[, layout$phi_num])
  }
  if (!is.null(layout$phi_denom)) {
    out$dispersion[[disp_names[["denom"]]]] <- exp(samples[, layout$phi_denom])
  }

  out$spatial <- unpack_areal(samples, layout$spatial, spatial_info,
                              icar_phi_star, bym2_theta_star)
  out$temporal <- unpack_temporal(samples, layout$temporal, temporal_info)
  out$zi <- block_draws(samples, layout$zi)
  out$oi <- block_draws(samples, layout$oi)
  out$gp <- unpack_gp(samples, layout$gp, gp_w_star)
  out$msgp <- unpack_msgp(samples, layout$msgp, spatial_info)
  out$ms_temporal <- unpack_ms_temporal(samples, layout$ms_temporal)
  out$svc <- unpack_svc(samples, layout$svc, spatial_info)
  out$latent <- unpack_latent(samples, layout$latent, latent_info)
  out$st <- unpack_st(samples, layout$st)
  out$hsgp <- unpack_hsgp(samples, layout$hsgp, spatial_info)
  out$tvc <- unpack_tvc(samples, layout$tvc)

  out
}


#' @keywords internal
unpack_re <- function(samples, lay, re_param) {
  if (identical(lay$kind, "none")) return(NULL)
  noncentered <- identical(re_param, "noncentered")
  n_samples <- nrow(samples)

  terms <- lapply(lay$terms, function(term) {
    sigma <- lapply(term$sigma, function(i) exp(samples[, i]))
    chol_raw <- lapply(term$chol, function(i) samples[, i])
    n_coefs <- term$n_coefs
    n_groups <- term$n_groups

    # Sampler layout within a term: group-major, [coef_1, ..., coef_k] per group
    coefs <- vector("list", n_coefs)
    for (c in seq_len(n_coefs)) {
      cols <- term$effects[seq(c, by = n_coefs, length.out = n_groups)]
      coefs[[c]] <- samples[, cols, drop = FALSE]
    }

    if (noncentered) {
      if (n_coefs == 2L && term$correlated && length(chol_raw) >= 1L) {
        # re = diag(sigma) L z, L = [[1, 0], [L21, sqrt(1 - L21^2)]]
        L21 <- tanh(chol_raw[[1]])
        L22 <- sqrt(pmax(0, 1 - L21^2))
        z1 <- coefs[[1]]
        z2 <- coefs[[2]]
        coefs[[1]] <- z1 * sigma[[1]]
        coefs[[2]] <- (z1 * L21 + z2 * L22) * sigma[[2]]
      } else {
        for (c in seq_len(n_coefs)) {
          coefs[[c]] <- coefs[[c]] * sigma[[c]]
        }
      }
    }

    list(
      sigma = sigma,
      chol = chol_raw,
      coefs = coefs,
      n_coefs = n_coefs,
      n_groups = n_groups,
      correlated = term$correlated,
      has_intercept = term$has_intercept,
      slope_names = term$slope_names
    )
  })

  list(kind = lay$kind, terms = terms, n_samples = n_samples)
}


#' @keywords internal
unpack_areal <- function(samples, lay, spatial_info, icar_phi_star = NULL,
                         bym2_theta_star = NULL) {
  if (identical(lay$type, "none")) return(NULL)
  out <- list(type = lay$type, collapsed = lay$collapsed)

  if (identical(lay$type, "bym2")) {
    sigma_total <- exp(samples[, lay$log_sigma])
    rho <- inv_logit(samples[, lay$logit_rho])
    out$sigma_total <- sigma_total
    out$rho <- rho
    out$sigma_s <- sigma_total * sqrt(rho)
    out$sigma_u <- sigma_total * sqrt(1 - rho)
    scale <- spatial_info$bym2_scale %||% 1.0
    if (lay$collapsed) {
      out$phi_scaled <- icar_phi_star
      out$theta <- bym2_theta_star
    } else {
      out$phi_scaled <- block_draws(samples, lay$phi)
      out$theta <- block_draws(samples, lay$theta)
    }
    out$field <- if (!is.null(out$phi_scaled) && !is.null(out$theta)) {
      out$sigma_s * out$phi_scaled * scale + out$sigma_u * out$theta
    } else {
      NULL
    }
    return(out)
  }

  out$tau <- exp(samples[, lay$log_tau])
  if (length(lay$logit_rho) > 0L) {
    rho_lower <- spatial_info$rho_lower %||% 0.0
    rho_upper <- spatial_info$rho_upper %||% 1.0
    out$rho <- rho_lower + (rho_upper - rho_lower) * inv_logit(samples[, lay$logit_rho])
  }
  out$field <- if (lay$collapsed) icar_phi_star else block_draws(samples, lay$phi)
  out
}


#' @keywords internal
unpack_temporal <- function(samples, lay, temporal_info) {
  if (identical(lay$type, "none")) return(NULL)
  out <- list(type = lay$type)

  if (identical(lay$type, "gp")) {
    out$sigma2 <- exp(samples[, lay$log_sigma2])
    phi_lower <- temporal_info$phi_prior_lower %||% 0.01
    phi_upper <- temporal_info$phi_prior_upper %||% 10.0
    out$phi <- phi_lower + (phi_upper - phi_lower) * inv_logit(samples[, lay$logit_phi])
    z <- block_draws(samples, lay$effects)
    param <- temporal_info$parameterization %||% "noncentered"
    out$field <- if (identical(param, "noncentered")) {
      temporal_gp_forward(z, out$sigma2, out$phi, temporal_info)
    } else {
      z
    }
    return(out)
  }

  out$tau <- exp(samples[, lay$log_tau])
  if (length(lay$logit_rho) > 0L) {
    # rho on (-1, 1), the mapping every AR1 correlation in the package uses.
    out$rho <- 2 * inv_logit(samples[, lay$logit_rho]) - 1
  }
  out$field <- block_draws(samples, lay$effects)
  out
}


#' Forward transform of the non-centred temporal GP
#'
#' `f[1] = sigma z[1]`, `f[t] = rho_t f[t-1] + sigma sqrt(1 - rho_t^2) z[t]`
#' with `rho_t = exp(-dt / phi)`, matching the state-space form the sampler uses.
#'
#' @keywords internal
temporal_gp_forward <- function(z, sigma2, phi, temporal_info) {
  n_samples <- nrow(z)
  T_times <- temporal_info$n_times
  n_groups <- temporal_info$n_groups
  time_vals <- temporal_info$time_values
  sigma <- sqrt(sigma2)

  f <- z
  for (g in seq_len(n_groups)) {
    off <- (g - 1L) * T_times
    f[, off + 1L] <- sigma * z[, off + 1L]
    if (T_times < 2L) next
    for (tt in seq.int(2L, T_times)) {
      dt <- time_vals[tt] - time_vals[tt - 1L]
      rho_t <- exp(-dt / phi)
      a_t <- sigma * sqrt(pmax(1 - rho_t^2, 1e-10))
      f[, off + tt] <- rho_t * f[, off + tt - 1L] + a_t * z[, off + tt]
    }
  }
  f
}


#' @keywords internal
unpack_gp <- function(samples, lay, gp_w_star = NULL) {
  if (identical(lay$type, "none")) return(NULL)
  list(
    type = "gp",
    collapsed = lay$collapsed,
    sigma2 = exp(samples[, lay$log_sigma2]),
    phi = exp(samples[, lay$log_phi]),
    field = if (lay$collapsed) gp_w_star else block_draws(samples, lay$w)
  )
}


#' @keywords internal
unpack_msgp <- function(samples, lay, spatial_info) {
  if (identical(lay$type, "none")) return(NULL)
  out <- list(type = "multiscale_gp", is_hsgp = lay$is_hsgp)
  out$sigma2_local <- exp(samples[, lay$local$log_sigma2])
  out$phi_local <- exp(samples[, lay$local$log_phi])
  out$w_local <- block_draws(samples, lay$local$w)
  out$sigma2_regional <- exp(samples[, lay$regional$log_sigma2])
  out$phi_regional <- exp(samples[, lay$regional$log_phi])
  out$w_regional <- block_draws(samples, lay$regional$w)

  if (lay$is_hsgp) {
    Phi <- spatial_info$msgp_hsgp_Phi
    eigenvalues <- spatial_info$msgp_hsgp_eigenvalues
    if (is.null(Phi) || is.null(eigenvalues)) {
      out$field <- NULL
      out$field_reason <- "hsgp_basis_unavailable"
    } else {
      out$field <-
        hsgp_field(out$w_local, out$sigma2_local, out$phi_local, Phi, eigenvalues) +
        hsgp_field(out$w_regional, out$sigma2_regional, out$phi_regional, Phi, eigenvalues)
    }
  } else {
    out$field <- out$w_local + out$w_regional
  }
  out
}


#' Reconstruct an HSGP field from its basis coefficients
#'
#' `f = Phi (sqrt(S) * beta)` with the squared-exponential spectral density
#' `S(omega) = sigma2 sqrt(2 pi) ell exp(-ell^2 omega^2 / 2)`, matching
#' `hsgp_evaluate()` in `src/hmc_hsgp.h`.
#'
#' @keywords internal
hsgp_field <- function(beta, sigma2, lengthscale, Phi, eigenvalues) {
  n_samples <- nrow(beta)
  out <- matrix(0, n_samples, nrow(Phi))
  for (s in seq_len(n_samples)) {
    S_k <- sigma2[s] * sqrt(2 * pi) * lengthscale[s] *
      exp(-0.5 * lengthscale[s]^2 * eigenvalues)
    out[s, ] <- as.numeric(Phi %*% (sqrt(pmax(S_k, 0)) * beta[s, ]))
  }
  out
}


#' @keywords internal
unpack_hsgp <- function(samples, lay, spatial_info) {
  if (identical(lay$type, "none")) return(NULL)
  out <- list(
    type = "hsgp",
    sigma2 = exp(samples[, lay$log_sigma2]),
    lengthscale = exp(samples[, lay$log_lengthscale]),
    beta = block_draws(samples, lay$beta)
  )
  Phi <- spatial_info$hsgp_Phi
  eigenvalues <- spatial_info$hsgp_eigenvalues
  if (is.null(Phi) || is.null(eigenvalues)) {
    out$field <- NULL
    out$field_reason <- "hsgp_basis_unavailable"
  } else {
    out$field <- hsgp_field(out$beta, out$sigma2, out$lengthscale, Phi, eigenvalues)
  }
  out
}


#' @keywords internal
unpack_ms_temporal <- function(samples, lay) {
  if (identical(lay$type, "none")) return(NULL)
  out <- list(type = "multiscale")
  for (comp in c("trend", "seasonal", "short")) {
    sub <- lay[[comp]]
    if (is.null(sub)) next
    entry <- list(
      sigma2 = exp(samples[, sub$log_sigma2]),
      field = block_draws(samples, sub$effects)
    )
    if (!is.null(sub$logit_rho) && length(sub$logit_rho) > 0L) {
      entry$rho <- 2 * inv_logit(samples[, sub$logit_rho]) - 1
    }
    out[[comp]] <- entry
  }
  out
}


#' @keywords internal
unpack_svc <- function(samples, lay, spatial_info) {
  if (identical(lay$type, "none")) return(NULL)
  n_svc <- lay$n_svc
  n_per_term <- lay$n_per_term
  out <- list(
    type = "svc",
    is_hsgp = lay$is_hsgp,
    n_svc = n_svc,
    names = lay$names,
    sigma2 = block_draws(samples, lay$log_sigma2),
    phi = block_draws(samples, lay$log_phi),
    w = block_draws(samples, lay$w)
  )
  out$sigma2 <- exp(out$sigma2)
  out$phi <- exp(out$phi)

  # Per-term field at the observation locations
  n_samples <- nrow(samples)
  N <- spatial_info$n_units
  fields <- vector("list", n_svc)
  if (lay$is_hsgp) {
    basis <- svc_hsgp_basis(spatial_info)
    for (j in seq_len(n_svc)) {
      beta_j <- out$w[, ((j - 1) * n_per_term + 1):(j * n_per_term), drop = FALSE]
      fields[[j]] <- if (is.null(basis)) {
        NULL
      } else {
        hsgp_field(beta_j, out$sigma2[, j], out$phi[, j], basis$phi, basis$eigenvalues)
      }
    }
  } else {
    for (j in seq_len(n_svc)) {
      fields[[j]] <- out$w[, ((j - 1) * n_per_term + 1):(j * n_per_term), drop = FALSE]
    }
  }
  out$fields <- fields
  out
}


#' Basis for an HSGP-approximated SVC term
#' @keywords internal
svc_hsgp_basis <- function(spatial_info) {
  coords <- spatial_info$svc_coords
  if (is.null(coords)) return(NULL)
  m <- as.integer(spatial_info$svc_hsgp_m)
  c_val <- spatial_info$svc_hsgp_c %||% 1.5
  hsgp_basis_2d(coords, m, c_val)
}


#' Laplacian eigenbasis on a rectangle
#'
#' Mirrors `setup_hsgp_data()` in `src/hmc_hsgp.h`: boundary `L = c * range / 2`
#' with a 0.1 floor, coordinates centred, `phi_j(x) = sin(pi j (x + L) / (2 L)) / sqrt(L)`
#' and the basis indexed `(j1 - 1) * m + j2`.
#'
#' @keywords internal
hsgp_basis_2d <- function(coords, m, c_boundary) {
  coords <- as.matrix(coords)
  x <- coords[, 1]
  y <- coords[, 2]
  L1 <- max(c_boundary * (max(x) - min(x)) / 2, 0.1)
  L2 <- max(c_boundary * (max(y) - min(y)) / 2, 0.1)
  x_c <- x - (max(x) + min(x)) / 2
  y_c <- y - (max(y) + min(y)) / 2

  m_total <- m * m
  eigenvalues <- numeric(m_total)
  phi <- matrix(0, length(x), m_total)
  for (j1 in seq_len(m)) {
    phi_x <- sin(pi * j1 * (x_c + L1) / (2 * L1)) / sqrt(L1)
    for (j2 in seq_len(m)) {
      k <- (j1 - 1L) * m + j2
      eigenvalues[k] <- (pi * j1 / (2 * L1))^2 + (pi * j2 / (2 * L2))^2
      phi[, k] <- phi_x * sin(pi * j2 * (y_c + L2) / (2 * L2)) / sqrt(L2)
    }
  }
  list(phi = phi, eigenvalues = eigenvalues, m_total = m_total)
}


#' @keywords internal
unpack_latent <- function(samples, lay, latent_info) {
  if (identical(lay$type, "none")) return(NULL)
  K <- lay$n_factors
  N <- lay$n_obs
  sigma <- exp(samples[, lay$log_sigma, drop = FALSE])
  factors <- block_draws(samples, lay$factors)

  # Identifiability constraint, applied as the sampler applies it
  constraint <- latent_info$constraint %||% "sum_to_zero"
  n_samples <- nrow(samples)
  eta <- matrix(0, n_samples, N)
  factor_list <- vector("list", K)
  for (k in seq_len(K)) {
    # factors are stored observation-major: f[i, k] at (i - 1) * K + k
    cols <- seq(k, by = K, length.out = N)
    f_k <- factors[, cols, drop = FALSE]
    f_k <- if (identical(constraint, "first_zero")) {
      f_k - f_k[, 1]
    } else {
      f_k - rowMeans(f_k)
    }
    factor_list[[k]] <- f_k
    eta <- eta + f_k * sigma[, k]
  }

  list(type = "latent", sigma = sigma, factors = factor_list, field = eta)
}


#' @keywords internal
unpack_st <- function(samples, lay) {
  if (identical(lay$type, "none")) return(NULL)
  out <- list(type = lay$type, is_hsgp = lay$is_hsgp)
  out$tau <- exp(samples[, lay$log_tau])
  if (length(lay$logit_rho) > 0L) {
    out$rho <- 2 * inv_logit(samples[, lay$logit_rho]) - 1
  }
  if (isTRUE(lay$is_gp)) {
    out$phi_space <- exp(samples[, lay$log_phi_space])
    out$phi_time <- exp(samples[, lay$log_phi_time])
  }
  if (isTRUE(lay$is_hsgp)) {
    out$sigma2_hsgp <- exp(samples[, lay$log_sigma2_hsgp])
    out$lengthscale_hsgp <- exp(samples[, lay$log_lengthscale_hsgp])
  }
  out$field <- block_draws(samples, lay$delta)
  out
}


#' @keywords internal
unpack_tvc <- function(samples, lay) {
  if (identical(lay$type, "none")) return(NULL)
  out <- list(
    type = "tvc",
    n_tvc = lay$n_tvc,
    n_times = lay$n_times,
    n_groups = lay$n_groups,
    names = lay$names,
    tau = exp(samples[, lay$log_tau, drop = FALSE]),
    field = block_draws(samples, lay$w)
  )
  if (length(lay$logit_rho) > 0L) {
    out$rho <- 2 * inv_logit(samples[, lay$logit_rho, drop = FALSE]) - 1
  }
  out
}


# ---------------------------------------------------------------------------
# Linear predictor
# ---------------------------------------------------------------------------

#' Design a linear predictor is assembled against
#'
#' @param X_num,X_denom Design matrices
#' @param re_group Group index per observation for a single RE term
#' @param re_group_matrix Observation-by-term group index matrix
#' @param slope_matrices Per-term slope covariate matrices
#' @param include_re Whether random effects enter the predictor
#' @param structures Whether the structured blocks enter the predictor
#' @param spatial_info,temporal_info,latent_info,st_info Info lists supplying
#'   the observation-to-unit maps
#'
#' @keywords internal
hmc_eta_design <- function(X_num, X_denom, re_group = NULL,
                           re_group_matrix = NULL, slope_matrices = NULL,
                           include_re = TRUE, structures = TRUE,
                           spatial_info = NULL, temporal_info = NULL,
                           latent_info = NULL, st_info = NULL,
                           ms_temporal_info = NULL) {
  list(
    X_num = X_num,
    X_denom = X_denom,
    N = nrow(X_num),
    re_group = re_group,
    re_group_matrix = re_group_matrix,
    slope_matrices = slope_matrices,
    include_re = include_re,
    structures = structures,
    spatial_info = spatial_info,
    temporal_info = temporal_info,
    latent_info = latent_info,
    st_info = st_info,
    ms_temporal_info = ms_temporal_info
  )
}


#' Unpack the posterior draws of a fitted object
#'
#' @param object A `ratiod_fit` from the HMC backend
#' @return Output of `hmc_unpack_draws()`
#' @keywords internal
hmc_fit_unpack <- function(object) {
  internal <- object$.internal
  hmc_unpack_draws(
    samples = internal$samples,
    layout = hmc_fit_layout(object),
    hmc_data = internal$hmc_data,
    spatial_info = object$spatial,
    temporal_info = object$temporal,
    model_type = internal$model_type,
    re_param = object$re_param %||% "noncentered",
    latent_info = internal$latent_info,
    st_info = object$spatiotemporal,
    ms_temporal_info = internal$ms_temporal_info,
    gp_w_star = internal$gp_w_star,
    icar_phi_star = internal$icar_phi_star,
    bym2_theta_star = internal$bym2_theta_star
  )
}


#' Design of the data a fit was made on
#' @keywords internal
hmc_fit_design <- function(object) {
  hmc_data <- object$.internal$hmc_data
  hmc_eta_design(
    X_num = hmc_data$X_num,
    X_denom = hmc_data$X_denom,
    re_group = hmc_data$re_group,
    re_group_matrix = hmc_data$re_group_matrix,
    slope_matrices = hmc_data$slope_matrices,
    include_re = TRUE,
    structures = TRUE,
    spatial_info = object$spatial,
    temporal_info = object$temporal,
    latent_info = object$.internal$latent_info,
    st_info = object$spatiotemporal,
    ms_temporal_info = object$.internal$ms_temporal_info
  )
}


#' Spread a per-unit effect over observations
#'
#' `index` is 1-based with 0 meaning "no unit"; those observations get zero.
#' @keywords internal
spread_effect <- function(effect, index, N) {
  if (is.null(effect)) return(NULL)
  idx <- as.integer(index)
  idx[is.na(idx) | idx < 0L] <- 0L
  if (max(idx) > ncol(effect)) {
    stop("Effect index ", max(idx), " exceeds the ", ncol(effect),
         " units the block holds.", call. = FALSE)
  }
  padded <- cbind(0, effect)
  padded[, idx + 1L, drop = FALSE]
}


#' Random-effect contribution to the linear predictor
#' @keywords internal
hmc_re_eta <- function(re, design) {
  if (is.null(re) || !isTRUE(design$include_re)) return(NULL)
  N <- design$N
  n_samples <- re$n_samples
  eta <- matrix(0, n_samples, N)

  group_matrix <- design$re_group_matrix
  for (t in seq_along(re$terms)) {
    term <- re$terms[[t]]
    g <- if (!is.null(group_matrix) && ncol(group_matrix) >= t) {
      group_matrix[, t]
    } else {
      design$re_group
    }
    if (is.null(g)) next

    coef_idx <- 1L
    if (term$has_intercept) {
      eta <- eta + spread_effect(term$coefs[[1]], g, N)
      coef_idx <- 2L
    }
    n_slopes <- term$n_coefs - coef_idx + 1L
    if (n_slopes > 0L) {
      Z <- design$slope_matrices[[t]]
      if (!is.null(Z)) {
        for (s in seq_len(n_slopes)) {
          x_s <- matrix(Z[, s], n_samples, N, byrow = TRUE)
          eta <- eta + spread_effect(term$coefs[[coef_idx + s - 1L]], g, N) * x_s
        }
      }
    }
  }
  eta
}


#' Structured contributions to the linear predictor
#'
#' Returns a list of `list(name, eta, shared)` entries, mirroring the
#' observation loop of `compute_log_post_impl()`. A structure whose field could
#' not be reconstructed contributes an entry with a `reason` and no `eta`.
#'
#' @keywords internal
hmc_structure_contributions <- function(unpacked, design) {
  if (!isTRUE(design$structures)) return(list())
  N <- design$N
  spatial_info <- design$spatial_info
  temporal_info <- design$temporal_info
  contribs <- list()

  add <- function(name, eta, shared, reason = NULL) {
    contribs[[length(contribs) + 1L]] <<- list(
      name = name, eta = eta, shared = shared, reason = reason
    )
  }

  # A restricted spatial field reaches the predictors projected onto the
  # complement of the restricted covariates, as the sampler reads it.
  Q <- spatial_info$rsr_basis
  restrict <- function(eta) {
    if (is.null(Q) || is.null(eta)) return(eta)
    eta - (eta %*% Q) %*% t(Q)
  }

  # Areal spatial: always shared between the two linear predictors
  if (!is.null(unpacked$spatial)) {
    field <- unpacked$spatial$field
    add("spatial", restrict(spread_effect(field, spatial_info$group, N)), TRUE,
        if (is.null(field)) "collapsed_field_unavailable" else NULL)
  }

  # GP
  if (!is.null(unpacked$gp)) {
    field <- unpacked$gp$field
    add("gp", restrict(spread_effect(field, spatial_info$group, N)),
        spatial_info$shared %||% TRUE,
        if (is.null(field)) "collapsed_field_unavailable" else NULL)
  }

  # HSGP: the basis is evaluated at the observations
  if (!is.null(unpacked$hsgp)) {
    add("hsgp", restrict(unpacked$hsgp$field), spatial_info$shared %||% TRUE,
        unpacked$hsgp$field_reason)
  }

  # Multi-scale GP
  if (!is.null(unpacked$msgp)) {
    field <- unpacked$msgp$field
    eta <- if (isTRUE(unpacked$msgp$is_hsgp)) {
      field
    } else {
      spread_effect(field, spatial_info$group, N)
    }
    add("msgp", restrict(eta), spatial_info$shared %||% TRUE,
        unpacked$msgp$field_reason)
  }

  # Temporal
  if (!is.null(unpacked$temporal)) {
    idx <- temporal_param_index(temporal_info)
    add("temporal", spread_effect(unpacked$temporal$field, idx, N),
        temporal_info$shared %||% TRUE)
  }

  # Multi-scale temporal
  if (!is.null(unpacked$ms_temporal)) {
    ms <- unpacked$ms_temporal
    info <- design$ms_temporal_info %||% temporal_info
    time_idx <- info$time_index %||% info$ms_time_index
    eta <- NULL
    if (!is.null(time_idx)) {
      eta <- matrix(0, unpacked$n_samples, N)
      if (!is.null(ms$trend)) {
        eta <- eta + spread_effect(ms$trend$field, time_idx, N)
      }
      if (!is.null(ms$short)) {
        eta <- eta + spread_effect(ms$short$field, time_idx, N)
      }
      if (!is.null(ms$seasonal)) {
        period <- ncol(ms$seasonal$field)
        season_idx <- ((as.integer(time_idx) - 1L) %% period) + 1L
        season_idx[as.integer(time_idx) <= 0L] <- 0L
        eta <- eta + spread_effect(ms$seasonal$field, season_idx, N)
      }
    }
    add("ms_temporal", eta, info$shared %||% TRUE,
        if (is.null(eta)) "time_index_unavailable" else NULL)
  }

  # Spatially-varying coefficients
  if (!is.null(unpacked$svc)) {
    X_svc <- spatial_info$X_svc
    eta <- matrix(0, unpacked$n_samples, N)
    reason <- NULL
    for (j in seq_len(unpacked$svc$n_svc)) {
      f_j <- unpacked$svc$fields[[j]]
      if (is.null(f_j)) {
        reason <- "hsgp_basis_unavailable"
        eta <- NULL
        break
      }
      x_j <- if (!is.null(X_svc)) X_svc[, j] else rep(1, N)
      eta <- eta + f_j * matrix(x_j, unpacked$n_samples, N, byrow = TRUE)
    }
    add("svc", eta, spatial_info$svc_shared %||% TRUE, reason)
  }

  # Temporally-varying coefficients
  if (!is.null(unpacked$tvc)) {
    eta <- tvc_eta(unpacked$tvc, temporal_info, design)
    add("tvc", eta, temporal_info$shared %||% TRUE,
        if (is.null(eta)) "tvc_design_unavailable" else NULL)
  }

  # Latent factors
  if (!is.null(unpacked$latent)) {
    add("latent", unpacked$latent$field, design$latent_info$shared %||% FALSE)
  }

  # Spatiotemporal interaction
  if (!is.null(unpacked$st)) {
    eta <- st_eta(unpacked$st, design$st_info, design)
    add("spatiotemporal", eta, design$st_info$shared %||% TRUE,
        if (is.null(eta)) "st_index_unavailable" else NULL)
  }

  contribs
}


#' Flat temporal parameter index per observation
#' @keywords internal
temporal_param_index <- function(temporal_info) {
  t_idx <- as.integer(temporal_info$time_index)
  g_idx <- as.integer(temporal_info$group_index)
  n_groups <- temporal_info$n_groups %||% 1L
  n_times <- temporal_info$n_times %||% 0L
  if (n_groups > 1L) {
    idx <- (g_idx - 1L) * n_times + t_idx
    idx[t_idx <= 0L] <- 0L
    idx
  } else {
    t_idx
  }
}


#' TVC contribution to the linear predictor
#'
#' The sampler stores `w[g, j, t]` at `g * n_tvc * n_times + j * n_times + t`.
#' @keywords internal
tvc_eta <- function(tvc, temporal_info, design) {
  X <- design$X_num
  idx_cols <- temporal_info$tvc_indices
  time_idx <- as.integer(temporal_info$time_index)
  group_idx <- as.integer(temporal_info$group_index %||% rep(1L, design$N))
  if (is.null(idx_cols) || is.null(time_idx)) return(NULL)

  N <- design$N
  n_samples <- nrow(tvc$field)
  n_tvc <- tvc$n_tvc
  n_times <- tvc$n_times

  eta <- matrix(0, n_samples, N)
  for (j in seq_len(n_tvc)) {
    x_j <- X[, idx_cols[j]]
    flat <- (group_idx - 1L) * n_tvc * n_times + (j - 1L) * n_times + time_idx
    flat[time_idx <= 0L] <- 0L
    w_obs <- spread_effect(tvc$field, flat, N)
    eta <- eta + w_obs * matrix(x_j, n_samples, N, byrow = TRUE)
  }
  eta
}


#' Spatiotemporal contribution to the linear predictor
#' @keywords internal
st_eta <- function(st, st_info, design) {
  if (is.null(st_info)) return(NULL)
  N <- design$N
  n_samples <- nrow(st$field)

  if (isTRUE(st$is_hsgp)) {
    coords <- st_info$hsgp_coords
    t_idx <- as.integer(st_info$t_idx)
    if (is.null(coords) || is.null(t_idx)) return(NULL)
    Phi <- hsgp_basis_2d(coords, as.integer(st_info$hsgp_m %||% 6L),
                         st_info$hsgp_c %||% 1.5)$phi
    T_st <- st_info$n_times
    M <- ncol(Phi)
    eta <- matrix(0, n_samples, N)
    for (i in seq_len(N)) {
      cols <- (seq_len(M) - 1L) * T_st + t_idx[i]
      eta[, i] <- st$field[, cols, drop = FALSE] %*% Phi[i, ]
    }
    return(eta)
  }

  st_flat <- as.integer(st_info$st_flat)
  if (is.null(st_flat)) return(NULL)
  spread_effect(st$field, st_flat, N)
}


#' Assemble the two linear predictors
#'
#' @return A list with `eta_num`, `eta_denom` (draws by observation) and
#'   `dropped`, the structures whose field could not be reconstructed.
#' @keywords internal
hmc_eta_draws <- function(unpacked, design) {
  eta_num <- unpacked$beta_num %*% t(design$X_num)
  eta_denom <- if (!is.null(unpacked$beta_denom) && ncol(design$X_denom) > 0L) {
    unpacked$beta_denom %*% t(design$X_denom)
  } else {
    matrix(0, nrow(eta_num), ncol(eta_num))
  }

  re_eta <- hmc_re_eta(unpacked$re, design)
  if (!is.null(re_eta)) {
    eta_num <- eta_num + re_eta
    eta_denom <- eta_denom + re_eta
  }

  dropped <- character(0)
  for (contrib in hmc_structure_contributions(unpacked, design)) {
    if (is.null(contrib$eta)) {
      dropped <- c(dropped, paste0(contrib$name, " (", contrib$reason, ")"))
      next
    }
    eta_num <- eta_num + contrib$eta
    if (isTRUE(contrib$shared)) {
      eta_denom <- eta_denom + contrib$eta
    }
  }

  list(eta_num = eta_num, eta_denom = eta_denom, dropped = dropped)
}


#' Response scale of a model type
#'
#' The numerator link is logistic for the binomial families and log for the
#' count and continuous ones; the denominator of a binomial model is its trials,
#' which are data.
#'
#' @keywords internal
model_type_link <- function(model_type) {
  if (model_type %in% c("binomial", "beta_binomial")) "logit" else "log"
}


#' Transform linear predictors to the response scale
#'
#' @param eta A list with `eta_num` and `eta_denom`
#' @param model_type Model type string
#' @param type `"response"` or `"link"`
#'
#' @keywords internal
hmc_response_draws <- function(eta, model_type, type = "response") {
  eta_num <- eta$eta_num
  eta_denom <- eta$eta_denom
  logit_link <- identical(model_type_link(model_type), "logit")

  if (identical(type, "link")) {
    ratio <- if (logit_link) eta_num else eta_num - eta_denom
    return(list(numerator = eta_num, denominator = eta_denom, ratio = ratio))
  }

  if (logit_link) {
    mu_num <- inv_logit(eta_num)
    mu_denom <- matrix(1, nrow(eta_num), ncol(eta_num))
    ratio <- mu_num
  } else {
    mu_num <- exp(eta_num)
    mu_denom <- exp(eta_denom)
    ratio <- mu_num / mu_denom
  }

  list(numerator = mu_num, denominator = mu_denom, ratio = ratio)
}


#' Warn once about structures dropped from a linear predictor
#' @keywords internal
warn_dropped_structures <- function(dropped) {
  if (length(dropped) == 0L) return(invisible(NULL))
  warning("Linear predictor is missing ", paste(dropped, collapse = ", "),
          ". The values below exclude ", if (length(dropped) > 1L) "those" else "that",
          " term.", call. = FALSE)
  invisible(NULL)
}
