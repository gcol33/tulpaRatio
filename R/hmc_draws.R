#' Flat and per-structure posterior draws of an HMC fit
#'
#' @description
#' Reporting side of the single parameter-vector walk in `hmc_unpack_draws()`:
#' the named columns of `fit$draws`, and the per-structure draw matrices the
#' `svc()` / `tvc()` / `temporal()` / `spatiotemporal_effects()` extractors read.
#'
#' @name hmc_draws
#' @keywords internal
NULL


#' Name a per-element block
#' @keywords internal
element_names <- function(prefix, n, labels = NULL) {
  if (is.null(labels)) return(paste0(prefix, "[", seq_len(n), "]"))
  paste0(prefix, "[", labels, "]")
}


#' Label for the j-th term of a named structure
#' @keywords internal
term_label <- function(names, j) {
  if (!is.null(names) && j <= length(names)) names[j] else as.character(j)
}


#' Coefficient labels of a random-effect term
#' @keywords internal
re_coef_labels <- function(term) {
  labels <- character(0)
  if (isTRUE(term$has_intercept)) labels <- "intercept"
  c(labels, term$slope_names)
}


#' Flat posterior draws of an unpacked fit
#'
#' Columns are named as the parameters are reported and ordered as the sampler
#' lays them out. The high-dimensional latent blocks -- the factor scores and
#' the spatiotemporal interaction -- are summarized here and reported in full
#' through `hmc_structure_draws()`.
#'
#' @param unpacked Output of `hmc_unpack_draws()`
#'
#' @return A named list of draw vectors.
#' @keywords internal
hmc_draws_list <- function(unpacked) {
  out <- list()
  put <- function(name, x) out[[name]] <<- x
  put_cols <- function(prefix, m, labels = NULL) {
    if (is.null(m)) return(invisible(NULL))
    nms <- element_names(prefix, ncol(m), labels)
    for (j in seq_len(ncol(m))) out[[nms[j]]] <<- m[, j]
    invisible(NULL)
  }

  put_cols("beta_num", unpacked$beta_num)
  put_cols("beta_denom", unpacked$beta_denom)

  # Random effects
  re <- unpacked$re
  if (!is.null(re)) {
    single <- identical(re$kind, "single")
    for (t in seq_along(re$terms)) {
      term <- re$terms[[t]]
      coef_labels <- re_coef_labels(term)
      for (c in seq_len(term$n_coefs)) {
        name <- if (single) {
          "sigma_re"
        } else if (term$n_coefs == 1L) {
          paste0("sigma_re[", t, "]")
        } else {
          paste0("sigma_re[", t, ",", coef_labels[c], "]")
        }
        put(name, term$sigma[[c]])
      }
      for (k in seq_along(term$cor)) {
        pair <- coef_labels[term$cor_pairs[k, 2:1]]
        put(paste0("cor_re[", t, ",", pair[1], ":", pair[2], "]"), term$cor[[k]])
      }
      for (c in seq_len(term$n_coefs)) {
        m <- term$coefs[[c]]
        for (g in seq_len(term$n_groups)) {
          name <- if (single) {
            paste0("re[", g, "]")
          } else if (term$n_coefs == 1L) {
            paste0("re[", t, ",", g, "]")
          } else {
            paste0("re[", t, ",", g, ",", coef_labels[c], "]")
          }
          put(name, m[, g])
        }
      }
    }
  }

  # Dispersion
  for (name in names(unpacked$dispersion)) {
    put(name, unpacked$dispersion[[name]])
  }

  # Areal spatial
  sp <- unpacked$spatial
  if (!is.null(sp)) {
    if (identical(sp$type, "bym2")) {
      put("sigma_spatial", sp$sigma_total)
      put("rho_spatial", sp$rho)
      put("sigma_s_spatial", sp$sigma_s)
      put("sigma_u_spatial", sp$sigma_u)
      if (!sp$collapsed) {
        put_cols("phi_scaled", sp$phi_scaled)
        put_cols("theta", sp$theta)
      }
    } else {
      put("tau_spatial", sp$tau)
      if (!is.null(sp$rho)) put("rho_spatial", sp$rho)
      if (!sp$collapsed) put_cols("phi_spatial", sp$field)
    }
  }

  # Temporal
  tp <- unpacked$temporal
  if (!is.null(tp)) {
    if (identical(tp$type, "gp")) {
      put("sigma2_temporal_gp", tp$sigma2)
      put("phi_temporal_gp", tp$phi)
      put_cols("temporal_gp", tp$field)
    } else {
      put("tau_temporal", tp$tau)
      if (!is.null(tp$rho)) put("rho_ar1", tp$rho)
      put_cols("temporal", tp$field)
    }
  }

  # Zero- and one-inflation
  put_cols("beta_zi", unpacked$zi)
  put_cols("beta_oi", unpacked$oi)

  # GP
  gp <- unpacked$gp
  if (!is.null(gp)) {
    put("sigma2_gp", gp$sigma2)
    put("phi_gp", gp$phi)
    if (!gp$collapsed) put_cols("gp_w", gp$field)
  }

  # Multi-scale GP
  ms <- unpacked$msgp
  if (!is.null(ms)) {
    if (ms$is_hsgp) {
      put("sigma2_hsgp_local", ms$sigma2_local)
      put("lengthscale_hsgp_local", ms$phi_local)
      put_cols("hsgp_beta_local", ms$w_local)
      put("sigma2_hsgp_regional", ms$sigma2_regional)
      put("lengthscale_hsgp_regional", ms$phi_regional)
      put_cols("hsgp_beta_regional", ms$w_regional)
    } else {
      put("sigma2_gp_local", ms$sigma2_local)
      put("phi_gp_local", ms$phi_local)
      put_cols("gp_local_w", ms$w_local)
      put("sigma2_gp_regional", ms$sigma2_regional)
      put("phi_gp_regional", ms$phi_regional)
      put_cols("gp_regional_w", ms$w_regional)
    }
  }

  # Multi-scale temporal
  mst <- unpacked$ms_temporal
  if (!is.null(mst)) {
    if (!is.null(mst$trend)) {
      put("sigma2_trend", mst$trend$sigma2)
      put_cols("trend", mst$trend$field)
    }
    if (!is.null(mst$seasonal)) {
      put("sigma2_seasonal", mst$seasonal$sigma2)
      put_cols("seasonal", mst$seasonal$field)
    }
    if (!is.null(mst$short)) {
      put("sigma2_short", mst$short$sigma2)
      if (!is.null(mst$short$rho)) put("rho_short", mst$short$rho)
      put_cols("short_term", mst$short$field)
    }
  }

  # Spatially-varying coefficients
  svc <- unpacked$svc
  if (!is.null(svc)) {
    phi_prefix <- if (svc$is_hsgp) "lengthscale_svc" else "phi_svc"
    for (j in seq_len(svc$n_svc)) {
      label <- term_label(svc$names, j)
      put(paste0("sigma2_svc[", label, "]"), svc$sigma2[, j])
      put(paste0(phi_prefix, "[", label, "]"), svc$phi[, j])
    }
    w_prefix <- if (svc$is_hsgp) "svc_beta" else "svc"
    n_per_term <- ncol(svc$w) / svc$n_svc
    for (j in seq_len(svc$n_svc)) {
      label <- term_label(svc$names, j)
      cols <- ((j - 1) * n_per_term + 1):(j * n_per_term)
      block <- svc$w[, cols, drop = FALSE]
      for (k in seq_len(n_per_term)) {
        put(paste0(w_prefix, "[", label, ",", k, "]"), block[, k])
      }
    }
  }

  # Latent factors
  lf <- unpacked$latent
  if (!is.null(lf)) {
    for (k in seq_len(ncol(lf$sigma))) {
      put(paste0("sigma_latent[", k, "]"), lf$sigma[, k])
    }
  }

  # Spatiotemporal interaction
  st <- unpacked$st
  if (!is.null(st)) {
    put("tau_st", st$tau)
    if (!is.null(st$rho)) put("rho_st", st$rho)
    if (!is.null(st$phi_space)) put("phi_st_space", st$phi_space)
    if (!is.null(st$phi_time)) put("phi_st_time", st$phi_time)
    if (!is.null(st$sigma2_hsgp)) put("sigma2_st_hsgp", st$sigma2_hsgp)
    if (!is.null(st$lengthscale_hsgp)) put("lengthscale_st_hsgp", st$lengthscale_hsgp)
  }

  # Temporally-varying coefficients
  tvc <- unpacked$tvc
  if (!is.null(tvc)) {
    for (j in seq_len(tvc$n_tvc)) {
      label <- term_label(tvc$names, j)
      put(paste0("tau_tvc[", label, "]"), tvc$tau[, j])
      if (!is.null(tvc$rho)) put(paste0("rho_tvc[", label, "]"), tvc$rho[, j])
    }
    for (g in seq_len(tvc$n_groups)) {
      for (j in seq_len(tvc$n_tvc)) {
        label <- term_label(tvc$names, j)
        for (t in seq_len(tvc$n_times)) {
          col <- (g - 1L) * tvc$n_tvc * tvc$n_times + (j - 1L) * tvc$n_times + t
          name <- if (tvc$n_groups > 1L) {
            paste0("tvc[", label, ",g", g, ",t", t, "]")
          } else {
            paste0("tvc[", label, ",t", t, "]")
          }
          put(name, tvc$field[, col])
        }
      }
    }
  }

  out
}


#' Per-structure posterior draws the extractors read
#'
#' Each entry is shaped the way its extractor indexes it: `svc_draws` is
#' `[draw, location, term]`, `tvc_draws` is `[draw, time, term, group]`,
#' `latent_draws` is `[draw, observation, factor]`, and the field draws are
#' `[draw, unit]`. A multiscale temporal fit reports a list of component
#' matrices named as `temporal_multiscale()` names its components.
#'
#' @param unpacked Output of `hmc_unpack_draws()`
#'
#' @return A named list holding, for each structure present, the draws of its
#'   own effects: `spatial_draws`, `temporal_draws`, `svc_draws`, `tvc_draws`,
#'   `latent_draws`, `spatiotemporal_draws`.
#' @keywords internal
hmc_structure_draws <- function(unpacked) {
  out <- list()

  spatial <- unpacked$spatial %||% unpacked$gp %||% unpacked$hsgp %||% unpacked$msgp
  if (!is.null(spatial)) out$spatial_draws <- spatial$field

  if (!is.null(unpacked$temporal)) out$temporal_draws <- unpacked$temporal$field

  mst <- unpacked$ms_temporal
  if (!is.null(mst)) {
    components <- list(trend = mst$trend, seasonal = mst$seasonal,
                       short_term = mst$short)
    components <- components[!vapply(components, is.null, logical(1))]
    out$temporal_draws <- lapply(components, function(x) x$field)
  }

  svc <- unpacked$svc
  if (!is.null(svc)) {
    out$svc_draws <- structure_array(svc$fields)
  }

  tvc <- unpacked$tvc
  if (!is.null(tvc)) {
    n_draws <- nrow(tvc$field)
    arr <- array(NA_real_, dim = c(n_draws, tvc$n_times, tvc$n_tvc, tvc$n_groups))
    for (g in seq_len(tvc$n_groups)) {
      for (j in seq_len(tvc$n_tvc)) {
        cols <- (g - 1L) * tvc$n_tvc * tvc$n_times + (j - 1L) * tvc$n_times +
          seq_len(tvc$n_times)
        arr[, , j, g] <- tvc$field[, cols, drop = FALSE]
      }
    }
    out$tvc_draws <- arr
  }

  if (!is.null(unpacked$latent)) {
    out$latent_draws <- structure_array(unpacked$latent$factors)
  }

  if (!is.null(unpacked$st)) out$spatiotemporal_draws <- unpacked$st$field

  out
}


#' Stack per-term draw matrices into a `[draw, unit, term]` array
#' @keywords internal
structure_array <- function(fields) {
  fields <- Filter(Negate(is.null), fields)
  if (length(fields) == 0L) return(NULL)
  n_draws <- nrow(fields[[1]])
  n_units <- ncol(fields[[1]])
  arr <- array(NA_real_, dim = c(n_draws, n_units, length(fields)))
  for (j in seq_along(fields)) arr[, , j] <- fields[[j]]
  arr
}
