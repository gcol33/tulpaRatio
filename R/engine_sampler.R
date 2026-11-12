engine_sampler_inputs <- function(inp, family) {
  model <- inp$model_type
  entry <- MODEL_TYPES[[model]]
  zi <- inp$zi_info$type %||% "none"
  spatial <- inp$spatial_info$type %||% "none"
  temporal <- inp$temporal_info$type %||% "none"
  bundles <- inp$bundles
  if (is.null(entry$n_processes)) stop("No engine likelihood for family '", model, "'.", call. = FALSE)
  if (!zi %in% entry$spec_inflation)
    stop("The engine likelihood for '", model, "' does not support inflation '", zi, "'.", call. = FALSE)
  main_spatial <- spatial %in% c("icar", "bym2", "car_proper", "gp", "hsgp", "multiscale_gp")
  if (!spatial %in% c("none", "svc", "icar", "bym2", "car_proper", "gp", "hsgp", "multiscale_gp"))
    stop("The engine does not support spatial structure '", spatial, "'.", call. = FALSE)
  if (!temporal %in% c("none", "rw1", "rw2", "ar1", "iid", "gp", "multiscale", "tvc"))
    stop("The engine does not support temporal structure '", temporal, "'.", call. = FALSE)
  if (identical(inp$spatial_info$parameterization, "collapsed")) {
    if (!spatial %in% c("icar", "bym2", "gp"))
      stop("Collapsed spatial '", spatial, "' has no engine representation.", call. = FALSE)
    inp$spatial_info$parameterization <- bundles$spatial_params$parameterization <- "auto"
    if (spatial == "gp") {
      inp$gp_info$parameterization <- bundles$gp_params$parameterization <- "auto"
      bundles$gp_params$conditional_coordinates <- TRUE
    }
  }
  st <- bundles$st_params
  if (isTRUE(st$has_spatiotemporal)) {
    if (!st$type %in% c("I", "II", "III", "IV", "separable", "nonsep_gp"))
      stop("The engine does not support spatiotemporal structure '", st$type, "'.", call. = FALSE)
    if (identical(st$parameterization, "collapsed")) {
      if (!st$type %in% c("separable", "nonsep_gp"))
        stop("Collapsed spatiotemporal '", st$type, "' has no engine representation.", call. = FALSE)
      inp$spatiotemporal_info$parameterization <- st$parameterization <- "noncentered"
    }
  }
  if (isTRUE(bundles$tvc_params$has_tvc) && identical(bundles$tvc_params$structure, "gp"))
    stop("The engine bridge does not support a GP time-varying coefficient.", call. = FALSE)
  inp$bundles <- bundles
  inp$bundles$st_params <- st
  links <- family_links(family, model)
  cfg <- list(family = model, zi = zi, initialize = TRUE,
    num_link = links$num, denom_link = links$denom %||% "log",
    zi_link = links$zi, oi_link = links$oi)
  if (identical(entry$responses[["denom"]], "exposure"))
    cfg$exposure <- as.numeric(inp$hmc_data$y_denom_cont)
  if (!zi %in% c("none", "oi_binomial")) cfg$X_zi <- inp$zi_info$X_zi
  if (zi %in% c("oi_binomial", "zoib")) cfg$X_oi <- inp$zi_info$X_oi
  if (zi != "none") cfg$zi_prior_sd <- cfg$oi_prior_sd <- inp$sp$zi_prior_sd
  spatial_params <- if (main_spatial) bundles$spatial_params else list(type = "none")
  if (main_spatial) {
    spatial_params$sigma_prior_U <- inp$sp$spatial_sigma_prior_U
    spatial_params$sigma_prior_alpha <- inp$sp$spatial_sigma_prior_alpha
    spatial_params$shared <- inp$spatial_info$shared %||% TRUE
    spatial_params$gp <- bundles$gp_params
    spatial_params$hsgp_m <- inp$spatial_info$hsgp_m
    spatial_params$basis <- inp$spatial_info$hsgp_Phi
    spatial_params$eigenvalues <- inp$spatial_info$hsgp_eigenvalues
  }
  temporal_params <- if (!temporal %in% c("none", "tvc")) bundles$temporal_params else list()
  if (length(temporal_params)) temporal_params$parameterization <- inp$temporal_info$parameterization %||% "centered"
  q <- initialize_hmc_params_full(inp$hmc_data, model, inp$spatial_info,
    inp$temporal_info, inp$zi_info, inp$latent_info, inp$spatiotemporal_info,
    inp$ms_temporal_info, links = links)
  hmc_layout <- hmc_param_layout(inp$hmc_data, inp$spatial_info, inp$temporal_info,
    inp$zi_info, model, inp$latent_info, inp$spatiotemporal_info, inp$ms_temporal_info)
  extra <- c(hmc_layout$phi_num, hmc_layout$phi_denom)
  permutation <- c(setdiff(seq_along(q), extra), extra)
  args <- list(y_num = as.integer(inp$hmc_data$y_num),
    y_denom = as.integer(inp$hmc_data$y_denom), y_num_cont = as.numeric(inp$hmc_data$y_num_cont),
    y_denom_cont = as.numeric(inp$hmc_data$y_denom_cont),
    X_list = if (entry$n_processes == 1L) list(inp$hmc_data$X_num) else
      list(inp$hmc_data$X_num, inp$hmc_data$X_denom), cfg_list = cfg, init = q[permutation],
    sigma_beta = inp$sp$sigma_beta, phi_prior_shape = inp$sp$phi_shape,
    phi_prior_rate = inp$sp$phi_rate, sigma_prior_scale = inp$sp$sigma_re_scale,
    re_params = bundles$re_params, spatial_params = spatial_params,
    temporal_params = temporal_params, svc_spec = bundles$svc_params,
    tvc_spec = bundles$tvc_params, latent_spec = bundles$latent_params, st_spec = st)
  if (temporal == "gp") inp$temporal_info$stored_parameterization <- "centered"
  if (isTRUE(st$has_spatiotemporal)) inp$spatiotemporal_info$stored_parameterization <- "centered"
  list(inputs = inp, args = args, restore_order = order(permutation), layout = hmc_layout)
}

engine_sampler_run <- function(setup, method, iter, warmup, seed, verbose,
                               options = list(), ...) {
  result <- do.call(cpp_tulpaRatio_sample, c(setup$args,
    list(method = method, sampler_options = options, n_iter = as.integer(iter),
         n_warmup = as.integer(warmup), seed = as.integer(seed), verbose = isTRUE(verbose)),
    list(...)))
  engine_sampler_result(result, setup)
}

engine_sampler_result <- function(result, setup, chains = 1L) {
  order <- setup$restore_order
  result$samples <- if (chains > 1L) lapply(result$samples, function(x) x[, order, drop = FALSE]) else
    result$samples[, order, drop = FALSE]
  if (!is.null(result$mu)) result$mu <- result$mu[order]
  if (!is.null(result$Sigma)) result$Sigma <- result$Sigma[order, order, drop = FALSE]
  if (!is.null(result$L) && nrow(result$L) == length(order)) result$L <- result$L[order, , drop = FALSE]
  if (length(result$d) == length(order)) result$d <- result$d[order]
  result
}

engine_raw_parameter_names <- function(layout) {
  out <- character(layout$total)
  indexed <- function(index, stem) {
    if (length(index)) out[index] <<- paste0(stem, "[", seq_along(index), "]")
  }
  scalar <- function(index, name) if (length(index)) out[index] <<- name
  indexed(layout$beta_num, "beta_num"); indexed(layout$beta_denom, "beta_denom")
  for (term in layout$re$terms) {
    scalar(term$sigma, "log_sigma_re"); indexed(term$effects, "re")
  }
  scalar(layout$phi_num, "log_phi_num"); scalar(layout$phi_denom, "log_phi_denom")
  spatial <- layout$spatial
  scalar(spatial$log_tau, "log_tau_spatial")
  scalar(spatial$log_sigma, "log_sigma_bym2")
  scalar(spatial$logit_rho, if (spatial$type == "bym2") "logit_rho_bym2" else "logit_rho_car")
  indexed(spatial$phi, "phi_spatial"); indexed(spatial$theta, "theta_bym2")
  temporal <- layout$temporal
  scalar(temporal$log_tau, "log_tau_temporal"); scalar(temporal$logit_rho, "logit_rho_ar1")
  scalar(temporal$log_sigma2, "log_sigma2_temporal_gp"); scalar(temporal$logit_phi, "logit_phi_temporal_gp")
  indexed(temporal$effects, "phi_temporal")
  indexed(layout$zi, "beta_zi"); indexed(layout$oi, "beta_oi")
  visit <- function(node, path = character()) {
    if (is.list(node)) {
      for (name in names(node)) {
        if (!name %in% c("total", "m") && !startsWith(name, "n_"))
          visit(node[[name]], c(path, name))
      }
    } else if (is.numeric(node) && length(node)) {
      index <- node[!nzchar(out[node])]
      if (length(index)) indexed(index, paste(path, collapse = "."))
    }
  }
  visit(layout)
  if (any(!nzchar(out))) stop("The sampler result contains an unnamed parameter block.", call. = FALSE)
  out
}
