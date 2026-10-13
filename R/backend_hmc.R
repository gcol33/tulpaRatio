#' HMC/NUTS Backend for ratio Models
#'
#' @description
#' Hamiltonian Monte Carlo with the No-U-Turn Sampler (NUTS), the backend
#' that fits every tulpaRatio family and structure.
#'
#' @details
#' This backend implements:
#' - Hand-coded and automatic-differentiation gradients of the log-posterior
#' - NUTS with dual-averaging step size and windowed metric adaptation
#' - OpenMP parallelization (within-chain for likelihood, across-chain for multi-chain)
#' - ICAR and BYM2 spatial random effects
#' - RW1/RW2/AR1 temporal random effects
#' - Zero-inflation and hurdle models
#'
#' **Supported model families:**
#' - `ratiod_binomial()` - Trial-based proportions
#' - `ratiod_negbin_negbin()` - Two-process count ratios
#' - `ratiod_poisson_gamma()` - Count/effort ratios (CPUE)
#'
#' **Spatial structures:**
#' - `spatial_car()` - ICAR prior for areal data
#' - `spatial_bym2()` - BYM2 prior (structured + unstructured)
#'
#' **Zero-inflation types:**
#' - `zi_poisson()` - Zero-inflated Poisson
#' - `zi_negbin()` - Zero-inflated negative binomial
#' - `hurdle_poisson()` - Hurdle Poisson
#' - `hurdle_negbin()` - Hurdle negative binomial
#'
#' **When to use:**
#' - When you need full MCMC (not just approximations)
#' - For parallel multi-chain sampling
#'
#' @name hmc_backend
#' @keywords internal
NULL


#' Fit ratio model using HMC/NUTS
#'
#' @param formula A ratiod_formula object
#' @param data Data frame
#' @param family Model family
#' @param spatial Optional spatial structure (spatial_car or spatial_bym2)
#' @param temporal Optional temporal structure (temporal_rw1, temporal_ar1, etc.)
#' @param zi Optional zero-inflation specification (ratiod_zi object)
#' @param priors Prior specification
#' @param iter Total number of iterations per chain
#' @param warmup Number of warmup iterations per chain
#' @param chains Number of chains to run (can be parallelized)
#' @param cores Core budget for the fit. Split as `concurrent_chains =
#'   min(cores, chains)` chains in flight at once, each running with
#'   `cores %/% concurrent_chains` threads for its own within-chain
#'   gradient/likelihood work. Defaults to the machine's OpenMP thread cap.
#' @param L Number of leapfrog steps per iteration
#' @param seed Random seed for reproducibility
#' @param verbose Print progress
#' @param gradient_mode Gradient computation method: "auto", "H", "A_r", "A", "N"
#' @param re_param Random effects parameterization: "centered" or "noncentered"
#' @param metric Mass matrix type: "auto" (default), "diag", "dense", or "block_diag"
#' @param walnuts Run each NUTS trajectory as a WALNUTS transition: TRUE or
#'   FALSE (default)
#'
#' @return A ratiod_fit object
#' @keywords internal
fit_hmc <- function(formula,
                    data,
                    family,
                    spatial = NULL,
                    temporal = NULL,
                    spatiotemporal = NULL,
                    zi = NULL,
                    latent = NULL,
                    priors = NULL,
                    iter = 2000,
                    warmup = floor(iter / 2),
                    chains = 4,
                    cores = NULL,
                    L = 0,
                    adapt_delta = NULL,
                    max_treedepth = NULL,
                    seed = NULL,
                    verbose = TRUE,
                    gradient_mode = "auto",
                    re_param = "noncentered",
                    metric = "auto",
                    walnuts = FALSE) {

  # Set cores. Defaults to the machine's OpenMP cap, matching how the PG and
  # Laplace backends default their own (purely within-chain) `cores` budget:
  # see fit_pg_binomial()'s `cpp_pg_get_max_threads()` default in backend_pg.R.
  if (is.null(cores)) {
    cores <- cpp_get_max_threads()
  }
  cores <- as.integer(cores)
  if (is.na(cores) || cores < 1L) cores <- 1L

  # Set seed
  if (is.null(seed)) {
    seed <- sample.int(.Machine$integer.max, 1)
  }

  # Resolve adapt_delta: NULL → -1.0 (C++ sentinel for auto-selection)
  adapt_delta_value <- if (is.null(adapt_delta)) -1.0 else as.double(adapt_delta)

  # Get model type
  model_type <- get_hmc_model_type(family)

  # Extract data
  hmc_data <- prepare_hmc_data(formula, data, family, model_type)

  # Random slopes are now supported - log info if present
  if (isTRUE(hmc_data$has_slopes) && verbose) {
    n_slope_terms <- sum(sapply(hmc_data$re_terms, function(t) length(t$slope_vars) > 0))
    n_corr <- sum(sapply(hmc_data$re_terms, function(t) isTRUE(t$correlated) && length(t$slope_vars) > 0))
    message("  Random slopes: ", n_slope_terms, " term(s) (",
            n_corr, " correlated, ", n_slope_terms - n_corr, " uncorrelated)")
  }

  # Prepare spatial structure
  # Check if GP-based spatial (requires GP sampler) or multiscale temporal
  # without areal spatial (ICAR/BYM2/pCAR can pair with MS_t via regular sampler)
  has_areal_spatial <- !is.null(spatial) && !is.null(spatial$type) &&
    spatial$type %in% c("car", "car_proper", "bym2")
  use_gp_sampler <- is_gp_spatial(spatial) ||
    (is_multiscale_temporal(temporal) && !has_areal_spatial)

  if (use_gp_sampler) {
    # GP spatial uses dedicated sampler
    gp_info <- prepare_gp_for_hmc(spatial, data, hmc_data$N)
    gp_n_units <- if (gp_info$n_unique > 0L) gp_info$n_unique else hmc_data$N
    gp_group <- if (length(gp_info$gp_obs_to_loc) > 0) gp_info$gp_obs_to_loc else seq_len(hmc_data$N)
    # HSGP basis for reconstructing the field from its coefficients, built
    # from the coordinates the sampler itself was handed
    hsgp_basis <- NULL
    if (gp_info$gp_type %in% c("multiscale_gp", "hsgp")) {
      hsgp_basis <- hsgp_basis_2d(
        coords = matrix(gp_info$coords, ncol = 2, byrow = TRUE),
        m = as.integer(gp_info$hsgp_m %||% 6L),
        c_boundary = gp_info$hsgp_c %||% 1.5
      )
    }

    spatial_info <- list(type = gp_info$gp_type, n_units = gp_n_units,
                         group = gp_group, adj_row_ptr = integer(1),
                         adj_col_idx = integer(0), n_neighbors = integer(0),
                         bym2_scale = 1.0,
                         hsgp_m = gp_info$hsgp_m,  # HSGP: basis functions per dim
                         hsgp_c = gp_info$hsgp_c,
                         shared = gp_info$shared %||% TRUE,
                         msgp_hsgp_Phi = hsgp_basis$phi,
                         msgp_hsgp_eigenvalues = hsgp_basis$eigenvalues,
                         hsgp_Phi = hsgp_basis$phi,
                         hsgp_eigenvalues = hsgp_basis$eigenvalues,
                         coord_vars = if (!is.null(spatial)) spatial$coord_vars else NULL,
                         parameterization = gp_info$parameterization,
                         # Range bounds for proper initialization (from prepare_gp_for_hmc)
                         range_local_lower = gp_info$range_local_lower,
                         range_local_upper = gp_info$range_local_upper,
                         range_regional_lower = gp_info$range_regional_lower,
                         range_regional_upper = gp_info$range_regional_upper,
                         rsr_basis = rsr_basis(spatial, data))
  } else {
    gp_info <- NULL
    spatial_info <- prepare_spatial_for_hmc(spatial, data, hmc_data$N)
  }
  if (!is.null(spatial_info$rsr_basis) &&
      identical(spatial_info$parameterization, "collapsed")) {
    stop("A restricted spatial field (`spatial_rsr()`) cannot be collapsed: ",
         "the collapsed marginal integrates the unrestricted field. ",
         "Use the default parameterization.", call. = FALSE)
  }

  # Auto-cap max_treedepth for GP/MSGP spatial models (many spatial params → deep trees)
  if (is.null(max_treedepth)) {
    gp_type <- if (!is.null(gp_info)) gp_info$gp_type else "none"
    if (gp_type %in% c("gp", "multiscale_gp")) {
      max_treedepth <- 8L
      if (verbose) {
        message("  Auto-capping max_treedepth to 8 for GP spatial model ",
                "(use max_treedepth = 10 to override)")
      }
    } else {
      max_treedepth <- 10L
    }
  }

  # Prepare temporal structure
  temporal_info <- prepare_temporal_for_hmc(temporal, data, hmc_data$N)
  temporal_info$time_levels <- temporal$time_levels

  # Prepare zero-inflation structure. Handles both an explicit `zi = `
  # argument and auto-detection from a ZI/hurdle/OI/ZOIB family (e.g.
  # ratiod_zinegbin()) when `zi` is not given.
  zi_info <- prepare_zi_for_hmc(zi, data, hmc_data$N, family)

  # Prepare latent factor structure. A binomial family has no denominator
  # predictor, so a shared factor loads on the numerator alone.
  latent_info <- prepare_latent_for_hmc(latent, hmc_data$N)
  if (model_type %in% c("binomial", "beta_binomial")) latent_info$shared <- FALSE

  # Prepare spatiotemporal structure
  spatiotemporal_info <- prepare_spatiotemporal_for_hmc(spatiotemporal, data)

  # Prepare multiscale temporal structure (both sampler paths allocate it)
  ms_temporal_info <- prepare_multiscale_temporal_for_hmc(temporal, data, hmc_data$N)

  # Prepare SVC structure (if spatial is an SVC specification)
  svc_info <- prepare_svc_for_hmc(spatial, data, hmc_data$N, hmc_data$X_num)

  # Prior parameters the sampler evaluates
  sp <- sampler_priors(priors, model_type)
  priors <- fill_priors(priors, model_type)
  sigma_beta <- sp$sigma_beta
  sigma_re_scale <- sp$sigma_re_scale
  phi_shape <- sp$phi_shape
  phi_rate <- sp$phi_rate

  tau_spatial_shape <- sp$tau_spatial_shape
  tau_spatial_rate <- sp$tau_spatial_rate


  # Initialize parameters
  q_init <- initialize_hmc_params_full(
    hmc_data, model_type, spatial_info, temporal_info, zi_info, latent_info, svc_info,
    spatiotemporal_info, links = family_links(family, model_type)
  )

  if (verbose) {
    sampler_name <- if (L == 0) "NUTS" else paste0("HMC (L=", L, ")")
    message("Running ", sampler_name, " sampler...")
    message("  Parameters: ", length(q_init))
    message("  Iterations: ", iter, " (warmup: ", warmup, ")")
    message("  Chains: ", chains, " (cores: ", cores, ")")
    if (spatial_info$type != "none") {
      message("  Spatial: ", spatial_info$type, " (",
              spatial_info$n_units, " units)")
    }
    if (temporal_info$type != "none") {
      message("  Temporal: ", temporal_info$type, " (",
              temporal_info$n_times, " time points)")
    }
    if (zi_info$type != "none") {
      message("  Zero-inflation: ", zi_info$type, " (",
              zi_info$p_zi, " predictors)")
    }
    if (latent_info$type != "none") {
      message("  Latent factors: ", latent_info$n_factors, " factor(s)")
    }
  }

  # `cores` is a budget split two ways rather than a chain-concurrency count
  # alone: concurrent_chains chains run at once, and each gets the rest of the
  # budget for its own within-chain gradient/likelihood parallelism. A single
  # chain (concurrent_chains == 1) gets the whole budget within-chain, which a
  # chain-count-only cap could never express (#17).
  concurrent_chains <- min(cores, chains)
  n_threads_within <- as.integer(cores %/% concurrent_chains)

  # Get temporal prior parameters
  tau_temporal_shape <- sp$tau_temporal_shape
  tau_temporal_rate <- sp$tau_temporal_rate
  # Beta anchors on u = (rho + 1) / 2 for an AR1 correlation. The block's own
  # rho_prior wins over the model-wide one; both default to Beta(2, 2).
  rho_temporal_ab <- rho_prior_anchors(temporal_info$rho_prior,
                                       priors$rho_temporal)

  # Prepare multi-term RE data
  n_re_terms <- hmc_data$n_re_terms %||% 0L
  if (n_re_terms > 1 && !is.null(hmc_data$re_group_matrix)) {
    re_group_matrix <- hmc_data$re_group_matrix
    re_n_groups_vec <- sapply(hmc_data$re_terms, function(x) x$n_groups)
  } else if (n_re_terms == 1 && !is.null(hmc_data$re_terms)) {
    # Single RE term: create matrix from first term
    re_group_matrix <- matrix(hmc_data$re_terms[[1]]$group_idx, ncol = 1L)
    re_n_groups_vec <- as.integer(hmc_data$re_terms[[1]]$n_groups)
  } else {
    # No RE: create dummy matrix
    re_group_matrix <- matrix(0L, nrow = hmc_data$N, ncol = 1L)
    re_n_groups_vec <- as.integer(hmc_data$n_re_groups)
  }

  # Prepare slope-related parameters
  has_re_slopes <- isTRUE(hmc_data$has_slopes)
  has_re_correlated_slopes <- isTRUE(hmc_data$has_correlated_slopes)

  if (has_re_slopes && n_re_terms > 0) {
    re_n_coefs_vec <- sapply(hmc_data$re_terms, function(x) x$n_coefs)
    re_correlated_vec <- sapply(hmc_data$re_terms, function(x) isTRUE(x$correlated))
    re_n_chol_vec <- sapply(hmc_data$re_terms, function(x) x$n_chol %||% 0L)
    slope_matrices_list <- hmc_data$slope_matrices
  } else {
    re_n_coefs_vec <- rep(1L, max(1L, n_re_terms))
    re_correlated_vec <- rep(FALSE, max(1L, n_re_terms))
    re_n_chol_vec <- rep(0L, max(1L, n_re_terms))
    slope_matrices_list <- list()
  }

  # Parameter bundles both sampler entry points read. Built once, above the
  # branch: a bundle a branch does not build is a block C++ leaves unwritten
  # while the layout above still names its columns.

  re_params <- list(
    group = as.integer(hmc_data$re_group),
    n_groups = as.integer(hmc_data$n_re_groups),
    n_terms = as.integer(n_re_terms),
    group_matrix = as.matrix(re_group_matrix),
    n_groups_vec = as.integer(re_n_groups_vec),
    has_slopes = has_re_slopes,
    has_correlated_slopes = has_re_correlated_slopes,
    n_coefs_vec = as.integer(re_n_coefs_vec),
    correlated_vec = as.logical(re_correlated_vec),
    n_chol_vec = as.integer(re_n_chol_vec),
    slope_matrices = slope_matrices_list,
    parameterization = as.integer(if (re_param == "centered") 0L else 1L)
  )

  temporal_params <- list(
    type = temporal_info$type,
    time_idx = as.integer(temporal_info$time_index),
    group_idx = as.integer(temporal_info$group_index),
    n_times = as.integer(temporal_info$n_times),
    n_groups = as.integer(temporal_info$n_groups),
    n_params = as.integer(temporal_info$n_temporal_params),
    cyclic = temporal_info$precision_structure$cyclic %||% FALSE,
    shared = temporal_info$shared %||% TRUE,
    tau_shape = tau_temporal_shape,
    tau_rate = tau_temporal_rate,
    rho_prior_a = rho_temporal_ab[["a"]],
    rho_prior_b = rho_temporal_ab[["b"]],
    # GP-specific fields (only used when type = "gp")
    time_values = temporal_info$time_values %||% numeric(0),
    cov_type = temporal_info$cov_type %||% "exponential",
    nu = temporal_info$nu %||% 1.5,
    period = temporal_info$period %||% 1.0,
    gp_sigma2_prior_U = priors$temporal_gp_sigma2_prior_U %||% 1.0,
    gp_sigma2_prior_alpha = priors$temporal_gp_sigma2_prior_alpha %||% 0.01,
    gp_phi_prior_lower = priors$temporal_gp_phi_prior_lower %||% 0.01,
    gp_phi_prior_upper = priors$temporal_gp_phi_prior_upper %||% 10.0,
    gp_parameterization = temporal_info$parameterization %||% "noncentered",
    # AR1's own coordinate, read only when type = "ar1"
    # The standalone temporal block's own coordinate, read for the structures
    # that offer one: AR1's innovations and the intrinsic walks' z.
    ar1_parameterization = if (temporal_info$type %in% c("ar1", "rw1", "rw2")) {
      temporal_info$parameterization %||% "centered"
    } else "centered",
    # Multiscale temporal fields (only used when type = "multiscale")
    ms_parameterization = if (identical(temporal_info$type, "multiscale")) {
      temporal_info$parameterization %||% "centered"
    } else "centered",
    trend_type = temporal_info$trend %||% "rw1",
    short_term_type = temporal_info$short_term %||% "ar1",
    seasonal_period = as.integer(temporal_info$precision_structure$seasonal_period %||% 0L),
    sigma2_trend_prior_U = priors$ms_sigma2_trend_prior_U %||% 1.0,
    sigma2_trend_prior_alpha = priors$ms_sigma2_trend_prior_alpha %||% 0.01,
    sigma2_short_prior_U = priors$ms_sigma2_short_prior_U %||% 1.0,
    sigma2_short_prior_alpha = priors$ms_sigma2_short_prior_alpha %||% 0.01
  )

  latent_params <- list(
    has_latent = latent_info$type != "none",
    n_factors = as.integer(latent_info$n_factors),
    shared = latent_info$shared,
    constraint = as.integer(ifelse(latent_info$constraint == "sum_to_zero", 0L, 1L)),
    sigma_prior_rate = latent_info$sigma_prior_rate,
    offdiag_sd = latent_info$offdiag_sd %||% 1
  )

  st_is_hsgp <- isTRUE(spatiotemporal_info$spatial_is_hsgp)
  st_gp <- spatiotemporal_info$gp %||% list()
  st_rho_ab <- rho_prior_anchors(spatiotemporal_info$rho_prior,
                                 priors$rho_temporal)
  st_params <- list(
    has_spatiotemporal = spatiotemporal_info$has_spatiotemporal %||% FALSE,
    type = spatiotemporal_info$type %||% "none",
    shared = spatiotemporal_info$shared %||% TRUE,
    parameterization = spatiotemporal_info$parameterization %||% "centered",
    n_spatial = as.integer(spatiotemporal_info$n_spatial %||% 0L),
    n_times = as.integer(spatiotemporal_info$n_times %||% 0L),
    n_params = as.integer(spatiotemporal_info$n_params %||% 0L),
    s_idx = as.integer(spatiotemporal_info$s_idx %||% integer(0)),
    t_idx = as.integer(spatiotemporal_info$t_idx %||% integer(0)),
    st_flat = as.integer(spatiotemporal_info$st_flat %||% integer(0)),
    temporal_type = spatiotemporal_info$temporal_type %||% "rw1",
    temporal_cyclic = spatiotemporal_info$temporal_cyclic %||% FALSE,
    adj_row_ptr = as.integer(spatiotemporal_info$spatial_Q$adj_row_ptr %||% integer(0)),
    adj_col_idx = as.integer(spatiotemporal_info$spatial_Q$adj_col_idx %||% integer(0)),
    sigma2_prior_U = priors$st_sigma2_prior_U %||% 1.0,
    sigma2_prior_alpha = priors$st_sigma2_prior_alpha %||% 0.01,
    rho_prior_a = st_rho_ab[["a"]],
    rho_prior_b = st_rho_ab[["b"]],
    # Kronecker precision mass data
    Qs_inv = spatiotemporal_info$spatial_Q$Q_inv,
    Ls = spatiotemporal_info$spatial_Q$L_Q,
    Qt_inv = spatiotemporal_info$temporal_Q$Q_time_inv,
    Lt = spatiotemporal_info$temporal_Q$L_time,
    # HSGP-ST fields (only used when spatial_is_hsgp = TRUE)
    st_is_hsgp = st_is_hsgp,
    hsgp_m = as.integer(spatiotemporal_info$hsgp_m %||% 0L),
    hsgp_c = spatiotemporal_info$hsgp_c %||% 1.5,
    hsgp_coords = if (st_is_hsgp) as.numeric(t(spatiotemporal_info$hsgp_coords)) else numeric(0),
    hsgp_scale_coords = spatiotemporal_info$hsgp_scale_coords %||% TRUE,
    # GP interaction fields (type "separable" / "nonsep_gp"). Empty for every
    # Knorr-Held type, which carries no geometry; the .Call boundary reads
    # them only for the GP types and refuses a structure that does not
    # describe the interaction's own index set.
    nn = as.integer(st_gp$nn %||% 0L),
    gp_coords = as.numeric(st_gp$coords %||% numeric(0)),
    gp_time_values = as.numeric(st_gp$time_values %||% numeric(0)),
    nn_idx = as.integer(st_gp$nn_idx %||% integer(0)),
    nn_dist_space = as.numeric(st_gp$nn_dist_space %||% numeric(0)),
    nn_dist_time = as.numeric(st_gp$nn_dist_time %||% numeric(0)),
    nn_order = as.integer(st_gp$nn_order %||% integer(0)),
    nn_order_inv = as.integer(st_gp$nn_order_inv %||% integer(0)),
    cov_space = as.integer(st_gp$cov_space %||% 0L),
    cov_time = as.integer(st_gp$cov_time %||% 0L),
    nonsep_type = st_gp$nonsep_type %||% "product",
    phi_space_prior_lower = priors$st_phi_space_prior_lower %||% 0.01,
    phi_space_prior_upper = priors$st_phi_space_prior_upper %||% 10.0,
    phi_time_prior_lower = priors$st_phi_time_prior_lower %||% 0.01,
    phi_time_prior_upper = priors$st_phi_time_prior_upper %||% 10.0
  )

  # TVC (Temporally-Varying Coefficients) parameters
  # Check if temporal is a TVC specification
  tvc_info <- prepare_tvc_for_hmc(temporal, data, hmc_data$N, hmc_data$X_num)
  tvc_params <- list(
    has_tvc = tvc_info$has_tvc %||% FALSE,
    n_tvc = as.integer(tvc_info$n_tvc %||% 0L),
    n_times = as.integer(tvc_info$n_times %||% 0L),
    n_groups = as.integer(tvc_info$n_groups %||% 1L),
    structure = tvc_info$structure %||% "rw1",
    shared = tvc_info$shared %||% TRUE,
    cyclic = tvc_info$cyclic %||% FALSE,
    tvc_indices = as.integer(tvc_info$tvc_indices %||% integer(0)),
    time_index = as.integer(tvc_info$time_index %||% integer(0)),
    group_index = as.integer(tvc_info$group_index %||% integer(0)),
    X_tvc = as.numeric(tvc_info$X_tvc %||% numeric(0)),
    tau_shape = priors$tvc_tau_shape %||% 2.0,
    tau_rate = priors$tvc_tau_rate %||% 0.5
  )

  # SVC (Spatially-Varying Coefficients) parameters
  # Check if spatial is an SVC specification
  svc_params <- list(
    has_svc = svc_info$has_svc %||% FALSE,
    n_svc = as.integer(svc_info$n_svc %||% 0L),
    nn = as.integer(svc_info$nn %||% 0L),
    shared = svc_info$shared %||% TRUE,
    cov_type = svc_info$cov_type %||% "exponential",
    coords = as.numeric(svc_info$coords %||% numeric(0)),
    svc_indices = as.integer(svc_info$svc_indices %||% integer(0)),
    X_svc = as.numeric(svc_info$X_svc %||% numeric(0)),
    nn_idx = as.integer(svc_info$nn_idx %||% integer(0)),
    nn_dist = as.numeric(svc_info$nn_dist %||% numeric(0)),
    nn_order = as.integer(svc_info$nn_order %||% integer(0)),
    nn_order_inv = as.integer(svc_info$nn_order_inv %||% integer(0)),
    parameterization = svc_info$parameterization,
    sigma2_prior_scale = svc_info$sigma2_prior_scale %||% 1.0,
    phi_prior_lower = svc_info$phi_prior_lower,
    phi_prior_upper = svc_info$phi_prior_upper,
    tau_shape = priors$svc_tau_shape %||% 1.0,
    tau_rate = priors$svc_tau_rate %||% 0.01,
    svc_approx = svc_info$svc_approx %||% "nngp",
    hsgp_m = as.integer(svc_info$hsgp_m %||% 0L),
    hsgp_c = svc_info$hsgp_c %||% 0.0
  )

  # Populate spatial_info with SVC details for output conversion
  if (svc_info$has_svc) {
    spatial_info$n_svc <- svc_info$n_svc
    spatial_info$svc_shared <- svc_info$shared %||% TRUE
    spatial_info$svc_spec <- svc_info$spec
    spatial_info$svc_names <- svc_info$svc_names
    spatial_info$svc_approx <- svc_info$svc_approx %||% "nngp"
    spatial_info$svc_hsgp_m <- svc_info$hsgp_m %||% 0L
    spatial_info$svc_hsgp_c <- svc_info$hsgp_c %||% 1.5
    spatial_info$svc_coords <- if (!is.null(svc_info$coords)) {
      matrix(svc_info$coords, ncol = 2, byrow = TRUE)
    } else {
      NULL
    }
    spatial_info$X_svc <- if (!is.null(svc_info$X_svc)) {
      matrix(svc_info$X_svc, nrow = hmc_data$N, ncol = svc_info$n_svc)
    } else {
      NULL
    }
  }

  # Run sampler - branch based on GP vs non-GP spatial
  if (use_gp_sampler) {
    # Bundle GP parameters into list for C++
    gp_params <- list(
      gp_type = gp_info$gp_type,
      coords = gp_info$coords,
      nn_idx = as.integer(gp_info$nn_idx),
      nn_dist = gp_info$nn_dist,
      nn_order = as.integer(gp_info$nn_order),
      nn_order_inv = as.integer(gp_info$nn_order_inv),
      nn_neighbor_dist = gp_info$nn_neighbor_dist,  # Phase 1.3: cached pairwise distances
      nn = as.integer(gp_info$nn),
      cov_type = gp_info$cov_type,
      nu = gp_info$nu %||% 1.5,  # Default nu for non-Matern covariances
      shared = gp_info$shared,
      sigma2_prior_U = priors$gp_sigma2_prior_U %||% 1.0,
      sigma2_prior_alpha = priors$gp_sigma2_prior_alpha %||% 0.01,
      phi_prior_lower = priors$gp_phi_prior_lower %||% 0.01,
      phi_prior_upper = priors$gp_phi_prior_upper %||% 100.0,
      # HSGP parameters
      hsgp_m = as.integer(gp_info$hsgp_m %||% 8L),
      hsgp_c = gp_info$hsgp_c %||% 1.5,
      # GP solver configuration
      # Observation-to-location mapping (1-based)
      gp_obs_to_loc = as.integer(gp_info$gp_obs_to_loc),
      n_unique = as.integer(gp_info$n_unique),
      # GP parameterization
      parameterization = gp_info$parameterization %||% "centered"
    )

    # Bundle multiscale GP parameters
    ms_gp_params <- list(
      range_local_lower = gp_info$range_local_lower,
      range_local_upper = gp_info$range_local_upper,
      range_regional_lower = gp_info$range_regional_lower,
      range_regional_upper = gp_info$range_regional_upper,
      sigma2_local_prior_U = priors$ms_sigma2_local_prior_U %||% 1.0,
      sigma2_local_prior_alpha = priors$ms_sigma2_local_prior_alpha %||% 0.01,
      sigma2_regional_prior_U = priors$ms_sigma2_regional_prior_U %||% 1.0,
      sigma2_regional_prior_alpha = priors$ms_sigma2_regional_prior_alpha %||% 0.01,
      sampler = gp_info$sampler %||% "auto"
    )

    # Bundle multiscale temporal parameters
    ms_temporal_params <- list(
      type = ms_temporal_info$ms_temporal_type,
      time_index = as.integer(ms_temporal_info$ms_time_index),
      group_index = as.integer(ms_temporal_info$ms_group_index),
      n_times = as.integer(ms_temporal_info$ms_n_times),
      n_groups = as.integer(ms_temporal_info$ms_n_groups),
      trend_type = ms_temporal_info$trend_type,
      seasonal_period = as.integer(ms_temporal_info$seasonal_period),
      short_term_type = ms_temporal_info$short_term_type,
      shared = ms_temporal_info$shared,
      sigma2_trend_prior_U = priors$ms_sigma2_trend_prior_U %||% 1.0,
      sigma2_trend_prior_alpha = priors$ms_sigma2_trend_prior_alpha %||% 0.01,
      sigma2_seasonal_prior_U = priors$ms_sigma2_seasonal_prior_U %||% 1.0,
      sigma2_seasonal_prior_alpha = priors$ms_sigma2_seasonal_prior_alpha %||% 0.01,
      sigma2_short_prior_U = priors$ms_sigma2_short_prior_U %||% 1.0,
      sigma2_short_prior_alpha = priors$ms_sigma2_short_prior_alpha %||% 0.01
    )

    rsr_params <- hmc_rsr_params(spatial_info)

    # Use O2-safe interface with single List parameter
    # This minimizes Rcpp template instantiation at ABI boundary
    fit_raw <- cpp_hmc_fit_gp_v2(list(
      q_init = q_init,
      y_num = as.integer(hmc_data$y_num),
      y_denom = as.integer(hmc_data$y_denom),
      y_num_cont = as.numeric(hmc_data$y_num_cont),
      y_denom_cont = hmc_data$y_denom_cont,
      X_num = hmc_data$X_num,
      X_denom = hmc_data$X_denom,
      re_params = re_params,
      family = family_params(family, model_type),
      # Bundled parameter lists
      gp_params = gp_params,
      ms_gp_params = ms_gp_params,
      ms_temporal_params = ms_temporal_params,
      rsr_params = rsr_params,
      temporal_params = temporal_params,
      latent_params = latent_params,
      st_params = st_params,
      tvc_params = tvc_params,
      svc_params = svc_params,
      # Priors
      sigma_beta = sigma_beta,
      sigma_re_scale = sigma_re_scale,
      phi_prior_shape = phi_shape,
      phi_prior_rate = phi_rate,
      # Zero-inflation
      zi_type_str = zi_info$type,
      X_zi = zi_info$X_zi,
      zi_prior_sd = sp$zi_prior_sd,
      # Sampler
      n_iter = as.integer(iter),
      n_warmup = as.integer(warmup),
      L = as.integer(L),
      n_chains = as.integer(chains),
      seed = as.integer(seed),
      n_threads = n_threads_within,
      verbose = verbose,
      gradient_mode_str = gradient_mode,
      max_treedepth = as.integer(max_treedepth),
      adapt_delta = adapt_delta_value,
      metric_str = metric
    ))
  } else {
    # Bundle parameters into lists to stay under R's 65-argument limit for .Call
    spatial_params <- list(
      type = spatial_info$type,
      group = as.integer(spatial_info$group),
      n_units = spatial_info$n_units,
      adj_row_ptr = as.integer(spatial_info$adj_row_ptr),
      adj_col_idx = as.integer(spatial_info$adj_col_idx),
      n_neighbors = as.integer(spatial_info$n_neighbors),
      bym2_scale = spatial_info$bym2_scale,
      component = spatial_info$component,
      node_prec = spatial_info$node_prec,
      Q_inv = spatial_info$Q_inv,
      L_Q = spatial_info$L_Q,
      parameterization = spatial_info$parameterization %||% "standard",
      rho_lower = spatial_info$rho_lower %||% 0.0,
      rho_upper = spatial_info$rho_upper %||% 1.0,
      rho_prior_a = spatial_info$rho_prior_a %||% 1.0,
      rho_prior_b = spatial_info$rho_prior_b %||% 1.0,
      center = spatial_info$center %||% TRUE,
      rsr = hmc_rsr_params(spatial_info)
    )

    prior_params <- list(
      sigma_beta = sigma_beta,
      sigma_re_scale = sigma_re_scale,
      phi_shape = phi_shape,
      phi_rate = phi_rate,
      tau_spatial_shape = tau_spatial_shape,
      tau_spatial_rate = tau_spatial_rate
    )

    zi_params <- list(
      type = zi_info$type,
      X = zi_info$X_zi,
      p_zi = zi_info$p_zi %||% as.integer(ncol(zi_info$X_zi)),  # Explicit p_zi for OI-only models
      prior_sd = sp$zi_prior_sd,
      # OI info for OI-binomial and ZOIB models
      X_oi = zi_info$X_oi,
      p_oi = zi_info$p_oi %||% 0L,
      oi_prior_sd = sp$zi_prior_sd
    )


    # B1b feature flag: route the 7 ratio families through the LikelihoodSpec
    # path (autodiff only — no spatial / temporal / RE / latent factors,
    # single chain, L = 0). B1c relaxed the ZI gate so the 8 ZI/hurdle/OI/ZOIB
    # variants flow through the same path when the family supports them. Off
    # by default; enabled only for the narrowest scope so the legacy backend
    # stays the default for every other config.
    SUPPORTED_SPEC_FAMILIES <- c("binomial", "poisson_gamma", "negbin_gamma",
                                 "negbin_negbin", "gamma_gamma", "lognormal",
                                 "beta_binomial")
    SPEC_ZI_COMPAT <- list(
      binomial      = c("none", "zi_binomial", "hurdle_binomial",
                        "oi_binomial", "zoib"),
      poisson_gamma = c("none", "zi_poisson", "hurdle_poisson"),
      negbin_gamma  = c("none", "zi_negbin", "hurdle_negbin"),
      negbin_negbin = c("none", "zi_negbin", "hurdle_negbin"),
      gamma_gamma   = "none",
      lognormal     = "none",
      beta_binomial = "none"
    )
    use_specs_path <- isTRUE(getOption("tulpaRatio.use_specs", FALSE)) ||
      identical(Sys.getenv("TULPARATIO_USE_SPECS"), "1")
    zi_type_str <- zi_info$type %||% "none"
    zi_supported <- (model_type %in% SUPPORTED_SPEC_FAMILIES) &&
      (zi_type_str %in% (SPEC_ZI_COMPAT[[model_type]] %||% "none"))
    # B1d-1: allow single-grouping-factor RE on the spec path — random
    # intercepts AND uncorrelated random slopes (n_coefs >= 1, correlated
    # off). Multi-term (crossed/nested) and correlated slopes (LKJ) still
    # fall back to legacy until B1d-2 wires them up.
    n_re_groups_v <- hmc_data$n_re_groups %||% 0L
    n_re_terms_v  <- hmc_data$n_re_terms %||% 0L
    re_supported <- (n_re_groups_v == 0L && n_re_terms_v == 0L) ||
      (n_re_groups_v > 0L && n_re_terms_v <= 1L &&
       isFALSE(has_re_correlated_slopes))
    # B1d Step 2: allow ICAR / BYM2 (non-collapsed) on the spec path. Collapsed
    # parameterisations, CAR_PROPER, GP / HSGP / multiscale GP, and RSR still
    # fall back to legacy.
    spatial_type_v <- spatial_info$type %||% "none"
    spatial_param_v <- spatial_info$parameterization %||% "standard"
    spatial_supported <- identical(spatial_type_v, "none") ||
      (spatial_type_v %in% c("icar", "bym2") &&
       !identical(spatial_param_v, "collapsed") &&
       !isTRUE(spatial_info$has_rsr))
    # B1d Step 3: allow RW1 / RW2 / AR1 temporal. Temporal GP / multiscale /
    # cyclic still fall back to legacy.
    temporal_type_v <- temporal_info$type %||% "none"
    temporal_supported <- identical(temporal_type_v, "none") ||
      (temporal_type_v %in% TEMPORAL_TYPES_SINGLE &&
       !isTRUE(temporal_info$precision_structure$cyclic))
    specs_eligible <- use_specs_path &&
      family_handcoded(family, model_type) &&
      zi_supported &&
      re_supported &&
      spatial_supported &&
      temporal_supported &&
      identical(latent_info$type, "none") &&
      isFALSE(svc_params$has_svc) &&
      isFALSE(tvc_params$has_tvc) &&
      isFALSE(st_params$has_spatiotemporal) &&
      as.integer(L) == 0L &&
      as.integer(chains) == 1L

    if (specs_eligible) {
      num_link   <- if (model_type %in% c("binomial", "beta_binomial")) "logit" else "log"
      denom_link <- "log"
      cfg <- list(family = model_type, zi = zi_type_str,
                  num_link = num_link, denom_link = denom_link)
      # OI uses X_oi only (no ZI block); ZOIB uses both; all other inflation
      # types use X_zi only. Mirrors zi_info shape from prepare_zi_for_hmc /
      # the family-built-in ZI path.
      has_zi_block <- !(zi_type_str %in% c("none", "oi_binomial"))
      has_oi_block <- zi_type_str %in% c("oi_binomial", "zoib")
      if (has_zi_block) cfg$X_zi <- zi_info$X_zi
      if (has_oi_block) cfg$X_oi <- zi_info$X_oi
      # Pass the same ZI / OI prior scales the legacy backend uses so a path
      # switch does not silently change the prior (the bridge defaults to
      # tulpa's engine default of 2.5 otherwise; legacy default is 10.0).
      if (has_zi_block || has_oi_block) {
        cfg$zi_prior_sd <- sp$zi_prior_sd
        cfg$oi_prior_sd <- sp$zi_prior_sd
      }
      # X_list carries one design matrix per process. Single-process families
      # (binomial / beta_binomial) pass length-1; two-process families append
      # X_denom. Adding a 3-process family is a list element, not a new arg.
      # NOTE: this set must mirror the n_processes value each builder returns
      # in lik_dispatch.cpp. The bridge cross-checks and errors on mismatch.
      ONE_PROCESS_FAMILIES <- c("binomial", "beta_binomial")
      X_list <- if (model_type %in% ONE_PROCESS_FAMILIES) {
        list(hmc_data$X_num)
      } else {
        list(hmc_data$X_num, hmc_data$X_denom)
      }

      # Reorder q_init from the legacy layout to tulpa's generic layout. With
      # only ZI/OI present (B1c) the difference reduces to extras position;
      # B1d Step 1 also adds an RE block (1 + n_re_groups) between betas and
      # extras in BOTH layouts. The reorder rule generalises:
      #   Legacy: [betas | re | extras | zi | oi]
      #   Spec  : [betas | re | zi | oi | extras]
      # i.e. every middle block keeps its position; only `extras` moves to the
      # tail. Slicing by known block sizes reconstructs the right ordering
      # without parsing the engine's ParamLayout struct.
      n_extra_per_family <- switch(model_type,
        binomial = 0L, beta_binomial = 1L, poisson_gamma = 1L,
        negbin_gamma = 2L, negbin_negbin = 2L, gamma_gamma = 2L,
        lognormal = 2L, 0L
      )
      p_total_design <- (hmc_data$p_num %||% 0L) +
        (if (model_type %in% ONE_PROCESS_FAMILIES) 0L else (hmc_data$p_denom %||% 0L))
      # RE block size on the engine's layout. With slopes, each group carries
      # n_coefs values (intercept + slope coefficients) and there is one
      # log_sigma per coefficient; without slopes the legacy single-term
      # branch is `1 + n_re_groups`.
      n_coefs_re_v <- if (has_re_slopes && n_re_terms_v > 0L) {
        as.integer(hmc_data$re_terms[[1]]$n_coefs)
      } else {
        1L
      }
      p_re_n <- if (n_re_groups_v > 0L) {
        if (has_re_slopes) {
          n_coefs_re_v + n_re_groups_v * n_coefs_re_v
        } else {
          1L + n_re_groups_v
        }
      } else {
        0L
      }
      p_spatial_n <- switch(spatial_type_v,
        icar = 1L + (spatial_info$n_units %||% 0L),
        bym2 = 2L + 2L * (spatial_info$n_units %||% 0L),
        0L
      )
      p_temporal_n <- if (temporal_type_v %in% TEMPORAL_TYPES_SINGLE) {
        hyper <- if (temporal_type_v == "ar1") 2L else 1L
        hyper + (temporal_info$n_temporal_params %||% 0L)
      } else 0L
      p_zi_n <- if (has_zi_block) ncol(zi_info$X_zi) else 0L
      p_oi_n <- if (has_oi_block) ncol(zi_info$X_oi) else 0L

      q_init_spec <- as.numeric(q_init)
      # Reorder rule (legacy q produced by initialize_hmc_params_full):
      #   Legacy: [betas | re | extras | spatial | temporal | zi | oi]
      #   Spec  : [betas | re | spatial | temporal | zi | oi | extras]
      # `extras` (phi/shape/sigma) sits BEFORE spatial in legacy and AT THE END
      # in the generic spec layout. Every other middle block keeps its slot.
      need_reorder <- (n_extra_per_family > 0L) &&
        (p_re_n > 0L || p_spatial_n > 0L || p_temporal_n > 0L ||
         has_zi_block || has_oi_block)
      if (need_reorder) {
        s <- 1L
        beta_seg <- q_init_spec[s:(s + p_total_design - 1L)]; s <- s + p_total_design
        re_seg <- if (p_re_n > 0L) {
          v <- q_init_spec[s:(s + p_re_n - 1L)]; s <- s + p_re_n; v
        } else numeric(0)
        extras_seg <- if (n_extra_per_family > 0L) {
          v <- q_init_spec[s:(s + n_extra_per_family - 1L)]; s <- s + n_extra_per_family; v
        } else numeric(0)
        spatial_seg <- if (p_spatial_n > 0L) {
          v <- q_init_spec[s:(s + p_spatial_n - 1L)]; s <- s + p_spatial_n; v
        } else numeric(0)
        temporal_seg <- if (p_temporal_n > 0L) {
          v <- q_init_spec[s:(s + p_temporal_n - 1L)]; s <- s + p_temporal_n; v
        } else numeric(0)
        zi_seg <- if (p_zi_n > 0L) {
          v <- q_init_spec[s:(s + p_zi_n - 1L)]; s <- s + p_zi_n; v
        } else numeric(0)
        oi_seg <- if (p_oi_n > 0L) {
          v <- q_init_spec[s:(s + p_oi_n - 1L)]; s <- s + p_oi_n; v
        } else numeric(0)
        q_init_spec <- c(beta_seg, re_seg, spatial_seg, temporal_seg,
                         zi_seg, oi_seg, extras_seg)
      }

      # Bridge accepts a re_params list shaped like the legacy backend's;
      # for B1d-1 we pass enough for both intercept-only and intercept+slopes
      # single-grouping-factor RE. When RE is absent we pass an empty list so
      # the bridge skips the RE block.
      re_params_spec <- if (p_re_n > 0L) {
        out <- list(
          group           = as.integer(hmc_data$re_group),
          n_groups        = as.integer(n_re_groups_v),
          n_terms         = as.integer(n_re_terms_v),
          has_slopes      = isTRUE(has_re_slopes),
          has_correlated_slopes = FALSE,  # B1d-1 gates correlated out
          parameterization = as.integer(if (re_param == "centered") 0L else 1L)
        )
        if (isTRUE(has_re_slopes)) {
          slope_mat <- hmc_data$slope_matrices[[1]]
          out$n_coefs       <- as.integer(n_coefs_re_v)
          out$slope_matrix  <- as.matrix(slope_mat)
        }
        out
      } else list()

      # Bridge accepts a spatial_params list mirroring the legacy backend.
      # ICAR / BYM2 (non-collapsed) only — both require adjacency + Q_inv/L_Q
      # (precomputed in R). Empty list means "no spatial".
      spatial_params_spec <- if (p_spatial_n > 0L) {
        list(
          type             = spatial_type_v,
          group            = as.integer(spatial_info$group),
          n_units          = as.integer(spatial_info$n_units),
          adj_row_ptr      = as.integer(spatial_info$adj_row_ptr),
          adj_col_idx      = as.integer(spatial_info$adj_col_idx),
          n_neighbors      = as.integer(spatial_info$n_neighbors),
          bym2_scale       = spatial_info$bym2_scale %||% 1.0,
          Q_inv            = spatial_info$Q_inv,
          L_Q              = spatial_info$L_Q,
          parameterization = spatial_param_v
        )
      } else list()

      # Bridge accepts a temporal_params list. RW1/RW2/AR1 only on the spec
      # path; GP / multiscale / cyclic still fall back to legacy (gated above).
      temporal_params_spec <- if (p_temporal_n > 0L) {
        list(
          type      = temporal_type_v,
          time_idx  = as.integer(temporal_info$time_index),
          group_idx = as.integer(temporal_info$group_index),
          n_times   = as.integer(temporal_info$n_times),
          n_groups  = as.integer(temporal_info$n_groups),
          n_params  = as.integer(temporal_info$n_temporal_params),
          cyclic    = isTRUE(temporal_info$precision_structure$cyclic),
          shared    = isTRUE(temporal_info$shared %||% TRUE),
          tau_shape = tau_temporal_shape,
          tau_rate  = tau_temporal_rate,
          rho_prior_a = rho_temporal_ab[["a"]],
          rho_prior_b = rho_temporal_ab[["b"]]
        )
      } else list()

      fit_raw <- cpp_tulpaRatio_run_nuts_specs(
        y_num             = as.integer(hmc_data$y_num),
        y_denom           = as.integer(hmc_data$y_denom),
        y_num_cont        = as.numeric(hmc_data$y_num_cont),
        y_denom_cont      = as.numeric(hmc_data$y_denom_cont),
        X_list            = X_list,
        cfg_list          = cfg,
        init              = q_init_spec,
        n_iter            = as.integer(iter),
        n_warmup          = as.integer(warmup),
        max_treedepth     = as.integer(max_treedepth %||% 10L),
        adapt_delta       = if (adapt_delta_value < 0) 0.8 else adapt_delta_value,
        seed              = as.integer(seed),
        verbose           = isTRUE(verbose),
        gradient_mode     = if (identical(gradient_mode, "auto")) "A_r" else gradient_mode,
        sigma_beta        = sigma_beta,
        phi_prior_shape   = phi_shape,
        phi_prior_rate    = phi_rate,
        sigma_prior_scale = sigma_re_scale,
        re_params         = re_params_spec,
        spatial_params    = spatial_params_spec,
        temporal_params   = temporal_params_spec
      )

      # Reorder samples columns from tulpa's generic layout
      # ([betas | re | spatial | temporal | zi | oi | extras]) back to legacy
      # ordering ([betas | re | extras | spatial | temporal | zi | oi]) so
      # hmc_param_layout() reads each block from the slot it expects.
      if (need_reorder) {
        S <- fit_raw$samples
        n_total <- ncol(S)
        off <- 0L
        beta_cols <- if (p_total_design > 0L) (off + 1L):(off + p_total_design) else integer(0); off <- off + p_total_design
        re_cols   <- if (p_re_n > 0L)         (off + 1L):(off + p_re_n)         else integer(0); off <- off + p_re_n
        sp_cols   <- if (p_spatial_n > 0L)    (off + 1L):(off + p_spatial_n)    else integer(0); off <- off + p_spatial_n
        t_cols    <- if (p_temporal_n > 0L)   (off + 1L):(off + p_temporal_n)   else integer(0); off <- off + p_temporal_n
        zi_cols   <- if (p_zi_n > 0L)         (off + 1L):(off + p_zi_n)         else integer(0); off <- off + p_zi_n
        oi_cols   <- if (p_oi_n > 0L)         (off + 1L):(off + p_oi_n)         else integer(0); off <- off + p_oi_n
        ex_cols   <- if (n_extra_per_family > 0L) (off + 1L):n_total            else integer(0)
        new_order <- c(beta_cols, re_cols, ex_cols,
                       sp_cols, t_cols, zi_cols, oi_cols)
        fit_raw$samples <- S[, new_order, drop = FALSE]
      }
    } else {
      fit_raw <- cpp_hmc_fit(
        q_init = q_init,
        y_num = as.integer(hmc_data$y_num),
        y_denom = as.integer(hmc_data$y_denom),
        y_num_cont = hmc_data$y_num_cont,
        y_denom_cont = hmc_data$y_denom_cont,
        X_num = hmc_data$X_num,
        X_denom = hmc_data$X_denom,
        family = family_params(family, model_type),
        re_params = re_params,
        spatial_params = spatial_params,
        temporal_params = temporal_params,
        prior_params = prior_params,
        zi_params = zi_params,
        latent_params = latent_params,
        st_params = st_params,
        tvc_params = tvc_params,
        svc_params = svc_params,
        n_iter = as.integer(iter),
        n_warmup = as.integer(warmup),
        L = as.integer(L),
        n_chains = as.integer(chains),
        seed = as.integer(seed),
        n_threads = n_threads_within,
        verbose = verbose,
        gradient_mode_str = gradient_mode,
        max_treedepth = as.integer(max_treedepth),
        metric_str = metric,
        adapt_delta = adapt_delta_value,
        walnuts = isTRUE(walnuts),
        n_cores = as.integer(concurrent_chains)
      )
    }
  }

  # Divergences bias the draws whether or not progress is printed
  if (L == 0) {
    n_div <- fit_raw$n_divergent %||%
      sum(unlist(lapply(fit_raw$divergent, function(x) sum(x))))
    if (n_div > 0) {
      warning(warningCondition(
        paste0(n_div, " divergent transition(s) after warmup; raise ",
               "`adapt_delta` or reparameterise."),
        class = "tulpaRatio_divergence"))
    }
  }

  # Convert to ratiod_fit
  fit <- convert_hmc_to_ratiod_fit_full(
    fit_raw = fit_raw,
    hmc_data = hmc_data,
    spatial_info = spatial_info,
    temporal_info = temporal_info,
    spatiotemporal_info = spatiotemporal_info,
    zi_info = zi_info,
    latent_info = latent_info,
    ms_temporal_info = ms_temporal_info,
    tvc_spec = if (inherits(temporal, "ratiod_tvc")) temporal else NULL,
    formula = formula,
    data = data,
    family = family,
    model_type = model_type,
    iter = iter,
    warmup = warmup,
    chains = chains,
    re_param = re_param
  )
  # The validated GP specification: coordinates, their scaling and the
  # neighbour structure prediction at new locations reads.
  fit$.internal$spatial_spec <- gp_info$spec

  return(fit)
}


#' Get HMC model type string
#' @keywords internal
get_hmc_model_type <- function(family) {
  num <- family$numerator$base_distribution %||% family$numerator$distribution
  num <- switch(num,
    negbin = , negative_binomial = "neg_binomial_2",
    zero_inflated_binomial = , hurdle_binomial = , one_inflated_binomial = ,
    zero_one_inflated_binomial = "binomial",
    num
  )
  den <- family$denominator$distribution
  key <- paste(num, den)
  type <- hmc_model_types[[key]]
  if (is.null(type)) {
    stop(sprintf("No sampler fits a %s numerator with a %s denominator.",
                 num, den), call. = FALSE)
  }
  type
}

# Numerator and denominator distribution pairs the samplers fit, each with
# its own likelihood.
hmc_model_types <- list(
  "binomial fixed" = "binomial",
  "beta_binomial fixed" = "beta_binomial",
  "poisson gamma" = "poisson_gamma",
  "poisson neg_binomial_2" = "poisson_negbin",
  "poisson fixed" = "poisson_fixed",
  "neg_binomial_2 neg_binomial_2" = "negbin_negbin",
  "neg_binomial_2 gamma" = "negbin_gamma",
  "neg_binomial_2 fixed" = "negbin_fixed",
  "gamma gamma" = "gamma_gamma",
  "lognormal lognormal" = "lognormal",
  "lognormal fixed" = "lognormal_fixed"
)


#' What the numerator and denominator responses of each model type are
#'
#' `count` an integer response, `positive` a positive continuous one,
#' `trials` the trial counts of a binomial numerator, `exposure` a known
#' positive exposure that scales the numerator mean. The data preparation and
#' the response transforms read this table.
#'
#' @keywords internal
MODEL_TYPE_RESPONSES <- list(
  binomial        = c(num = "count",    denom = "trials"),
  beta_binomial   = c(num = "count",    denom = "trials"),
  negbin_negbin   = c(num = "count",    denom = "count"),
  poisson_negbin  = c(num = "count",    denom = "count"),
  poisson_gamma   = c(num = "count",    denom = "positive"),
  negbin_gamma    = c(num = "count",    denom = "positive"),
  gamma_gamma     = c(num = "positive", denom = "positive"),
  lognormal       = c(num = "positive", denom = "positive"),
  poisson_fixed   = c(num = "count",    denom = "exposure"),
  negbin_fixed    = c(num = "count",    denom = "exposure"),
  lognormal_fixed = c(num = "positive", denom = "exposure")
)


#' Prepare data for HMC sampler
#' @keywords internal
prepare_hmc_data <- function(formula, data, family, model_type) {
  # Extract responses
  y_num <- formula$numerator$response
  y_denom <- formula$denominator$response

  # Design matrices
  X_num <- formula$numerator$X
  X_denom <- formula$denominator$X

  # A denominator that is data (trials or a known exposure) has no
  # coefficients
  responses <- MODEL_TYPE_RESPONSES[[model_type]]
  if (responses[["denom"]] %in% c("trials", "exposure")) {
    X_denom <- matrix(numeric(0), nrow = nrow(data), ncol = 0)
  } else if (is.null(X_denom) || ncol(X_denom) == 0) {
    X_denom <- matrix(1, nrow = nrow(data), ncol = 1)
    colnames(X_denom) <- "(Intercept)_denom"
  }

  # Random effects
  re_info <- extract_re_for_hmc(formula)

  # Responses, by what each one is for this model type. The sampler reads
  # integer responses from y_num / y_denom and continuous ones from
  # y_num_cont / y_denom_cont; the slot a type does not read holds a
  # placeholder.
  N_obs <- length(y_num)
  positive <- function(y, what) {
    y <- as.numeric(y)
    n_bad <- sum(y <= 0)
    if (n_bad > 0) {
      warning(sprintf("%d %s value(s) <= 0 (the density needs y > 0). Clamping to 1e-6.",
                      n_bad, what), call. = FALSE)
      y[y <= 0] <- 1e-6
    }
    y
  }

  if (identical(responses[["num"]], "count")) {
    y_num_cont <- rep(0.0, N_obs)
    y_num <- as.integer(y_num)
  } else {
    y_num_cont <- positive(y_num, "numerator")
    y_num <- rep(0L, N_obs)
  }

  denom_kind <- responses[["denom"]]
  if (denom_kind %in% c("count", "trials")) {
    y_denom <- as.integer(y_denom)
    y_denom_cont <- rep(0.0, N_obs)
  } else if (identical(denom_kind, "exposure")) {
    y_denom_cont <- as.numeric(y_denom)
    if (anyNA(y_denom_cont) || any(y_denom_cont <= 0)) {
      stop("A known exposure must be positive; it enters the numerator mean ",
           "as a factor.", call. = FALSE)
    }
    y_denom <- rep(1L, N_obs)
  } else {
    y_denom_cont <- positive(y_denom, "denominator")
    y_denom <- rep(1L, N_obs)
  }

  # Build slope design matrices for random effects with slopes
  # These are pre-computed during formula parsing with proper expansion
  slope_matrices <- NULL
  if (re_info$has_slopes && re_info$n_re_terms > 0) {
    slope_matrices <- vector("list", re_info$n_re_terms)
    for (t in seq_len(re_info$n_re_terms)) {
      term <- re_info$re_terms[[t]]
      if (length(term$slope_vars) > 0) {
        # Use the pre-computed slope_matrix from formula parsing
        # This handles interactions (x*z), polynomials (poly(x,2)), etc.
        raw_re <- formula$numerator$random_effects[[t]]
        if (!is.null(raw_re$slope_matrix)) {
          slope_matrices[[t]] <- raw_re$slope_matrix
        } else {
          # Fallback: build design matrix manually (for backwards compatibility)
          slope_mat <- matrix(0, nrow = length(y_num), ncol = length(term$slope_vars))
          for (s in seq_along(term$slope_vars)) {
            var_spec <- term$slope_vars[s]
            if (!var_spec %in% names(data)) {
              stop(sprintf("Slope variable '%s' not found in data", var_spec),
                   call. = FALSE)
            }
            slope_mat[, s] <- as.numeric(data[[var_spec]])
          }
          colnames(slope_mat) <- term$slope_vars
          slope_matrices[[t]] <- slope_mat
        }
      } else {
        slope_matrices[[t]] <- NULL
      }
    }
  }

  list(
    y_num = y_num,
    y_denom = y_denom,
    y_num_cont = y_num_cont,
    y_denom_cont = y_denom_cont,
    X_num = X_num,
    X_denom = X_denom,
    # Legacy single-term RE fields (for backwards compatibility)
    re_group = re_info$group_idx,
    n_re_groups = re_info$n_groups,
    re_var = re_info$group_var,
    # New multi-term RE fields
    n_re_terms = re_info$n_re_terms,
    re_terms = re_info$re_terms,
    re_group_matrix = re_info$group_idx_matrix,
    total_re_groups = re_info$total_groups %||% re_info$n_groups,
    total_re_params = re_info$total_re_params %||% re_info$total_groups,
    total_sigma_params = re_info$total_sigma_params %||% re_info$n_re_terms,
    total_chol_params = re_info$total_chol_params %||% 0L,
    has_slopes = re_info$has_slopes %||% FALSE,
    has_correlated_slopes = re_info$has_correlated_slopes %||% FALSE,
    slope_matrices = slope_matrices,
    N = length(y_num),
    p_num = ncol(X_num),
    p_denom = ncol(X_denom)
  )
}


#' Extract RE info for HMC
#'
#' @description
#' Extracts random effects information for the HMC backend.
#' Supports:
#' - Multiple crossed random effects: `(1|site) + (1|year)`
#' - Correlated random slopes: `(1 + x | group)` - estimates intercept-slope correlation
#' - Uncorrelated random slopes: `(1 + x || group)` - no correlation estimated
#'
#' For correlated slopes, the covariance matrix is parameterized via Cholesky decomposition:
#' `Sigma = diag(sigma) * L * L' * diag(sigma)` where L is lower-triangular with unit diagonal
#' (LKJ correlation prior).
#'
#' @param formula A ratiod_formula object
#' @return List with RE structure for HMC backend
#' @keywords internal
extract_re_for_hmc <- function(formula) {
  re_terms <- formula$numerator$random_effects
  n_obs <- length(formula$numerator$response)

  if (is.null(re_terms) || length(re_terms) == 0) {
    return(list(
      # Legacy single-term fields (for backwards compatibility)
      group_idx = rep(0L, n_obs),
      n_groups = 0L,
      group_var = NULL,
      # New multi-term fields
      n_re_terms = 0L,
      re_terms = list(),
      has_slopes = FALSE,
      has_correlated_slopes = FALSE,
      total_re_params = 0L,
      total_sigma_params = 0L,
      total_chol_params = 0L
    ))
  }

  n_re_terms <- length(re_terms)

  # Check for slopes and correlation structure
  has_slopes <- FALSE
  has_correlated_slopes <- FALSE
  for (term in re_terms) {
    if (length(term$slope_vars) > 0) {
      has_slopes <- TRUE
      if (isTRUE(term$correlated)) {
        has_correlated_slopes <- TRUE
      }
    }
  }

  # Process RE terms with full slope support
  re_terms_processed <- vector("list", n_re_terms)
  total_groups <- 0L
  total_re_params <- 0L  # Total RE parameters including slopes
  total_sigma_params <- 0L  # Total variance parameters (one per coefficient type)
  total_chol_params <- 0L  # Total Cholesky lower-triangular parameters (for correlations)

  for (t in seq_len(n_re_terms)) {
    term <- re_terms[[t]]
    n_groups_t <- as.integer(term$n_groups)

    # Number of RE coefficients per group: intercept + slopes (expanded)
    # Use the expanded slope_vars which handles interactions (x*z -> x, z, x:z)
    n_slopes <- if (!is.null(term$slope_vars)) length(term$slope_vars) else 0L
    n_coefs <- (if (isTRUE(term$has_intercept)) 1L else 0L) + n_slopes

    # Total parameters for this term: n_groups * n_coefs
    n_params_t <- n_groups_t * n_coefs

    # Number of Cholesky parameters for correlated slopes
    # For a k×k correlation matrix, we need k*(k-1)/2 off-diagonal elements
    # (diagonal is implicitly 1 for correlation matrix parameterization)
    n_chol_t <- if (n_coefs > 1 && isTRUE(term$correlated)) {
      as.integer(n_coefs * (n_coefs - 1) / 2)
    } else {
      0L
    }

    # Number of sigma parameters for this term
    n_sigma_t <- n_coefs

    re_terms_processed[[t]] <- list(
      group_var = term$group_var,
      group_idx = as.integer(term$group),
      n_groups = n_groups_t,
      has_intercept = isTRUE(term$has_intercept),
      slope_vars = term$slope_vars,  # Expanded slope names from model.matrix
      slope_vars_clean = term$slope_vars_clean,  # Cleaned names for parameter output
      correlated = isTRUE(term$correlated),
      n_coefs = n_coefs,  # Coefficients per group (1 for intercept-only, more for slopes)
      n_sigma = n_sigma_t,  # Number of sigma parameters
      n_chol = n_chol_t,  # Number of Cholesky parameters (0 if uncorrelated)
      re_offset = total_re_params,  # Offset in flattened RE parameter vector
      sigma_offset = total_sigma_params,  # Offset for variance parameters
      chol_offset = total_chol_params  # Offset for Cholesky parameters
    )

    total_groups <- total_groups + n_groups_t
    total_re_params <- total_re_params + n_params_t
    total_sigma_params <- total_sigma_params + n_sigma_t
    total_chol_params <- total_chol_params + n_chol_t
  }

  # Build group index matrix: [n_obs, n_re_terms]
  group_idx_matrix <- matrix(0L, nrow = n_obs, ncol = n_re_terms)
  for (t in seq_len(n_re_terms)) {
    group_idx_matrix[, t] <- as.integer(re_terms[[t]]$group)
  }

  list(
    # Legacy single-term fields (use first term for backwards compatibility)
    group_idx = as.integer(re_terms[[1]]$group),
    n_groups = as.integer(re_terms[[1]]$n_groups),
    group_var = re_terms[[1]]$group_var,
    # New multi-term fields
    n_re_terms = n_re_terms,
    re_terms = re_terms_processed,
    group_idx_matrix = group_idx_matrix,
    total_groups = total_groups,
    total_re_params = total_re_params,  # Total RE effects (intercepts + slopes)
    total_sigma_params = total_sigma_params,  # Total variance parameters
    total_chol_params = total_chol_params,  # Total Cholesky correlation parameters
    has_slopes = has_slopes,
    has_correlated_slopes = has_correlated_slopes
  )
}


#' Prepare spatial structure for HMC
#' @keywords internal
prepare_spatial_for_hmc <- function(spatial, data, N) {
  if (is.null(spatial)) {
    return(list(
      type = "none",
      group = rep(0L, N),
      n_units = 0L,
      adj_row_ptr = integer(1),
      adj_col_idx = integer(0),
      n_neighbors = integer(0),
      bym2_scale = 1.0
    ))
  }

  # Extract spatial type - check $type attribute or class
  spatial_type <- if (inherits(spatial, "ratiod_gp")) {
    "gp"
  } else if (inherits(spatial, "ratiod_svc")) {
    "svc"
  } else if (inherits(spatial, "ratiod_spatial_bym2") ||
             (!is.null(spatial$type) && spatial$type == "bym2")) {
    "bym2"
  } else if (!is.null(spatial$type) && spatial$type == "car_proper") {
    "car_proper"
  } else if (inherits(spatial, "ratiod_spatial_car") ||
             inherits(spatial, "ratiod_spatial_icar") ||
             (!is.null(spatial$type) && spatial$type %in% c("car", "car_proper"))) {
    "icar"
  } else {
    stop("Unknown spatial structure type")
  }

  # Get group mapping
  if (!is.null(spatial$group_var)) {
    spatial_group <- as.integer(factor(data[[spatial$group_var]]))
    n_units <- max(spatial_group)
  } else {
    # Observation-level spatial
    spatial_group <- seq_len(N)
    n_units <- N
  }

  # Convert adjacency matrix to CSR format
  adj <- spatial$adjacency
  if (inherits(adj, "Matrix")) {
    adj <- as.matrix(adj)
  }

  # Build CSR structure
  n_neighbors <- integer(n_units)
  adj_col_idx <- integer(0)
  adj_row_ptr <- integer(n_units + 1)

  for (i in seq_len(n_units)) {
    neighbors <- which(adj[i, ] > 0)
    n_neighbors[i] <- length(neighbors)
    adj_col_idx <- c(adj_col_idx, neighbors - 1L)  # Convert to 0-based for C++
    adj_row_ptr[i + 1] <- adj_row_ptr[i] + n_neighbors[i]
  }

  # BYM2 scale factor: the one the specification carries, which every
  # backend multiplies into the structured component
  bym2_scale <- 1.0
  if (spatial_type == "bym2") {
    bym2_scale <- spatial$scale_factor %||% compute_bym2_scale(adj)
  }

  # Precision mass matrix: compute Q_inv and L_Q for ICAR/BYM2
  # This provides the known correlation structure to the HMC mass matrix,
  # avoiding under-determined OAS estimation for 50+ spatial params.
  Q_inv_flat <- NULL
  L_Q_flat <- NULL
  if (spatial_type %in% c("icar", "bym2", "car", "car_proper") && n_units >= 5) {
    mass <- ridged_precision_factors(diag(n_neighbors) - adj)
    Q_inv_flat <- mass$Q_inv
    L_Q_flat <- mass$L
  }

  rho_bounds <- spatial$rho_bounds
  list(
    type = spatial_type,
    group = spatial_group,
    n_units = n_units,
    adj_row_ptr = adj_row_ptr,
    adj_col_idx = adj_col_idx,
    n_neighbors = n_neighbors,
    bym2_scale = bym2_scale,
    # A disconnected graph: the sum-to-zero blocks and the per-unit ICAR
    # precision; NULL on a connected one
    component = if (spatial_type %in% c("icar", "bym2", "car")) spatial$component,
    node_prec = if (spatial_type %in% c("icar", "bym2", "car")) spatial$node_prec,
    group_var = spatial$group_var,  # Preserve for prediction lookup
    Q_inv = Q_inv_flat,
    L_Q = L_Q_flat,
    parameterization = spatial$parameterization %||% "standard",
    rho_lower = if (spatial_type == "car_proper") unname(rho_bounds["lower"]) %||% 0.0 else NULL,
    rho_upper = if (spatial_type == "car_proper") unname(rho_bounds["upper"]) %||% 1.0 else NULL,
    rho_prior_a = 1.0,
    rho_prior_b = 1.0,
    center = spatial$center %||% TRUE,
    rsr_basis = rsr_basis(spatial, data)
  )
}


#' Prepare temporal structure for HMC
#' @keywords internal
prepare_temporal_for_hmc <- function(temporal, data, N) {
  if (is.null(temporal)) {
    return(list(
      type = "none",
      time_index = rep(0L, N),
      group_index = rep(0L, N),
      n_times = 0L,
      n_groups = 0L,
      n_temporal_params = 0L,
      precision_diag = numeric(0),
      precision_offdiag = numeric(0),
      precision_structure = list(cyclic = FALSE),
      shared = TRUE,
      # TVC fields (not used for regular temporal)
      n_tvc = 0L,
      structure = "none"
    ))
  }

  # Temporal has already been validated by tratio()
  # Check if it's a TVC specification
  if (inherits(temporal, "ratiod_tvc")) {
    # TVC-specific fields
    return(list(
      type = "tvc",
      time_index = temporal$time_index,
      group_index = temporal$group_index,
      n_times = temporal$n_times,
      n_groups = temporal$n_groups,
      n_temporal_params = temporal$n_temporal_params,
      precision_structure = NULL,
      shared = temporal$shared,
      # TVC-specific
      n_tvc = temporal$n_tvc,
      structure = temporal$structure,
      tvc_indices = temporal$tvc_indices,
      tvc_names = temporal$tvc_names
    ))
  }

  # Check if it's a temporal GP specification
  if (inherits(temporal, "ratiod_temporal_gp")) {
    return(list(
      type = "gp",
      time_index = temporal$time_index,  # Maps obs to unique time (1-based)
      group_index = temporal$group_index,
      n_times = temporal$n_times,
      n_groups = temporal$n_groups,
      n_temporal_params = temporal$n_times * temporal$n_groups,  # One effect per unique time
      precision_structure = list(cyclic = FALSE),
      shared = temporal$shared,
      # GP-specific fields
      time_values = temporal$unique_time_values,  # Unique time values (length n_times)
      cov_type = temporal$cov,
      nu = temporal$nu %||% 1.5,
      period = temporal$period %||% 1.0,
      parameterization = temporal$parameterization %||% "noncentered",
      phi_prior_lower = temporal$phi_prior_lower %||% 0.01,
      phi_prior_upper = temporal$phi_prior_upper %||% 10.0,
      # TVC fields (not used)
      n_tvc = 0L,
      structure = "gp"
    ))
  }

  # Check if it's multiscale temporal
  if (inherits(temporal, "ratiod_temporal_multiscale")) {
    return(list(
      type = "multiscale",
      time_index = temporal$time_index,
      group_index = temporal$group_index,
      n_times = temporal$n_times,
      n_groups = temporal$n_groups,
      n_temporal_params = temporal$n_temporal_params,
      precision_structure = temporal$precision_structure,
      shared = temporal$shared,
      # Multiscale-specific fields needed for parameter initialization
      trend = temporal$trend,
      seasonal = temporal$seasonal,
      short_term = temporal$short_term,
      parameterization = temporal$parameterization %||% "centered",
      components = temporal$components,
      # TVC fields (not used)
      n_tvc = 0L,
      structure = "multiscale"
    ))
  }

  # Regular temporal (RW1, RW2, AR1, etc.)
  list(
    type = temporal$type,
    time_index = temporal$time_index,
    group_index = temporal$group_index,
    n_times = temporal$n_times,
    n_groups = temporal$n_groups,
    n_temporal_params = temporal$n_temporal_params,
    precision_structure = temporal$precision_structure,
    shared = temporal$shared,
    rho_prior = temporal$rho_prior,
    parameterization = temporal$parameterization %||% "centered",
    # TVC fields (not used for regular temporal)
    n_tvc = 0L,
    structure = temporal$type  # Use type as structure for consistency
  )
}


#' Prepare TVC (Temporally-Varying Coefficients) for HMC
#' @keywords internal
prepare_tvc_for_hmc <- function(temporal, data, N, X_num) {
  # Check if temporal is a TVC specification
  if (is.null(temporal) || !inherits(temporal, "ratiod_tvc")) {
    return(list(
      has_tvc = FALSE,
      n_tvc = 0L,
      n_times = 0L,
      n_groups = 1L,
      structure = "none",
      shared = TRUE,
      cyclic = FALSE,
      tvc_indices = integer(0),
      time_index = integer(0),
      group_index = integer(0),
      X_tvc = numeric(0)
    ))
  }

  # TVC has already been validated by tratio()
  # Extract TVC-specific fields
  n_tvc <- temporal$n_tvc
  n_times <- temporal$n_times
  n_groups <- temporal$n_groups %||% 1L
  tvc_indices <- temporal$tvc_indices
  time_index <- temporal$time_index
  group_index <- temporal$group_index %||% rep(1L, N)

  # Extract the X_tvc design matrix subset (columns from X_num at tvc_indices)
  # Store as flat vector in row-major order for C++
  X_tvc <- as.numeric(t(X_num[, tvc_indices, drop = FALSE]))

  list(
    has_tvc = TRUE,
    n_tvc = as.integer(n_tvc),
    n_times = as.integer(n_times),
    n_groups = as.integer(n_groups),
    structure = temporal$structure %||% "rw1",
    shared = temporal$shared %||% TRUE,
    cyclic = temporal$cyclic %||% FALSE,
    tvc_indices = as.integer(tvc_indices),
    time_index = as.integer(time_index),
    group_index = as.integer(group_index),
    X_tvc = X_tvc
  )
}


#' Prepare SVC (Spatially-Varying Coefficients) for HMC
#' @keywords internal
prepare_svc_for_hmc <- function(svc, data, N, X_num) {
  # Check if spatial is a SVC specification
  if (is.null(svc) || !inherits(svc, "ratiod_svc")) {
    return(list(
      has_svc = FALSE,
      n_svc = 0L,
      n_obs = as.integer(N),
      nn = 0L,
      shared = TRUE,
      cov_type = "exponential",
      coords = numeric(0),
      svc_indices = integer(0),
      X_svc = numeric(0),
      nn_idx = integer(0),
      nn_dist = numeric(0),
      nn_order = integer(0),
      nn_order_inv = integer(0),
      sigma2_prior_scale = 1.0,
      phi_prior_lower = svc_default_range()[1],
      phi_prior_upper = svc_default_range()[2],
      tau_shape = 1.0,
      tau_rate = 0.01
    ))
  }

  # Validate SVC if not already done (coords_matrix will be NULL if not validated)
  if (is.null(svc$coords_matrix)) {
    svc <- validate_svc(svc, data, X_num)
  }

  # Extract SVC-specific fields
  n_svc <- svc$n_svc
  nn <- svc$nn
  svc_indices <- svc$svc_indices
  coords_matrix <- svc$coords_matrix  # N x 2 matrix
  neighbor_info <- svc$neighbor_info

  # Extract the X_svc design matrix subset (columns from X_num at svc_indices)
  # Store as flat vector in row-major order for C++
  X_svc <- as.numeric(t(X_num[, svc_indices, drop = FALSE]))

  # Flatten coords to row-major for C++ (x1, y1, x2, y2, ...)
  coords_flat <- as.numeric(t(coords_matrix))

  approx <- svc$approx %||% "nngp"
  # A spec built before `range` existed carries none; the default is the value
  # those fits already ran under.
  svc_range <- svc$range %||% svc_default_range()

  if (approx == "hsgp") {
    list(
      has_svc = TRUE,
      spec = svc,
      svc_approx = "hsgp",
      n_svc = as.integer(n_svc),
      n_obs = as.integer(N),
      nn = 0L,
      shared = svc$shared %||% TRUE,
      cov_type = "exponential",
      coords = coords_flat,
      svc_indices = as.integer(svc_indices),
      X_svc = X_svc,
      nn_idx = integer(0),
      nn_dist = numeric(0),
      nn_order = integer(0),
      nn_order_inv = integer(0),
      sigma2_prior_scale = 1.0,
      phi_prior_lower = svc_range[1],
      phi_prior_upper = svc_range[2],
      tau_shape = 1.0,
      tau_rate = 0.01,
      hsgp_m = as.integer(svc$m %||% 6L),
      hsgp_c = svc$c_boundary %||% 1.5
    )
  } else {
    # Flatten neighbor indices and distances (N x nn matrices)
    nn_idx_flat <- as.integer(t(neighbor_info$nn_idx))
    nn_dist_flat <- as.numeric(t(neighbor_info$nn_dist))

    list(
      has_svc = TRUE,
      spec = svc,
      svc_approx = "nngp",
      n_svc = as.integer(n_svc),
      n_obs = as.integer(N),
      nn = as.integer(nn),
      shared = svc$shared %||% TRUE,
      cov_type = svc$cov %||% "exponential",
      coords = coords_flat,
      svc_indices = as.integer(svc_indices),
      X_svc = X_svc,
      nn_idx = nn_idx_flat,
      nn_dist = nn_dist_flat,
      nn_order = as.integer(neighbor_info$nn_order - 1L),
      nn_order_inv = as.integer(neighbor_info$nn_order_inv - 1L),
      # A spec built before `parameterization` existed ran centred.
      parameterization = svc$parameterization %||% "centered",
      sigma2_prior_scale = 1.0,
      phi_prior_lower = svc_range[1],
      phi_prior_upper = svc_range[2],
      tau_shape = 1.0,
      tau_rate = 0.01,
      hsgp_m = 0L,
      hsgp_c = 0.0
    )
  }
}


#' Check if HMC backend can be used
#' @keywords internal
can_use_hmc_backend <- function(family) {
  # HMC works for all families

  TRUE
}


#' Get maximum number of threads available
#' @keywords internal
get_max_threads <- function() {
  cpp_get_max_threads()
}


#' Prepare zero-inflation structure for HMC
#'
#' Single source of truth for turning either an explicit `zi = ` argument
#' or a ZI/hurdle/OI/ZOIB *family* (e.g. `ratiod_zinegbin()`, with no
#' explicit `zi = `) into the `zi_info` list every backend consumes.
#' @param family Optional `ratiod_family` object, used for auto-detection
#'   when `zi` is NULL (see [family_zi_info()])
#' @keywords internal
prepare_zi_for_hmc <- function(zi, data, N, family = NULL) {
  if (is.null(zi) && !is.null(family) &&
      (isTRUE(family$zero_inflated) || isTRUE(family$one_inflated))) {
    return(family_zi_info(family, N))
  }

  if (is.null(zi)) {
    return(list(
      type = "none",
      X_zi = matrix(0, nrow = N, ncol = 1),
      p_zi = 1L,
      coef_names = NULL,
      X_oi = NULL,
      p_oi = 0L,
      coef_names_oi = NULL
    ))
  }

  # ZI is a ratiod_zi object with formula and type
  zi_type <- zi$type

  if (identical(zi_type, "oi_binomial")) {
    # OI-only: no ZI coefficient, only OI coefficient
    X_oi <- build_zi_design_matrix(zi$formula, data, N, "_oi")
    return(list(
      type = zi_type,
      X_zi = matrix(0, nrow = N, ncol = 1),  # Placeholder (not used)
      p_zi = 0L,
      coef_names = NULL,
      X_oi = X_oi,
      p_oi = ncol(X_oi),
      coef_names_oi = colnames(X_oi)
    ))
  }

  if (identical(zi_type, "zoib")) {
    # ZOIB: both ZI and OI coefficients, from separate formulas
    X_zi <- build_zi_design_matrix(zi$formula, data, N, "_zi")
    X_oi <- build_zi_design_matrix(zi$oi_formula, data, N, "_oi")
    return(list(
      type = zi_type,
      X_zi = X_zi,
      p_zi = ncol(X_zi),
      coef_names = colnames(X_zi),
      X_oi = X_oi,
      p_oi = ncol(X_oi),
      coef_names_oi = colnames(X_oi)
    ))
  }

  # zi_poisson / zi_negbin / hurdle_poisson / hurdle_negbin / zi_binomial / hurdle_binomial:
  # only a ZI coefficient
  X_zi <- build_zi_design_matrix(zi$formula, data, N, "_zi")
  list(
    type = zi_type,
    X_zi = X_zi,
    p_zi = ncol(X_zi),
    coef_names = colnames(X_zi),
    # OI info (not used for standard ZI, but needed for consistent interface)
    X_oi = NULL,
    p_oi = 0L,
    coef_names_oi = NULL
  )
}


#' Initialize parameters for HMC with full feature support
#' @keywords internal
initialize_hmc_params_full <- function(hmc_data, model_type, spatial_info,
                                        temporal_info, zi_info, latent_info = NULL,
                                        svc_info = NULL,
                                        spatiotemporal_info = NULL,
                                        links = family_links(NULL, model_type)) {
  # Start building parameter vector with data-driven initialization
  q_init <- numeric(0)
  idx <- 1

  # Fixed effects: each intercept starts at the link of its response's mean,
  # so the sampler does not begin at a mean of one when the data's is far
  # from it. Over trial counts that mean is the success proportion, over a
  # known exposure the rate.
  responses <- MODEL_TYPE_RESPONSES[[model_type]]
  start_mean <- function(y, scale = NULL) {
    keep <- is.finite(y) & y > 0
    if (!any(keep)) return(NA_real_)
    if (is.null(scale)) mean(y[keep]) else sum(y[keep]) / sum(scale[keep])
  }
  y_num_obs <- if (identical(responses[["num"]], "count")) hmc_data$y_num else hmc_data$y_num_cont
  num_mean <- switch(responses[["denom"]],
    trials = {
      p_hat <- sum(hmc_data$y_num) / max(1, sum(hmc_data$y_denom))
      max(0.01, min(0.99, p_hat))
    },
    exposure = start_mean(y_num_obs, hmc_data$y_denom_cont),
    start_mean(y_num_obs)
  )
  denom_mean <- switch(responses[["denom"]],
    count = start_mean(hmc_data$y_denom),
    positive = start_mean(hmc_data$y_denom_cont),
    NA_real_
  )

  p_num <- hmc_data$p_num
  if (p_num > 0) {
    beta_num_init <- rep(0.0, p_num)
    start <- pp_linkfun(links$num)(num_mean)
    if (is.finite(start)) beta_num_init[1] <- start
    q_init <- c(q_init, beta_num_init)
  }

  p_denom <- hmc_data$p_denom
  if (p_denom > 0) {
    beta_denom_init <- rep(0.0, p_denom)
    start <- pp_linkfun(links$denom %||% "log")(denom_mean)
    if (is.finite(start)) beta_denom_init[1] <- start
    q_init <- c(q_init, beta_denom_init)
  }

  # Random effects (supports multi-term RE with slopes and correlations)
  n_re_terms <- hmc_data$n_re_terms %||% 0L
  has_slopes <- hmc_data$has_slopes %||% FALSE
  has_correlated_slopes <- hmc_data$has_correlated_slopes %||% FALSE

  if (n_re_terms > 0) {
    if (has_slopes) {
      q_init <- c(q_init, rep(0.0, hmc_data$total_sigma_params + hmc_data$total_re_params))
      if (has_correlated_slopes) {
        q_init <- c(q_init, rep(0.0, hmc_data$total_chol_params))
      }
    } else if (n_re_terms > 1) {
      q_init <- c(q_init, rep(0.0, n_re_terms + hmc_data$total_re_groups))
    } else if (hmc_data$n_re_groups > 0) {
      q_init <- c(q_init, rep(0.0, 1 + hmc_data$n_re_groups))
    }
  } else if (hmc_data$n_re_groups > 0) {
    # Single-term legacy path (used by GP interface where n_re_terms = 0)
    q_init <- c(q_init, rep(0.0, 1 + hmc_data$n_re_groups))
  }

  # Overdispersion / shape / sigma parameters, numerator then denominator, on
  # the log scale: a moderate overdispersion or shape of 2, a log-scale SD of
  # 0.5, a beta-binomial precision of 10
  dispersion <- MODEL_DISPERSION_NAMES[[model_type]]
  dispersion <- dispersion[intersect(c("num", "denom"), names(dispersion))]
  dispersion_start <- c(phi_num = log(2), phi_denom = log(2), shape = log(2),
                        shape_num = log(2), shape_denom = log(2),
                        sigma_num = log(0.5), sigma_denom = log(0.5),
                        phi = log(10))
  q_init <- c(q_init, unname(dispersion_start[dispersion]))

  # Spatial
  if (spatial_info$type == "icar") {
    icar_param <- spatial_info$parameterization %||% "standard"
    if (icar_param == "collapsed") {
      q_init <- c(q_init, rep(0.0, 1))  # log_tau only
    } else {
      q_init <- c(q_init, rep(0.0, 1 + spatial_info$n_units))
    }
  } else if (spatial_info$type == "car_proper") {
    q_init <- c(q_init, rep(0.0, 2 + spatial_info$n_units))  # log_tau + logit_rho + phi
  } else if (spatial_info$type == "bym2") {
    bym2_param <- spatial_info$parameterization %||% "standard"
    if (bym2_param == "collapsed") {
      q_init <- c(q_init, rep(0.0, 2))  # log_sigma_total, logit_rho only
    } else {
      q_init <- c(q_init, rep(0.0, 2 + 2 * spatial_info$n_units))
    }
  } else if (spatial_info$type == "gp") {
    # Layout: log_sigma2, log_phi, w[n_units] (or just log_sigma2, log_phi for collapsed)
    gp_param <- spatial_info$parameterization %||% "centered"
    if (gp_param == "collapsed") {
      # Collapsed: only hyperparams, GP effects marginalized
      q_init <- c(q_init, rep(0.0, 2))
    } else {
      q_init <- c(q_init, rep(0.0, 2 + spatial_info$n_units))
    }
  } else if (spatial_info$type == "multiscale_gp") {
    # Initialize phi/lengthscale to geometric mean of range bounds (on log scale)
    range_local_lower <- spatial_info$range_local_lower %||% 0.01
    range_local_upper <- spatial_info$range_local_upper %||% 1.0
    range_regional_lower <- spatial_info$range_regional_lower %||% 1.0
    range_regional_upper <- spatial_info$range_regional_upper %||% 10.0
    log_phi_local_init <- log(sqrt(range_local_lower * range_local_upper))
    log_phi_regional_init <- log(sqrt(range_regional_lower * range_regional_upper))

    # m^2 basis coefficients per scale
    hsgp_m <- spatial_info$hsgp_m %||% 6L
    n_per_scale <- hsgp_m * hsgp_m

    # Layout: log_sigma2_local, log_ls_local, beta_local[n], log_sigma2_regional, log_ls_regional, beta_regional[n]
    q_init <- c(q_init,
                0.0, log_phi_local_init, rep(0.0, n_per_scale),
                0.0, log_phi_regional_init, rep(0.0, n_per_scale))
  } else if (spatial_info$type == "hsgp") {
    hsgp_m <- spatial_info$hsgp_m %||% 8L
    q_init <- c(q_init, rep(0.0, 2 + hsgp_m * hsgp_m))
  }

  # Temporal (regular temporal effects, not TVC or multiscale)
  # Note: multiscale temporal is handled by the GP sampler path which has its own param layout
  if (temporal_info$type != "none" && temporal_info$type != "tvc" &&
      temporal_info$type != "multiscale") {
    if (temporal_info$type == "gp") {
      # Temporal GP: log_sigma2 + log_phi + N effects
      n_temporal_params <- 2 + temporal_info$n_temporal_params
    } else {
      # RW1/RW2/AR1: log_tau + effects (+ rho for AR1)
      n_temporal_params <- 1 + temporal_info$n_temporal_params
      if (temporal_info$type == "ar1") {
        n_temporal_params <- n_temporal_params + 1
      }
    }
    q_init <- c(q_init, rep(0.0, n_temporal_params))
  }

  # Multiscale temporal: handled by GP sampler path
  # Layout: [log_sigma2_trend, trend[n_times], log_sigma2_seasonal, seasonal[period],
  #          log_sigma2_short, (logit_rho if AR1), short[n_times]]
  if (temporal_info$type == "multiscale") {
    n_times <- temporal_info$n_times %||% 0L
    trend_type <- temporal_info$trend %||% "none"
    seasonal_period <- temporal_info$seasonal %||% 0L
    short_type <- temporal_info$short_term %||% "none"

    n_ms_params <- 0L
    # Trend component
    if (trend_type != "none") {
      n_ms_params <- n_ms_params + 1 + n_times  # log_sigma2_trend + effects
    }
    # Seasonal component
    if (!is.null(seasonal_period) && seasonal_period >= 2) {
      n_ms_params <- n_ms_params + 1 + seasonal_period  # log_sigma2_seasonal + effects
    }
    # Short-term component
    if (short_type != "none") {
      n_ms_params <- n_ms_params + 1 + n_times  # log_sigma2_short + effects
      if (short_type == "ar1") {
        n_ms_params <- n_ms_params + 1  # logit_rho
      }
    }
    q_init <- c(q_init, rep(0.0, n_ms_params))
  }

  # TVC (Temporally-Varying Coefficients)
  if (!is.null(temporal_info$type) && temporal_info$type == "tvc") {
    n_tvc <- temporal_info$n_tvc %||% 0L
    n_times <- temporal_info$n_times %||% 0L
    n_groups <- temporal_info$n_groups %||% 1L
    structure <- temporal_info$structure %||% "rw1"

    if (n_tvc > 0) {
      n_tvc_params <- n_tvc
      if (structure == "ar1") {
        n_tvc_params <- n_tvc_params + 1
      }
      n_tvc_params <- n_tvc_params + n_times * n_tvc * n_groups
      q_init <- c(q_init, rep(0.0, n_tvc_params))
    }
  }

  # Zero-inflation
  if (zi_info$type != "none") {
    q_init <- c(q_init, rep(0.0, zi_info$p_zi))
    if (!is.null(zi_info$p_oi) && zi_info$p_oi > 0) {
      q_init <- c(q_init, rep(0.0, zi_info$p_oi))
    }
  }

  # Latent factors
  if (!is.null(latent_info) && latent_info$type != "none") {
    K <- latent_info$n_factors
    N <- latent_info$n_obs
    n_proc <- if (isTRUE(latent_info$shared)) 2L else 1L
    q_init <- c(q_init, rep(0.0, latent_n_loadings(n_proc, K) + N * K))
  }

  # Spatiotemporal interaction (Knorr-Held Type I-IV)
  if (!is.null(spatiotemporal_info) &&
      isTRUE(spatiotemporal_info$has_spatiotemporal)) {
    st_type <- spatiotemporal_info$type %||% "none"
    if (st_type != "none") {
      n_st_spatial <- spatiotemporal_info$n_spatial %||% 0L
      n_st_times <- spatiotemporal_info$n_times %||% 0L
      n_st_delta <- spatiotemporal_info$n_params %||% (n_st_spatial * n_st_times)

      # log_tau_st (precision for interaction)
      q_init <- c(q_init, 0.0)

      # tau_st2 removed — single tau for all ST types

      # logit_rho_st, where the interaction's time margin reads one
      if (st_has_rho(spatiotemporal_info)) {
        q_init <- c(q_init, 0.0)
      }

      # GP range parameters for separable/non-separable GP
      if (st_type %in% c("separable", "nonsep_gp")) {
        q_init <- c(q_init, 0.0, 0.0)  # log_phi_space, log_phi_time
      }

      # ST interaction effects delta[S*T], unless integrated out
      if (!identical(spatiotemporal_info$parameterization, "collapsed")) {
        q_init <- c(q_init, rep(0.0, n_st_delta))
      }
    }
  }

  # SVC (Spatially-Varying Coefficients)
  if (!is.null(svc_info) && isTRUE(svc_info$has_svc)) {
    n_svc <- svc_info$n_svc %||% 0L
    n_obs <- svc_info$n_obs %||% hmc_data$N
    if (n_svc > 0) {
      svc_approx <- svc_info$svc_approx %||% "nngp"
      if (svc_approx == "hsgp") {
        m_total <- as.integer(svc_info$hsgp_m)^2
        # log_sigma2_svc (n_svc) + log_lengthscale_svc (n_svc) + beta (n_svc * m^2)
        q_init <- c(q_init, rep(0.0, 2 * n_svc + n_svc * m_total))
      } else {
        # log_sigma2_svc (n_svc) + range_svc (n_svc: the logit of phi on its
        # prior interval, so 0 is the interval's midpoint) + svc_w (n_svc * n_obs)
        q_init <- c(q_init, rep(0.0, 2 * n_svc + n_svc * n_obs))
      }
    }
  }

  q_init
}


#' Convert HMC output to ratiod_fit (full feature support)
#' @keywords internal
convert_hmc_to_ratiod_fit_full <- function(fit_raw, hmc_data, spatial_info,
                                           temporal_info,
                                           spatiotemporal_info = NULL,
                                           zi_info,
                                           latent_info = NULL,
                                           ms_temporal_info = NULL,
                                           tvc_spec = NULL,
                                           formula, data, family,
                                           model_type, iter, warmup, chains,
                                           re_param = "noncentered") {
  # Handle multi-chain case
  if (chains > 1) {
    # Combine chains
    all_samples <- do.call(rbind, fit_raw$samples)
    all_log_prob <- unlist(fit_raw$log_prob)
    all_accept <- unlist(fit_raw$accept_prob)
    all_n_leapfrog <- unlist(fit_raw$n_leapfrog)
    all_treedepth <- unlist(fit_raw$treedepth)
    all_divergent <- unlist(fit_raw$divergent)
    all_energy <- unlist(fit_raw$energy)
    energy_chain <- rep(seq_along(fit_raw$log_prob), lengths(fit_raw$log_prob))
    epsilon <- mean(fit_raw$epsilon)
  } else {
    all_samples <- fit_raw$samples
    all_log_prob <- fit_raw$log_prob
    all_accept <- fit_raw$accept_prob
    all_n_leapfrog <- fit_raw$n_leapfrog
    all_treedepth <- fit_raw$treedepth
    all_divergent <- fit_raw$divergent
    all_energy <- fit_raw$energy
    energy_chain <- rep(1L, length(all_log_prob))
    epsilon <- fit_raw$epsilon
  }

  n_samples <- nrow(all_samples)

  # One layout, read by every consumer of the sample matrix
  layout <- hmc_param_layout(
    hmc_data = hmc_data, spatial_info = spatial_info,
    temporal_info = temporal_info, zi_info = zi_info,
    model_type = model_type, latent_info = latent_info,
    st_info = spatiotemporal_info, ms_temporal_info = ms_temporal_info
  )
  if (layout$total != ncol(all_samples)) {
    stop("Parameter layout describes ", layout$total, " parameters but the ",
         "sampler returned ", ncol(all_samples), ". The layout in ",
         "hmc_param_layout() and compute_param_layout() in ",
         "src/hmc_sampler.cpp have diverged.", call. = FALSE)
  }

  # Fields the inner Laplace optimization of a collapsed parameterization
  # leaves behind (NULL otherwise): one matrix per chain, stacked as the
  # samples are
  star <- function(name, chain = NULL) {
    x <- fit_raw[[name]]
    if (is.null(x) || chains == 1) return(x)
    if (is.null(chain)) do.call(rbind, x) else x[[chain]]
  }
  gp_w_star <- star("gp_w_star")
  icar_phi_star <- star("icar_phi_star")
  bym2_theta_star <- star("bym2_theta_star")
  st_delta_draws <- star("st_delta_draws")

  unpack <- function(samples, chain = NULL) {
    hmc_unpack_draws(
      samples = samples, layout = layout, hmc_data = hmc_data,
      spatial_info = spatial_info, temporal_info = temporal_info,
      model_type = model_type, re_param = re_param,
      latent_info = latent_info, st_info = spatiotemporal_info,
      ms_temporal_info = ms_temporal_info,
      gp_w_star = star("gp_w_star", chain),
      icar_phi_star = star("icar_phi_star", chain),
      bym2_theta_star = star("bym2_theta_star", chain),
      st_delta_draws = star("st_delta_draws", chain)
    )
  }

  unpacked <- unpack(all_samples)
  draws_list <- hmc_draws_list(unpacked)
  draws <- do.call(cbind, draws_list)
  colnames(draws) <- names(draws_list)

  design <- hmc_eta_design(
    X_num = hmc_data$X_num, X_denom = hmc_data$X_denom,
    re_group = hmc_data$re_group,
    re_group_matrix = hmc_data$re_group_matrix,
    slope_matrices = hmc_data$slope_matrices,
    spatial_info = spatial_info, temporal_info = temporal_info,
    latent_info = latent_info, st_info = spatiotemporal_info,
    ms_temporal_info = ms_temporal_info
  )
  eta <- hmc_eta_draws(unpacked, design)
  warn_dropped_structures(eta$dropped)

  ratio_draws <- hmc_response_draws(eta, family_links(family, model_type),
                                    exposure = hmc_exposure(hmc_data, model_type))$ratio
  colnames(ratio_draws) <- paste0("ratio[", seq_len(ncol(ratio_draws)), "]")

  # Diagnostics
  n_divergent <- sum(all_divergent)
  avg_accept <- mean(all_accept)

  # One matrix of named draws per chain
  if (chains > 1) {
    samples_list <- lapply(seq_len(chains), function(c) {
      chain_draws <- hmc_draws_list(unpack(fit_raw$samples[[c]], c))
      chain_mat <- do.call(cbind, chain_draws)
      colnames(chain_mat) <- names(chain_draws)
      chain_mat
    })
  } else {
    samples_list <- list(draws)
  }

  structure_draws <- hmc_structure_draws(unpacked)

  fit <- list(
    draws = draws,
    samples = samples_list,
    ratio_draws = ratio_draws,
    formula = formula,
    data = data,
    family = family,
    spatial = spatial_info,
    temporal = temporal_info,
    spatiotemporal = spatiotemporal_info,
    svc = if (!is.null(unpacked$svc)) spatial_info$svc_spec else NULL,
    tvc = if (!is.null(unpacked$tvc)) tvc_spec else NULL,
    zi = zi_info,
    latent = latent_info,
    backend = "hmc",
    algorithm = fit_raw$sampler %||% "HMC",
    re_param = re_param,
    iter = iter,
    warmup = warmup,
    chains = chains,
    n_save = n_samples,
    epsilon = epsilon,
    diagnostics = list(
      algorithm = fit_raw$sampler %||% "HMC",
      n_divergent = n_divergent,
      avg_accept_prob = avg_accept,
      n_leapfrog = all_n_leapfrog,
      treedepth = all_treedepth,
      divergent = all_divergent,
      log_posterior = all_log_prob,
      energy = all_energy,
      energy_chain = if (!is.null(all_energy)) energy_chain
    ),
    log_prob = all_log_prob,
    .internal = c(
      list(
        samples = all_samples,
        hmc_data = hmc_data,
        model_type = model_type,
        latent_info = latent_info,
        ms_temporal_info = ms_temporal_info,
        layout = layout,
        eta_dropped = eta$dropped,
        gp_w_star = gp_w_star,
        icar_phi_star = icar_phi_star,
        bym2_theta_star = bym2_theta_star,
        st_delta_draws = st_delta_draws
      ),
      structure_draws
    )
  )

  class(fit) <- "ratiod_fit"
  return(fit)
}


# =============================================================================
# GP Spatial Support (v0.9.4+)
# =============================================================================

#' Check if spatial specification is GP-based
#' @keywords internal
is_gp_spatial <- function(spatial) {
  if (is.null(spatial)) return(FALSE)
  inherits(spatial, "ratiod_gp") ||
    inherits(spatial, "ratiod_multiscale") ||
    inherits(spatial, "ratiod_hsgp")
}

#' Check if temporal specification is multiscale
#' @keywords internal
is_multiscale_temporal <- function(temporal) {
  if (is.null(temporal)) return(FALSE)
  inherits(temporal, "ratiod_temporal_multiscale")
}

#' Prepare GP spatial structure for HMC
#'
#' Computes nearest neighbor structure for NNGP approximation.
#'
#' @param gp A ratiod_gp or ratiod_multiscale spatial specification
#' @param data Data frame with coordinate columns
#' @param N Number of observations
#' @return List of GP parameters for cpp_hmc_fit_gp
#' @keywords internal
prepare_gp_for_hmc <- function(gp, data, N) {
  if (is.null(gp)) {
    return(list(
      gp_type = "none",
      coords = numeric(0),
      nn_idx = integer(0),
      nn_dist = numeric(0),
      nn_order = integer(0),
      nn_order_inv = integer(0),
      nn_neighbor_dist = numeric(0),  # Phase 1.3
      nn = 0L,
      range_local_lower = 0,
      range_local_upper = 1,
      range_regional_lower = 1,
      range_regional_upper = 10,
      cov_type = "exponential",
      nu = 1.5,
      shared = TRUE,
      sampler = "auto",
      gp_obs_to_loc = integer(0),
      n_unique = 0L
    ))
  }

  # Handle HSGP specially (no neighbor computation needed)
  if (inherits(gp, "ratiod_hsgp")) {
    validated <- validate_hsgp(gp, data)
    coords_mat <- validated$coords_matrix
    coords_flat <- as.vector(t(coords_mat))

    return(list(
      gp_type = "hsgp",
      spec = validated,
      coords = coords_flat,
      hsgp_m = as.integer(gp$m),
      hsgp_c = gp$c,
      shared = gp$shared,
      # Empty placeholders for standard GP fields
      nn_idx = integer(0),
      nn_dist = numeric(0),
      nn_order = integer(0),
      nn_order_inv = integer(0),
      nn_neighbor_dist = numeric(0),  # Phase 1.3
      nn = 0L,
      range_local_lower = 0,
      range_local_upper = 1,
      range_regional_lower = 1,
      range_regional_upper = 10,
      cov_type = "exponential",
      nu = 1.5,
      sampler = "auto",
      # Solver config (HSGP uses Cholesky internally, but keep consistent interface)
      gp_obs_to_loc = integer(0),
      n_unique = 0L
    ))
  }

  if (inherits(gp, "ratiod_multiscale")) {
    # Two HSGP evaluations over a shared basis. The basis is built from the
    # observation coordinates, so no neighbour structure is computed.
    validated_hsgp <- validate_hsgp(gp, data)
    coords_flat_hsgp <- as.vector(t(validated_hsgp$coords_matrix))

    return(list(
      gp_type = "multiscale_gp",
      spec = validated_hsgp,
      hsgp_m = as.integer(gp$m),
      hsgp_c = gp$c_boundary,
      coords = coords_flat_hsgp,
      # Range constraints (used for lengthscale prior means)
      range_local_lower = gp$range_local[1],
      range_local_upper = gp$range_local[2],
      range_regional_lower = gp$range_regional[1],
      range_regional_upper = gp$range_regional[2],
      # Common params
      cov_type = gp$cov %||% "exponential",
      nu = gp$nu %||% 1.5,
      shared = gp$shared,
      sampler = gp$sampler %||% "auto",
      parameterization = "noncentered",  # HSGP always NC
      # Empty NNGP placeholders
      nn_idx = integer(0), nn_dist = numeric(0),
      nn_order = integer(0), nn_order_inv = integer(0),
      nn_neighbor_dist = numeric(0), nn = 0L,
      gp_obs_to_loc = integer(0),
      n_unique = 0L
    ))
  }

  # Validate GP specification (computes neighbor structure)
  validated <- validate_gp(gp, data)

  # Extract unique coordinates (NNGP computed on unique locations only)
  coords_mat <- validated$unique_coords
  coords_flat <- as.vector(t(coords_mat))  # Row-major flatten

  nn_info <- validated$neighbor_info

  # Phase 1.3: Flatten nn_neighbor_dist from N x nn x nn for C++ row-major access
  # C++ accesses as: i * nn * nn + j1 * nn + j2 (0-indexed, i slowest, j2 fastest)
  # R stores column-major: arr[i, j1, j2] at index (j2-1)*N*nn + (j1-1)*N + i (1-indexed)
  # Use C++ helper for fast flattening
  nn <- validated$nn
  N_gp <- nrow(nn_info$nn_idx)
  nn_neighbor_dist_flat <- cpp_flatten_3d_rowmajor(
    as.vector(nn_info$nn_neighbor_dist), N_gp, nn, nn
  )

  list(
    gp_type = "gp",
    spec = validated,
    coords = coords_flat,
    nn_idx = as.integer(as.vector(t(nn_info$nn_idx))),
    nn_dist = as.vector(t(nn_info$nn_dist)),
    nn_order = as.integer(nn_info$nn_order),
    nn_order_inv = as.integer(nn_info$nn_order_inv),
    nn_neighbor_dist = nn_neighbor_dist_flat,  # Phase 1.3: cached pairwise distances
    nn = as.integer(validated$nn),
    # Multi-scale params (not used for single-scale)
    range_local_lower = 0,
    range_local_upper = 1,
    range_regional_lower = 1,
    range_regional_upper = 10,
    # Common params
    cov_type = gp$cov,
    nu = gp$nu,
    shared = gp$shared,
    parameterization = gp$parameterization %||% "centered",
    # Solver config
    # Observation-to-location mapping (1-based)
    gp_obs_to_loc = as.integer(validated$obs_to_loc),
    n_unique = validated$n_unique
  )
}

#' Prepare multiscale temporal structure for HMC
#' @keywords internal
prepare_multiscale_temporal_for_hmc <- function(temporal, data, N) {
  if (is.null(temporal) || !inherits(temporal, "ratiod_temporal_multiscale")) {
    return(list(
      ms_temporal_type = "none",
      ms_time_index = integer(N),
      ms_group_index = integer(N),
      ms_n_times = 0L,
      ms_n_groups = 1L,
      trend_type = "none",
      seasonal_period = 0L,
      short_term_type = "none",
      shared = TRUE
    ))
  }

  # Validate temporal specification
  validated <- validate_temporal_multiscale(temporal, data)

  list(
    ms_temporal_type = "multiscale",
    ms_time_index = as.integer(validated$time_index),
    ms_group_index = as.integer(validated$group_index),
    ms_n_times = as.integer(validated$n_times),
    ms_n_groups = as.integer(validated$n_groups),
    trend_type = temporal$trend,
    seasonal_period = as.integer(temporal$seasonal %||% 0L),
    short_term_type = temporal$short_term,
    shared = temporal$shared,
    parameterization = temporal$parameterization %||% "centered"
  )
}

#' RSR block of the sampler's data
#'
#' The restricted covariate basis `Q` (`N x r`, orthonormal columns) as the
#' sampler reads it: `basis` row-major, so observation `i`'s row is
#' `basis[i * r + (0:(r - 1))]`.
#'
#' @param spatial_info Output of `prepare_spatial_for_hmc()` or the GP branch
#' @keywords internal
hmc_rsr_params <- function(spatial_info) {
  Q <- spatial_info$rsr_basis
  if (is.null(Q)) {
    return(list(has_rsr = FALSE, basis = numeric(0), rank = 0L))
  }
  list(has_rsr = TRUE, basis = as.numeric(t(Q)), rank = ncol(Q))
}
