fixture_re_params <- function(n, group = rep(0L, n), n_groups = 0L, n_terms = 0L) {
  list(
    group = as.integer(group),
    n_groups = as.integer(n_groups),
    n_terms = as.integer(n_terms),
    group_matrix = matrix(0L, nrow = n, ncol = 1),
    n_groups_vec = as.integer(n_groups),
    has_slopes = FALSE,
    has_correlated_slopes = FALSE,
    n_coefs_vec = integer(0),
    correlated_vec = logical(0),
    n_chol_vec = integer(0),
    slope_matrices = list(),
    parameterization = 0L  # 0 = centered (default)
  )
}
fixture_spatial_params <- function(n, type = "none") {
  list(
    type = type,
    group = rep(0L, n),
    n_units = 0L,
    adj_row_ptr = 0L,
    adj_col_idx = integer(0),
    n_neighbors = integer(0),
    bym2_scale = 1.0
  )
}
fixture_temporal_params <- function(n, time_idx = rep(0L, n),
                                 type = if (missing(time_idx)) "none" else "rw1",
                                 n_times = 0L, cyclic = FALSE, group_idx = NULL,
                                 n_groups = if (type == "none") 0L else 1L) {
  if (is.null(group_idx)) group_idx <- rep(if (type == "none") 0L else 1L, n)
  list(
    type = type,
    time_idx = as.integer(time_idx),
    group_idx = as.integer(group_idx),
    n_times = as.integer(n_times),
    n_groups = as.integer(n_groups),
    n_params = as.integer(n_times * n_groups),
    cyclic = cyclic,
    shared = TRUE,
    tau_shape = 2.0,
    tau_rate = 0.5,
    # GP-specific fields (required even for non-GP types)
    time_values = numeric(0),
    cov_type = "exponential",
    nu = 1.5,
    period = 1.0,
    gp_sigma2_prior_U = 1.0,
    gp_sigma2_prior_alpha = 0.01,
    gp_phi_prior_lower = 0.01,
    gp_phi_prior_upper = 10.0
  )
}

fixture_prior_params <- function() {
  list(
    sigma_beta = 10.0,
    sigma_re_scale = 2.5,
    phi_shape = 2.0,
    phi_rate = 0.1,
    tau_spatial_shape = 1.0,
    tau_spatial_rate = 0.01
  )
}
fixture_zi_params <- function(n) {
  list(
    type = "none",
    X = matrix(0, nrow = n, ncol = 1),
    p_zi = 1L,
    prior_sd = 10.0,
    # OI fields (required even for non-OI types)
    X_oi = NULL,
    p_oi = 0L,
    oi_prior_sd = 10.0
  )
}
fixture_latent_params <- function() {
  list(
    has_latent = FALSE,
    n_factors = 0L,
    shared = FALSE,
    scale = TRUE,
    constraint = 0L,
    sigma_prior_rate = 0.0
  )
}
fixture_st_params <- function() {
  list(
    has_spatiotemporal = FALSE,
    type = "none",
    shared = TRUE,
    n_spatial = 0L,
    n_times = 0L,
    n_params = 0L,
    s_idx = integer(0),
    t_idx = integer(0),
    st_flat = integer(0),
    temporal_type = "rw1",
    temporal_cyclic = FALSE,
    adj_row_ptr = integer(0),
    adj_col_idx = integer(0),
    sigma2_prior_U = 1.0,
    sigma2_prior_alpha = 0.01
  )
}
fixture_tvc_params <- function() {
  list(
    has_tvc = FALSE,
    n_tvc = 0L,
    n_times = 0L,
    n_groups = 1L,
    structure = "rw1",
    time_idx = integer(0),
    group_idx = integer(0),
    X_tvc = matrix(0, nrow = 1, ncol = 1),
    sigma2_prior_U = 1.0,
    sigma2_prior_alpha = 0.01
  )
}
fixture_svc_params <- function() {
  list(
    has_svc = FALSE,
    n_svc = 0L,
    nn = 0L,
    shared = TRUE,
    cov_type = "exponential",
    spatial_idx = integer(0),
    X_svc = matrix(0, nrow = 1, ncol = 1),
    coords = matrix(0, nrow = 1, ncol = 2),
    sigma2_prior_U = 1.0,
    sigma2_prior_alpha = 0.01,
    phi_prior_shape = 3.0,
    phi_prior_rate = 1.0
  )
}
