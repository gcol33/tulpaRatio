#' @keywords internal
"_PACKAGE"

#' @useDynLib tulpaRatio, .registration = TRUE
#' @importFrom Rcpp sourceCpp
#' @importFrom stats as.formula model.frame model.matrix na.pass quantile sd optim fitted predict rnorm rnbinom var ar cor pnorm qnorm reshape qlogis coef vcov nobs formula
#' @importFrom utils head
#' @importFrom grDevices adjustcolor
#' @importFrom graphics abline lines polygon
NULL

# The diagnostic front door is tulpa's generic; tulpaRatio contributes the
# `ratiod_fit` method (see diagnostics.ratiod_fit). Re-exported so
# `library(tulpaRatio)` puts the same object on the search path rather than a
# same-named function that would mask the engine's.
#' @importFrom tulpa diagnostics
#' @export
tulpa::diagnostics

#' tulpaRatio: Bayesian Hierarchical Models for Ratios, Rates, and Proportions
#'
#' @description
#' An opinionated framework for Bayesian hierarchical modelling of ratios,
#' rates, and proportions. The core premise is non-negotiable:
#'
#' **Ratios are not data. They are derived quantities.**
#'
#' All inference is performed on the latent processes generating the numerator
#' and the denominator, never on their quotient. Ratios, rates, and proportions
#' are computed post hoc with full uncertainty propagation.
#'
#' @section Model families:
#' \describe{
#'   \item{ratiod_binomial()}{Trial-based ratios (successes/trials)}
#'   \item{ratiod_negbin_negbin()}{Two-process ratios (count/count)}
#'   \item{ratiod_poisson_gamma()}{Count/effort ratios (e.g., CPUE)}
#' }
#'
#' @section Key features:
#' \itemize{
#'   \item Shared random effects between numerator and denominator (default)
#'   \item Spatial structure via CAR/BYM2
#'   \item Full posterior inference on derived ratios
#'   \item Explicit rejection of offset-based approaches
#' }
#'
#' @name tulpaRatio-package
#' @aliases tulpaRatio-package
NULL

utils::globalVariables(c(
  ".",
  "value",
  "variable",
  ".data",
  "cpp_flatten_3d_rowmajor",
  "cpp_sample_crt",
  "cpp_sample_crt_sum",
  "cpp_crt_mean",
  "cpp_enum_names",
  "cpp_ess_fit",
  "cpp_gibbs_spatial",
  "cpp_gpu_available",
  "cpp_gpu_info",
  "cpp_hmc_fit",
  "cpp_get_max_threads",
  "cpp_kriging_predict",
  "cpp_pg_binomial_gibbs",
  "cpp_pg_get_max_threads",
  "cpp_pg_binomial_gibbs_spatial",
  "cpp_pg_binomial_gibbs_bym2",
  "cpp_pg_nngp_conditional_probe",
  "cpp_pg_nngp_log_density_probe",
  "cpp_pg_binomial_gibbs_gp",
  "cpp_pg_binomial_gibbs_temporal",
  "cpp_pg_binomial_gibbs_rsr",
  "cpp_pg_negbin_gibbs",
  "cpp_pg_negbin_negbin_gibbs",
  "cpp_pg_negbin_gibbs_spatial",
  "cpp_rpg1",
  "cpp_rpg",
  "cpp_sghmc_fit",
  "cpp_sgld_fit",
  "cpp_temporal_gp_field",
  "cpp_temporal_gp_log_prior",
  "cpp_chain_concurrency",
  "cpp_chain_visits",
  "cpp_set_max_threads",
  "cpp_within_chain_team",
  "cpp_num_procs",
  "cpp_gradient_fields",
  "cpp_gradient_dispatch",
  "cpp_gradient_check",
  "cpp_gradient_race",
  "cpp_time_gradient",
  "cpp_logpost_at",
  "cpp_st_gp_nc_coordinate_spread",
  "cpp_st_collapsed_exactness",
  "cpp_ms_temporal_nc_coordinate_spread",
  "cpp_polygamma",
  "cpp_test_negbin_h_kernels",
  "cpp_test_negbin_jet",
  "cpp_test_negbin_paths",
  "cpp_test_cholesky",
  "cpp_test_inverse_logit",
  "cpp_test_integrator",
  "cpp_test_autodiff_unary",
  "cpp_test_autodiff_binary",
  "cpp_test_autodiff_vector",
  "cpp_test_pg_update_beta",
  "cpp_test_pg_update_re",
  "cpp_test_pg_sigma_halfcauchy_chain",
  "cpp_test_dot_product",
  "cpp_test_norm_squared",
  "cpp_test_vector_sum",
  "cpp_test_axpy",
  "cpp_test_scale",
  "cpp_test_linalg_matvec",
  "cpp_test_linalg_matvec_add",
  "cpp_test_linalg_matvec_transpose",
  "cpp_test_sparse_laplacian_quadform",
  "cpp_test_linalg_log_sum_exp",
  "cpp_test_linalg_log_sum_exp_vec",
  "cpp_test_softmax_inplace",
  "cpp_test_compute_linear_predictors",
  "cpp_test_rw1_quadratic_form",
  "cpp_test_rw2_quadratic_form",
  "cpp_test_ar1_log_density",
  "cpp_test_temporal_log_prior",
  "cpp_test_ms_rw_log_lik",
  "cpp_test_log1pexp",
  "cpp_test_log_logistic",
  "cpp_test_log1m_logistic",
  "cpp_test_zi_poisson_lpmf",
  "cpp_test_zi_negbin_lpmf",
  "cpp_test_binomial_lpmf",
  "cpp_test_beta_binomial_lpmf",
  "cpp_test_zi_poisson_lpmf_logit",
  "cpp_test_zi_negbin_lpmf_logit",
  "cpp_test_zi_negbin_grad_logit_zi",
  "cpp_test_truncated_poisson_lpmf",
  "cpp_test_truncated_negbin_lpmf",
  "cpp_test_hurdle_poisson_lpmf_logit",
  "cpp_test_hurdle_negbin_lpmf_logit",
  "cpp_test_zi_log_likelihood",
  "cpp_test_get_max_threads",
  "cpp_test_parallel_dot_products",
  "cpp_test_parallel_likelihood",
  "cpp_test_st_gp_log_lik",
  "cpp_test_st_gp_margin_normalizer",
  "cpp_test_st_ar1_margin_normalizer",
  "cpp_test_negbin_math",
  "cpp_test_negbin_spec",
  "cpp_tls_workspace_shrink",
  "cpp_tulpa_abi_version",
  "cpp_tulpa_compiled_abi_version",
  "cpp_tulpa_runtime_abi_version",
  "cpp_tulpa_pg_binomial_gibbs",
  "cpp_tulpaRatio_run_nuts_specs",
  "cpp_vi_fit",
  "cpp_ratio_precision_density",
  "cpp_test_precision_h",
  "cpp_test_precision_legacy",
  "cpp_test_precision_gibbs",
  "cpp_test_precision_beta_legacy",
  "cpp_test_precision_jet",
  "cpp_test_precision_spec_h",
  "cpp_test_ratio_margin",
  "cpp_test_metric_step",
  "cpp_test_metric_epsilon",
  "cpp_test_metric_geometry",
  "cpp_test_residual_accumulation",
  "cpp_test_precision_spec",
  "cpp_test_precision_beta_spec",
  "cpp_test_precision_beta_spec_h",
  "cpp_scale_prior_check",
  "cpp_scale_conditional_draws"
))
