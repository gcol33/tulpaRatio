test_that("the native bridge refuses collapsed spatial requests for every family", {
  entries <- tulpaRatio:::cpp_ratio_family_entries()
  for (family in names(entries)) {
    processes <- entries[[family]]$n_processes
    for (type in c("icar", "bym2", "gp")) for (chains in 1:2) {
      expect_error(tulpaRatio:::cpp_tulpaRatio_sample(
        y_num = rep(1L, 4), y_denom = rep(2L, 4),
        y_num_cont = rep(1, 4), y_denom_cont = rep(2, 4),
        X_list = rep(list(matrix(1, 4, 1)), processes),
        cfg_list = list(family = family), init = numeric(), n_chains = chains,
        n_iter = 20L, n_warmup = 10L,
        spatial_params = list(type = type, parameterization = "collapsed")),
        "native collapsed spatial integration is unavailable.*full-field NUTS")
    }
  }
})

test_that("the NUTS bridge reports its joint target and matching engine ABI", {
  expect_identical(tulpaRatio:::cpp_tulpa_compiled_abi_version(), 61L)
  expect_identical(tulpaRatio:::cpp_tulpa_runtime_abi_version(), 61L)
  for (chains in 1:2) {
    fit <- tulpaRatio:::cpp_tulpaRatio_sample(
      y_num = rep(1L, 4), y_denom = rep(2L, 4),
      y_num_cont = numeric(), y_denom_cont = numeric(),
      X_list = list(matrix(1, 4, 1)), cfg_list = list(family = "binomial"),
      init = 0, n_chains = chains, n_iter = 30L, n_warmup = 15L, seed = 1177L)
    expect_identical(fit$target, "Joint posterior")
    expect_identical(fit$inference_tier, 1L)
    expect_false(any(c("icar_phi_star", "bym2_theta_star", "gp_w_star") %in% names(fit)))
  }
})
