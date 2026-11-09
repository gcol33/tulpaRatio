test_that("automatic field-coordinate gradients differentiate the transformed density", {
  fields <- c("rw1", "rw2", "temporal_ar1", "icar", "icar_rw1", "car_proper",
              "gp", "gp_matern", "gp_gaussian", "gp_spherical",
              "tvc", "tvc_ar1", "tvc_iid", "tvc_rw2", "ms_temporal", "svc",
              "temporal_gp", "temporal_gp_matern32")
  for (field in fields) for (mode in c("H", "A_r", "N")) {
    if (field == "car_proper" && mode == "A_r") next
    r <- tulpaRatio:::cpp_gradient_check(paste0(field, "_auto"), mode = mode,
      n_obs = 40, n_units = 8, n_times = 5, seed = 17)
    expect_lt(max(grad_rel_dev(r)), 1e-4, label = paste(field, mode))
    expect_equal(r$log_post_mode, r$log_post, tolerance = 1e-8)
  }
})

test_that("field constructors select automatic coordinates by default", {
  expect_identical(temporal_rw1("time")$parameterization, "auto")
  expect_identical(temporal_rw2("time")$parameterization, "auto")
  expect_identical(temporal_ar1("time")$parameterization, "auto")
  expect_identical(temporal_multiscale("time")$parameterization, "auto")
  expect_identical(spatial_gp(~ lon + lat)$parameterization, "auto")
  expect_identical(spatial_svc(~ lon + lat, terms = ~ x)$parameterization, "auto")
  expect_identical(temporal_gp("time")$parameterization, "auto")
})

test_that("automatic fields use the NUTS field-coordinate transform", {
  skip_on_cran()
  set.seed(29)
  df <- data.frame(y = rbinom(40, 20, .6), n = 20, time = rep(1:5, each = 8))
  fit <- (function() {
    op <- options(tulpaRatio.use_specs = TRUE)
    on.exit(options(op), add = TRUE)
    calls <- 0L
    trace('cpp_tulpaRatio_run_nuts_specs', where = asNamespace('tulpaRatio'),
      tracer = function() calls <<- calls + 1L, print = FALSE)
    on.exit(untrace('cpp_tulpaRatio_run_nuts_specs', where = asNamespace('tulpaRatio')),
      add = TRUE)
    fit <- tratio(y | n ~ 1, data = df, family = ratiod_binomial(),
      temporal = temporal_rw1("time"), mode = "hmc",
      control = list(iter = 200, warmup = 100, chains = 1, seed = 7, verbose = FALSE))
    expect_equal(calls, 1L)
    fit
  })()
  expect_identical(fit$algorithm, "NUTS")
  expect_equal(nrow(as.matrix(fit$draws)), 100L)
})
