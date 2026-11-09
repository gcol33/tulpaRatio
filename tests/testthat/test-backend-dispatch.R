test_that("backend argument builders forward each control to its fitter", {
  inputs <- list(formula = quote(y | n ~ x), data = data.frame(x = seq_len(25)),
    family = ratiod_binomial(), spatial = NULL, temporal = NULL,
    spatiotemporal = NULL, zi = NULL, latent = NULL, priors = ratiod_priors())
  control <- tulpaRatio:::.tratio_control(list(iter = 140, warmup = 60, chains = 2,
    cores = 1, thin = 2, seed = 181, L = 7L, adapt_delta = 0.91,
    max_treedepth = 9L, metric = "dense", gradient_mode = "A", re_param = "centered",
    vi_variant = "meanfield", batch_size = 11L, epsilon = 0.002, alpha = 0.2,
    schedule_a = 0.03, schedule_b = 2, schedule_gamma = 0.7, use_schedule = FALSE,
    verbose = FALSE))
  registry <- tulpaRatio:::TRATIO_BACKENDS
  expect_setequal(names(registry), names(tulpaRatio:::BACKEND_STRUCTURE_SUPPORT))
  expected <- list(
    gibbs = c("iter", "warmup", "thin", "chains", "cores", "seed", "verbose"),
    pg = c("iter", "warmup", "thin", "chains", "cores", "seed", "verbose"),
    laplace = c("cores", "verbose"),
    hmc = c("iter", "warmup", "chains", "cores", "L", "adapt_delta", "max_treedepth",
      "gradient_mode", "re_param", "metric", "walnuts", "seed", "verbose"),
    vi = c("seed", "verbose"), ess = c("iter", "warmup", "chains", "seed", "verbose"),
    sghmc = c("iter", "warmup", "seed", "verbose", "batch_size", "epsilon", "alpha", "L"),
    sgld = c("iter", "warmup", "seed", "verbose", "batch_size", "epsilon", "schedule_a",
      "schedule_b", "schedule_gamma", "use_schedule"))
  for (backend in names(registry)) {
    spec <- registry[[backend]]
    args <- spec$args(list(inputs = inputs, control = control))
    expect_false(anyDuplicated(names(args)) > 0L, info = backend)
    expect_true(all(names(args) %in% names(formals(get(spec$fit, envir = asNamespace("tulpaRatio"))))),
      info = backend)
    expect_identical(args[expected[[backend]]], control[expected[[backend]]], info = backend)
  }
  laplace <- registry$laplace$args(list(inputs = inputs, control = control))
  expect_identical(laplace$n_samples, 80L)
  vi <- registry$vi$args(list(inputs = inputs, control = control))
  expect_identical(vi$variant, control$vi_variant)
  expect_identical(vi$max_iter, control$iter)
  pg <- registry$pg$args(list(inputs = inputs, control = control))
  prior <- tulpaRatio:::sampler_priors(inputs$priors, "binomial")
  expect_identical(pg$prior_beta_sd, prior$sigma_beta)
  expect_identical(pg$prior_sigma_scale, prior$sigma_re_scale)
  expect_identical(pg$prior_sigma_spatial_U, prior$spatial_sigma_prior_U)
  expect_identical(pg$prior_sigma_spatial_alpha, prior$spatial_sigma_prior_alpha)
})

test_that("stochastic gradient builders share the data-sized batch default", {
  context <- list(inputs = list(formula = NULL, data = data.frame(x = seq_len(81)),
    family = ratiod_binomial(), spatial = NULL, temporal = NULL, zi = NULL, priors = NULL),
    control = tulpaRatio:::.tratio_control(list(iter = 100, chains = 1, verbose = FALSE)))
  for (backend in c("sghmc", "sgld")) {
    args <- tulpaRatio:::TRATIO_BACKENDS[[backend]]$args(context)
    expect_identical(args$batch_size, 9)
  }
})
