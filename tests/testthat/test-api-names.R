test_that("model accessors and summaries expose consistent argument names", {
  accessors <- c("validate_mode", "plot_rhat", "plot_ess", "plot_acf", "plot_pairs",
                 "plot_divergences", "plot_energy", "diagnostic_summary",
                 "plot_diagnostics", "geweke_test", "check_diagnostics", "n_divergent")
  for (name in accessors) {
    expect_identical(names(formals(get(name, asNamespace("tulpaRatio"))))[1], "object", info = name)
  }
  expect_true("probs" %in% names(formals(getS3method("summary", "ratiod_fit"))))
  expect_true("probs" %in% names(formals(getS3method("summary", "ratiod_vi"))))
  expect_true("probs" %in% names(formals(draws_interval)))
  expect_true("verbose" %in% names(formals(check_diagnostics)))
  expect_false("quiet" %in% names(formals(check_diagnostics)))
  expect_false("zi" %in% names(formals(tratio)))
  expect_true("spatial_level" %in% names(formals(spatial_car)))
  expect_true("spatial_level" %in% names(formals(spatial_bym2)))
  expect_true("level" %in% names(formals(getS3method("confint", "ratiod_fit"))))
  for (generic in c("diagnostics", "latent_factors")) {
    expect_identical(names(formals(getS3method(generic, "ratiod_fit")))[1],
                     names(formals(get(generic, asNamespace("tulpa"))))[1])
  }
})

test_that("inflation has one exported family route with explicit formulas", {
  constructors <- c("ratiod_zi_poisson", "ratiod_zi_negbin", "ratiod_hurdle_poisson",
                    "ratiod_hurdle_negbin", "ratiod_zi_binomial", "ratiod_oi_binomial",
                    "ratiod_zoi_binomial", "ratiod_hurdle_binomial")
  exports <- getNamespaceExports("tulpaRatio")
  expect_true(all(constructors %in% exports))
  expect_false("zi" %in% names(formals(tratio)))
  for (name in constructors) {
    fun <- get(name, asNamespace("tulpaRatio"))
    args <- if (name == "ratiod_oi_binomial") list(oi_formula = ~ x) else list(zi_formula = ~ x)
    if (name == "ratiod_zoi_binomial") args$oi_formula <- ~ z
    family <- do.call(fun, args)
    spec <- tulpaRatio:::family_inflation_spec(family)
    expect_identical(all.vars(spec$formula), "x")
    if (name == "ratiod_zoi_binomial") expect_identical(all.vars(spec$oi_formula), "z")
    design <- tulpaRatio:::prepare_zi_for_hmc(NULL, data.frame(x = 1:3, z = 3:1), 3, family)
    X <- if (name == "ratiod_oi_binomial") design$X_oi else design$X_zi
    expect_identical(unname(X), unname(model.matrix(~ x, data.frame(x = 1:3))))
  }
})

test_that("benchmark budgets are forwarded and seeded calls preserve RNG state", {
  captures <- list()
  local_mocked_bindings(tratio = function(formula, data, family, control) {
    captures[[length(captures) + 1L]] <<- list(data = data, control = control)
    list(draws = matrix(rnorm(8), 4, 2), diagnostics = list(divergent = FALSE))
  }, .package = "tulpaRatio")
  set.seed(871)
  state <- .Random.seed
  first <- ratiod_benchmark(N = 12, p = 2, iter = 10, warmup = 4, chains = 2,
                            cores = 2, seed = 42, verbose = TRUE)
  expect_identical(.Random.seed, state)
  second <- ratiod_benchmark(N = 12, p = 2, iter = 10, warmup = 4, chains = 2,
                             cores = 2, seed = 42, verbose = TRUE)
  expect_identical(.Random.seed, state)
  expect_identical(captures[[1]], captures[[2]])
  expect_identical(captures[[1]]$control,
                   list(iter = 10, warmup = 4, chains = 2, cores = 2, verbose = TRUE))
  ratiod_benchmark(N = 12, p = 2, iter = 10, warmup = 4, seed = 43)
  expect_false(identical(captures[[1]]$data, captures[[3]]$data))
  expect_identical(.Random.seed, state)
  expect_identical(first$config$seed, 42)
  expect_identical(first$config[c("iter", "warmup", "chains", "cores")],
                   list(iter = 10, warmup = 4, chains = 2, cores = 2))
  ratiod_benchmark_compare(list(list(N = 12, p = 2)), iter = 10, warmup = 4,
                           chains = 2, cores = 2, seed = 44)
  expect_identical(tail(captures, 1)[[1]]$control$cores, 2)
  expect_identical(.Random.seed, state)
})

test_that("benchmark restores absent RNG state and restores it after an error", {
  set.seed(319)
  state <- .Random.seed
  local_mocked_bindings(tratio = function(...) stop("model failure"), .package = "tulpaRatio")
  expect_error(ratiod_benchmark(N = 12, p = 2), "model failure")
  expect_identical(.Random.seed, state)
  rm(".Random.seed", envir = .GlobalEnv)
  tryCatch({
    expect_error(ratiod_benchmark(N = 12, p = 2), "model failure")
    expect_false(exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE))
  }, finally = assign(".Random.seed", state, envir = .GlobalEnv))
})

test_that("one-inflated pointwise likelihood matches its closed-form mixture", {
  draws <- cbind("beta_oi[1]" = c(-1, 0.5), "beta_oi[2]" = c(0.2, -0.4))
  X <- cbind(1, c(-1, 0, 1))
  object <- list(zi = list(X_zi = matrix(0, 3, 1), X_oi = X))
  hmc_data <- list(y_num = c(0, 1, 3), y_denom = rep(3, 3))
  mu <- matrix(c(0.2, 0.4, 0.6, 0.3, 0.5, 0.7), 2, 3, byrow = TRUE)
  actual <- tulpaRatio:::.ratiod_inflated_num_loglik(object, draws, mu, list(),
      hmc_data, "oi_binomial", list(zi = "logit", oi = "logit"))
  expected <- matrix(NA_real_, 2, 3)
  for (s in 1:2) for (i in 1:3) {
    pi <- plogis(sum(draws[s, ] * X[i, ]))
    pmf <- (1 - pi) * dbinom(hmc_data$y_num[i], 3, mu[s, i])
    if (hmc_data$y_num[i] == 3) pmf <- pmf + pi
    expected[s, i] <- log(pmf)
  }
  expect_equal(unname(actual), expected, tolerance = 1e-14)
})

test_that("automatic mode carries the family's inflation likelihood", {
  local_mocked_bindings(
    fit_hmc = function(...) list(backend = "hmc"),
    fit_laplace = function(...) list(backend = "laplace"),
    .package = "tulpaRatio"
  )
  data <- data.frame(count = rep(c(0L, 4L), length.out = 50001L), effort = 1)
  fit <- suppressMessages(tratio(count | effort ~ 1, data = data,
      family = ratiod_zi_poisson(),
      control = list(iter = 20, warmup = 10, chains = 1, cores = 1)))
  expect_identical(fit$backend, "hmc")
  expect_identical(fit$inference_tier, 1L)
})
