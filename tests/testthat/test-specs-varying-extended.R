test_that("SVC and factor bridges agree across seeds and recover planted predictors", {
  skip_on_cran()
  cases <- list(svc_nngp = "nngp", svc_hsgp = "hsgp",
    svc_matern = "matern", svc_gaussian = "gaussian", svc_spherical = "spherical",
    svc_shared = "shared", svc_unshared = "unshared",
    tvc_shared = "tvc_shared", tvc_unshared = "tvc_unshared",
    latent_single = "single", latent_first_zero = "first_zero",
    latent_shared = "factor_shared", latent_two = "factor_two")
  selected <- Sys.getenv("TULPA_BRIDGE_ROUTE")
  if (nzchar(selected)) {
    wanted <- strsplit(selected, ",", fixed = TRUE)[[1L]]
    stopifnot(all(wanted %in% names(cases)))
    cases <- cases[wanted]
  }
  for (seed in c(18L, 181L, 182L)) for (name in names(cases)) {
    kind <- cases[[name]]
    factor <- startsWith(name, "latent")
    tvc <- startsWith(name, "tvc")
    paired <- kind %in% c("shared", "unshared", "factor_shared", "factor_two", "tvc_shared", "tvc_unshared")
    n <- if (factor) 60L else 24L
    set.seed(seed)
    d <- data.frame(x = rnorm(n), lon = runif(n), lat = runif(n), n = 80L)
    d$time <- rep(seq_len(6L), length.out = n)
    stopifnot(nrow(d) == n)
    field <- .25 * cos(2 * pi * d$lon)
    if (tvc) field <- .25 * cos(2 * pi * d$time / 6)
    field <- field - mean(field)
    if (factor) {
      field <- .3 * rnorm(n)
      field <- field - mean(field)
    }
    num_eta <- (if (paired) 3 else .2) + (if (paired) .4 else .5) * d$x +
      if (factor) field else field * d$x
    if (paired) {
      denominator_field <- if (kind %in% c("unshared", "tvc_unshared")) 0 else
        if (factor) -.6 * field else field * d$x
      if (kind == "factor_two") denominator_field <- denominator_field + .2 * rnorm(n)
      denom_eta <- 1 + .2 * d$x + denominator_field
      d$y <- rpois(n, exp(num_eta))
      d$denom <- rgamma(n, shape = 30, rate = 30 / exp(denom_eta))
      formula <- y | denom ~ x
      family <- ratiod_poisson(denom_family = "gamma")
      truth <- exp(num_eta)
      beta <- c(3, .4)
    } else {
      truth <- plogis(num_eta)
      d$y <- rbinom(n, d$n, truth)
      formula <- y | n ~ x
      family <- ratiod_binomial()
      beta <- c(.2, .5)
    }
    args <- list(formula = formula, data = d, family = family, mode = "hmc",
      control = list(iter = 6000L, warmup = 2000L, chains = 1L, cores = 1L,
        seed = seed, verbose = FALSE, adapt_delta = .99, gradient_mode = "A_r"))
    if (factor) args$latent <- latent_factor(if (kind == "factor_two") 2L else 1L,
      constraint = if (kind == "first_zero") "first_zero" else "sum_to_zero") else if (tvc)
      args$temporal <- temporal_tvc("time", ~ x, structure = "iid", shared = kind == "tvc_shared") else
      args$spatial <- suppressWarnings(spatial_svc(~ lon + lat, terms = "x", nn = 4L,
        approx = if (kind == "hsgp") "hsgp" else "nngp", m = 3L,
        cov = if (kind %in% c("matern", "gaussian", "spherical")) kind else "exponential",
        shared = kind != "unshared"))
    fit <- bridge19_pair(args, name, seed)
    b <- as.matrix(fit$draws)
    expect_true(all(abs(colMeans(b)[1:2] - beta) < .3), info = paste(name, seed))
    recovered <- colMeans(fitted(fit, summary = FALSE)$numerator)
    expect_lt(sqrt(mean(((recovered - truth) / if (paired) truth else 1)^2)),
      if (paired) .3 else .12, label = paste(name, seed, "recovery"))
  }
})
