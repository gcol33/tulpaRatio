test_that("field scale priors carry the SD density to every sampled coordinate", {
  fields <- c("rw1", "rw1_nc", "rw2", "rw2_nc", "temporal_ar1",
              "temporal_ar1_nc", "temporal_iid", "tvc", "tvc_iid", "tvc_ar1",
              "svc", "svc_nc", "svc_hsgp", "hsgp", "icar", "bym2", "car_proper",
              "icar_collapsed", "bym2_collapsed", "icar_rw1", "bym2_ms",
              "tvc_st4", "svc_ms")
  U <- 0.8
  alpha <- 0.15
  lambda <- -log(alpha) / U
  lambda0 <- -log(0.01)
  for (field in fields) for (x in c(-2, 0, 1.5)) {
    z <- tulpaRatio:::cpp_scale_prior_check(field, x, U, alpha)
    sigma <- exp(z$slopes * x)
    expect_true(length(sigma) > 0, info = field)
    reference <- dexp(sigma, lambda, log = TRUE) + log(abs(z$slopes)) + z$slopes * x
    expect_equal(z$density, reference, tolerance = 1e-12, info = field)
    expect_equal(z$gradient, z$slopes * (1 - lambda * sigma),
                 tolerance = 1e-12, info = field)
    expect_equal(z$delta, sum(log(lambda / lambda0) - (lambda - lambda0) * sigma),
                 tolerance = 1e-9, info = field)
  }
})

test_that("Gibbs precision conditionals match normalized quadrature", {
  set.seed(217)
  for (rank in c(0L, 3L, 8L)) {
    quad <- if (rank == 0) 0 else 1.4
    U <- 0.8
    alpha <- 0.15
    x <- seq(-18, 18, length.out = 20001)
    sigma <- exp(-x / 2)
    logp <- dexp(sigma, -log(alpha) / U, log = TRUE) - log(2) - x / 2 +
      rank * x / 2 - quad * exp(x) / 2
    w <- exp(logp - max(logp))
    cdf <- cumsum(w) / sum(w)
    probabilities <- c(0.1, 0.5, 0.9)
    keep <- !duplicated(cdf)
    reference <- approx(cdf[keep], x[keep], probabilities)$y
    dr <- log(tulpaRatio:::cpp_scale_conditional_draws(40000L, quad, rank, U, alpha))
    expect_lt(max(abs(quantile(dr, probabilities) - reference)) / sd(dr), 0.06)
  }
})

test_that("field SD anchors reach every R sampler bundle", {
  p <- ratiod_priors(sigma_temporal = prior_pc(0.7, 0.2),
                    sigma_spatial = prior_pc(1.2, 0.1),
                    sigma_tvc = prior_pc(0.6, 0.15),
                    sigma_svc = prior_pc(0.4, 0.05))
  sp <- tulpaRatio:::sampler_priors(p, "binomial")
  expect_equal(unname(unlist(sp[c("temporal_sigma_prior_U", "temporal_sigma_prior_alpha",
                                  "spatial_sigma_prior_U", "spatial_sigma_prior_alpha",
                                  "tvc_sigma_prior_U", "tvc_sigma_prior_alpha",
                                  "svc_sigma_prior_U", "svc_sigma_prior_alpha")])),
               c(0.7, 0.2, 1.2, 0.1, 0.6, 0.15, 0.4, 0.05))
  defaults <- tulpaRatio:::fill_priors(NULL, "binomial")
  expect_equal(vapply(defaults[c("sigma_temporal", "sigma_spatial", "sigma_tvc", "sigma_svc")],
                       `[[`, numeric(1), "U"),
               c(sigma_temporal = 3, sigma_spatial = 3, sigma_tvc = 1, sigma_svc = 1))
  expect_error(ratiod_priors(sigma_temporal = prior_gamma()), "prior_pc")
  expect_error(tulpaRatio:::sampler_priors(list(tau_temporal_shape = 2), "binomial"),
               "removed")
  expect_output(print(p), "sigma_temporal")

  d <- data.frame(y = 1L, trials = 2L, x = seq(-1, 1, length.out = 12),
                  lon = rep(1:4, 3), lat = rep(1:3, each = 4))
  formula <- tulpaRatio:::ratiod_formula(y | trials ~ x, data = d)
  for (approx in c("nngp", "hsgp")) {
    svc <- spatial_svc(~ lon + lat, terms = ~ x - 1, approx = approx, nn = 3)
    inputs <- tulpaRatio:::sampler_inputs(formula, d, ratiod_binomial(), spatial = svc,
                                        priors = p)
    expect_equal(inputs$bundles$svc_params$sigma_prior_U, 0.4)
    expect_equal(inputs$bundles$svc_params$sigma_prior_alpha, 0.05)
  }
})

scale_temporal_reference <- function(n_eta = 121L, n_sigma = 161L, n_rho = 81L) {
  eta <- seq(-5, 5, length.out = n_eta)
  grid <- expand.grid(a = eta, b = eta)
  squares <- grid$a^2 + grid$b^2
  products <- grid$a * grid$b
  loglik <- dbinom(5, 40, plogis(grid$a), log = TRUE) +
    dbinom(35, 40, plogis(grid$b), log = TRUE)
  sigma <- exp(seq(-4, 2, length.out = n_sigma))
  rho <- seq(-0.995, 0.995, length.out = n_rho)
  weights <- matrix(0, n_sigma, n_rho)
  for (j in seq_along(rho)) for (i in seq_along(sigma)) {
    variance <- sigma[i]^2 / (1 - rho[j]^2)
    v <- variance + 0.5^2
    covariance <- rho[j] * variance + 0.5^2
    determinant <- (v - covariance) * (v + covariance)
    log_density <- -log(2 * pi) - log(determinant) / 2 -
      (v * squares - 2 * covariance * products) / (2 * determinant)
    weights[i, j] <- sum(exp(loglik + log_density)) *
      dexp(sigma[i], -log(0.15) / 0.8) * sigma[i] *
      dbeta((rho[j] + 1) / 2, 3, 4) / 2
  }
  weighted_quantile <- function(x, w) {
    cdf <- cumsum(w) / sum(w)
    approx(cdf, x, c(0.25, 0.5, 0.75))$y
  }
  cbind(sigma = weighted_quantile(sigma, rowSums(weights)),
        rho = weighted_quantile(rho, colSums(weights)))
}

test_that("temporal spec and legacy quantiles agree with dense quadrature", {
  skip_on_cran()
  if (!identical(Sys.getenv("TULPA_SLOW_TESTS"), "true")) skip("Set TULPA_SLOW_TESTS=true")
  d <- data.frame(y = c(5L, 35L), trials = 40L, time = 1:2)
  p <- ratiod_priors(beta = prior_normal(0, 0.5),
                    sigma_temporal = prior_pc(0.8, 0.15),
                    rho_temporal = prior_beta(3, 4))
  old <- options(tulpaRatio.use_specs = FALSE)
  on.exit(options(old), add = TRUE)
  fit_one <- function(specs, seed) {
    options(tulpaRatio.use_specs = specs)
    fit <- tratio(y | trials ~ 1, data = d, family = ratiod_binomial(),
                  temporal = temporal_ar1("time"), priors = p, mode = "hmc",
                  control = list(iter = 16000L, warmup = 4000L, chains = 1L,
                                 seed = seed, verbose = FALSE, gradient_mode = "A_r",
                                 adapt_delta = 0.99))
    dr <- as.matrix(fit$draws)
    cbind(sigma = 1 / sqrt(dr[, "tau_temporal"]),
          rho = dr[, "rho_ar1"])
  }
  a <- fit_one(FALSE, 1701L)
  b <- fit_one(TRUE, 1702L)
  probabilities <- c(0.25, 0.5, 0.75)
  qa <- apply(a, 2, quantile, probs = probabilities)
  qb <- apply(b, 2, quantile, probs = probabilities)
  reference <- scale_temporal_reference()
  sd_reference <- sqrt((apply(a, 2, var) + apply(b, 2, var)) / 2)
  expect_lt(max(abs(sweep(qa - reference, 2, sd_reference, `/`))), 0.2)
  expect_lt(max(abs(sweep(qb - reference, 2, sd_reference, `/`))), 0.2)
  expect_lt(max(abs(sweep(qa - qb, 2, sqrt((apply(a, 2, var) + apply(b, 2, var)) / 2), `/`))), 0.2)
})
