simulate_for_family <- function(name, n = 200) {
  set.seed(20260503)
  x1 <- rnorm(n); x2 <- rnorm(n)
  if (name == "binomial") {
    eta <- 0.4 + 0.8 * x1 - 0.5 * x2
    nt  <- sample(5:20, n, replace = TRUE)
    y   <- rbinom(n, nt, plogis(eta))
    list(formula = y | n_trials ~ x1 + x2,
         data = data.frame(y = y, n_trials = nt, x1 = x1, x2 = x2),
         family = tulpaRatio::ratiod_binomial())
  } else if (name == "poisson_gamma") {
    mu_n  <- exp(1.0 + 0.4 * x1)
    mu_d  <- exp(0.5 + 0.3 * x2)
    yn    <- rpois(n, mu_n)
    yd    <- rgamma(n, shape = 4.0, rate = 4.0 / mu_d)
    list(formula = y_num | y_denom ~ x1 + x2,
         data = data.frame(y_num = yn, y_denom = yd, x1 = x1, x2 = x2),
         family = tulpaRatio::ratiod_poisson_gamma())
  } else if (name == "negbin_gamma") {
    mu_n  <- exp(1.0 + 0.4 * x1)
    mu_d  <- exp(0.5 + 0.3 * x2)
    yn    <- rnbinom(n, size = 5.0, mu = mu_n)
    yd    <- rgamma(n, shape = 4.0, rate = 4.0 / mu_d)
    list(formula = y_num | y_denom ~ x1 + x2,
         data = data.frame(y_num = yn, y_denom = yd, x1 = x1, x2 = x2),
         family = tulpaRatio::ratiod_negbin_gamma())
  } else if (name == "negbin_negbin") {
    mu_n <- exp(1.0 + 0.4 * x1); mu_d <- exp(0.7 + 0.3 * x2)
    yn   <- rnbinom(n, size = 5.0, mu = mu_n)
    yd   <- rnbinom(n, size = 5.0, mu = mu_d)
    list(formula = y_num | y_denom ~ x1 + x2,
         data = data.frame(y_num = yn, y_denom = yd, x1 = x1, x2 = x2),
         family = tulpaRatio::ratiod_negbin_negbin())
  } else if (name == "gamma_gamma") {
    mu_n <- exp(1.0 + 0.4 * x1); mu_d <- exp(0.5 + 0.3 * x2)
    yn   <- rgamma(n, shape = 4.0, rate = 4.0 / mu_n)
    yd   <- rgamma(n, shape = 5.0, rate = 5.0 / mu_d)
    list(formula = y_num | y_denom ~ x1 + x2,
         data = data.frame(y_num = yn, y_denom = yd, x1 = x1, x2 = x2),
         family = tulpaRatio::ratiod_gamma_gamma())
  } else if (name == "lognormal") {
    mu_ln <- 1.0 + 0.4 * x1; mu_ld <- 0.5 + 0.3 * x2
    yn    <- exp(rnorm(n, mu_ln, 0.5))
    yd    <- exp(rnorm(n, mu_ld, 0.5))
    list(formula = y_num | y_denom ~ x1 + x2,
         data = data.frame(y_num = yn, y_denom = yd, x1 = x1, x2 = x2),
         family = tulpaRatio::ratiod_lognormal())
  } else if (name == "beta_binomial") {
    eta <- 0.4 + 0.8 * x1 - 0.5 * x2
    p   <- plogis(eta)
    nt  <- sample(5:20, n, replace = TRUE)
    phi <- 10
    y   <- vapply(seq_len(n), function(i) {
      pp <- rbeta(1, p[i] * phi, (1 - p[i]) * phi)
      rbinom(1, nt[i], pp)
    }, integer(1))
    list(formula = y | n_trials ~ x1 + x2,
         data = data.frame(y = y, n_trials = nt, x1 = x1, x2 = x2),
         family = tulpaRatio::ratiod_beta_binomial())
  } else stop("unknown family: ", name)
}

fit_one <- function(spec, seed_val) {
  fit <- tulpaRatio::tratio(
    formula = spec$formula, data = spec$data, family = spec$family,
    mode = "hmc",
    control = list(iter = 2000L, warmup = 500L, chains = 1L, seed = seed_val, verbose = FALSE, gradient_mode = "A_r")
  )
  list(mean = colMeans(fit$draws), sd = apply(fit$draws, 2, stats::sd))
}

families <- c("binomial", "poisson_gamma", "negbin_gamma", "negbin_negbin",
              "gamma_gamma", "lognormal", "beta_binomial")
for (family in families) local({
  name <- family
  test_that(paste("engine recovers fixed effects for", name), {
    skip_on_cran()
    spec <- simulate_for_family(name)
    fitted <- fit_one(spec, 42L)$mean
    truth <- if (name %in% c("binomial", "beta_binomial")) c(.4, .8, -.5) else
      c(1, .4, 0, if (name == "negbin_negbin") .7 else .5, 0, .3)
    expect_lt(max(abs(fitted[seq_along(truth)] - truth)), .25)
    if (name == "poisson_gamma") expect_lt(abs(tail(fitted, 1L) - 4), .6)
  })
})
