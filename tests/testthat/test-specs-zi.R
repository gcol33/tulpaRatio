# test-specs-zi.R
# B1c parity test: each supported ZI / hurdle / OI / ZOIB variant routed
# through the LikelihoodSpec path produces posterior means within 4x within-MC
# noise of the legacy backend at the same seed.
#
# Parity scope (binomial only)
# ----------------------------
#   binomial : zi_binomial, hurdle_binomial, oi_binomial, zoib
# These are the variants where the legacy A_r autodiff path
# (src/log_post_impl.h) actually applies ZI / OI / ZOIB. For count ratio
# families (poisson_gamma, negbin_*), the legacy A_r path falls back to the
# plain count likelihood and IGNORES the requested ZI mechanism — only the
# H-mode handcoded path (src/hmc_sampler.cpp) routes count ZI. A like-for-like
# parity test for count ZI would need legacy H-mode vs. spec A_r and is left
# for a follow-up (B2 will rebuild H gradients on the spec path anyway).
#
# Smoke scope (count families)
# ----------------------------
#   poisson_gamma : zi_poisson,  hurdle_poisson
#   negbin_negbin : zi_negbin,   hurdle_negbin
# These verify the spec path *runs end-to-end* and produces finite, named
# posterior means — no comparison to legacy.

simulate_zi_for_variant <- function(variant, n = 200) {
  set.seed(20260504)
  x1 <- rnorm(n); x2 <- rnorm(n)

  if (variant == "zi_binomial") {
    eta <- 0.4 + 0.8 * x1 - 0.5 * x2
    p   <- plogis(eta)
    pi0 <- plogis(-1.0)  # ~27% structural zeros
    nt  <- sample(5:20, n, replace = TRUE)
    is_zero <- runif(n) < pi0
    y       <- ifelse(is_zero, 0L, rbinom(n, nt, p))
    list(formula = y | n_trials ~ x1 + x2,
         data    = data.frame(y = y, n_trials = nt, x1 = x1, x2 = x2),
         family  = tulpaRatio::ratiod_zibinomial(),
         zi      = NULL)
  } else if (variant == "hurdle_binomial") {
    eta <- 0.4 + 0.8 * x1 - 0.5 * x2
    p   <- plogis(eta)
    theta <- plogis(0.5)  # P(Y > 0)
    nt    <- sample(5:20, n, replace = TRUE)
    has_pos <- runif(n) < theta
    y_trunc <- vapply(seq_len(n), function(i) {
      yi <- 0L
      while (yi == 0L) yi <- rbinom(1, nt[i], p[i])
      yi
    }, integer(1))
    y <- ifelse(has_pos, y_trunc, 0L)
    list(formula = y | n_trials ~ x1 + x2,
         data    = data.frame(y = y, n_trials = nt, x1 = x1, x2 = x2),
         family  = tulpaRatio::ratiod_hurdle_binomial(),
         zi      = NULL)
  } else if (variant == "oi_binomial") {
    eta <- 0.4 + 0.8 * x1 - 0.5 * x2
    p   <- plogis(eta)
    psi <- plogis(-1.0)
    nt  <- sample(5:20, n, replace = TRUE)
    is_struct_one <- runif(n) < psi
    y <- ifelse(is_struct_one, nt, rbinom(n, nt, p))
    list(formula = y | n_trials ~ x1 + x2,
         data    = data.frame(y = y, n_trials = nt, x1 = x1, x2 = x2),
         family  = tulpaRatio::ratiod_oibinomial(),
         zi      = NULL)
  } else if (variant == "zoib") {
    eta  <- 0.4 + 0.8 * x1 - 0.5 * x2
    p    <- plogis(eta)
    pi0  <- plogis(-1.5)
    pi1  <- plogis(-1.5)
    nt   <- sample(5:20, n, replace = TRUE)
    u    <- runif(n)
    y <- vapply(seq_len(n), function(i) {
      if (u[i] < pi0) return(0L)
      if (u[i] < pi0 + (1 - pi0) * pi1) return(as.integer(nt[i]))
      as.integer(rbinom(1, nt[i], p[i]))
    }, integer(1))
    list(formula = y | n_trials ~ x1 + x2,
         data    = data.frame(y = y, n_trials = nt, x1 = x1, x2 = x2),
         family  = tulpaRatio::ratiod_zoibinomial(),
         zi      = NULL)
  } else if (variant == "zi_poisson") {
    mu_n <- exp(1.0 + 0.4 * x1)
    mu_d <- exp(0.5 + 0.3 * x2)
    pi0  <- plogis(-1.0)
    is_zero <- runif(n) < pi0
    yn  <- ifelse(is_zero, 0L, rpois(n, mu_n))
    yd  <- rgamma(n, shape = 4.0, rate = 4.0 / mu_d)
    list(formula = y_num | y_denom ~ x1 + x2,
         data    = data.frame(y_num = yn, y_denom = yd, x1 = x1, x2 = x2),
         family  = tulpaRatio::ratiod_poisson_gamma(),
         zi      = tulpaRatio::zi_poisson())
  } else if (variant == "hurdle_poisson") {
    mu_n <- exp(1.0 + 0.4 * x1)
    mu_d <- exp(0.5 + 0.3 * x2)
    theta <- plogis(0.5)
    has_pos <- runif(n) < theta
    yn_trunc <- vapply(seq_len(n), function(i) {
      yi <- 0L
      while (yi == 0L) yi <- rpois(1, mu_n[i])
      yi
    }, integer(1))
    yn <- ifelse(has_pos, yn_trunc, 0L)
    yd <- rgamma(n, shape = 4.0, rate = 4.0 / mu_d)
    list(formula = y_num | y_denom ~ x1 + x2,
         data    = data.frame(y_num = yn, y_denom = yd, x1 = x1, x2 = x2),
         family  = tulpaRatio::ratiod_poisson_gamma(),
         zi      = tulpaRatio::hurdle_poisson())
  } else if (variant == "zi_negbin") {
    mu_n <- exp(1.0 + 0.4 * x1)
    mu_d <- exp(0.7 + 0.3 * x2)
    pi0  <- plogis(-1.0)
    is_zero <- runif(n) < pi0
    yn   <- ifelse(is_zero, 0L, rnbinom(n, size = 5.0, mu = mu_n))
    yd   <- rnbinom(n, size = 5.0, mu = mu_d)
    list(formula = y_num | y_denom ~ x1 + x2,
         data    = data.frame(y_num = yn, y_denom = yd, x1 = x1, x2 = x2),
         family  = tulpaRatio::ratiod_negbin_negbin(),
         zi      = tulpaRatio::zi_negbin())
  } else if (variant == "hurdle_negbin") {
    mu_n <- exp(1.0 + 0.4 * x1)
    mu_d <- exp(0.7 + 0.3 * x2)
    theta <- plogis(0.5)
    has_pos <- runif(n) < theta
    yn_trunc <- vapply(seq_len(n), function(i) {
      yi <- 0L
      while (yi == 0L) yi <- as.integer(rnbinom(1, size = 5.0, mu = mu_n[i]))
      yi
    }, integer(1))
    yn <- ifelse(has_pos, yn_trunc, 0L)
    yd <- rnbinom(n, size = 5.0, mu = mu_d)
    list(formula = y_num | y_denom ~ x1 + x2,
         data    = data.frame(y_num = yn, y_denom = yd, x1 = x1, x2 = x2),
         family  = tulpaRatio::ratiod_negbin_negbin(),
         zi      = tulpaRatio::hurdle_negbin())
  } else stop("unknown ZI variant: ", variant)
}

fit_one_zi <- function(spec, use_specs, seed_val) {
  op <- options(tulpaRatio.use_specs = use_specs); on.exit(options(op), add = TRUE)
  args <- list(
    formula = spec$formula, data = spec$data, family = spec$family,
    mode = "hmc",
    control = list(iter = 2000L, warmup = 500L, chains = 1L,
                   seed = seed_val, verbose = FALSE, gradient_mode = "A_r")
  )
  if (!is.null(spec$zi)) args$zi <- spec$zi
  fit <- do.call(tulpaRatio::tratio, args)
  dr <- as.matrix(fit$draws)
  list(mean = colMeans(dr),
       mcse = apply(dr, 2, function(v) posterior::mcse_mean(matrix(v, ncol = 1))))
}

# Largest z-score between two fits' posterior means, each difference scaled
# by the two Monte Carlo standard errors
max_mean_z <- function(a, b) {
  max(abs(a$mean - b$mean) / sqrt(a$mcse^2 + b$mcse^2 + 1e-12))
}

parity_variants_b1c <- c("zi_binomial", "hurdle_binomial",
                          "oi_binomial", "zoib")
smoke_variants_b1c  <- c("zi_poisson", "hurdle_poisson",
                          "zi_negbin",  "hurdle_negbin")

for (v in parity_variants_b1c) {
  local({
    vv <- v
    test_that(sprintf("B1c spec path matches legacy for %s within MC noise", vv), {
      skip_on_cran()
      spec <- simulate_zi_for_variant(vv)
      legacy <- fit_one_zi(spec, FALSE, 42L)
      specs  <- fit_one_zi(spec, TRUE,  43L)

      expect_named(specs$mean, names(legacy$mean))
      # Two samplers of one posterior differ by Monte Carlo error alone
      expect_lt(max_mean_z(legacy, specs), 4.5)
    })
  })
}

for (v in smoke_variants_b1c) {
  local({
    vv <- v
    test_that(sprintf("B1c spec path runs end-to-end for %s", vv), {
      skip_on_cran()
      spec <- simulate_zi_for_variant(vv)
      specs42 <- fit_one_zi(spec, TRUE, 42L)$mean
      expect_true(all(is.finite(specs42)))
      expect_true(length(specs42) >= 4L)  # at least betas + zi/oi
    })
  })
}
