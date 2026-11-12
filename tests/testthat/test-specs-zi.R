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
         family  = tulpaRatio::ratiod_zi_binomial())
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
         family  = tulpaRatio::ratiod_hurdle_binomial())
  } else if (variant == "oi_binomial") {
    eta <- 0.4 + 0.8 * x1 - 0.5 * x2
    p   <- plogis(eta)
    psi <- plogis(-1.0)
    nt  <- sample(5:20, n, replace = TRUE)
    is_struct_one <- runif(n) < psi
    y <- ifelse(is_struct_one, nt, rbinom(n, nt, p))
    list(formula = y | n_trials ~ x1 + x2,
         data    = data.frame(y = y, n_trials = nt, x1 = x1, x2 = x2),
         family  = tulpaRatio::ratiod_oi_binomial())
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
         family  = tulpaRatio::ratiod_zoi_binomial())
  } else if (variant == "zi_poisson") {
    mu_n <- exp(1.0 + 0.4 * x1)
    mu_d <- exp(0.5 + 0.3 * x2)
    pi0  <- plogis(-1.0)
    is_zero <- runif(n) < pi0
    yn  <- ifelse(is_zero, 0L, rpois(n, mu_n))
    yd  <- rgamma(n, shape = 4.0, rate = 4.0 / mu_d)
    list(formula = y_num | y_denom ~ x1 + x2,
         data    = data.frame(y_num = yn, y_denom = yd, x1 = x1, x2 = x2),
         family  = tulpaRatio::ratiod_zi_poisson())
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
         family  = tulpaRatio::ratiod_hurdle_poisson())
  } else if (variant == "zi_negbin") {
    mu_n <- exp(1.0 + 0.4 * x1)
    mu_d <- exp(0.7 + 0.3 * x2)
    pi0  <- plogis(-1.0)
    is_zero <- runif(n) < pi0
    yn   <- ifelse(is_zero, 0L, rnbinom(n, size = 5.0, mu = mu_n))
    yd   <- rnbinom(n, size = 5.0, mu = mu_d)
    list(formula = y_num | y_denom ~ x1 + x2,
         data    = data.frame(y_num = yn, y_denom = yd, x1 = x1, x2 = x2),
         family  = tulpaRatio::ratiod_zi_negbin())
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
         family  = tulpaRatio::ratiod_hurdle_negbin())
  } else stop("unknown ZI variant: ", variant)
}

fit_one_zi <- function(spec, seed_val) {
  args <- list(
    formula = spec$formula, data = spec$data, family = spec$family,
    mode = "hmc",
    control = list(iter = 2000L, warmup = 500L, chains = 1L,
                   seed = seed_val, verbose = FALSE, gradient_mode = "A_r")
  )
  fit <- do.call(tulpaRatio::tratio, args)
  dr <- as.matrix(fit$draws)
  list(mean = colMeans(dr),
       mcse = apply(dr, 2, function(v) posterior::mcse_mean(matrix(v, ncol = 1))))
}

parity_variants_b1c <- c("zi_binomial", "hurdle_binomial",
                          "oi_binomial", "zoib")
smoke_variants_b1c  <- c("zi_poisson", "hurdle_poisson",
                          "zi_negbin",  "hurdle_negbin")

for (v in parity_variants_b1c) {
  local({
    vv <- v
    test_that(sprintf("engine recovers inflated binomial coefficients for %s", vv), {
      skip_on_cran()
      spec <- simulate_zi_for_variant(vv)
      estimate <- fit_one_zi(spec, 43L)$mean
      expect_lt(max(abs(estimate[1:3] - c(.4, .8, -.5))), .25)
    })
  })
}

for (v in smoke_variants_b1c) {
  local({
    vv <- v
    test_that(sprintf("B1c spec path runs end-to-end for %s", vv), {
      skip_on_cran()
      spec <- simulate_zi_for_variant(vv)
      specs42 <- fit_one_zi(spec, 42L)$mean
      expect_true(all(is.finite(specs42)))
      expect_true(length(specs42) >= 4L)  # at least betas + zi/oi
    })
  })
}
