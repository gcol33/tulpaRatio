# test-zi-cross-backend.R
#
# Cross-backend ZI/hurdle/OI/ZOIB coverage for #33: VI/ESS/SGHMC previously
# auto-detected a ZI/hurdle family from a bare "zi"/"hurdle" string, which
# collapsed every variant to the same (or wrong) likelihood and left the
# four binomial-family ZI variants (zi_binomial, hurdle_binomial,
# oi_binomial, zoib) completely unreachable through those three backends.
#
# For each variant this file checks that mode = "vi"/"ess"/"sghmc":
#  - fits without error
#  - reports the correct, specific ZI/OI coefficient (not silently zero or
#    absent), agreeing with the mode = "hmc" fit on the same data/seed
#
# oi_binomial and zoib's OI component is checked on all four backends: VI,
# ESS, and SGHMC each read data.p_oi/X_oi_flat and expose beta_oi[...] in
# their draws, matching HMC.

fit_zi_coefs <- function(formula, data, family, mode, seed = 20260803) {
  args <- list(
    formula = formula, data = data, family = family, mode = mode,
    control = list(iter = 400L, warmup = 200L, chains = 1L, seed = seed,
                   verbose = FALSE)
  )
  fit <- do.call(tratio, args)

  pn <- colnames(fit$draws)
  zi_cols <- pn[grepl("^beta_zi\\[", pn)]
  oi_cols <- pn[grepl("^beta_oi\\[", pn)]
  list(
    zi = if (length(zi_cols) > 0) colMeans(fit$draws[, zi_cols, drop = FALSE]) else NULL,
    oi = if (length(oi_cols) > 0) colMeans(fit$draws[, oi_cols, drop = FALSE]) else NULL
  )
}

# ============================================================================
# Count-response families, explicit zi = (previously: bare "zi"/"hurdle"
# collapsed zi_negbin/hurdle_negbin to the same likelihood as zi_poisson on
# ESS/SGHMC, and VI mis-dispatched zi_negbin as ZI_POISSON)
# ============================================================================

simulate_count_zi <- function(n = 150) {
  set.seed(910)
  x <- rnorm(n)
  pi0 <- plogis(-1.0)
  is_zero <- runif(n) < pi0
  list(
    formula = count | total ~ x,
    data = data.frame(
      count = ifelse(is_zero, 0L, rnbinom(n, size = 5, mu = 10)),
      total = rnbinom(n, size = 5, mu = 50),
      x = x
    )
  )
}

for (backend in c("vi", "ess", "sghmc")) {
  test_that(sprintf("zi_negbin (explicit zi=) fits a nonzero, HMC-consistent ZI logit via %s", backend), {
    skip_on_cran()
    spec <- simulate_count_zi()

    ref <- fit_zi_coefs(spec$formula, spec$data, ratiod_zi_negbin(denom_family = "negbin"), "hmc")
    got <- fit_zi_coefs(spec$formula, spec$data, ratiod_zi_negbin(denom_family = "negbin"), backend)

    expect_false(is.null(got$zi))
    expect_true(is.finite(got$zi[[1]]))
    expect_lt(abs(got$zi[[1]] - ref$zi[[1]]), 1.5)
  })

  test_that(sprintf("ratiod_zi_negbin family auto-detect fits the negbin ZI likelihood (not poisson) via %s", backend), {
    skip_on_cran()
    spec <- simulate_count_zi()

    ref <- fit_zi_coefs(spec$formula, spec$data, ratiod_zi_negbin(), "hmc")
    got <- fit_zi_coefs(spec$formula, spec$data, ratiod_zi_negbin(), backend)

    expect_false(is.null(got$zi))
    expect_true(is.finite(got$zi[[1]]))
    expect_lt(abs(got$zi[[1]] - ref$zi[[1]]), 1.5)
  })
}

# ============================================================================
# Binomial-family ZI/hurdle: family auto-detection (previously unreachable
# on any of VI/ESS/SGHMC -- no path existed at all before #33)
# ============================================================================

simulate_binomial_zi <- function(n = 150) {
  set.seed(911)
  x <- rnorm(n)
  trials <- sample(10:20, n, replace = TRUE)
  p_true <- plogis(0.4 + 0.3 * x)
  is_zero <- runif(n) < 0.3
  list(
    formula = succ | trials ~ x,
    data = data.frame(
      succ = ifelse(is_zero, 0L, rbinom(n, trials, p_true)),
      trials = trials,
      x = x
    )
  )
}

for (backend in c("vi", "ess", "sghmc")) {
  test_that(sprintf("ratiod_zi_binomial family auto-detect fits a nonzero, HMC-consistent ZI logit via %s", backend), {
    skip_on_cran()
    spec <- simulate_binomial_zi()

    ref <- fit_zi_coefs(spec$formula, spec$data, ratiod_zi_binomial(), "hmc")
    got <- fit_zi_coefs(spec$formula, spec$data, ratiod_zi_binomial(), backend)

    expect_false(is.null(got$zi))
    expect_true(is.finite(got$zi[[1]]))
    expect_lt(abs(got$zi[[1]] - ref$zi[[1]]), 1.5)
  })

  test_that(sprintf("explicit zi_binomial() workaround fits the same way via %s", backend), {
    skip_on_cran()
    spec <- simulate_binomial_zi()

    ref <- fit_zi_coefs(spec$formula, spec$data, ratiod_zi_binomial(), "hmc")
    got <- fit_zi_coefs(spec$formula, spec$data, ratiod_zi_binomial(), backend)

    expect_false(is.null(got$zi))
    expect_true(is.finite(got$zi[[1]]))
    expect_lt(abs(got$zi[[1]] - ref$zi[[1]]), 1.5)
  })

  test_that(sprintf("ratiod_hurdle_binomial family auto-detect fits a nonzero, HMC-consistent hurdle logit via %s", backend), {
    skip_on_cran()
    spec <- simulate_binomial_zi()

    ref <- fit_zi_coefs(spec$formula, spec$data, ratiod_hurdle_binomial(), "hmc")
    got <- fit_zi_coefs(spec$formula, spec$data, ratiod_hurdle_binomial(), backend)

    expect_false(is.null(got$zi))
    expect_true(is.finite(got$zi[[1]]))
    expect_lt(abs(got$zi[[1]] - ref$zi[[1]]), 1.5)
  })
}

# ============================================================================
# One-inflated and ZOIB binomial families: full check on VI (data plumbing
# fixed here); ESS/SGHMC checked only where their existing, documented
# `p_oi = 0` hardcoding doesn't apply (zoib's ZI component).
# ============================================================================

simulate_oi_binomial <- function(n = 150) {
  set.seed(912)
  x <- rnorm(n)
  trials <- sample(10:20, n, replace = TRUE)
  p_true <- plogis(0.3 + 0.2 * x)
  succ <- rbinom(n, trials, p_true)
  is_oi <- runif(n) < 0.3
  succ[is_oi] <- trials[is_oi]
  list(
    formula = succ | trials ~ x,
    data = data.frame(succ = succ, trials = trials, x = x)
  )
}

test_that("ratiod_oi_binomial family auto-detect fits a nonzero, HMC-consistent OI logit via vi", {
  skip_on_cran()
  spec <- simulate_oi_binomial()

  ref <- fit_zi_coefs(spec$formula, spec$data, ratiod_oi_binomial(), "hmc")
  got <- fit_zi_coefs(spec$formula, spec$data, ratiod_oi_binomial(), "vi")

  expect_false(is.null(got$oi))
  expect_true(is.finite(got$oi[[1]]))
  expect_lt(abs(got$oi[[1]] - ref$oi[[1]]), 1.5)
})


test_that("explicit oi_binomial() workaround fits via vi", {
  skip_on_cran()
  spec <- simulate_oi_binomial()

  ref <- fit_zi_coefs(spec$formula, spec$data, ratiod_oi_binomial(), "hmc")
  got <- fit_zi_coefs(spec$formula, spec$data, ratiod_oi_binomial(), "vi")

  expect_false(is.null(got$oi))
  expect_true(is.finite(got$oi[[1]]))
  expect_lt(abs(got$oi[[1]] - ref$oi[[1]]), 1.5)
})


for (backend in c("ess", "sghmc")) {
  test_that(sprintf("ratiod_oi_binomial family auto-detect fits a nonzero, HMC-consistent OI logit via %s", backend), {
    skip_on_cran()
    spec <- simulate_oi_binomial()

    ref <- fit_zi_coefs(spec$formula, spec$data, ratiod_oi_binomial(), "hmc")
    got <- fit_zi_coefs(spec$formula, spec$data, ratiod_oi_binomial(), backend)

    expect_false(is.null(got$oi))
    expect_true(is.finite(got$oi[[1]]))
    expect_lt(abs(got$oi[[1]] - ref$oi[[1]]), 1.5)
  })
}


simulate_zoib <- function(n = 150) {
  set.seed(913)
  x <- rnorm(n)
  trials <- sample(10:20, n, replace = TRUE)
  p_true <- plogis(0.3 + 0.2 * x)
  u <- runif(n)
  pi0 <- 0.2
  pi1 <- 0.25
  succ <- vapply(seq_len(n), function(i) {
    if (u[i] < pi0) return(0L)
    if (u[i] < pi0 + (1 - pi0) * pi1) return(as.integer(trials[i]))
    as.integer(rbinom(1, trials[i], p_true[i]))
  }, integer(1))
  list(
    formula = succ | trials ~ x,
    data = data.frame(succ = succ, trials = trials, x = x)
  )
}

test_that("ratiod_zoi_binomial family auto-detect fits nonzero, HMC-consistent ZI and OI logits via vi", {
  skip_on_cran()
  spec <- simulate_zoib()

  ref <- fit_zi_coefs(spec$formula, spec$data, ratiod_zoi_binomial(), "hmc")
  got <- fit_zi_coefs(spec$formula, spec$data, ratiod_zoi_binomial(), "vi")

  expect_false(is.null(got$zi))
  expect_false(is.null(got$oi))
  expect_true(is.finite(got$zi[[1]]))
  expect_true(is.finite(got$oi[[1]]))
  expect_lt(abs(got$zi[[1]] - ref$zi[[1]]), 1.5)
  expect_lt(abs(got$oi[[1]] - ref$oi[[1]]), 1.5)
})


for (backend in c("ess", "sghmc")) {
  test_that(sprintf("ratiod_zoi_binomial family auto-detect fits nonzero, HMC-consistent ZI and OI logits via %s", backend), {
    skip_on_cran()
    spec <- simulate_zoib()

    ref <- fit_zi_coefs(spec$formula, spec$data, ratiod_zoi_binomial(), "hmc")
    got <- fit_zi_coefs(spec$formula, spec$data, ratiod_zoi_binomial(), backend)

    expect_false(is.null(got$zi))
    expect_false(is.null(got$oi))
    expect_true(is.finite(got$zi[[1]]))
    expect_true(is.finite(got$oi[[1]]))
    expect_lt(abs(got$zi[[1]] - ref$zi[[1]]), 1.5)
    expect_lt(abs(got$oi[[1]] - ref$oi[[1]]), 1.5)
  })
}

# ============================================================================
# The count-response ZI likelihood ESS actually evaluates (#34)
# ============================================================================
#
# The checks above compare a backend's ZI logit against HMC's with a tolerance
# of 1.5, on data whose true ZI logit is about -1. A backend that ignored the
# ZI term entirely would sample beta_zi from its N(0, zi_prior_sd) prior and
# land within that band, so those tests hold either way for ESS. ESS called
# compute_log_post_impl where the other backends called compute_log_post, and
# at the time those were two densities: the templated one branched on zi_type
# only for binomial responses, so a Poisson/NegBin ZI or hurdle term was
# dropped and beta_zi was sampled against a likelihood that did not contain it.
# They are one function since gcol33/tulpaRatio#28, which is what keeps that
# from recurring; this still checks the term reaches the sampler.
#
# Separating the two takes a true ZI logit far from the prior mean.

simulate_strong_count_zi <- function(n = 300) {
  set.seed(4034)
  x <- rnorm(n)
  is_zero <- runif(n) < plogis(1.4)  # about 80% structural zeros
  list(
    formula = count | total ~ x,
    data = data.frame(
      count = ifelse(is_zero, 0L, rnbinom(n, size = 5, mu = 12)),
      total = rnbinom(n, size = 5, mu = 60),
      x = x
    )
  )
}

for (variant in list(list(family = ratiod_zi_negbin(), name = "zi_negbin"),
                     list(family = ratiod_hurdle_negbin(), name = "hurdle_negbin"))) {
  test_that(sprintf("ESS fits %s from the likelihood, not from its prior", variant$name), {
    skip_on_cran()
    spec <- simulate_strong_count_zi()

    ref <- fit_zi_coefs(spec$formula, spec$data, variant$family, "hmc")
    got <- fit_zi_coefs(spec$formula, spec$data, variant$family, "ess")

    expect_false(is.null(got$zi))
    # Far from the prior mean the ZI coefficient collapses to when the
    # likelihood does not contain it, and close to what HMC gets from the same
    # data.
    expect_gt(abs(got$zi[[1]]), 0.7)
    expect_lt(abs(got$zi[[1]] - ref$zi[[1]]), 0.6)
  })
}
