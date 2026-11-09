# Recovery against simulated truth for models carrying a constrained
# (rank-deficient) field. Shape and class assertions pass just as happily when
# the intercept is unidentified, which is how a missing sum-to-zero constraint
# reached a release: nothing asserted that the intercept lands near the value it
# was simulated from.

TRUE_INTERCEPT <- 0.5
TRUE_SLOPE <- 0.3

sim_data <- function(seed, N = 500, S = 50, TT = 20) {
  set.seed(seed)
  x <- rnorm(N)
  site <- factor(rep(seq_len(S), length.out = N))
  time <- factor(rep(seq_len(TT), length.out = N))
  trials <- sample(10:50, N, replace = TRUE)
  eta <- TRUE_INTERCEPT + TRUE_SLOPE * x
  data.frame(y = rbinom(N, trials, plogis(eta)), trials = trials,
             x = x, site = site, time = time, spatial_site = site,
             lon = runif(N), lat = runif(N))
}

grid_adj <- function(S) {
  n <- ceiling(sqrt(S))
  g <- expand.grid(lon = seq_len(n), lat = seq_len(n))[seq_len(S), ]
  A <- matrix(0, S, S)
  for (i in seq_len(S)) for (j in seq_len(S)) {
    if (i != j && sqrt((g$lon[i] - g$lon[j])^2 + (g$lat[i] - g$lat[j])^2) <= 1.5)
      A[i, j] <- 1
  }
  dimnames(A) <- list(as.character(seq_len(S)), as.character(seq_len(S)))
  A
}

# Gaussian importance integration over the fixed effects and the constrained
# field, followed by trapezoidal integration over a log scale. Whitening by
# the positive eigenvalues of Q keeps the dense solve conditioned as tau grows.
constrained_field_reference <- function(df, Q, group, n_draws = 2048L,
                                        seed = 222L,
                                        flat = matrix(0, nrow(Q), 0L),
                                        log_sd_slope = -0.5, U = 3,
                                        grid = seq(-4, 50, by = 0.25)) {
  force(df)
  old_seed <- get0(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  on.exit({
    if (is.null(old_seed)) rm(".Random.seed", envir = .GlobalEnv)
    else assign(".Random.seed", old_seed, envir = .GlobalEnv)
  }, add = TRUE)
  set.seed(seed)
  eigen_Q <- eigen(Q, symmetric = TRUE)
  rank <- nrow(Q) - 1L - ncol(flat)
  positive <- seq_len(rank)
  field <- sweep(eigen_Q$vectors[, positive, drop = FALSE],
                  2, sqrt(eigen_Q$values[positive]), "/")
  dimension <- rank + 2L + ncol(flat)
  z <- matrix(0, dimension, n_draws)
  odd <- seq.int(1L, n_draws, by = 2L)
  z[, odd] <- matrix(rnorm(dimension * length(odd)), dimension)
  z[, odd + 1L] <- -z[, odd]
  softplus <- function(x) pmax(x, 0) + log1p(exp(-abs(x)))
  x <- grid
  log_marginal <- matrix(0, length(x), 4L)
  field_first <- field_second <- array(0, c(length(x), nrow(Q), 4L))
  at <- numeric(dimension)
  precision <- c(rep(1 / 100, 2L), rep(0, ncol(flat)), rep(1, rank))
  lambda <- -log(0.01) / U
  for (i in seq_along(x)) {
    field_design <- cbind(flat, exp(log_sd_slope * x[i]) * field)
    design <- cbind(1, df$x, field_design[group, , drop = FALSE])
    log_joint <- function(b) {
      eta <- drop(design %*% b)
      sum(df$y * eta - df$trials * softplus(eta)) - sum(precision * b^2) / 2
    }
    for (j in seq_len(30L)) {
      p <- plogis(drop(design %*% at))
      grad <- drop(crossprod(design, df$y - df$trials * p)) - precision * at
      hessian <- crossprod(design, design * (df$trials * p * (1 - p))) +
        diag(precision)
      delta <- solve(hessian, grad)
      fraction <- 1
      current <- log_joint(at)
      while (log_joint(at + fraction * delta) < current - 1e-10)
        fraction <- fraction / 2
      at <- at + fraction * delta
      if (max(abs(grad)) < 1e-8 || max(abs(delta)) < 1e-12) break
    }
    p <- plogis(drop(design %*% at))
    hessian <- crossprod(design, design * (df$trials * p * (1 - p))) +
      diag(precision)
    factor <- chol(hessian)
    b <- at + backsolve(factor, z)
    eta <- design %*% b
    joint <- colSums(df$y * eta - df$trials * softplus(eta)) -
      colSums(precision * b^2) / 2
    correction <- exp(joint - log_joint(at) + colSums(z^2) / 2)
    log_prior <- log(lambda * abs(log_sd_slope)) + log_sd_slope * x[i] -
      lambda * exp(log_sd_slope * x[i])
    log_marginal[i, ] <- log_joint(at) - sum(log(diag(factor))) +
      log_prior + log(colMeans(matrix(correction, ncol = 4L)))
    phi <- field_design %*% b[-c(1L, 2L), , drop = FALSE]
    batches <- split(seq_len(n_draws), rep(seq_len(4L), each = n_draws / 4L))
    for (k in seq_along(batches)) {
      index <- batches[[k]]
      w <- correction[index] / sum(correction[index])
      field_first[i, , k] <- drop(phi[, index, drop = FALSE] %*% w)
      field_second[i, , k] <- drop(phi[, index, drop = FALSE]^2 %*% w)
    }
  }
  moments <- function(logp, index = seq_along(x)) {
    w <- exp(logp[index] - max(logp[index]))
    w[c(1L, length(w))] <- w[c(1L, length(w))] / 2
    w <- w / sum(w)
    mean <- sum(w * x[index])
    list(mean = mean, sd = sqrt(sum(w * (x[index] - mean)^2)), weights = w)
  }
  logp <- log(rowMeans(exp(log_marginal - max(log_marginal))))
  reference <- moments(logp)
  batch_means <- apply(log_marginal, 2, function(a) moments(a)$mean)
  field_moments <- function(index = seq_along(x)) {
    w <- exp(log_marginal[index, , drop = FALSE] - max(log_marginal))
    w[c(1L, nrow(w)), ] <- w[c(1L, nrow(w)), ] / 2
    w <- w / sum(w)
    first <- vapply(seq_len(nrow(Q)), function(j)
      sum(w * field_first[index, j, ]), numeric(1))
    second <- vapply(seq_len(nrow(Q)), function(j)
      sum(w * field_second[index, j, ]), numeric(1))
    list(mean = first, sd = sqrt(second - first^2))
  }
  field_result <- field_moments()
  field_batches <- vapply(seq_len(4L), function(k)
    drop(crossprod(moments(log_marginal[, k])$weights, field_first[, , k])),
    numeric(nrow(Q)))
  coarse <- seq.int(1L, length(x), by = 2L)
  list(mean = reference$mean, sd = reference$sd,
       mcse = sd(batch_means) / 2,
       grid_error = abs(reference$mean - moments(logp, coarse)$mean),
       field_mean = field_result$mean, field_sd = field_result$sd,
       field_mcse = apply(field_batches, 1L, sd) / 2,
       field_grid_error = abs(field_result$mean - field_moments(coarse)$mean),
       tail_slope = diff(tail(logp, 2L)) / diff(tail(x, 2L)))
}

fit_one <- function(df, ..., control = list()) {
  control <- utils::modifyList(
    list(iter = 1000, warmup = 500, chains = 4, verbose = FALSE), control)
  f <- tratio(y | trials ~ x, data = df, family = ratiod_binomial(), ...,
              control = control)
  dr <- as.matrix(f$draws)
  per <- nrow(dr) / f$chains
  summarise <- function(v, transform = identity) {
    values <- transform(dr[, v])
    m <- matrix(values, nrow = per, ncol = f$chains)
    list(mean = mean(values),
         lo = unname(quantile(values, 0.025)),
         hi = unname(quantile(values, 0.975)),
         rhat = posterior::rhat(m),
         ess = posterior::ess_bulk(m),
         mcse = posterior::mcse_mean(m))
  }
  list(backend = f$backend,
       intercept = summarise("beta_num[1]"),
       slope = summarise("beta_num[2]"),
       log_tau_temporal = if ("tau_temporal" %in% colnames(dr))
         summarise("tau_temporal", log) else NULL)
}

test_that("RW1 recovers the intercept and slope with calibrated intervals", {
  skip_on_cran()
  skip_if_not_installed("posterior")

  seeds <- c(11, 22, 33)
  cover_i <- 0L; cover_s <- 0L
  for (s in seeds) {
    df <- sim_data(s)
    # At zero temporal signal the PC scale prior reaches the neck of the
    # centred funnel. Standard normal increments remove its scale dependence.
    r <- fit_one(df, temporal = temporal_rw1("time", parameterization = "noncentered"),
                 control = list(iter = 4000, warmup = 1000, adapt_delta = 0.95))
    expect_identical(r$backend, "hmc")

    # Convergence must hold before recovery means anything.
    expect_lt(r$intercept$rhat, 1.05)
    expect_lt(r$slope$rhat, 1.05)
    expect_gt(r$intercept$ess, 400)

    reference <- constrained_field_reference(
      df, crossprod(diff(diag(20))), as.integer(df$time))
    expect_lt(reference$mcse / reference$sd, 0.001)
    expect_lt(reference$grid_error / reference$sd, 0.001)
    expect_lt(r$log_tau_temporal$rhat, 1.01)
    expect_lt(r$log_tau_temporal$mcse, reference$sd / 30)
    expect_lt(abs(r$log_tau_temporal$mean - reference$mean), 0.1 * reference$sd)

    # The intercept is the quantity a missing sum-to-zero constraint destroys.
    expect_equal(r$intercept$mean, TRUE_INTERCEPT, tolerance = 0.08)
    expect_equal(r$slope$mean, TRUE_SLOPE, tolerance = 0.08)

    cover_i <- cover_i + (r$intercept$lo <= TRUE_INTERCEPT && TRUE_INTERCEPT <= r$intercept$hi)
    cover_s <- cover_s + (r$slope$lo <= TRUE_SLOPE && TRUE_SLOPE <= r$slope$hi)
  }
  expect_gte(cover_i, length(seeds) - 1L)
  expect_gte(cover_s, length(seeds) - 1L)
})

test_that("ICAR combined with RW1 recovers the intercept", {
  skip_on_cran()
  skip_if_not_installed("posterior")
  r <- fit_one(sim_data(11), spatial = spatial_car(grid_adj(50), spatial_level = "group",
                                                   group_var = "spatial_site"),
               temporal = temporal_rw1("time"))
  expect_lt(r$intercept$rhat, 1.05)
  expect_equal(r$intercept$mean, TRUE_INTERCEPT, tolerance = 0.08)
})

test_that("multiscale trend and seasonal recover the intercept", {
  skip_on_cran()
  skip_if_not_installed("posterior")

  # Trend (RW1) and seasonal (cyclic RW1) are both intrinsic and both land on
  # the same linear predictor, so each carries a constant null direction that is
  # unidentified against the intercept AND against the other. With neither
  # pinned the intercept is free to move along a two-dimensional ridge; the
  # shape of the fitted series is unaffected, which is why only a recovery
  # assertion on the intercept catches it.
  r <- fit_one(sim_data(11),
               temporal = temporal_multiscale("time", trend = "rw1",
                                              seasonal = 4, short_term = "none"))
  expect_lt(r$intercept$rhat, 1.05)
  expect_equal(r$intercept$mean, TRUE_INTERCEPT, tolerance = 0.08)
  expect_equal(r$slope$mean, TRUE_SLOPE, tolerance = 0.08)
})

test_that("multiscale RW2 trend recovers the intercept", {
  skip_on_cran()
  skip_if_not_installed("posterior")

  r <- fit_one(sim_data(22),
               temporal = temporal_multiscale("time", trend = "rw2",
                                              seasonal = NULL, short_term = "none"))
  expect_lt(r$intercept$rhat, 1.05)
  expect_equal(r$intercept$mean, TRUE_INTERCEPT, tolerance = 0.08)
})

test_that("a TVC term recovers the intercept and its own slope", {
  skip_on_cran()
  skip_if_not_installed("posterior")

  # A TVC trajectory is an intrinsic RW over time per (group, term). Its mean is
  # unidentified against the fixed coefficient on the same covariate, so the
  # quantity that moves when the pin is absent is the SLOPE rather than the
  # intercept.
  r <- fit_one(sim_data(11),
               temporal = temporal_tvc("time", terms = "x", structure = "rw1"))
  expect_lt(r$intercept$rhat, 1.05)
  expect_lt(r$slope$rhat, 1.05)
  expect_equal(r$intercept$mean, TRUE_INTERCEPT, tolerance = 0.1)
  expect_equal(r$slope$mean, TRUE_SLOPE, tolerance = 0.1)
})

test_that("a TVC term with structure = 'iid' recovers the intercept and slope (gcol33/tulpaRatio#32)", {
  skip_on_cran()
  skip_if_not_installed("posterior")

  # TemporalType::IID had a full log-prior and gradient in C++ but
  # temporal_tvc()'s match.arg blocked "iid" from R entirely.
  r <- fit_one(sim_data(11),
               temporal = temporal_tvc("time", terms = "x", structure = "iid"))
  expect_lt(r$intercept$rhat, 1.05)
  expect_lt(r$slope$rhat, 1.05)
  expect_equal(r$intercept$mean, TRUE_INTERCEPT, tolerance = 0.1)
  expect_equal(r$slope$mean, TRUE_SLOPE, tolerance = 0.1)
})

test_that("an SVC term recovers the intercept and its own slope (gcol33/tulpaRatio#25)", {
  skip_on_cran()
  skip_if_not_installed("posterior")

  # An NNGP field's prior is PROPER, so unlike the intrinsic fields above its
  # mean needs no pin supplying a prior -- it needs removing from the
  # likelihood, where it aliases with the coefficient on the covariate the
  # field rides on. The engine centres it on its way into eta, so the quantity
  # that moves when the identification is wrong is the SLOPE.
  #
  # What the slope should equal is the value THIS SAMPLE implies, not the value
  # it was simulated from. The fixture is 80 observations, where the maximum
  # likelihood slope carries a standard error of about 0.05, so a realization
  # sits a fraction of that away from 0.3 for reasons that have nothing to do
  # with the field: seed 11 implies 0.2533 and seed 22 implies 0.2062, and an
  # SVC fit reproducing 0.3 on either would be reporting the prior rather than
  # the data. Asserting against the unpenalized MLE of the same sample instead
  # removes that noise and sharpens the test by an order of magnitude -- the
  # agreement measured below is within 0.002, where a tolerance against 0.3 has
  # to allow 0.05. A field whose level reached the likelihood would pull the
  # coefficient off the MLE, which is what this catches.
  #
  # Recovery of the simulated values is still asserted, as interval coverage
  # across seeds, which is the form that fixture noise does not break.

  seeds <- c(11, 22, 33)
  cover_i <- 0L; cover_s <- 0L
  for (seed in seeds) {
    df <- sim_data(seed, N = 80)
    mle <- summary(stats::glm(cbind(y, trials - y) ~ x,
                              family = stats::binomial, data = df))$coefficients

    set.seed(seed)
    f <- tratio(y | trials ~ x, data = df, family = ratiod_binomial(),
                spatial = spatial_svc(~ lon + lat, terms = ~ x - 1, nn = 8),
                control = list(iter = 3000, warmup = 1500, chains = 4,
                               verbose = FALSE))
    expect_identical(f$backend, "hmc")

    dr <- as.matrix(f$draws)
    per <- nrow(dr) / f$chains
    summarise <- function(v) {
      m <- matrix(dr[, v], nrow = per, ncol = f$chains)
      list(mean = mean(dr[, v]),
           lo = unname(stats::quantile(dr[, v], 0.025)),
           hi = unname(stats::quantile(dr[, v], 0.975)),
           rhat = posterior::rhat(m), ess = posterior::ess_bulk(m))
    }
    int <- summarise("beta_num[1]")
    slope <- summarise("beta_num[2]")

    expect_lt(int$rhat, 1.05)
    expect_lt(slope$rhat, 1.05)
    expect_gt(slope$ess, 100)

    # Within a quarter of the sample's own standard error on the coefficient
    # the field rides on, and a third on the intercept.
    expect_lt(abs(slope$mean - mle["x", "Estimate"]),
              0.25 * mle["x", "Std. Error"])
    expect_lt(abs(int$mean - mle["(Intercept)", "Estimate"]),
              0.35 * mle["(Intercept)", "Std. Error"])

    cover_i <- cover_i + (int$lo <= TRUE_INTERCEPT && TRUE_INTERCEPT <= int$hi)
    cover_s <- cover_s + (slope$lo <= TRUE_SLOPE && TRUE_SLOPE <= slope$hi)
  }
  expect_gte(cover_i, length(seeds) - 1L)
  expect_gte(cover_s, length(seeds) - 1L)
})

test_that("a spatiotemporal interaction recovers the intercept", {
  skip_on_cran()
  skip_if_not_installed("posterior")

  # The interaction is pinned along both margins. A space margin sums S terms
  # and a time margin T, so the two carry their own constants; sharing one
  # leaves whichever margin is longer under-identified. Kept to 9 x 5 so the
  # interaction has fewer coefficients than the data has observations.
  #
  # This data carries no true space-time signal, so the interaction's
  # posterior for tau drifts to large values and, under the centered
  # parameterization, dragged rhat to 1.14 (and to 2.05 over a 4x longer run,
  # with the per-chain SD shrinking) -- a funnel between tau and the field's
  # conditional scale that gcol33/tulpaRatio#24 traced to no single step size
  # spanning both regimes. parameterization = "noncentered" samples a tau-free
  # z and reconstructs the interaction as z / sqrt(tau), which decouples the
  # two: rhat drops to 1.004 (intercept) / 1.000 (slope) at the same 1000-iter
  # budget this test uses.
  df <- sim_data(11, S = 9, TT = 5)
  r <- fit_one(df,
               spatiotemporal = spatiotemporal(
                 spatial = spatial_car(grid_adj(9), spatial_level = "group",
                                       group_var = "spatial_site"),
                 temporal = temporal_rw1("time"),
                 type = "IV",
                 parameterization = "noncentered"))
  expect_equal(r$intercept$mean, TRUE_INTERCEPT, tolerance = 0.1)
  expect_equal(r$slope$mean, TRUE_SLOPE, tolerance = 0.1)
  expect_lt(r$intercept$rhat, 1.05)
  expect_lt(r$slope$rhat, 1.05)
})

test_that("proper CAR recovers rho and the intercept (gcol33/tulpaRatio#31)", {
  skip_on_cran()
  skip_if_not_installed("posterior")

  # Simulate directly from Q(rho) = D - rho*W so the field itself, not just
  # the regression coefficients, carries a known truth to recover. rho_true
  # kept at 0.5 rather than close to 1: Q(rho)'s smallest eigenvalue shrinks
  # as rho -> 1 (a grid graph's leading eigenvector is the constant
  # direction), creating a near-collinearity between the intercept and phi's
  # mean that is a real property of proper CAR, not something this recovery
  # test is meant to stress.
  S <- 36L
  A <- grid_adj(S)
  D <- diag(rowSums(A))
  rho_true <- 0.5
  tau_true <- 3.0
  set.seed(44)
  Q <- D - rho_true * A
  # A draw through the Cholesky factor is the same on every platform, which
  # an eigendecomposition (MASS::mvrnorm) is not
  phi_true <- backsolve(chol(tau_true * Q), rnorm(S))
  # The field is centred in eta, so the intercept is the level of the
  # linear predictor: the true intercept plus the drawn field's mean
  level_true <- TRUE_INTERCEPT + mean(phi_true)

  n_per <- 25L
  N <- S * n_per
  site <- rep(seq_len(S), each = n_per)
  x <- rnorm(N)
  trials <- sample(15:50, N, replace = TRUE)
  eta <- TRUE_INTERCEPT + TRUE_SLOPE * x + phi_true[site]
  df <- data.frame(y = rbinom(N, trials, plogis(eta)), trials = trials, x = x,
                   site = factor(site))

  # Proper CAR has no specialized mass-matrix warm-start (unlike ICAR/BYM2),
  # so NUTS adapts a generic one from scratch; a longer warmup (rather than a
  # seed pick) is what buys reliable convergence on rho/tau's slower geometry.
  f <- tratio(y | trials ~ x, data = df, family = ratiod_binomial(),
             spatial = spatial_car(A, spatial_level = "group", group_var = "site", proper = TRUE),
             mode = "hmc",
             control = list(iter = 2000, warmup = 1500, chains = 4, seed = 5,
                            verbose = FALSE))
  expect_identical(f$backend, "hmc")

  dr <- as.matrix(f$draws)
  expect_true("rho_spatial" %in% colnames(dr))
  expect_true("tau_spatial" %in% colnames(dr))
  per <- nrow(dr) / f$chains
  summarise <- function(v) {
    m <- matrix(dr[, v], nrow = per, ncol = f$chains)
    list(mean = mean(dr[, v]),
         lo = unname(quantile(dr[, v], 0.025)),
         hi = unname(quantile(dr[, v], 0.975)),
         rhat = posterior::rhat(m),
         ess = posterior::ess_bulk(m))
  }
  r_int <- summarise("beta_num[1]")
  r_rho <- summarise("rho_spatial")
  r_tau <- summarise("tau_spatial")

  # Q(rho) is full rank, so the prior identifies the field's mean, but the
  # LIKELIHOOD cannot see it apart from the intercept: eta reads beta_0 + phi_s,
  # and moving the intercept by c against every phi by -c leaves eta alone. With
  # the mean left in, the two correlate at -0.977, each carries a posterior sd
  # near 0.06 while their sum carries 0.013, and the intercept is a coordinate
  # along a ridge rather than a quantity the data pin. spatial_car() removes it
  # from eta by default now, so the intercept IS the level and converges.
  expect_lt(r_int$rhat, 1.05)
  expect_gt(r_int$ess, 1000)

  expect_lt(r_rho$rhat, 1.1)
  expect_equal(r_int$mean, level_true, tolerance = 0.15)
  # Within three posterior SD: one seed's 95% interval misses the truth in
  # about one run in twenty by construction
  expect_lt(abs(r_int$mean - level_true) / sd(dr[, "beta_num[1]"]), 3)

  # rho is weakly identified even at this size (a well-known property of
  # proper CAR, not specific to this implementation), so the interval, not
  # the point estimate, is what recovery means here.
  expect_true(r_rho$lo <= rho_true && rho_true <= r_rho$hi,
              label = sprintf("rho 95%% CI [%.3f, %.3f] vs true %.2f",
                              r_rho$lo, r_rho$hi, rho_true))
  expect_true(r_tau$lo <= tau_true && tau_true <= r_tau$hi,
              label = sprintf("tau 95%% CI [%.3f, %.3f] vs true %.2f",
                              r_tau$lo, r_tau$hi, tau_true))
})

test_that("a spatial-only model reaches a sampler that reports its chains", {
  skip_on_cran()
  # Spatial-only models route to the Gibbs backend, which must honour chains=
  # and expose the spatial field so the convergence diagnostics above apply to
  # them too.
  df <- sim_data(11)
  f <- tratio(y | trials ~ x, data = df, family = ratiod_binomial(),
              spatial = spatial_car(grid_adj(50), spatial_level = "group",
                                    group_var = "spatial_site"),
              control = list(iter = 1000, warmup = 500, chains = 4, verbose = FALSE))
  expect_identical(as.integer(f$chains), 4L)
  expect_true(any(grepl("^phi_spatial\\[", colnames(as.matrix(f$draws)))))
})

test_that("an intrinsic field is reported as the field the likelihood saw", {
  skip_on_cran()

  # The intrinsic fields enter eta centred, and hmc_unpack adds the STORED
  # field straight back into eta for fitted(), ratio() and predict(). A draw
  # that kept its constant would report -- and predict from -- a series that
  # differs from the fitted one by a level the likelihood never saw, so the
  # constraint has to hold on the stored draws and not only inside the density.
  block_means <- function(f, prefix) {
    dr <- as.matrix(f$draws)
    cols <- which(startsWith(colnames(dr), paste0(prefix, "[")))
    expect_gt(length(cols), 1L)
    rowMeans(dr[, cols, drop = FALSE])
  }

  df <- sim_data(11)
  ctl <- list(iter = 400, warmup = 200, chains = 2, seed = 3, verbose = FALSE)

  f_rw1 <- tratio(y | trials ~ x, data = df, family = ratiod_binomial(),
                  temporal = temporal_rw1("time"), control = ctl)
  expect_lt(max(abs(block_means(f_rw1, "temporal"))), 1e-9)

  f_ms <- tratio(y | trials ~ x, data = df, family = ratiod_binomial(),
                 temporal = temporal_multiscale("time", trend = "rw1",
                                                seasonal = 4,
                                                short_term = "none"),
                 control = ctl)
  expect_lt(max(abs(block_means(f_ms, "trend"))), 1e-9)
  expect_lt(max(abs(block_means(f_ms, "seasonal"))), 1e-9)

  # AR1 is proper and identifies its own level, so its draws are NOT centred:
  # centring it would impose a constraint the model does not carry.
  f_ar1 <- tratio(y | trials ~ x, data = df, family = ratiod_binomial(),
                  temporal = temporal_ar1("time"), control = ctl)
  expect_gt(max(abs(block_means(f_ar1, "temporal"))), 1e-9)
})


# The PC prior's log-precision density has tail exp(-x/2). The likelihood
# approaches a finite limit as the field shrinks to zero. Counting the constant
# direction twice multiplies the marginal by exp(x/2), leaving an improper tail.
test_that("sampled and collapsed ICAR recover the dense log-precision posterior", {
  skip_on_cran()
  skip_if_not_installed("posterior")
  if (!identical(Sys.getenv("TULPA_SLOW_TESTS"), "true"))
    skip("Set TULPA_SLOW_TESTS=true")

  S <- 9L
  df <- sim_data(11, N = 300, S = S)
  A <- grid_adj(S)
  reference <- constrained_field_reference(
    df, diag(rowSums(A)) - A, as.integer(df$site))
  expect_lt(reference$mcse / reference$sd, 0.001)
  expect_lt(reference$grid_error / reference$sd, 0.001)
  expect_equal(reference$tail_slope, -0.5, tolerance = 1e-8)

  fit_tau <- function(parameterization, seed, walnuts = FALSE) {
    f <- tratio(y | trials ~ x, data = df, family = ratiod_binomial(),
                mode = "hmc",
                spatial = spatial_car(A, spatial_level = "group", group_var = "site",
                                      parameterization = parameterization),
                control = list(iter = if (walnuts) 50000 else 22000,
                               warmup = 2000, chains = 4, seed = seed,
                               adapt_delta = if (walnuts) 0.8 else 0.99,
                               walnuts = walnuts, verbose = FALSE))
    expect_identical(f$backend, "hmc")
    if (walnuts) expect_identical(f$diagnostics$algorithm, "WALNUTS")
    dr <- as.matrix(f$draws)
    log_tau <- log(dr[, "tau_spatial"])
    m <- matrix(log_tau, nrow = nrow(dr) / f$chains, ncol = f$chains)
    list(draws = m, intercept = mean(dr[, "beta_num[1]"]))
  }

  summarise_tau <- function(fits) {
    m <- do.call(cbind, lapply(fits, `[[`, "draws"))
    list(mean = mean(m),
         mcse = posterior::mcse_mean(m),
         ess = posterior::ess_bulk(m),
         rhat = posterior::rhat(m),
         intercept = mean(vapply(fits, `[[`, numeric(1), "intercept")))
  }

  # Local leapfrog subdivisions adapt to the field curvature at each precision.
  sampled_fits <- lapply(c(7L, 17L), function(seed) fit_tau("standard", seed, TRUE))
  standard <- summarise_tau(sampled_fits)
  collapsed <- summarise_tau(list(fit_tau("collapsed", 7L)))

  for (fit in sampled_fits) {
    result <- summarise_tau(list(fit))
    expect_lt(result$rhat, 1.01)
    expect_lt(abs(result$mean - reference$mean), 0.1 * reference$sd)
  }

  # A mean accurate to 0.1 posterior SD at three Monte Carlo standard errors
  # requires MCSE below SD/30. Each route is checked against the integrated
  # posterior independently.
  for (result in list(standard, collapsed)) {
    error <- abs(result$mean - reference$mean)
    expect_lt(result$rhat, 1.01)
    expect_lt(result$mcse, reference$sd / 30)
    expect_lt(error, 0.1 * reference$sd)
    expect_lt(error, 3 * sqrt(result$mcse^2 + reference$mcse^2) +
                reference$grid_error)
  }

  # The pin holds the constant at fixed precision rather than exactly zero, so
  # the collapsed marginal is the hard-constrained one plus a leak. It shows up
  # as a level shared between the intercept and the field constant, which is
  # what this bounds.
  expect_lt(abs(collapsed$intercept - standard$intercept), 0.005)

  # Marginalizing phi is what the collapsed path is for.
  expect_gt(collapsed$ess, standard$ess)
})


test_that("multiscale RW2 funnel diagnostics and posterior agree with dense integration", {
  skip_on_cran()
  skip_if_not_installed("posterior")
  if (!identical(Sys.getenv("TULPA_SLOW_TESTS"), "true"))
    skip("Set TULPA_SLOW_TESTS=true")

  df <- sim_data(22)
  fit <- function(parameterization, seed, iter = 1000L, warmup = 500L,
                  adapt_delta = NULL) {
    f <- tratio(y | trials ~ x, data = df, family = ratiod_binomial(),
      temporal = temporal_multiscale("time", trend = "rw2", seasonal = NULL,
        short_term = "none", parameterization = parameterization),
      control = list(iter = iter, warmup = warmup, chains = 4, seed = seed,
        adapt_delta = adapt_delta, verbose = FALSE))
    expect_identical(f$backend, "hmc")
    dr <- as.matrix(f$draws)
    log_sigma2 <- log(dr[, "sigma2_trend"])
    list(draws = dr, log_sigma2 = matrix(log_sigma2, ncol = f$chains),
         divergent = as.logical(f$diagnostics$divergent),
         neck = log_sigma2 < unname(quantile(log_sigma2, 1 / 3)))
  }

  # The lower tertile defines the neck within each fit. Pooling independent
  # seeds measures its divergence share across sampling trajectories.
  centred <- lapply(c(3L, 7L, 17L, 23L, 31L, 43L, 59L),
                    function(seed) fit("centered", seed))
  divergent <- unlist(lapply(centred, `[[`, "divergent"))
  neck <- unlist(lapply(centred, `[[`, "neck"))
  expect_lt(mean(divergent), 0.05)
  expect_gte(sum(divergent), 5L)
  expect_gt(mean(neck[divergent]), 0.8)
  for (result in centred) {
    if (sum(result$divergent) >= 5L)
      expect_gt(mean(result$neck[result$divergent]), mean(result$neck))
  }
  centred_draws <- do.call(rbind, lapply(centred, `[[`, "draws"))
  expect_lt(median(centred_draws[, "sigma2_trend"]), 1e-2)

  # RW2 leaves a flat linear direction after its constant is constrained.
  n <- nlevels(df$time)
  ramp <- seq_len(n) - mean(seq_len(n))
  ramp <- matrix(ramp / sqrt(sum(ramp^2)), ncol = 1L)
  reference <- constrained_field_reference(df,
    crossprod(diff(diag(n), differences = 2)), as.integer(df$time),
    seed = 228L, flat = ramp, log_sd_slope = 0.5, U = 1,
    grid = seq(-60, 0, by = 0.25))
  expect_lt(reference$mcse / reference$sd, 0.001)
  expect_lt(reference$grid_error / reference$sd, 0.001)
  expect_lt(max(reference$field_mcse / reference$field_sd), 0.001)
  expect_lt(max(reference$field_grid_error / reference$field_sd), 0.001)

  # A 0.1 posterior-SD mean error at three Monte Carlo standard errors requires
  # MCSE below SD/30. The field moments also integrate over log variance.
  nc <- lapply(c(3L, 7L, 17L), function(seed)
    fit("noncentered", seed, iter = 4000L, warmup = 1000L, adapt_delta = 0.99))
  log_sigma2 <- do.call(cbind, lapply(nc, `[[`, "log_sigma2"))
  error <- abs(mean(log_sigma2) - reference$mean)
  mcse <- posterior::mcse_mean(log_sigma2)
  expect_lt(posterior::rhat(log_sigma2), 1.01)
  expect_lt(mcse, reference$sd / 30)
  expect_lt(error, 0.1 * reference$sd)
  expect_lt(error, 3 * sqrt(mcse^2 + reference$mcse^2) + reference$grid_error)
  expect_lt(mean(unlist(lapply(nc, `[[`, "divergent"))), 0.005)
  nc_draws <- do.call(rbind, lapply(nc, `[[`, "draws"))
  field <- nc_draws[, startsWith(colnames(nc_draws), "trend["), drop = FALSE]
  expect_lt(max(abs(colMeans(field) - reference$field_mean) / reference$field_sd), 0.1)
  for (result in nc) {
    expect_lt(posterior::rhat(result$log_sigma2), 1.01)
    expect_lt(abs(mean(result$log_sigma2) - reference$mean), 0.1 * reference$sd)
  }
  for (dr in list(centred_draws, nc_draws)) {
    expect_equal(mean(dr[, "beta_num[1]"]), TRUE_INTERCEPT, tolerance = 0.08)
    expect_equal(mean(dr[, "beta_num[2]"]), TRUE_SLOPE, tolerance = 0.08)
  }
})


# center = FALSE keeps the field's mean in the linear predictor, which is what
# spatial_car(proper = TRUE) did before. The two parameterisations describe the
# same identified posterior and differ in how the level is attributed, so this
# asserts both halves: the level agrees, and the intercept alone only converges
# under centring.

test_that("centring a proper CAR field moves the level onto the intercept", {
  skip_on_cran()
  skip_if_not_installed("posterior")

  S <- 36L
  A <- grid_adj(S)
  rho_true <- 0.5
  tau_true <- 3.0
  set.seed(44)
  phi_true <- backsolve(chol(tau_true * (diag(rowSums(A)) - rho_true * A)),
                        rnorm(S))

  n_per <- 25L
  N <- S * n_per
  site <- rep(seq_len(S), each = n_per)
  x <- rnorm(N)
  trials <- sample(15:50, N, replace = TRUE)
  df <- data.frame(
    y = rbinom(N, trials, plogis(TRUE_INTERCEPT + TRUE_SLOPE * x + phi_true[site])),
    trials = trials, x = x, site = factor(site))

  fit <- function(center) {
    f <- tratio(y | trials ~ x, data = df, family = ratiod_binomial(), mode = "hmc",
                spatial = spatial_car(A, spatial_level = "group", group_var = "site",
                                      proper = TRUE, center = center),
                control = list(iter = 2000, warmup = 1500, chains = 4, seed = 5,
                               verbose = FALSE))
    dr <- as.matrix(f$draws)
    per <- nrow(dr) / f$chains
    b0 <- dr[, "beta_num[1]"]
    field_mean <- rowMeans(dr[, grep("^phi_spatial", colnames(dr)), drop = FALSE])
    # The level eta actually carries, written in each parameterisation's terms.
    level <- if (center) b0 else b0 + field_mean
    list(intercept_ess = posterior::ess_bulk(matrix(b0, per, f$chains)),
         intercept_rhat = posterior::rhat(matrix(b0, per, f$chains)),
         level = mean(level),
         slope = mean(dr[, "beta_num[2]"]),
         rho = mean(dr[, "rho_spatial"]))
  }

  raw <- fit(FALSE)
  centred <- fit(TRUE)

  # Same identified posterior. Centring is a change of coordinates on the level,
  # not a change of model fit.
  expect_equal(centred$level, raw$level, tolerance = 0.02)
  expect_equal(centred$slope, raw$slope, tolerance = 0.02)
  expect_equal(centred$rho, raw$rho, tolerance = 0.15)

  # What it buys. Measured on this fixture: ESS 71 -> 1953, rhat 1.053 -> 1.001.
  expect_lt(raw$intercept_ess, 400)
  expect_gt(centred$intercept_ess, 1000)
  expect_gt(centred$intercept_ess, 4 * raw$intercept_ess)
  expect_lt(centred$intercept_rhat, 1.05)

  # An intrinsic field has no such choice: its mean is unidentified until it is
  # removed, so there is no model on the other side of the switch.
  expect_error(spatial_car(A, spatial_level = "group", group_var = "site", center = FALSE),
               "proper = TRUE")
})
