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

fit_one <- function(df, ...) {
  f <- tratio(y | trials ~ x, data = df, family = ratiod_binomial(), ...,
              control = list(iter = 1000, warmup = 500, chains = 4, verbose = FALSE))
  dr <- as.matrix(f$draws)
  per <- nrow(dr) / f$chains
  summarise <- function(v) {
    m <- matrix(dr[, v], nrow = per, ncol = f$chains)
    list(mean = mean(dr[, v]),
         lo = unname(quantile(dr[, v], 0.025)),
         hi = unname(quantile(dr[, v], 0.975)),
         rhat = posterior::rhat(m),
         ess = posterior::ess_bulk(m))
  }
  list(backend = f$backend,
       intercept = summarise("beta_num[1]"),
       slope = summarise("beta_num[2]"))
}

test_that("RW1 recovers the intercept and slope with calibrated intervals", {
  skip_on_cran()
  skip_if_not_installed("posterior")

  seeds <- c(11, 22, 33)
  cover_i <- 0L; cover_s <- 0L
  for (s in seeds) {
    r <- fit_one(sim_data(s), temporal = temporal_rw1("time"))
    expect_identical(r$backend, "hmc")

    # Convergence must hold before recovery means anything.
    expect_lt(r$intercept$rhat, 1.05)
    expect_lt(r$slope$rhat, 1.05)
    expect_gt(r$intercept$ess, 400)

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
  r <- fit_one(sim_data(11), spatial = spatial_car(grid_adj(50), level = "group",
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
                 spatial = spatial_car(grid_adj(9), level = "group",
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
             spatial = spatial_car(A, level = "group", group_var = "site", proper = TRUE),
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
              spatial = spatial_car(grid_adj(50), level = "group",
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


# The sampled ICAR block identifies the field by augment-and-centre and the
# collapsed block keeps the soft pin (hmc_icar_collapsed.h). The two are claimed
# to produce the same marginal: the freed constant is centred out of eta, so it
# factors off as an independent N(0, 1/tau) integrating to 1, which turns the
# augmented normalizer S/2 * log(tau) into the (S - 1)/2 * log(tau) the collapsed
# prior carries. If that identity broke -- the collapsed rank counting the pinned
# direction, or the pin picking up a tau it should not -- the prior would gain or
# lose half a power of tau and the tau posterior would move.
#
# The intercept cannot see that. It is separately identified, so it tests only
# the pin's residual leak, and a normalizer off by one rank leaves it where it
# was. tau is the quantity that moves, and it moves by roughly
# digamma(S/2) - digamma((S-1)/2), which is ~1/S: 0.133 at S = 9 against a
# Monte Carlo error of ~0.04, but 0.021 at S = 50 against ~0.05, where it would
# be invisible. So this asserts on a small field.

test_that("the collapsed ICAR marginal agrees with the sampled one on tau", {
  skip_on_cran()
  skip_if_not_installed("posterior")

  S <- 9L
  df <- sim_data(11, N = 300, S = S)
  A <- grid_adj(S)

  fit_tau <- function(parameterization) {
    set.seed(7)
    f <- tratio(y | trials ~ x, data = df, family = ratiod_binomial(),
                mode = "hmc",
                spatial = spatial_car(A, level = "group", group_var = "site",
                                      parameterization = parameterization),
                control = list(iter = 2000, warmup = 1000, chains = 4,
                               verbose = FALSE))
    expect_identical(f$backend, "hmc")
    dr <- as.matrix(f$draws)
    log_tau <- log(dr[, "tau_spatial"])
    m <- matrix(log_tau, nrow = nrow(dr) / f$chains, ncol = f$chains)
    list(mean = mean(log_tau),
         mcse = posterior::mcse_mean(m),
         ess = posterior::ess_bulk(m),
         intercept = mean(dr[, "beta_num[1]"]))
  }

  standard <- fit_tau("standard")
  collapsed <- fit_tau("collapsed")

  rank_error_shift <- digamma(S / 2) - digamma((S - 1) / 2)
  diff <- collapsed$mean - standard$mean
  mcse <- sqrt(standard$mcse^2 + collapsed$mcse^2)

  # Well inside Monte Carlo error, and comfortably below the shift a rank that
  # counted the pinned direction would produce.
  expect_lt(abs(diff), 4 * mcse)
  expect_lt(abs(diff), 0.6 * rank_error_shift)

  # The pin holds the constant at fixed precision rather than exactly zero, so
  # the collapsed marginal is the hard-constrained one plus a leak. It shows up
  # as a level shared between the intercept and the field constant, which is
  # what this bounds.
  expect_lt(abs(collapsed$intercept - standard$intercept), 0.005)

  # Marginalizing phi is what the collapsed path is for.
  expect_gt(collapsed$ess, standard$ess)
})


# The multiscale RW2 fixture simulates eta = TRUE_INTERCEPT + TRUE_SLOPE * x
# with no trend, so sigma2_trend's true value is zero and the fit sits in the
# neck of a centred-parameterisation funnel by construction. Augmenting the
# constant raised the step size, which is what let the sampler reach the neck at
# all: the divergence count went 15 -> 38 out of 2000 while the intercept
# recovery was unchanged.
#
# A count alone cannot separate "reached further into a known funnel" from "a
# new pathology somewhere else", and it is not comparable across step sizes.
# What distinguishes them is WHERE the divergences sit. They belong at the small
# end of sigma2_trend and nowhere else; divergences spread across the range
# would be a different fault. The trend arm's non-centred coordinate is the fix
# (gcol33/tulpaRatio#79).

test_that("multiscale RW2 divergences stay in the sigma2_trend funnel neck", {
  skip_on_cran()

  set.seed(3)
  f <- tratio(y | trials ~ x, data = sim_data(22), family = ratiod_binomial(),
              temporal = temporal_multiscale("time", trend = "rw2",
                                             seasonal = NULL, short_term = "none"),
              control = list(iter = 1000, warmup = 500, chains = 4,
                             verbose = FALSE))
  expect_identical(f$backend, "hmc")

  divergent <- as.logical(f$diagnostics$divergent)
  log_sigma2 <- log(as.matrix(f$draws)[, "sigma2_trend"])

  # The arm is shrunk to its true value of zero, which is where the neck is.
  expect_lt(median(exp(log_sigma2)), 1e-2)

  # Bounded, so the rate cannot climb unnoticed.
  expect_lt(mean(divergent), 0.05)

  skip_if(sum(divergent) < 5, "too few divergences to locate")

  # Every divergence in the bottom third of sigma2_trend is the funnel. Any
  # appreciable share above it is not, and is what this guards against.
  cutoff <- unname(quantile(log_sigma2, 1 / 3))
  expect_gt(mean(log_sigma2[divergent] < cutoff), 0.8)

  # Same statement the other way: the divergent draws sit well below the clean
  # ones on the scale the funnel is in.
  shift <- (mean(log_sigma2[divergent]) - mean(log_sigma2[!divergent])) /
    stats::sd(log_sigma2)
  expect_lt(shift, -1)

  # None of this is allowed to cost the recovery the fixture is for.
  dr <- as.matrix(f$draws)
  expect_equal(mean(dr[, "beta_num[1]"]), TRUE_INTERCEPT, tolerance = 0.08)
  expect_equal(mean(dr[, "beta_num[2]"]), TRUE_SLOPE, tolerance = 0.08)
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
                spatial = spatial_car(A, level = "group", group_var = "site",
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
  expect_error(spatial_car(A, level = "group", group_var = "site", center = FALSE),
               "proper = TRUE")
})


# The centred coordinate cannot reach the neck of a variance shrunk toward zero,
# so it reports that variance too large. This fixture's trend is exactly zero,
# which puts the whole posterior in the neck, and the non-centred coordinate is
# what lets the sampler descend into it (gcol33/tulpaRatio#79).

test_that("the multi-scale trend's non-centred coordinate drains the funnel", {
  skip_on_cran()
  skip_if_not_installed("posterior")

  fit <- function(parameterization) {
    set.seed(3)
    f <- tratio(y | trials ~ x, data = sim_data(22), family = ratiod_binomial(),
                temporal = temporal_multiscale("time", trend = "rw2",
                                               seasonal = NULL,
                                               short_term = "none",
                                               parameterization = parameterization),
                control = list(iter = 1000, warmup = 500, chains = 4,
                               verbose = FALSE))
    dr <- as.matrix(f$draws)
    per <- nrow(dr) / f$chains
    log_s2 <- log(dr[, "sigma2_trend"])
    list(divergent = mean(as.logical(f$diagnostics$divergent)),
         ess = posterior::ess_bulk(matrix(log_s2, per, f$chains)),
         rhat = posterior::rhat(matrix(log_s2, per, f$chains)),
         median = median(dr[, "sigma2_trend"]),
         intercept = mean(dr[, "beta_num[1]"]),
         slope = mean(dr[, "beta_num[2]"]))
  }

  centred <- fit("centered")
  nc <- fit("noncentered")

  # Measured: 30/2000 divergences -> 3/2000, ESS 21 -> 310, rhat 1.126 -> 1.013.
  expect_lt(nc$divergent, 0.005)
  expect_lt(nc$divergent, centred$divergent)
  expect_gt(nc$ess, 150)
  expect_gt(nc$ess, 4 * centred$ess)
  expect_lt(nc$rhat, 1.05)

  # The coordinate the centred sampler could not reach is the small end, so what
  # it reports is biased upward: 1.6e-04 against 4.2e-05 here. Only the
  # non-centred run has converged on that parameter, which is what makes it the
  # one to believe.
  expect_lt(nc$median, centred$median)

  # Everything the fixture is actually about is unmoved.
  expect_equal(nc$intercept, centred$intercept, tolerance = 0.05)
  expect_equal(nc$slope, centred$slope, tolerance = 0.05)
  expect_equal(nc$intercept, TRUE_INTERCEPT, tolerance = 0.08)
  expect_equal(nc$slope, TRUE_SLOPE, tolerance = 0.08)
})
