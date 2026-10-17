test_that("cpp_test_st_gp_log_lik is the covariance it claims", {
  # The NNGP over a field's index set is EXACT when every location conditions
  # on all of its predecessors. At nn = n - 1 the shipped density therefore has
  # to reproduce the dense multivariate normal built from the same kernel, and
  # that identity is what pins the covariance assembly, the ordering and the
  # neighbour Cholesky against the definition instead of against themselves.
  #
  # The reference below is written from the kernel formulas, not from the
  # engine: unit-variance kernels multiplied (or Gneiting 2002 eq. 14 at
  # a = c = tau = 1, alpha = 1, gamma = 1/2, beta = 1) and scaled by sigma2.

  unit_kernel <- function(d, phi, cov) {
    switch(cov,
      exponential = exp(-d / phi),
      matern      = (1 + sqrt(3) * d / phi) * exp(-sqrt(3) * d / phi),
      gaussian    = exp(-0.5 * (d / phi)^2),
      spherical   = ifelse(d >= phi, 0, 1 - 1.5 * (d / phi) + 0.5 * (d / phi)^3),
      stop("unknown kernel")
    )
  }

  st_corr_ref <- function(h, u, phi_s, phi_t, nonsep, cov_s, cov_t) {
    if (nonsep == "gneiting") {
      base <- (u / phi_t)^2 + 1
      return(exp(-(h / phi_s) / sqrt(base)) / base)
    }
    unit_kernel(h, phi_s, cov_s) * unit_kernel(u, phi_t, cov_t)
  }

  dense_logpdf <- function(w, Sigma) {
    L <- chol(Sigma)
    z <- backsolve(L, w, transpose = TRUE)
    -0.5 * length(w) * log(2 * pi) - sum(log(diag(L))) - 0.5 * sum(z^2)
  }

  set.seed(11)
  S <- 5L; T <- 4L; n <- S * T
  xy <- cbind(runif(S), runif(S))
  coords <- xy[rep(seq_len(S), each = T), , drop = FALSE]
  times <- rep(seq_len(T) / T, times = S)

  H <- as.matrix(dist(coords))
  U <- abs(outer(times, times, "-"))

  nb <- tulpaRatio:::compute_st_nngp_neighbors(coords, times, k = n - 1L)
  w <- rnorm(n)

  nngp <- function(sigma2, phi_s, phi_t, nonsep, cov) {
    tulpaRatio:::cpp_test_st_gp_log_lik(
      w = w, sigma2 = sigma2, phi_space = phi_s, phi_time = phi_t,
      coords = coords, time_values = times, nn = n - 1L,
      nn_idx = as.integer(t(nb$nn_idx)),
      nn_dist_space = as.numeric(t(nb$nn_dist_space)),
      nn_dist_time = as.numeric(t(nb$nn_dist_time)),
      nn_order = as.integer(nb$nn_order),
      nonsep_type = nonsep, cov_space = cov, cov_time = cov
    )
  }

  # The gaussian kernel is left out of the identity and measured on its own
  # below: it is smooth enough that the neighbour blocks' 1e-8 ridge moves the
  # density by more than any tolerance worth calling exact.
  grid <- expand.grid(
    nonsep = c("product", "gneiting"),
    cov = c("exponential", "matern"),
    sigma2 = c(0.4, 2.5),
    stringsAsFactors = FALSE
  )

  for (i in seq_len(nrow(grid))) {
    nonsep <- grid$nonsep[i]; cov <- grid$cov[i]; sigma2 <- grid$sigma2[i]
    phi_s <- 0.35; phi_t <- 0.6

    R <- matrix(st_corr_ref(as.vector(H), as.vector(U), phi_s, phi_t,
                            nonsep, cov, cov), n, n)
    expect_equal(nngp(sigma2, phi_s, phi_t, nonsep, cov),
                 dense_logpdf(w, sigma2 * R),
                 tolerance = 1e-5,
                 info = paste(nonsep, cov, "sigma2 =", sigma2))
  }

  # What separates the ridge from a missing term: a term absent from one side is
  # wrong by a fixed amount, while the ridge's effect falls with the
  # conditioning it is being felt through. Measured on this fixture at
  # phi_space 0.35 / 0.15, gaussian product: condition number 3.35e5 / 3.84e4,
  # deviation 2.07e-01 / 2.37e-02 -- the same factor of 8.7 in both.
  dev_at <- function(phi_s) {
    R <- matrix(st_corr_ref(as.vector(H), as.vector(U), phi_s, 0.6,
                            "product", "gaussian", "gaussian"), n, n)
    e <- eigen(R, symmetric = TRUE, only.values = TRUE)$values
    c(cond = max(e) / min(e),
      dev = abs(nngp(0.4, phi_s, 0.6, "product", "gaussian") -
                  dense_logpdf(w, 0.4 * R)))
  }
  coarse <- dev_at(0.35)
  fine <- dev_at(0.15)
  expect_gt(coarse[["cond"]] / fine[["cond"]], 5)
  expect_equal(coarse[["dev"]] / fine[["dev"]],
               coarse[["cond"]] / fine[["cond"]], tolerance = 0.25)
})


test_that("the density falls to -Inf on a structure that is not filled", {
  # Every GP member is defaulted and st_gp_structure_filled() is the check, so
  # a half-filled structure is a rejected parameter state rather than a read
  # past the end of an empty vector (gcol33/tulpaRatio#69).
  set.seed(3)
  n <- 6L
  coords <- cbind(runif(n), runif(n))
  times <- as.numeric(seq_len(n))
  w <- rnorm(n)

  got <- tulpaRatio:::cpp_test_st_gp_log_lik(
    w = w, sigma2 = 1, phi_space = 0.5, phi_time = 0.5,
    coords = coords, time_values = times, nn = 2L,
    nn_idx = integer(0),          # the structure the R door refuses to build
    nn_dist_space = numeric(0),
    nn_dist_time = numeric(0),
    nn_order = seq_len(n)
  )
  expect_true(is.infinite(got) && got < 0)
})


test_that("an additive space-time kernel cannot carry an interaction", {
  # Why "sum" is not among the nonsep_type choices: on a complete S x T grid
  # the additive kernel has rank S + T - 1, and every direction it drops is an
  # interaction. Shown here on the kernel itself, with the product and Gneiting
  # arms as controls.
  set.seed(5)
  S <- 6L; T <- 5L
  xy <- cbind(runif(S), runif(S))
  coords <- xy[rep(seq_len(S), each = T), , drop = FALSE]
  times <- rep(seq_len(T), times = S)
  H <- as.matrix(dist(coords)); U <- abs(outer(times, times, "-"))
  ks <- exp(-H / 0.3); kt <- exp(-U / 0.4)

  rank_of <- function(M) {
    e <- eigen(M, symmetric = TRUE, only.values = TRUE)$values
    sum(e > 1e-8 * max(e))
  }

  expect_equal(rank_of(0.5 * (ks + kt)), S + T - 1L)
  expect_equal(rank_of(ks * kt), S * T)

  # And the direction it drops is literally an interaction contrast.
  v <- numeric(S * T)
  v[1] <- 1; v[2] <- -1; v[T + 1L] <- -1; v[T + 2L] <- 1
  expect_lt(abs(as.numeric(t(v) %*% (0.5 * (ks + kt)) %*% v)), 1e-10)
  expect_gt(as.numeric(t(v) %*% (ks * kt) %*% v), 1e-3)
})

test_that("the margin penalty's normalizer is 0.5 log|I + B' Sigma B| with its derivatives (#151)", {
  # At nn = n - 1 the NNGP is the dense GP, so the normalizer has to equal the
  # log-determinant written from its definition: B carries 1 / (kappa * S) on
  # every cell of each time slice and 1 / (kappa * T) on every cell of each
  # region, kappa = 0.001.
  set.seed(12)
  S <- 5L; T <- 4L; n <- S * T
  xy <- cbind(runif(S), runif(S))
  coords <- xy[rep(seq_len(S), each = T), , drop = FALSE]
  times <- rep(seq_len(T) / T, times = S)
  H <- as.matrix(dist(coords))
  U <- abs(outer(times, times, "-"))
  nb <- tulpaRatio:::compute_st_nngp_neighbors(coords, times, k = n - 1L)

  slice <- rep(seq_len(T), times = S)
  region <- rep(seq_len(S), each = T)
  B <- cbind(sapply(seq_len(T), function(t) (slice == t) / (0.001 * S)),
             sapply(seq_len(S), function(s) (region == s) / (0.001 * T)))

  dense <- function(tau, phi_s, phi_t) {
    base <- (U / phi_t)^2 + 1
    Sigma <- exp(-(H / phi_s) / sqrt(base)) / base / tau
    0.5 * as.numeric(determinant(diag(ncol(B)) + t(B) %*% Sigma %*% B)$modulus)
  }
  engine <- function(tau, phi_s, phi_t) {
    tulpaRatio:::cpp_test_st_gp_margin_normalizer(
      tau = tau, phi_space = phi_s, phi_time = phi_t, n_spatial = S, n_times = T,
      coords = coords, time_values = times, nn = n - 1L,
      nn_idx = as.integer(t(nb$nn_idx)),
      nn_dist_space = as.numeric(t(nb$nn_dist_space)),
      nn_dist_time = as.numeric(t(nb$nn_dist_time)),
      nn_order = as.integer(nb$nn_order))
  }

  for (p in list(c(0.5, 0.2, 0.3), c(3, 2, 0.1), c(0.25, 8, 1))) {
    r <- engine(p[1], p[2], p[3])
    expect_equal(r$value, dense(p[1], p[2], p[3]), tolerance = 1e-8)
    h <- 1e-5
    fd <- sapply(1:3, function(k) {
      up <- dn <- log(p)
      up[k] <- up[k] + h; dn[k] <- dn[k] - h
      (dense(exp(up[1]), exp(up[2]), exp(up[3])) -
         dense(exp(dn[1]), exp(dn[2]), exp(dn[3]))) / (2 * h)
    })
    expect_equal(r$grad, fd, tolerance = 1e-5)
  }

  # it moves with the range, which a count of pinned directions cannot
  expect_gt(abs(engine(1, 8, 0.3)$value - engine(1, 0.2, 0.3)$value), 1)
})

test_that("an AR1 time margin's normalizer term matches the dense integral", {
  # The penalised Knorr-Held prior on an S x T grid integrates to
  # exp(normalizer) * |Q|^(1/2) |Q + P|^(-1/2), so with the rho term added the
  # integral must not move with rho.
  S <- 5L; T <- 4L; tau <- 2
  ar1_R <- function(rho) {
    R <- diag(c(1, rep(1 + rho^2, T - 2), 1))
    for (i in 1:(T - 1)) R[i, i + 1] <- R[i + 1, i] <- -rho
    R
  }
  rows <- sapply(1:T, function(t) kronecker(rep(1, S), diag(T)[, t]))
  cols <- sapply(1:S, function(s) kronecker(diag(S)[, s], rep(1, T)))
  P <- tcrossprod(rows) / (0.001 * S)^2 + tcrossprod(cols) / (0.001 * T)^2
  Qicar <- matrix(0, S, S)
  for (i in 1:(S - 1)) Qicar[i, i + 1] <- Qicar[i + 1, i] <- -1
  diag(Qicar) <- -rowSums(Qicar)
  ldet <- function(M) as.numeric(determinant(M)$modulus)
  log_mass <- function(rho, type) {
    Qt <- if (type == "II") diag(S) else Qicar
    n_copies <- if (type == "II") S else S - 1
    0.5 * n_copies * log(1 - rho^2) -
      0.5 * ldet(tau * kronecker(Qt, ar1_R(rho)) + P) +
      tulpaRatio:::cpp_test_st_ar1_margin_normalizer(rho, S, T, type)[1]
  }

  for (type in c("II", "IV")) {
    m <- sapply(c(-0.6, 0, 0.5, 0.9), log_mass, type = type)
    expect_lt(diff(range(m)), 1e-3)
    for (rho in c(-0.6, 0.3, 0.8)) {
      h <- 1e-6
      fd <- (tulpaRatio:::cpp_test_st_ar1_margin_normalizer(rho + h, S, T, type)[1] -
               tulpaRatio:::cpp_test_st_ar1_margin_normalizer(rho - h, S, T, type)[1]) / (2 * h)
      expect_equal(tulpaRatio:::cpp_test_st_ar1_margin_normalizer(rho, S, T, type)[2],
                   fd, tolerance = 1e-6)
    }
  }
})
