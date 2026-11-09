test_that("HSGP spectra reproduce the marginal variance in one and two dimensions", {
  sigma2 <- 1.7
  ell <- c(0.25, 0.5, 1)
  L <- 8
  j <- seq_len(400L)
  x <- c(-1, 0, 0.7, 1)
  phi1 <- outer(x, j, function(x, j) sin(pi * j * (x + L) / (2 * L))) / sqrt(L)
  eig1 <- (pi * j / (2 * L))^2
  basis <- tulpaRatio:::hsgp_basis_2d(cbind(x, rev(x)), length(j), L)
  for (dimension in 1:2) for (kernel in 0:2) {
    Phi <- if (dimension == 1L) phi1 else basis$phi
    eig <- if (dimension == 1L) eig1 else basis$eigenvalues
    S <- tulpaRatio:::cpp_hsgp_spectrum(eig, rep(sigma2, length(ell)), ell,
                                       dimension, kernel)
    variance <- S %*% t(Phi^2)
    expect_lt(max(abs(variance / sigma2 - 1)), 0.003,
              label = paste("dimension", dimension, "kernel", kernel))
  }
})

test_that("HSGP spectral gradients include the spatial dimension", {
  eig <- c(0, 0.2, 1, 3)
  for (dimension in 1:2) for (kernel in 0:2) {
    ell <- 0.6
    h <- 1e-5
    S <- function(l) tulpaRatio:::cpp_hsgp_spectrum(eig, 1, l, dimension, kernel)
    fd <- (log(S(ell * exp(h))) - log(S(ell * exp(-h)))) / (2 * h)
    if (kernel == 0L) expected <- dimension - ell^2 * eig else {
      nu <- if (kernel == 1L) 1.5 else 2.5
      expected <- -2 * nu + (2 * nu + dimension) * 2 * nu / (2 * nu + ell^2 * eig)
    }
    expect_equal(drop(fd), expected, tolerance = 1e-8)
  }
})

test_that("a sampled two-dimensional HSGP covers the simulated field SD", {
  skip_on_cran()
  skip_if_not(Sys.getenv("TULPA_SLOW_TESTS") == "true")
  set.seed(812)
  n <- 160L
  df <- data.frame(lon = runif(n, -1, 1), lat = runif(n, -1, 1))
  sigma <- 0.7
  ell <- 0.25
  K <- sigma^2 * exp(-as.matrix(dist(df))^2 / (2 * ell^2))
  field <- drop(t(chol(K + diag(1e-10, n))) %*% rnorm(n))
  df$trials <- 100L
  df$y <- rbinom(n, df$trials, plogis(0.3 + field))
  fit <- tratio(y | trials ~ 1, data = df, family = ratiod_binomial(),
    spatial = spatial_hsgp(~ lon + lat, m = 14, c_boundary = 1.5, scale_coords = FALSE),
    control = list(iter = 4000, warmup = 1500, chains = 4, seed = 17, verbose = FALSE))
  log_sigma2 <- fit$.internal$samples[, fit$.internal$layout$hsgp$log_sigma2]
  sigma_draw <- exp(log_sigma2 / 2)
  interval <- quantile(sigma_draw, c(0.025, 0.975))
  expect_lt(interval[1], sigma)
  expect_gt(interval[2], sigma)
  chains <- matrix(log_sigma2 / 2, ncol = 4L)
  expect_lt(posterior::rhat(chains), 1.01)
  expect_gt(posterior::ess_bulk(chains), 900)
})
