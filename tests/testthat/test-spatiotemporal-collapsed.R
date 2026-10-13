# The GP space-time interaction integrated out by a Laplace approximation
# inside HMC (gcol33/tulpaRatio#143).

collapsed_fixture <- function(seed = 7) {
  set.seed(seed)
  coords <- expand.grid(x = 1:3, y = 1:3)
  S <- nrow(coords)
  adj <- 1 * (as.matrix(dist(coords)) <= 1.01)
  diag(adj) <- 0
  d <- expand.grid(region = seq_len(S), year = 2015:2018, survey = 1:3)
  d$lon <- coords$x[d$region]
  d$lat <- coords$y[d$region]
  d$x <- rnorm(nrow(d))
  delta <- matrix(rnorm(S * 4, sd = 0.4), S, 4)
  eta <- 0.3 * d$x + delta[cbind(d$region, d$year - 2014)]
  d$effort <- rgamma(nrow(d), shape = 5, rate = 1 / exp(eta))
  d$count <- rpois(nrow(d), exp(2 + eta))
  list(data = d, adj = adj)
}

collapsed_spec <- function(adj, parameterization) {
  spatiotemporal(
    spatial = spatial_car(adj, level = "group", group_var = "region"),
    temporal = temporal_rw1("year"),
    type = "separable", coords = ~ lon + lat, nn = 5,
    parameterization = parameterization
  )
}

test_that("for a Gaussian likelihood the collapsed density is exact", {
  # The Laplace step is exact when the likelihood is Gaussian in the
  # interaction, so the centred density at any delta less N(delta; dh, H^-1)
  # is the collapsed one; a missing normalizer or penalty term shows as a gap.
  for (field in c("stgp_collapsed", "stgp_gneiting_collapsed")) {
    expect_lt(tulpaRatio:::cpp_st_collapsed_exactness(field), 1e-8)
  }
})

test_that("the collapsed parameterization is refused where it does not apply", {
  f <- collapsed_fixture()
  expect_error(
    spatiotemporal(
      spatial = spatial_car(f$adj, level = "group", group_var = "region"),
      temporal = temporal_rw1("year"), type = "IV",
      parameterization = "collapsed"),
    "integrates out a GP interaction")
  skip_on_cran()
  f$data$trials <- f$data$count + 5L
  expect_error(
    tratio(count | trials ~ x, data = f$data, family = ratiod_beta_binomial(),
           spatiotemporal = collapsed_spec(f$adj, "collapsed"), mode = "hmc",
           control = list(iter = 20, chains = 1, seed = 1)),
    "log-concave family")
})

test_that("a collapsed interaction fit agrees with the centred one", {
  skip_on_cran()
  f <- collapsed_fixture()
  fit_with <- function(par) {
    suppressWarnings(tratio(
      count | effort ~ x, data = f$data, family = ratiod_poisson_gamma(),
      spatiotemporal = collapsed_spec(f$adj, par), mode = "hmc",
      control = list(iter = 800, chains = 1, seed = 3)))
  }
  cen <- fit_with("centered")
  col <- fit_with("collapsed")

  # The collapsed fit reports the interaction as draws from its Gaussian
  # approximation, one per saved iteration
  st_col <- col$.internal$spatiotemporal_draws
  st_cen <- cen$.internal$spatiotemporal_draws
  expect_equal(dim(st_col), dim(st_cen))
  expect_true(all(is.finite(st_col)))
  expect_gt(stats::cor(colMeans(st_col), colMeans(st_cen)), 0.9)

  b_col <- coef(col)
  b_cen <- coef(cen)
  expect_lt(abs(b_col[["num:x"]] - b_cen[["num:x"]]), 0.05)
  tau <- function(fit) stats::median(as.matrix(fit$draws)[, "tau_st"])
  expect_lt(abs(log(tau(col)) - log(tau(cen))), log(1.5))
})

test_that("the polygamma the third-order jets read matches R's", {
  x <- c(0.05, 0.3, 1, 2.5, 7, 11.9, 12, 40, 300)
  for (n in 1:4) {
    expect_equal(tulpaRatio:::cpp_polygamma(n, x), psigamma(x, n),
                 tolerance = 1e-10, info = sprintf("n = %d", n))
  }
})
