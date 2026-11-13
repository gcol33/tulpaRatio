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
    spatial = spatial_car(adj, spatial_level = "group", group_var = "region"),
    temporal = temporal_rw1("year"),
    type = "separable", coords = ~ lon + lat, nn = 5,
    parameterization = parameterization
  )
}

test_that("the engine Gaussian interaction matches its dense conditional", {
  # The Laplace step is exact when the likelihood is Gaussian in the
  # interaction, so the centred density at any delta less N(delta; dh, H^-1)
  # is the collapsed one; a missing normalizer or penalty term shows as a gap.
  for (field in c("stgp", "stgp_gneiting")) {
    for (seed in c(42L, 181L)) {
      expect_lt(tulpaRatio:::cpp_st_gaussian_conditional_exactness(field, n_draws = 8L,
                                                      seed = seed), 1e-8)
    }
  }
})

test_that("the collapsed parameterization is refused where it does not apply", {
  f <- collapsed_fixture()
  expect_error(
    spatiotemporal(
      spatial = spatial_car(f$adj, spatial_level = "group", group_var = "region"),
      temporal = temporal_rw1("year"), type = "IV",
      parameterization = "collapsed"),
    "selects full-field GP coordinates")
  skip_on_cran()
  f$data$trials <- f$data$count + 5L
  fit <- suppressWarnings(
    tratio(count | trials ~ x, data = f$data, family = ratiod_beta_binomial(),
           spatiotemporal = collapsed_spec(f$adj, "collapsed"), mode = "hmc",
           control = list(iter = 20, chains = 1, seed = 1)))
  expect_identical(fit$spatiotemporal$parameterization, "noncentered")
  expect_equal(ncol(fit$.internal$spatiotemporal_draws), nrow(f$adj) * 4L)
  expect_true(all(is.finite(fit$.internal$spatiotemporal_draws)))
})


test_that("the polygamma the third-order jets read matches R's", {
  x <- c(0.05, 0.3, 1, 2.5, 7, 11.9, 12, 40, 300)
  for (n in 1:4) {
    expect_equal(tulpaRatio:::cpp_polygamma(n, x), psigamma(x, n),
                 tolerance = 1e-10, info = sprintf("n = %d", n))
  }
})
