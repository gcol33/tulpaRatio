# The centred and the non-centred interaction are two coordinates of one
# posterior, so they must agree on tau. Types I and III are the ones whose
# sum-to-zero constraints act on directions Q makes proper, where the centred
# normalizer and the non-centred tau power both have to count them.

st_parameterization_data <- function() {
  coords <- expand.grid(x = 1:5, y = 1:5)
  adj <- 1 * (as.matrix(dist(coords)) <= sqrt(2) + 0.01)
  diag(adj) <- 0
  phi <- 0.4 * sin(coords$x) + 0.4 * cos(coords$y)
  set.seed(3)
  df <- expand.grid(region = 1:25, year = 2015:2020)
  df$delta <- rnorm(nrow(df), sd = 0.5)
  df <- df[rep(seq_len(nrow(df)), 2), ]
  df$x <- rnorm(nrow(df))
  eta <- phi[df$region] + 0.1 * (df$year - 2017.5) + df$delta
  df$effort <- rgamma(nrow(df), shape = 5, rate = 1 / exp(eta))
  df$count <- rpois(nrow(df), exp(3 + 0.3 * df$x + eta))
  list(df = df, adj = adj)
}

test_that("centred and non-centred interactions give the same tau", {
  skip_on_cran()
  d <- st_parameterization_data()
  for (ty in c("I", "III")) {
    tau <- lapply(c("centered", "noncentered"), function(par) {
      fit <- suppressWarnings(tratio(
        count | effort ~ x, data = d$df, family = ratiod_poisson_gamma(),
        spatiotemporal = spatiotemporal(
          spatial = spatial_car(d$adj, spatial_level = "group", group_var = "region"),
          temporal = temporal_rw1("year"), type = ty, parameterization = par),
        control = list(iter = 1000, chains = 2, seed = 1)))
      log(as.matrix(fit$draws)[, "tau_st"])
    })
    expect_lt(abs(mean(tau[[1]]) - mean(tau[[2]])) / sd(tau[[1]]), 0.5,
              label = sprintf("Type %s: centred log precision %.2f, non-centred %.2f",
                              ty, mean(tau[[1]]), mean(tau[[2]])))
  }
})

test_that("the non-centred GP interaction is a change of variables of the centred one", {
  # At fixed hyperparameters the non-centred density at z and the centred
  # density at delta(z) differ by the log-Jacobian alone, the same for every z.
  for (field in c("stgp_nc", "stgp_gneiting_nc")) {
    expect_lt(tulpaRatio:::cpp_st_gp_nc_coordinate_spread(field), 1e-8,
              label = field)
  }
})

test_that("cyclic RW2 interaction gradients match the scored density", {
  for (field in c("st4_rw2", "st4_rw2_nc", "st4_rw2_cyclic", "st4_rw2_cyclic_nc"))
    for (mode in c("handcoded", "arena")) {
      r <- tulpaRatio:::cpp_gradient_check(field, mode = mode)
      expect_lt(max(abs(r$analytic - r$finite_diff) / pmax(1, abs(r$finite_diff))),
        1e-4, label = paste(field, mode))
    }
})

test_that("both AR1 interaction coordinates normalize the constrained Gaussian", {
  correlations <- c(-.6, 0, .5, .9)
  for (field in c("st2_ar1", "st2_ar1_nc", "st4_ar1", "st4_ar1_nc")) {
    r <- tulpaRatio:::cpp_st_ar1_zero_density(field, correlations)
    S <- r$S; nt <- r$T
    Qs <- diag(S)
    if (startsWith(field, "st4")) {
      Qs <- matrix(0, S, S)
      for (s in seq_len(S)) {
        at <- r$row_ptr[s] + seq_len(r$row_ptr[s + 1L] - r$row_ptr[s])
        neighbors <- r$col_idx[at]
        Qs[s, neighbors] <- -1
        Qs[s, s] <- length(neighbors)
      }
    }
    rows <- kronecker(matrix(1, S, S), diag(nt)) / (.001 * S)^2
    columns <- kronecker(diag(S), matrix(1, nt, nt)) / (.001 * nt)^2
    reference <- vapply(correlations, function(rho) {
      Qt <- diag(c(1, rep(1 + rho^2, nt - 2L), 1))
      Qt[row(Qt) == col(Qt) + 1L | col(Qt) == row(Qt) + 1L] <- -rho
      .5 * as.numeric(determinant(kronecker(Qs, Qt) + rows + columns)$modulus)
    }, numeric(1))
    expect_lt(max(abs((r$value - r$value[2L]) - (reference - reference[2L]))),
      1e-3, label = field)
  }
})
