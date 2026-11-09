nb_count_reference <- function(y, mu, phi) {
  sum(log1p((seq_len(y) - 1) / phi)) - lgamma(y + 1) +
    y * log(mu) - (phi + y) * log1p(mu / phi)
}

nb_inflated_reference <- function(y, mu, phi, z, type) {
  lp0 <- -phi * log1p(mu / phi)
  if (type == "plain") return(nb_count_reference(y, mu, phi))
  if (type == "zi_negbin") {
    if (y == 0) return(log(plogis(z) + plogis(-z) * exp(lp0)))
    return(-log1p(exp(z)) + nb_count_reference(y, mu, phi))
  }
  if (y == 0) return(-log1p(exp(z)))
  -log1p(exp(-z)) + nb_count_reference(y, mu, phi) - log(-expm1(lp0))
}

test_that("count normalizers and polygamma differences match finite sums", {
  for (y in c(0L, 1L, 3L, 32L, 33L, 200L, 5000L)) {
    for (phi in c(0.001, 0.7, 9.5, 50, 999, 1000, 1e8, 1e17)) {
      v <- tulpaRatio:::cpp_test_negbin_math(y, phi)
      terms <- phi + seq_len(y) - 1
      expect_lt(abs(v[1] - sum(log(terms))), 2e-10)
      for (k in 0:3) {
        ref <- sum((-1)^k * factorial(k) / terms^(k + 1))
        expect_lt(abs(v[k + 2] - ref) / max(abs(ref), 1e-300), 1e-11)
      }
    }
  }
})

test_that("all negative binomial density and autodiff paths preserve large sizes", {
  for (type in c("plain", "zi_negbin", "hurdle_negbin")) {
    for (y in c(0L, 1L, 3L, 33L, 200L)) {
      for (phi in c(0.1, 2, 999, 1000, 1e8, 1e12, 1e17)) {
        mu <- 2; z <- -0.3
        ref <- nb_inflated_reference(y, mu, phi, z, type)
        p <- tulpaRatio:::cpp_test_negbin_paths(y, mu, phi, z, type)
        s <- tulpaRatio:::cpp_test_negbin_spec(y, mu, phi, z, type)
        vals <- c(p$legacy_double, p$autodiff[, "value"], p$H[1], s[, "value"])
        expect_lt(max(abs(vals - ref)), 1e-10)
        expect_lt(abs(p$H_table[1] - nb_count_reference(y, mu, phi)), 1e-10)
        expect_lt(abs(tulpaRatio:::cpp_test_negbin_loglik(y, mu, phi) - nb_count_reference(y, mu, phi)), 1e-10)
        expect_lt(abs(tulpaRatio:::cpp_test_autodiff_negbin_loglik(y, mu, phi)$value - nb_count_reference(y, mu, phi)), 1e-10)
      }
    }
  }
})

test_that("negative binomial H tables serve fused and vectorized kernels", {
  for (y in c(0L, 3L, 33L, 200L)) for (phi in c(0.1, 2, 1000, 1e8, 1e17)) {
    h <- tulpaRatio:::cpp_test_negbin_h_kernels(y, 2, phi)
    expect_lt(max(abs(h[, 1] - nb_count_reference(y, 2, phi))), 1e-10)
    expect_lt(max(abs(h[, 2] - phi * (y - 2) / (phi + 2))), 1e-10)
    score <- phi * (sum(1 / (phi + seq_len(y) - 1)) - log1p(2 / phi) + (2 - y) / (phi + 2))
    expect_lt(max(abs(h[, 3] - score)), 1e-10)
  }
})

test_that("ZI and hurdle negative binomial gradients match exact-reference differences", {
  for (type in c("plain", "zi_negbin", "hurdle_negbin")) {
    for (y in c(0L, 3L, 33L)) for (phi in c(0.1, 2, 1000, 1e8, 1e17)) {
      x <- c(log(2), log(phi), -0.3)
      f <- function(x) nb_inflated_reference(y, exp(x[1]), exp(x[2]), x[3], type)
      fd <- vapply(seq_along(x), function(j) {
        plus <- minus <- x
        plus[j] <- plus[j] + 1e-5; minus[j] <- minus[j] - 1e-5
        (f(plus) - f(minus)) / 2e-5
      }, numeric(1))
      p <- tulpaRatio:::cpp_test_negbin_paths(y, 2, phi, -0.3, type)
      s <- tulpaRatio:::cpp_test_negbin_spec(y, 2, phi, -0.3, type)
      grads <- rbind(p$autodiff[, -1], p$H[-1], s[-1, -1])
      expect_lt(max(abs(sweep(grads, 2, fd))), 3e-7)
    }
  }
})

test_that("negative binomial ZI and hurdle C++ unit entries use normalized densities", {
  for (phi in c(0.2, 2, 1000, 1e8, 1e17)) for (mu in c(0.001, 2, 30)) {
    for (z in c(-6, -0.3, 5)) for (y in c(0L, 1L, 3L, 33L)) {
      zi <- tulpaRatio:::cpp_test_zi_negbin_lpmf_logit(y, mu, phi, z)
      hurdle <- tulpaRatio:::cpp_test_hurdle_negbin_lpmf_logit(y, mu, phi, z)
      expect_lt(abs(zi - nb_inflated_reference(y, mu, phi, z, "zi_negbin")), 1e-10)
      expect_lt(abs(hurdle - nb_inflated_reference(y, mu, phi, z, "hurdle_negbin")), 1e-10)
      trunc <- tulpaRatio:::cpp_test_truncated_negbin_lpmf(y, mu, phi)
      if (y == 0) expect_identical(trunc, -Inf)
      else expect_lt(abs(trunc - nb_count_reference(y, mu, phi) + log(-expm1(-phi * log1p(mu / phi)))), 1e-10)
      grad <- tulpaRatio:::cpp_test_zi_negbin_grad_logit_zi(y, mu, phi, z)
      fd <- (nb_inflated_reference(y, mu, phi, z + 1e-5, "zi_negbin") -
               nb_inflated_reference(y, mu, phi, z - 1e-5, "zi_negbin")) / 2e-5
      expect_lt(abs(grad - fd), 1e-8)
    }
  }
  for (type in c("zi_negbin", "hurdle_negbin")) {
    lp <- vapply(0:400, function(y) {
      if (type == "zi_negbin") tulpaRatio:::cpp_test_zi_negbin_lpmf_logit(y, 2, 2, -0.3)
      else tulpaRatio:::cpp_test_hurdle_negbin_lpmf_logit(y, 2, 2, -0.3)
    }, numeric(1))
    expect_lt(abs(sum(exp(lp)) - 1), 1e-12)
  }
})

test_that("Taylor count likelihoods retain the Poisson limit and its curvature", {
  for (y in c(0L, 3L, 33L)) for (phi in c(2, 1000, 1e8, 1e17)) {
    v <- tulpaRatio:::cpp_test_negbin_jet(y, 2, phi)
    expect_lt(abs(v[1] - nb_count_reference(y, 2, phi)), 1e-10)
    expect_lt(abs(v[2] - phi * (y - 2) / (phi + 2)), 1e-10)
    expect_lt(abs(v[3] + (y + phi) * phi * 2 / (phi + 2)^2), 1e-10)
    expect_lt(abs(v[4] + (y + phi) * phi * 2 * (phi - 2) / (phi + 2)^3), 1e-10)
  }
})

test_that("CRT means retain their finite-sum expectation at a large size", {
  for (y in c(0L, 3L, 33L, 200L)) for (phi in c(0.1, 2, 1000, 1e8, 1e17)) {
    ref <- sum(phi / (phi + seq_len(y) - 1))
    expect_lt(abs(tulpaRatio:::cpp_crt_mean(y, phi) - ref), 1e-10)
  }
})
