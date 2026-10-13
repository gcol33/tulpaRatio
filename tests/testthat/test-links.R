# Non-canonical links: the gradient of every link matches a finite difference
# of the log posterior, a model simulated under a link is recovered under it,
# and the samplers written for the canonical links refuse the others.

test_that("probability links have the gradient of their log posterior", {
  for (link in c("probit", "cloglog")) {
    for (field in c("none", "re", "icar")) {
      r <- tulpaRatio:::cpp_gradient_check(field, family = "binomial",
                                           link_num = link)
      expect_lt(max(grad_rel_dev(r)), 1e-4, label = paste(field, link))
    }
    for (zi in c("zi_binomial", "hurdle_binomial", "zoib")) {
      r <- tulpaRatio:::cpp_gradient_check("none", family = "binomial", zi = zi,
                                           link_num = link, link_inflation = link)
      expect_lt(max(grad_rel_dev(r)), 1e-4, label = paste(zi, link))
    }
    r <- tulpaRatio:::cpp_gradient_check("none", family = "poisson_gamma",
                                         zi = "zi_poisson", link_inflation = link)
    expect_lt(max(grad_rel_dev(r)), 1e-4, label = paste("zi_poisson", link))
  }
})

test_that("gamma mean links have the gradient of their log posterior", {
  for (link in c("identity", "inverse")) {
    for (family in c("poisson_gamma", "negbin_gamma")) {
      for (field in c("none", "re")) {
        r <- tulpaRatio:::cpp_gradient_check(field, family = family,
                                             link_denom = link)
        expect_lt(max(grad_rel_dev(r)), 1e-4, label = paste(family, field, link))
      }
    }
  }
})

test_that("families accept the links their likelihoods implement", {
  expect_equal(ratiod_binomial("probit")$numerator$link, "probit")
  expect_equal(ratiod_gamma_gamma("identity", "inverse")$denominator$link, "inverse")
  expect_equal(ratiod_zipois(link_zi = "cloglog")$numerator$link_zi, "cloglog")
  expect_equal(ratiod_zinegbin(denom_family = "gamma",
                               link_denom = "identity")$denominator$link, "identity")
  expect_error(ratiod_poisson_gamma(link_num = "identity"), "not supported")
  expect_error(ratiod_negbin_negbin(link_denom = "identity"), "not supported")
  expect_error(ratiod_zinegbin(link_denom = "identity"), "not supported")
  expect_error(ratiod_binomial("log"), "not supported")
})

test_that("a probit binomial is recovered under the probit link", {
  skip_on_cran()
  set.seed(11)
  n <- 400
  x <- rnorm(n)
  trials <- rep(20L, n)
  d <- data.frame(x = x, trials = trials,
                  y = rbinom(n, trials, stats::pnorm(-0.3 + 0.6 * x)))
  probit <- tratio(y | trials ~ x, data = d, family = ratiod_binomial("probit"),
                   control = list(iter = 1000, chains = 2, seed = 1))
  logit <- tratio(y | trials ~ x, data = d, family = ratiod_binomial(),
                  control = list(iter = 1000, chains = 2, seed = 1))
  b <- coef(probit)
  expect_equal(unname(b[c("num:(Intercept)", "num:x")]), c(-0.3, 0.6),
               tolerance = 0.1)
  # On the logit scale the same probabilities need coefficients about 1.6 to
  # 1.8 times larger
  expect_gt(unname(coef(logit)["num:x"] / b["num:x"]), 1.4)
  # Fitted probabilities agree, since both links fit the same data
  expect_equal(colMeans(fitted(probit, summary = FALSE)$ratio),
               colMeans(fitted(logit, summary = FALSE)$ratio), tolerance = 0.05)
})

test_that("an identity-link gamma denominator is recovered", {
  skip_on_cran()
  set.seed(12)
  n <- 300
  x <- runif(n)
  mu_d <- 4 + 3 * x
  d <- data.frame(x = x,
                  count = rpois(n, exp(1 + 0.5 * x)),
                  effort = rgamma(n, shape = 6, rate = 6 / mu_d))
  fit <- tratio(count | effort ~ x, data = d,
                family = ratiod_poisson_gamma(link_denom = "identity"),
                control = list(iter = 1000, chains = 2, seed = 1))
  b <- coef(fit)
  # The denominator likelihood is the identity-link gamma GLM's, so with a
  # wide prior the posterior mean sits on its maximum-likelihood estimate
  mle <- stats::coef(stats::glm(effort ~ x, data = d,
                                family = stats::Gamma(link = "identity")))
  expect_equal(unname(b[c("denom:(Intercept)", "denom:x")]), unname(mle),
               tolerance = 0.05)
  expect_equal(unname(b[c("num:(Intercept)", "num:x")]), c(1, 0.5),
               tolerance = 0.2)
  # The fitted denominator mean is the linear predictor itself
  mu <- fitted(fit, summary = FALSE)$denominator
  expect_equal(mean(mu), mean(mu_d), tolerance = 0.05)
})

test_that("samplers written for the canonical links refuse the others", {
  set.seed(13)
  n <- 40
  d <- data.frame(y = rbinom(n, 10, 0.4), trials = 10L, x = rnorm(n))
  expect_error(
    tratio(y | trials ~ x, data = d, family = ratiod_binomial("probit"),
           mode = "laplace"),
    "canonical links")
  expect_error(
    tratio(y | trials ~ x, data = d, family = ratiod_binomial("probit"),
           mode = "pg"),
    "canonical links")
})
