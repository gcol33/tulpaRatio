linked_reference <- function(family, y, n, eta, phi, link, inflation = "none",
                             zi = -.6, oi = -1, zi_link = "logit", oi_link = "logit") {
  probability <- function(x, link) switch(link, logit = plogis(x),
    probit = pnorm(x), cloglog = -expm1(-exp(x)))
  mu <- switch(link, log = exp(eta), identity = eta, inverse = 1 / eta,
    probability(eta, link))
  density <- function(y) switch(family,
    poisson = dpois(y, mu), negbin = dnbinom(y, mu = mu, size = phi),
    gamma = dgamma(y, shape = phi, rate = phi / mu),
    lognormal = dlnorm(y, eta, phi) * sqrt(2 * pi), binomial = dbinom(y, n, mu),
    beta_binomial = exp(lchoose(n, y) + lbeta(y + mu * phi, n - y + (1 - mu) * phi) -
      lbeta(mu * phi, (1 - mu) * phi)))
  pz <- probability(zi, zi_link)
  po <- probability(oi, oi_link)
  base <- density(y)
  value <- if (inflation == "none") base else if (inflation == "zoib") {
    if (y == 0) pz + (1 - pz) * (1 - po) * base else
      (1 - pz) * (if (y == n) po + (1 - po) * base else (1 - po) * base)
  } else if (inflation == "oi_binomial") {
    if (y == n) po + (1 - po) * base else (1 - po) * base
  } else if (startsWith(inflation, "hurdle")) {
    if (y == 0) 1 - pz else pz * base / (1 - density(0))
  } else if (y == 0) pz + (1 - pz) * base else (1 - pz) * base
  log(value)
}

test_that("one typed margin result carries linked and inflated derivatives", {
  cases <- list()
  for (family in c("poisson", "negbin", "gamma", "lognormal", "binomial", "beta_binomial")) {
    links <- if (family %in% c("binomial", "beta_binomial")) c("logit", "probit", "cloglog") else
      if (family == "lognormal") "log" else c("log", "identity", "inverse")
    inflation <- switch(family,
      poisson = c("none", "zi_poisson", "hurdle_poisson"),
      negbin = c("none", "zi_negbin", "hurdle_negbin"),
      binomial = c("none", "zi_binomial", "hurdle_binomial", "oi_binomial", "zoib"), "none")
    for (link in links) for (type in inflation) for (y in c(0, 3, 8)) {
      if (family %in% c("gamma", "lognormal") && y == 0) next
      args <- list(family = family, y = y, n = 8L, eta = .6, phi = 2.3,
        link = link, inflation = type, zi = -.6, oi = -1,
        zi_link = "probit", oi_link = "cloglog")
      r <- do.call(tulpaRatio:::cpp_test_linked_margin, args)
      f <- function(key, delta) {
        a <- args; a[[key]] <- a[[key]] + delta
        do.call(linked_reference, a)
      }
      reference <- do.call(linked_reference, args)
      context <- paste(family, link, type, y)
      expect_equal(unname(r["ll"]), reference, tolerance = 1e-10, info = context)
      for (key in c("eta", "phi", "zi", "oi")) {
        h <- 1e-5
        derivative <- (f(key, h) - f(key, -h)) / (2 * h)
        if (family == "lognormal" && key == "phi") derivative <- derivative * args$phi
        column <- switch(key, eta = "score", phi = "extra", key)
        expect_lt(abs(unname(r[column]) - derivative), 1e-7,
          label = paste(context, key, sprintf("actual=%.17g expected=%.17g", r[column], derivative)))
      }
      h <- 2e-4
      second <- (-f("eta", 2 * h) + 16 * f("eta", h) - 30 * reference +
        16 * f("eta", -h) - f("eta", -2 * h)) / (12 * h^2)
      expect_lt(abs(unname(r["curvature"]) + second),
        3e-6, label = context)
    }
  }
})
