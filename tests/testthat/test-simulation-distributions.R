simulation_raw_moments <- function(distribution, mu, phi, minimum = NULL) {
  orders <- seq_len(4)
  if (distribution == "lognormal") {
    return(exp(orders * log(mu) + orders^2 * phi^2 / 2))
  }
  if (distribution == "gamma") {
    raw <- vapply(orders, function(r) {
      moment <- mu^r * prod(phi + seq.int(0, r - 1)) / phi^r
      if (is.null(minimum)) return(moment)
      minimum^r * pgamma(minimum, phi, rate = phi / mu) +
        moment * pgamma(minimum, phi + r, rate = phi / mu, lower.tail = FALSE)
    }, 1.0)
    return(raw)
  }
  factorial <- vapply(orders, function(r) {
    switch(distribution,
      poisson = mu^r,
      neg_binomial_2 = mu^r * prod(1 + seq.int(0, r - 1) / phi),
      binomial = mean(prod(seq_len(r)) * choose(10:50, r)) * mu^r,
      beta_binomial = mean(prod(seq_len(r)) * choose(10:50, r)) *
        prod(mu * phi + seq.int(0, r - 1)) / prod(phi + seq.int(0, r - 1)))
  }, 1.0)
  raw <- c(factorial[1], factorial[1] + factorial[2],
           factorial[1] + 3 * factorial[2] + factorial[3],
           factorial[1] + 7 * factorial[2] + 6 * factorial[3] + factorial[4])
  if (!is.null(minimum)) {
    raw <- raw + dnbinom(0, mu = mu, size = phi)
  }
  raw
}

test_that("simulation moments and dispersions match each arm's exact distribution", {
  families <- list(ratiod_negbin_negbin(), ratiod_binomial(),
                    ratiod_poisson_gamma(), ratiod_negbin_gamma(),
                    ratiod_gamma_gamma(), ratiod_lognormal(),
                    ratiod_lognormal(denom_fixed = TRUE), ratiod_poisson(),
                    ratiod_poisson(denom_family = "negbin"), ratiod_negbin(),
                    ratiod_beta_binomial())
  n <- 100000L
  for (family in families) {
    num_dist <- family$numerator$distribution
    probability <- num_dist %in% c("binomial", "beta_binomial")
    mu_num <- if (probability) .35 else 8
    beta_num <- if (probability) qlogis(mu_num) else log(mu_num)
    phi_num <- if (num_dist == "lognormal") 1.2 else 2.5
    phi_denom <- if (family$denominator$distribution == "lognormal") .8 else 3
    sim <- sim_ratiod(n, family, beta_num = c(beta_num, 0),
                      beta_denom = c(log(10), 0), sigma_re = 0,
                      phi_num = phi_num, phi_denom = phi_denom, seed = 182)
    expect_identical(sim$true_params$phi_num, phi_num)
    expect_identical(sim$true_params$phi_denom, phi_denom)
    for (arm in c("num", "denom")) {
      distribution <- family[[if (arm == "num") "numerator" else "denominator"]]$distribution
      y <- sim$data[[paste0("y_", arm)]]
      mu <- if (arm == "num") mu_num else 10
      phi <- if (arm == "num") phi_num else phi_denom
      if (distribution == "fixed") {
        if (probability) expect_true(all(y %in% 10:50)) else expect_identical(y, rep(1, n))
        next
      }
      if (distribution == "lognormal") {
        y <- log(y)
        raw <- c(log(mu), log(mu)^2 + phi^2,
                 log(mu)^3 + 3 * log(mu) * phi^2,
                 log(mu)^4 + 6 * log(mu)^2 * phi^2 + 3 * phi^4)
      } else {
        minimum <- if (arm == "denom" && distribution == "neg_binomial_2") 1 else
          if (distribution == "gamma") {
            if (arm == "num" || num_dist == "gamma") .001 else .01
          } else NULL
        raw <- simulation_raw_moments(distribution, mu, phi, minimum)
      }
      variance <- raw[2] - raw[1]^2
      central4 <- raw[4] - 4 * raw[1] * raw[3] + 6 * raw[1]^2 * raw[2] - 3 * raw[1]^4
      label <- paste(family$name, arm)
      expect_lt(abs(mean(y) - raw[1]), 6 * sqrt(variance / n), label = label)
      expect_lt(abs(mean((y - raw[1])^2) - variance),
                 6 * sqrt((central4 - variance^2) / n), label = label)
    }
  }
})

test_that("calibration simulations use the sampled lognormal dispersions", {
  priors <- ratiod_priors(beta = prior_normal(0, .1), sigma = prior_half_normal(.1),
                         phi = prior_gamma(10000, 10000))
  simulations <- sim_ratiod_sbc(n_sims = 3, n = 30000, family = ratiod_lognormal(),
                                priors = priors, n_groups = 1, seed = 182)
  for (simulation in simulations) for (arm in c("num", "denom")) {
    phi <- simulation$true_params[[paste0("phi_", arm)]]
    beta <- simulation$true_params[[paste0("beta_", arm)]]
    eta <- beta[1] + beta[2] * simulation$data$x1 +
      simulation$true_params$re[as.integer(simulation$data$group)]
    residuals <- log(simulation$data[[paste0("y_", arm)]]) - eta
    expect_gt(phi, .9)
    expect_lt(phi, 1.1)
    expect_lt(abs(sd(residuals) - phi),
               6 * phi / sqrt(2 * (simulation$n - 1)))
  }
})
