test_that("noncanonical numerator and denominator links agree across seeds", {
  skip_on_cran()
  cases <- list(identity_inverse = c("identity", "inverse"), inverse_identity = c("inverse", "identity"))
  bridge19_inventory("links", paste0("links_", names(cases)))
  for (seed in c(18L, 181L, 182L)) for (name in names(cases)) {
    set.seed(seed)
    d <- data.frame(x = runif(160L, -.8, .8))
    links <- cases[[name]]
    numerator <- if (links[1L] == "identity") 3 + .3 * d$x else 1 / (.8 + .1 * d$x)
    denominator <- if (links[2L] == "identity") 2 + .2 * d$x else 1 / (.8 + .1 * d$x)
    d$y <- rgamma(nrow(d), 30, rate = 30 / numerator)
    d$denom <- rgamma(nrow(d), 30, rate = 30 / denominator)
    args <- list(formula = y | denom ~ x, data = d,
      family = ratiod_gamma_gamma(link_num = links[1L], link_denom = links[2L]), mode = "hmc",
      control = list(iter = 6000L, warmup = 2000L, chains = 1L, cores = 1L,
        seed = seed, verbose = FALSE, adapt_delta = .99, gradient_mode = "A_r"))
    fit <- bridge19_pair(args, paste0("links_", name), seed)
    recovery <- colMeans(as.matrix(fit$draws))[1:4]
    truth <- c(if (links[1L] == "identity") c(3, .3) else c(.8, .1),
      if (links[2L] == "identity") c(2, .2) else c(.8, .1))
    expect_true(all(abs(recovery - truth) < .2))
    prediction <- fitted(fit, summary = FALSE)
    numerator_error <- sqrt(mean(((colMeans(prediction$numerator) - numerator) / numerator)^2))
    denominator_error <- sqrt(mean(((colMeans(prediction$denominator) - denominator) / denominator)^2))
    bridge19_record_recovery(fit, max(abs(recovery - truth)), numerator_error,
      denominator_error, scale = "relative_rmse")
    expect_lt(numerator_error, .15)
    expect_lt(denominator_error, .15)
  }
})
