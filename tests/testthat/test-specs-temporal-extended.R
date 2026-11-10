test_that("temporal and linked routes agree across seeds and recover planted predictors", {
  skip_on_cran()
  cases <- bridge19_cases()
  bridge19_inventory("temporal", names(cases))
  selected <- Sys.getenv("TULPA_BRIDGE_ROUTE")
  if (nzchar(selected)) {
    wanted <- strsplit(selected, ",", fixed = TRUE)[[1L]]
    stopifnot(all(wanted %in% names(cases)))
    cases <- cases[wanted]
  }
  for (seed in c(18L, 181L, 182L)) for (name in names(cases)) {
    case <- cases[[name]]
    d <- bridge19_fixture(seed, case)
    args <- list(formula = y | n ~ x, data = d,
      family = ratiod_binomial(case$link %||% "logit"), temporal = case$temporal,
      mode = "hmc", control = list(iter = 6000L, warmup = 2000L,
        chains = 1L, cores = 1L, seed = seed, verbose = FALSE,
        adapt_delta = .99, gradient_mode = "A_r"))
    fit <- bridge19_pair(args, name, seed)
    b <- as.matrix(fit$draws)
    expect_true(all(abs(colMeans(b)[1:2] - c(.2, .5)) < .25), info = paste(name, seed))
    recovered <- colMeans(fitted(fit, summary = FALSE)$numerator)
    bridge19_record_recovery(fit, max(abs(colMeans(b)[1:2] - c(.2, .5))),
      max(abs(recovered - d$truth)), scale = "probability")
    expect_lt(max(abs(recovered - d$truth)), .12, label = paste(name, seed, "recovery"))
  }
})
