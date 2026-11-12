test_that("temporal and linked routes recover planted predictors", {
  skip_on_cran()
  cases <- bridge19_cases()
  bridge19_inventory("temporal", names(cases))
  cases <- bridge19_select_cases(cases)
  for (seed in c(18L, 181L, 182L)) for (name in names(cases)) {
    case <- cases[[name]]
    d <- bridge19_fixture(seed, case)
    args <- list(formula = y | n ~ x, data = d,
      family = ratiod_binomial(case$link %||% "logit"), temporal = case$temporal,
      mode = "hmc")
    fit <- engine_recovery_fit(args, name, seed)
    b <- as.matrix(fit$draws)
    recovery <- bridge19_temporal_recovery(fit, d, case$link %||% "logit")
    bridge19_write_evidence(cbind(attr(fit, "bridge19"), recovery),
      "temporal-coefficient-recovery.tsv")
    expect_lt(recovery$predictor_coefficient_error, .25,
      label = paste(name, seed, "predictor coefficients"))
    recovered <- colMeans(fitted(fit, summary = FALSE)$numerator)
    bridge19_record_recovery(fit, max(abs(colMeans(b)[1:2] - c(.2, .5))),
      max(abs(recovered - d$truth)), scale = "probability")
    expect_lt(max(abs(recovered - d$truth)), .12, label = paste(name, seed, "recovery"))
  }
})
