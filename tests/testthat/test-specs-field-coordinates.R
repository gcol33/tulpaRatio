test_that("automatic and cyclic fields reach the engine and recover simulated effects", {
  skip_on_cran()
  calls <- 0L
  trace('cpp_tulpaRatio_run_nuts_specs', where = asNamespace('tulpaRatio'),
    tracer = function() calls <<- calls + 1L, print = FALSE)
  on.exit(untrace('cpp_tulpaRatio_run_nuts_specs', where = asNamespace('tulpaRatio')))
  S <- 6L
  adjacency <- matrix(0, S, S)
  for (s in seq_len(S - 1L)) adjacency[s, s + 1L] <- adjacency[s + 1L, s] <- 1
  disconnected <- adjacency
  disconnected[3L, 4L] <- disconnected[4L, 3L] <- 0
  disconnected[5L, 6L] <- disconnected[6L, 5L] <- 0
  fields <- list(
    icar = list(spatial = spatial_car(adjacency, group_var = 'unit')),
    bym2_disconnected = list(spatial = suppressWarnings(
      spatial_bym2(disconnected, group_var = 'unit')), control = list(adapt_delta = .99)),
    rw1 = list(temporal = temporal_rw1('time')),
    rw2 = list(temporal = temporal_rw2('time')),
    ar1 = list(temporal = temporal_ar1('time')),
    cyclic_rw1 = list(temporal = temporal_rw1('time', cyclic = TRUE)),
    cyclic_rw2 = list(temporal = temporal_rw2('time', cyclic = TRUE)))
  for (seed in c(18L, 181L, 182L)) {
    set.seed(seed)
    d <- data.frame(unit = rep(seq_len(S), each = 30L), n = 20L)
    d$time <- d$unit
    d$x <- rnorm(nrow(d))
    truth <- c(-.6, -.3, -.1, .1, .3, .6)
    d$y <- rbinom(nrow(d), d$n, plogis(.2 + .5 * d$x + truth[d$unit]))
    for (name in names(fields)) {
      case <- fields[[name]]
      control <- utils::modifyList(list(iter = 6000L, warmup = 2000L,
        chains = 1L, cores = 1L, seed = seed, verbose = FALSE,
        adapt_delta = .95, gradient_mode = 'A_r'), case$control %||% list())
      case$control <- NULL
      args <- c(list(formula = y | n ~ x, data = d, family = ratiod_binomial(),
        mode = 'hmc', control = control), case)
      before <- calls
      fit <- do.call(tratio, args)
      expect_equal(calls, before + 1L)
      b <- as.matrix(fit$draws)
      expect_equal(fit$diagnostics$n_divergent, 0L)
      expect_true(all(abs(colMeans(b)[1:2] - c(.2, .5)) < .15))
      if (name != 'bym2_disconnected') {
        field_columns <- grep('^(phi_spatial|temporal)\\[', colnames(b))
        expect_length(field_columns, S)
        recovered <- mean(b[, 1L]) + colMeans(b[, field_columns, drop = FALSE])
        expect_lt(max(abs(recovered - (.2 + truth))), .3)
      }
      probability <- fitted(fit, summary = FALSE)$numerator
      expect_true(all(is.finite(probability)))
    }
  }
})
