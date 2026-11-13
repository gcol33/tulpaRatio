test_that("collapsed spatial requests recover planted fields through the engine", {
  skip_on_cran()
  calls <- 0L
  trace('cpp_tulpaRatio_sample', where = asNamespace('tulpaRatio'),
    tracer = function() calls <<- calls + 1L, print = FALSE)
  on.exit(untrace('cpp_tulpaRatio_sample', where = asNamespace('tulpaRatio')), add = TRUE)
  fixture <- spatial_bridge_fixture(1919L)
  adjacency <- fixture$adjacency
  truth <- fixture$truth
  d <- fixture$data
  fields <- list(
    icar = function(parameterization) spatial_car(adjacency, group_var = 'unit',
      parameterization = parameterization),
    bym2 = function(parameterization) spatial_bym2(adjacency, group_var = 'unit',
      parameterization = parameterization),
    gp = function(parameterization) spatial_gp(~ lon + lat, scale_coords = FALSE, nn = 5L,
      parameterization = parameterization))
  for (seed in c(18L, 181L, 182L)) for (name in names(fields)) {
    fit_one <- function() {
      parameterization <- 'collapsed'
      before <- calls
      timing <- system.time(fit <- suppressWarnings(tratio(y | n ~ x, data = d,
        family = ratiod_binomial(), spatial = fields[[name]](parameterization), mode = 'hmc',
        control = list(iter = 8000L, warmup = 2500L, chains = 1L, cores = 1L,
          seed = seed, verbose = FALSE, adapt_delta = .99,
          gradient_mode = 'A_r'))))
      expect_equal(calls, before + 1L, info = name)
      list(fit = fit, seconds = timing[['elapsed']])
    }
    b <- fit_one()
    expect_identical(b$fit$spatial$parameterization, 'auto')
    expect_identical(b$fit$target, 'Joint posterior')
    expect_identical(b$fit$inference_tier, 1L)
    y <- spatial_comparison_draws(b$fit$draws)
    expect_lt(abs(mean(y[, 'beta_num[2]']) - .5), .15, label = name)
    eta <- .2 + .5 * d$x + truth[d$unit]
    fitted_eta <- qlogis(fitted(b$fit, summary = FALSE)$numerator)
    expect_lt(sqrt(mean((colMeans(fitted_eta) - eta)^2)), .2, label = name)
  }
  for (name in names(fields)) {
    before <- calls
    fit <- suppressWarnings(tratio(y | n ~ x, data = d, family = ratiod_binomial(),
      spatial = fields[[name]]('collapsed'), mode = 'hmc', control = list(iter = 2000L,
        warmup = 750L, chains = 2L, cores = 2L, seed = 1919L, adapt_delta = .99,
        gradient_mode = 'A_r', verbose = FALSE)))
    expect_equal(calls, before + 1L, info = name)
    expect_equal(nrow(as.matrix(fit$draws)), 2500L)
    eta <- .2 + .5 * d$x + truth[d$unit]
    fitted_eta <- qlogis(fitted(fit, summary = FALSE)$numerator)
    expect_lt(sqrt(mean((colMeans(fitted_eta) - eta)^2)), .2, label = name)
  }
})
