test_that("GP coordinate choices accept positive identity and inverse predictors", {
  skip_on_cran()
  if (!'st_spec' %in% names(formals(tulpaRatio:::cpp_tulpaRatio_run_nuts_specs)))
    skip('Requires linked likelihood callbacks from the concurrent bridge for gcol33/tulpaRatio#176')
  op <- options(tulpaRatio.use_specs = TRUE)
  on.exit(options(op))
  calls <- 0L
  trace('cpp_tulpaRatio_run_nuts_specs', where = asNamespace('tulpaRatio'),
    tracer = function() calls <<- calls + 1L, print = FALSE)
  on.exit(untrace('cpp_tulpaRatio_run_nuts_specs', where = asNamespace('tulpaRatio')), add = TRUE)
  for (parameterization in c('auto', 'collapsed')) for (seed in c(18L, 181L, 182L)) for (links in list(c('identity', 'identity'),
      c('identity', 'inverse'), c('inverse', 'identity'))) {
    set.seed(seed)
    d <- data.frame(unit = rep(1:5, each = 8), x = runif(40, -.5, .5))
    d$lon <- c(0, .2, .4, .7, 1)[d$unit]; d$lat <- c(.2, .8, .1, .6, .3)[d$unit]
    mu_num <- if (links[1] == 'identity') 3 + .3 * d$x else 1 / (.8 + .1 * d$x)
    mu_den <- if (links[2] == 'identity') 2 + .2 * d$x else 1 / (.8 + .1 * d$x)
    d$y <- rgamma(40, 20, rate = 20 / mu_num)
    d$denom <- rgamma(40, 20, rate = 20 / mu_den)
    fit_one <- function(bridge) {
      options(tulpaRatio.use_specs = bridge)
      suppressWarnings(tratio(y | denom ~ x, data = d,
      family = ratiod_gamma_gamma(link_num = links[1], link_denom = links[2]),
      spatial = spatial_gp(~ lon + lat, scale_coords = FALSE, nn = 4L, parameterization = parameterization),
      mode = 'hmc', control = list(iter = 500L, warmup = 250L, chains = 1L, cores = 1L,
        seed = seed, verbose = FALSE, adapt_delta = .99, gradient_mode = 'A_r')))
    }
    if (parameterization == 'collapsed' && seed == 18L && identical(links, c('identity', 'identity')))
      expect_error(fit_one(FALSE), 'canonical links')
    before <- calls
    fit <- fit_one(TRUE)
    expect_equal(calls, before + 1L)
    expect_true(all(is.finite(as.matrix(fit$draws))))
    expect_equal(nrow(fit$draws), 250L)
  }
})
