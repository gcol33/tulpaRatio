test_that("supported structures use the engine by default and honor the switch", {
  old <- options(tulpaRatio.use_specs = NULL)
  on.exit(options(old))
  calls <- 0L
  trace("cpp_tulpaRatio_run_nuts_specs", where = asNamespace("tulpaRatio"),
    tracer = function() calls <<- calls + 1L, print = FALSE)
  on.exit(untrace("cpp_tulpaRatio_run_nuts_specs", where = asNamespace("tulpaRatio")), add = TRUE)
  d <- data.frame(time = rep(1:4, 8), n = 12L, y = rep(c(3L, 4L, 6L, 7L), 8))
  adjacency <- matrix(c(0, 1, 0, 1, 0, 1, 0, 1, 0), 3)
  gp_data <- expand.grid(time = 1:4, unit = 1:3, replicate = 1:2)
  gp_data$lon <- c(.1, .5, .9)[gp_data$unit]
  gp_data$lat <- c(.2, -.3, .6)[gp_data$unit]
  gp_data$n <- 12L; gp_data$y <- rep(c(3L, 4L, 6L, 7L), length.out = nrow(gp_data))
  main <- spatial_car(adjacency, group_var = "unit")
  cases <- list(list(data = d, temporal = temporal_iid("time")),
    list(data = gp_data, spatial = main,
      spatiotemporal = spatiotemporal(main, temporal_rw1("time"), type = "separable",
        parameterization = "noncentered", coords = ~ lon + lat, nn = 4L)),
    list(data = gp_data, spatial = main,
      spatiotemporal = spatiotemporal(spatial_hsgp(~ lon + lat, m = 3L),
        temporal_rw1("time"), type = "IV")))
  fit <- function(case) suppressWarnings(do.call(tratio, c(list(formula = y | n ~ 1,
    family = ratiod_binomial(), mode = "hmc",
    control = list(iter = 40L, warmup = 20L, chains = 1L, cores = 1L, seed = 19L,
      gradient_mode = "A_r", verbose = FALSE)), case)))
  for (i in seq_along(cases)) {
    options(tulpaRatio.use_specs = NULL)
    engine <- fit(cases[[i]])
    expect_equal(calls, i)
    options(tulpaRatio.use_specs = FALSE)
    legacy <- fit(cases[[i]])
    expect_equal(calls, i)
    expect_identical(dim(engine$draws), dim(legacy$draws))
  }
})

test_that("the model sampler rejects incomplete parameter states", {
  old <- options(tulpaRatio.use_specs = FALSE)
  on.exit(options(old))
  trace("cpp_hmc_fit", where = asNamespace("tulpaRatio"),
    tracer = quote(q_init <- q_init[-length(q_init)]), print = FALSE)
  on.exit(untrace("cpp_hmc_fit", where = asNamespace("tulpaRatio")), add = TRUE)
  d <- data.frame(n = 12L, y = c(3L, 4L, 6L, 7L))
  expect_error(tratio(y | n ~ 1, data = d, family = ratiod_binomial(), mode = "hmc",
    control = list(iter = 20L, warmup = 10L, chains = 1L, cores = 1L, verbose = FALSE)),
    "Initial parameter vector length .* does not match layout size")
})
