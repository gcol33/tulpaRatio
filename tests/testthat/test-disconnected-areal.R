# An areal field on a graph with several connected components and an island:
# centred per component, the island on a proper prior, BYM2 scaled per
# component (Freni-Sterrantino, Ventrucci and Rue 2018).

disconnected_map <- function() {
  # Two 3 x 3 grids and one island
  grid <- expand.grid(x = 1:3, y = 1:3)
  block <- 1 * (as.matrix(dist(grid)) <= 1.01)
  diag(block) <- 0
  A <- matrix(0, 19, 19)
  A[1:9, 1:9] <- block
  A[10:18, 10:18] <- block
  dimnames(A) <- list(as.character(1:19), as.character(1:19))
  A
}

disconnected_data <- function(seed = 51) {
  set.seed(seed)
  A <- disconnected_map()
  field <- c(scale(1:9)[, 1], -scale(1:9)[, 1], 0) * 0.6
  df <- data.frame(region = factor(rep(1:19, each = 12), levels = 1:19))
  df$x <- rnorm(nrow(df))
  df$trials <- 20L
  df$y <- rbinom(nrow(df), df$trials,
                 stats::plogis(-0.4 + 0.5 * df$x + field[as.integer(df$region)]))
  list(df = df, adj = A, field = field)
}

test_that("components and islands are found", {
  A <- disconnected_map()
  comp <- tulpaRatio:::areal_components(A)
  expect_equal(comp, c(rep(0L, 9), rep(1L, 9), -1L))
  expect_null(tulpaRatio:::areal_components(A[1:9, 1:9]))

  car <- spatial_car(A, spatial_level = "group", group_var = "region")
  expect_equal(car$component, comp)
  expect_equal(car$node_prec, rep(1, 19))
  bym <- spatial_bym2(A, spatial_level = "group", group_var = "region")
  expect_equal(bym$component, comp)
  # Two identical components share the reference scale; the island's
  # multiplier brings it to unit variance
  expect_equal(bym$node_prec[1:18], rep(1, 18))
  expect_equal(bym$node_prec[19], bym$scale_factor^2)
  expect_identical(spatial_car(A, spatial_level = "group", group_var = "region",
                              parameterization = "collapsed")$parameterization, "collapsed")
})

test_that("collapsed requests on disconnected graphs use joint engine fields", {
  skip_on_cran()
  d <- disconnected_data()
  op <- options(tulpaRatio.use_specs = FALSE)
  on.exit(options(op))
  fields <- list(
    spatial_car(d$adj, group_var = 'region', parameterization = 'collapsed'),
    spatial_bym2(d$adj, group_var = 'region', parameterization = 'collapsed'))
  for (spatial in fields) {
    expect_error(tratio(y | trials ~ x, data = d$df, family = ratiod_binomial(),
      spatial = spatial, mode = 'hmc', control = list(iter = 20L, chains = 1L,
        cores = 1L, seed = 18L, verbose = FALSE)), 'connected')
  }
  options(tulpaRatio.use_specs = TRUE)
  for (seed in c(18L, 181L, 182L)) for (spatial in fields) {
    fit <- suppressWarnings(tratio(y | trials ~ x, data = d$df,
      family = ratiod_binomial(), spatial = spatial, mode = 'hmc',
      control = list(iter = 2500L, warmup = 750L, chains = 1L, cores = 1L,
        seed = seed, adapt_delta = .99, verbose = FALSE)))
    expect_identical(fit$spatial$parameterization, 'auto')
    expect_lt(abs(coef(fit)[['num:x']] - .5), .2)
    eta <- -.4 + .5 * d$df$x + d$field[as.integer(d$df$region)]
    fitted_eta <- qlogis(fitted(fit, summary = FALSE)$numerator)
    expect_lt(sqrt(mean((colMeans(fitted_eta) - eta)^2)), .2)
  }
})

test_that("an ICAR and a BYM2 field on a disconnected map fit and recover", {
  skip_on_cran()
  d <- disconnected_data()
  for (spatial in list(spatial_car(d$adj, spatial_level = "group", group_var = "region"),
                       spatial_bym2(d$adj, spatial_level = "group", group_var = "region"))) {
    fit <- suppressWarnings(tratio(
      y | trials ~ x, data = d$df, family = ratiod_binomial(), spatial = spatial,
      control = list(iter = 1000, chains = 2, seed = 1)))
    expect_identical(fit$backend, "hmc")
    b <- coef(fit)
    expect_equal(unname(b["num:x"]), 0.5, tolerance = 0.2)
    expect_equal(unname(b["num:(Intercept)"]), -0.4, tolerance = 0.3)
  }
})

test_that("samplers that centre over the whole graph refuse a disconnected map", {
  d <- disconnected_data()
  car <- spatial_car(d$adj, spatial_level = "group", group_var = "region")
  expect_error(
    tratio(y | trials ~ x, data = d$df, family = ratiod_binomial(),
           spatial = car, mode = "gibbs",
           control = list(iter = 20, chains = 1, seed = 1)),
    "connected components")
  selection <- tulpaRatio:::select_inference_mode(
    mode = "auto", family = ratiod_binomial(), n_obs = nrow(d$df),
    structures = list(spatial = car))
  expect_equal(selection$backend, "hmc")
})

test_that("areal samplers fit the fields declared in their support table", {
  skip_on_cran()
  d <- disconnected_data()
  keep <- as.integer(d$df$region) <= 9
  df <- d$df[keep, ]
  df$region <- factor(as.integer(df$region), levels = 1:9)
  A <- d$adj[1:9, 1:9]
  fields <- list(
    icar = spatial_car(A, spatial_level = "group", group_var = "region"),
    car = spatial_car(A, spatial_level = "group", group_var = "region", proper = TRUE),
    bym2 = spatial_bym2(A, spatial_level = "group", group_var = "region"),
    bym2_disconnected = spatial_bym2(d$adj, spatial_level = "group", group_var = "region"))
  for (nm in names(fields)) {
    data <- if (nm == "bym2_disconnected") d$df else df
    for (m in c("ess", "sghmc", "sgld", "vi")) {
      run <- function() suppressWarnings(tratio(
        y | trials ~ x, data = data, family = ratiod_binomial(),
        spatial = fields[[nm]], mode = m,
        control = list(iter = 300, chains = 1, seed = 1)))
      reasons <- tulpaRatio:::unsupported_structures(m, list(spatial = fields[[nm]]))
      if (length(reasons)) {
        expect_error(run(), reasons[["spatial"]], fixed = TRUE, info = paste(nm, m))
      } else {
        fit <- run()
        expect_true(all(is.finite(coef(fit))), info = paste(nm, m))
      }
    }
  }
})

test_that("ESS moves the unstructured BYM2 component (#145)", {
  skip_on_cran()
  d <- disconnected_data()
  fit <- suppressWarnings(tratio(
    y | trials ~ x, data = d$df, family = ratiod_binomial(),
    spatial = spatial_bym2(d$adj, spatial_level = "group", group_var = "region"),
    mode = "ess", control = list(iter = 300, chains = 1, seed = 1)))
  dr <- as.matrix(fit$draws)
  theta <- dr[, startsWith(colnames(dr), "theta["), drop = FALSE]
  expect_equal(ncol(theta), 19L)
  expect_true(all(apply(theta, 2, stats::sd) > 0))
})
