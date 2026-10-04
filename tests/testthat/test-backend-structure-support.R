# A backend that cannot carry a structure must refuse the model, never fit it
# without the structure and report success.

chain_adjacency <- function(n) {
  A <- matrix(0L, n, n)
  for (i in seq_len(n - 1)) {
    A[i, i + 1] <- 1L
    A[i + 1, i] <- 1L
  }
  A
}

car_spec <- function(n = 10) {
  spatial_car(chain_adjacency(n), group_var = "region")
}

# ---------------------------------------------------------------------------
# The capability table
# ---------------------------------------------------------------------------

test_that("every backend has an entry covering every structure", {
  support <- tulpaRatio:::BACKEND_STRUCTURE_SUPPORT
  backends <- c("hmc", "ess", "pg", "gibbs", "sghmc", "sgld", "laplace", "vi")

  expect_setequal(names(support), backends)
  for (backend in backends) {
    expect_setequal(names(support[[backend]]), tulpaRatio:::MODEL_STRUCTURES)
  }
})


test_that("HMC carries every structure", {
  structures <- list(
    spatial = car_spec(),
    temporal = temporal_rw1("year"),
    spatiotemporal = structure(list(), class = "ratiod_spatiotemporal"),
    zi = structure(list(), class = "ratiod_zi"),
    latent = structure(list(), class = "ratiod_latent")
  )

  expect_length(tulpaRatio:::unsupported_structures("hmc", structures), 0L)
  expect_true(tulpaRatio:::backend_fits_structures("hmc", structures))
})


test_that("unsupported_structures names each structure the backend drops", {
  structures <- list(
    zi = structure(list(), class = "ratiod_zi"),
    latent = structure(list(), class = "ratiod_latent")
  )

  reasons <- tulpaRatio:::unsupported_structures("laplace", structures)

  expect_setequal(names(reasons), c("zi", "latent"))
  expect_true(all(nzchar(reasons)))
})


test_that("a model with no structure is fit by every backend", {
  for (backend in names(tulpaRatio:::BACKEND_STRUCTURE_SUPPORT)) {
    expect_true(tulpaRatio:::backend_fits_structures(backend, list()))
  }
})


test_that("unsupported_structures rejects an unknown backend", {
  expect_error(
    tulpaRatio:::unsupported_structures("nosuchbackend", list()),
    "Unknown backend"
  )
})


test_that("Gibbs carries TVC temporal only", {
  tvc <- temporal_tvc("year", terms = 1)
  rw1 <- temporal_rw1("year")

  expect_true(tulpaRatio:::backend_fits_structures("gibbs", list(temporal = tvc)))
  expect_false(tulpaRatio:::backend_fits_structures("gibbs", list(temporal = rw1)))
  expect_match(
    tulpaRatio:::unsupported_structures("gibbs", list(temporal = rw1))[["temporal"]],
    "temporal_tvc"
  )
})


test_that("Laplace carries a multi-scale temporal term beside a spatial field", {
  ms <- temporal_multiscale("year", trend = "rw1", short_term = "none")
  expect_true(tulpaRatio:::backend_fits_structures(
    "laplace", list(spatial = car_spec(), temporal = ms)
  ))
})


test_that("Laplace and PG carry multi-scale temporal on its own", {
  ms <- temporal_multiscale("year", trend = "rw1", short_term = "none")
  rw1 <- temporal_rw1("year")
  spat <- car_spec()

  for (backend in c("laplace", "pg")) {
    expect_true(tulpaRatio:::backend_fits_structures(backend, list(temporal = ms)))
    expect_false(tulpaRatio:::backend_fits_structures(backend, list(temporal = rw1)))
  }
  # The sampler dispatches on spatial ahead of temporal, so the two together
  # would fit the spatial model alone.
  expect_false(tulpaRatio:::backend_fits_structures(
    "pg", list(spatial = spat, temporal = ms)
  ))
})


test_that("assert_backend_fits_structures names the structure and a way out", {
  err <- expect_error(
    tulpaRatio:::assert_backend_fits_structures(
      "laplace",
      list(latent = structure(list(), class = "ratiod_latent"))
    )
  )

  expect_match(conditionMessage(err), "laplace")
  expect_match(conditionMessage(err), "`latent`")
  expect_match(conditionMessage(err), "mode = \"hmc\"", fixed = TRUE)
})


test_that("assert_backend_fits_structures passes a model the backend fits", {
  expect_true(tulpaRatio:::assert_backend_fits_structures(
    "laplace",
    list(spatial = car_spec())
  ))
})


# ---------------------------------------------------------------------------
# Auto selection never downgrades a structure away
# ---------------------------------------------------------------------------

test_that("auto does not send a temporal model to Laplace on row count alone", {
  selection <- tulpaRatio:::select_inference_mode(
    mode = "auto",
    family = ratiod_binomial(),
    n_obs = 60000,
    structures = list(temporal = temporal_rw1("year"))
  )

  expect_equal(selection$backend, "hmc")
})


test_that("auto still sends an unstructured large model to Laplace", {
  selection <- tulpaRatio:::select_inference_mode(
    mode = "auto",
    family = ratiod_binomial(),
    n_obs = 60000,
    structures = list()
  )

  expect_equal(selection$backend, "laplace")
})


test_that("auto sends a large multi-scale temporal model to Laplace", {
  selection <- tulpaRatio:::select_inference_mode(
    mode = "auto",
    family = ratiod_binomial(),
    n_obs = 60000,
    structures = list(temporal = temporal_multiscale("year", trend = "rw1",
                                                     short_term = "none"))
  )

  expect_equal(selection$backend, "laplace")
})


test_that("auto does not send a latent or zi model to Gibbs", {
  spat <- car_spec()

  for (extra in list(
    list(latent = structure(list(), class = "ratiod_latent")),
    list(zi = structure(list(), class = "ratiod_zi"))
  )) {
    selection <- tulpaRatio:::select_inference_mode(
      mode = "auto",
      family = ratiod_binomial(),
      n_obs = 500,
      structures = c(list(spatial = spat), extra)
    )
    expect_equal(selection$backend, "hmc")
  }
})


test_that("auto still sends an ICAR model to Gibbs", {
  selection <- tulpaRatio:::select_inference_mode(
    mode = "auto",
    family = ratiod_binomial(),
    n_obs = 500,
    structures = list(spatial = car_spec())
  )

  expect_equal(selection$backend, "gibbs")
})


# ---------------------------------------------------------------------------
# tratio() refuses before fitting
# ---------------------------------------------------------------------------

structure_test_data <- function(n = 60, n_times = 6) {
  set.seed(41)
  data.frame(
    successes = rbinom(n, size = 20, prob = 0.3),
    trials = rep(20L, n),
    x = rnorm(n),
    year = rep(seq_len(n_times), length.out = n),
    region = factor(rep(seq_len(n / n_times), each = n_times))
  )
}

test_that("a named backend refuses a structure it cannot carry", {
  df <- structure_test_data()

  refusals <- list(
    list(mode = "laplace", args = list(temporal = temporal_rw1("year")),
         named = "`temporal`"),
    list(mode = "laplace", args = list(latent = latent_factor(n_factors = 1)),
         named = "`latent`"),
    list(mode = "gibbs",
         args = list(spatial = car_spec(),
                     temporal = temporal_rw1("year")),
         named = "`temporal`"),
    list(mode = "pg", args = list(temporal = temporal_rw1("year")),
         named = "`temporal`"),
    list(mode = "sghmc", args = list(latent = latent_factor(n_factors = 1)),
         named = "`latent`"),
    list(mode = "sgld", args = list(latent = latent_factor(n_factors = 1)),
         named = "`latent`"),
    list(mode = "ess", args = list(latent = latent_factor(n_factors = 1)),
         named = "`latent`"),
    list(mode = "vi", args = list(latent = latent_factor(n_factors = 1)),
         named = "`latent`")
  )

  for (case in refusals) {
    err <- expect_error(do.call(tratio, c(list(
      formula = successes | trials ~ x,
      data = df,
      family = ratiod_binomial(),
      mode = case$mode,
      control = list(verbose = FALSE, iter = 20, warmup = 10, chains = 1)
    ), case$args)), info = case$mode)

    expect_match(conditionMessage(err), case$named, fixed = TRUE,
                 info = case$mode)
    expect_match(conditionMessage(err), case$mode, fixed = TRUE,
                 info = case$mode)
  }
})


test_that("mode = 'structured' refuses a structure Laplace cannot carry", {
  df <- structure_test_data()

  expect_error(
    tratio(successes | trials ~ x, data = df, family = ratiod_binomial(),
           zi = zi_binomial(), mode = "structured",
           control = list(verbose = FALSE)),
    "`zi`"
  )
})


# ---------------------------------------------------------------------------
# The structures these backends do carry are fitted
# ---------------------------------------------------------------------------

test_that("Laplace fits a multi-scale temporal model", {
  skip_on_cran()
  df <- structure_test_data()

  fit <- tratio(
    successes | trials ~ x,
    data = df,
    family = ratiod_binomial(),
    temporal = temporal_multiscale("year", trend = "rw1", short_term = "none"),
    mode = "laplace",
    control = list(verbose = FALSE, iter = 200, warmup = 100, chains = 1)
  )

  expect_s3_class(fit, "ratiod_fit")
  expect_equal(fit$backend, "laplace")
  expect_equal(ncol(fit$.internal$temporal_draws$trend), 6L)
  expect_equal(sum(grepl("^trend\\[", colnames(fit$draws))), 6L)
  expect_true(all(is.finite(fit$.internal$temporal_draws$trend)))
})


test_that("PG fits a multi-scale temporal model", {
  skip_on_cran()
  df <- structure_test_data()

  fit <- tratio(
    successes | trials ~ x,
    data = df,
    family = ratiod_binomial(),
    temporal = temporal_multiscale("year", trend = "rw1", short_term = "none"),
    mode = "pg",
    control = list(verbose = FALSE, iter = 200, warmup = 100, chains = 1)
  )

  expect_s3_class(fit, "ratiod_fit")
  expect_equal(fit$backend, "pg")
  expect_true("sigma2_trend" %in% dimnames(fit$draws)$variable)
  expect_equal(ncol(fit$.internal$temporal_draws$trend), 6L)
  expect_true(all(is.finite(fit$.internal$temporal_draws$trend)))
})
