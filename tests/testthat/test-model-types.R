test_that("model metadata resolves every distribution pair and dispersion", {
  models <- tulpaRatio:::MODEL_TYPES
  expect_setequal(names(models), c("binomial", "beta_binomial", "poisson_gamma",
    "poisson_negbin", "poisson_fixed", "negbin_negbin", "negbin_gamma",
    "negbin_fixed", "gamma_gamma", "lognormal", "lognormal_fixed"))
  for (type in names(models)) {
    spec <- models[[type]]
    probability <- spec$responses[["denom"]] == "trials"
    family <- tulpaRatio:::new_ratiod_family(type,
      list(num = if (probability) "logit" else "log", denom = "log"))
    expect_identical(tulpaRatio:::get_hmc_model_type(family), type)
    expect_identical(family$numerator$distribution, spec$distributions[["num"]])
    expect_identical(family$denominator$distribution, spec$distributions[["denom"]])
    expect_identical(tulpaRatio:::gibbs_model_type(family),
                     if ("gibbs" %in% spec$samplers) type else NA_character_)
    prior <- tulpaRatio:::default_prior("phi", type)
    expect_identical(prior$shape, 2)
    expect_identical(prior$rate, switch(type, lognormal = 2,
      poisson_gamma = , gamma_gamma = 0.5, 0.1))
  }
})

test_that("auto selects only a sampler implemented for the family", {
  for (type in names(tulpaRatio:::MODEL_TYPES)) {
    spec <- tulpaRatio:::MODEL_TYPES[[type]]
    family <- tulpaRatio:::new_ratiod_family(type, list(
      num = if (spec$responses[["denom"]] == "trials") "logit" else "log", denom = "log"))
    for (n_obs in c(100, 60000)) {
      backend <- tulpaRatio:::auto_select_mode(family, n_obs)$backend
      expect_true(backend %in% spec$samplers, info = paste(type, n_obs))
    }
  }
})

test_that("exported family constructors each call the shared builder", {
  constructors <- getNamespaceExports("tulpaRatio")
  constructors <- constructors[startsWith(constructors, "ratiod_")]
  constructors <- setdiff(constructors, c("ratiod_formula", "ratiod_priors", "ratiod_threads",
    "ratiod_average", "ratiod_benchmark", "ratiod_benchmark_compare", "ratiod_compare"))
  for (name in constructors) {
    fn <- getExportedValue("tulpaRatio", name)
    calls <- as.list(body(fn))
    expect_equal(length(calls), 2L, info = name)
    expect_identical(calls[[2L]][[1L]], as.name("new_ratiod_family"), info = name)
  }
})

test_that("equivalent count constructors have identical metadata and labels", {
  expect_identical(ratiod_poisson(denom_family = "gamma"), ratiod_poisson_gamma())
  expect_identical(ratiod_negbin(denom_family = "gamma"), ratiod_negbin_gamma())
  expect_identical(ratiod_negbin(denom_family = "negbin"), ratiod_negbin_negbin())
  for (den in c("fixed", "gamma", "negbin")) {
    for (base in c("poisson", "negbin")) {
      plain <- get(paste0("ratiod_", base))(denom_family = den)
      for (prefix in c("zi", "hurdle")) {
        inflated <- get(paste0("ratiod_", prefix, "_", base))(denom_family = den)
        expect_identical(inflated$denominator, plain$denominator)
        expect_identical(inflated$numerator$base_distribution, plain$numerator$distribution)
        expect_true(endsWith(inflated$description, plain$description))
      }
    }
  }
})

test_that("inflation formulas and links reach each predictor", {
  family <- ratiod_zoi_binomial(link_num = "probit", link_zi = "cloglog",
    link_oi = "probit", zi_formula = ~ x, oi_formula = ~ z)
  expect_identical(family$zi_formula, ~ x)
  expect_identical(family$oi_formula, ~ z)
  expect_identical(family$numerator$link, "probit")
  expect_identical(family$numerator$link_zi, "cloglog")
  expect_identical(family$numerator$link_oi, "probit")
  expect_true(family$one_inflated)
  expect_true(family$zero_inflated)
  expect_error(tulpaRatio:::new_ratiod_family("absent", list()), "Unknown model type")
  expect_error(tulpaRatio:::new_ratiod_family("gamma_gamma", list(num = "log", denom = "log"),
    list(type = "mixture")), "does not support")
})
