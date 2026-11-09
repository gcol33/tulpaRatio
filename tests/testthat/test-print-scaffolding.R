print_scaffolding_cases <- function() {
  draws <- matrix(seq_len(120) / 10, 10, 12,
                  dimnames = list(NULL, paste0("beta", seq_len(12))))
  point <- list(family = ratiod_binomial(), n_obs = 25, n_params = 12,
                batch_size = 10, iter = 50, warmup = 25, final_epsilon = 0.1234567,
                draws = draws)
  vi <- c(point, list(vi_variant = "lowrank", vi_rank = 2, elbo = -100.125,
                      vi_iterations = 10, vi_converged = TRUE, psis_k = 0.4,
                      param_names = colnames(draws), mu = colMeans(draws),
                      Sigma = diag(12)))
  posterior <- list(draws = array(seq_len(120) / 10, c(10, 6, 2)),
                    term_names = c("a", "b"), n_obs = 6, n_times = 6,
                    n_svc = 2, n_tvc = 2, n_draws = 10,
                    coords = cbind(seq_len(6), seq_len(6) * 2),
                    time_levels = 2011:2016, cov = "matern", structure = "rw1")
  adjacency <- matrix(c(0, 1, 1, 0), 2)
  cases <- list(
    sghmc = structure(point, class = "ratiod_sghmc"),
    sgld = structure(point, class = "ratiod_sgld"),
    vi = structure(vi, class = "ratiod_vi"),
    spatial = spatial_car(adjacency, group_var = "site"),
    gp = spatial_gp(~ lon + lat),
    hsgp = spatial_hsgp(~ lon + lat),
    multiscale = spatial_multiscale(~ lon + lat),
    svc = spatial_svc(~ lon + lat, terms = 1),
    svc_posterior = structure(posterior, class = "ratiod_svc_posterior"),
    temporal = temporal_rw1("year"),
    temporal_multiscale = temporal_multiscale("year"),
    temporal_gp = temporal_gp("year"),
    tvc = temporal_tvc("year", terms = 1),
    tvc_posterior = structure(posterior, class = "ratiod_tvc_posterior"),
    latent = latent_factor(2),
    priors = ratiod_priors(),
    zi = tulpaRatio:::family_inflation_spec(ratiod_zi_poisson(denom_family = "negbin")),
    gpu = structure(list(available = FALSE), class = "tulpaRatio_gpu_info"),
    ratio = structure(list(type = "log", n_obs = 6, n_draws = 10), class = "ratiod_ratio"),
    ratio_summary = structure(data.frame(mean = 1:2), type = "response",
                              class = c("ratiod_ratio_summary", "data.frame")),
    contrast = structure(data.frame(mean = 1), type = "ratio", contrast_var = "site",
                         ref = "a", class = c("ratiod_contrast", "data.frame")),
    fitted = structure(data.frame(mean = 1), n_draws = 10,
                       class = c("ratiod_fitted", "data.frame")),
    prediction = structure(data.frame(mean = 1), n_draws = 10, newdata = data.frame(x = 1),
                           class = c("ratiod_prediction", "data.frame")),
    summary = structure(list(tier = NA, backend = "hmc", family = "binomial"),
                         class = "summary.ratiod_fit"),
    diagnostics = structure(data.frame(parameter = "beta", rhat = 1, ess_bulk = 500,
                                       ess_tail = 500), class = c("ratiod_diagnostics", "data.frame")),
    temporal_posterior = structure(list(type = "rw1", n_times = 6, n_groups = 1,
                                        n_draws = 10), class = "ratiod_temporal_posterior"),
    geweke = structure(data.frame(parameter = "beta", z_score = 0.125, p_value = 0.9),
                        class = c("ratiod_geweke", "data.frame")),
    average = structure(list(weights_method = "stacking", n_models = 2, type = "ratio",
                              weights = c(0.3, 0.7), models = c("a", "b"),
                              predictions = data.frame(mean = 1:3)), class = "ratiod_average"),
    formula = ratiod_formula(y | e ~ x, data = data.frame(y = 1:3, e = 4:6, x = 1:3)),
    spatiotemporal = spatiotemporal(spatial_car(adjacency, group_var = "site"),
                                    temporal_rw1("year"))
  )
  sim <- list(family = ratiod_binomial(), n = 3, n_groups = 1, seed = 12,
              true_params = list(beta_num = c(1, 0.3), beta_denom = 2, sigma_re = 0.5),
              data = data.frame(y = 1:3, n = 5))
  cases$simulated <- structure(sim, class = "ratiod_simdata")
  cases$sbc <- structure(list(sim), n_sims = 1, family = ratiod_binomial(),
                         class = "ratiod_sbc_sims")
  cases
}

test_that("print scaffolding preserves its full output", {
  cases <- print_scaffolding_cases()
  for (name in names(cases)) {
    actual <- capture.output(print(cases[[name]]))
    expected <- dget(test_path("fixtures", paste0("print-", name, ".txt")))
    expect_identical(actual, expected, info = name)
  }
})

test_that("varying coefficient summaries retain exact sample summaries", {
  cases <- print_scaffolding_cases()
  for (name in c("svc_posterior", "tvc_posterior")) {
    object <- cases[[name]]
    actual <- summary(object, probs = c(0.1, 0.5, 0.9))
    for (j in seq_along(object$term_names)) {
      rows <- actual$term == object$term_names[j]
      term_draws <- object$draws[, , j]
      expect_identical(actual$mean[rows], unname(colMeans(term_draws)))
      expect_identical(actual$sd[rows], unname(apply(term_draws, 2, sd)))
      for (prob in c(0.1, 0.5, 0.9)) {
        expect_identical(actual[[paste0("q", prob * 100)]][rows],
                         unname(apply(term_draws, 2, quantile, probs = prob)))
      }
    }
  }
})
