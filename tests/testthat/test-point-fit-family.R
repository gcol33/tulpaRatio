test_that("point-fit printers report the model family name", {
  draws <- matrix(seq_len(12), 6, 2,
                  dimnames = list(NULL, c("beta_num[1]", "beta_denom[1]")))
  for (family in list(ratiod_binomial(), ratiod_poisson_gamma(),
                      ratiod_zi_negbin())) {
    point <- list(family = family, n_obs = 25, n_params = 2,
                  batch_size = 10, iter = 50, warmup = 25,
                  final_epsilon = 0.1, draws = draws)
    vi <- c(point, list(vi_variant = "lowrank", vi_rank = 1, elbo = -10,
                        vi_iterations = 10, vi_converged = TRUE, psis_k = 0.4,
                        param_names = colnames(draws), mu = colMeans(draws),
                        Sigma = diag(2)))
    for (class in c("ratiod_sghmc", "ratiod_sgld", "ratiod_vi")) {
      object <- structure(if (class == "ratiod_vi") vi else point, class = class)
      output <- capture.output(result <- print(object))
      expect_identical(result, object)
      expect_identical(output[startsWith(output, "Family:")],
                       paste("Family:", family$name, ""))
    }
  }
})
