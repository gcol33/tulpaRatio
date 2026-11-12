test_that("a correlation prior must be a Beta", {
    expect_error(temporal_ar1("year", rho_prior = prior_gamma(2, 1)), "Beta prior")
    expect_error(temporal_ar1("year", rho_prior = 0.8), "Beta prior")
    expect_null(temporal_ar1("year")$rho_prior)
})

test_that("the block's prior wins over the model-wide one", {
    block <- prior_beta(9, 1)
    model <- prior_beta(1, 4)
    expect_equal(unname(tulpaRatio:::rho_prior_anchors(block, model)), c(9, 1))
    expect_equal(unname(tulpaRatio:::rho_prior_anchors(NULL, model)), c(1, 4))
    expect_equal(unname(tulpaRatio:::rho_prior_anchors(NULL, NULL)), c(2, 2))
})

test_that("a negatively autocorrelated series recovers a negative rho", {
    skip_on_cran()
    set.seed(20260824)
    n_times <- 24
    n_per_time <- 40
    n <- n_times * n_per_time
    rho_true <- -0.75
    temporal_effect <- numeric(n_times)
    temporal_effect[1] <- rnorm(1, 0, 0.6)
    for (t in 2:n_times) {
        temporal_effect[t] <- rho_true * temporal_effect[t - 1] + rnorm(1, 0, 0.4)
    }
    time_idx <- rep(seq_len(n_times), each = n_per_time)
    x <- rnorm(n)
    p <- plogis(-0.2 + 0.4 * x + temporal_effect[time_idx])
    trials <- rpois(n, 20) + 10L
    successes <- rbinom(n, trials, p)
    X <- cbind(1, x)
    result <- tulpaRatio:::cpp_tulpaRatio_sample(y_num = as.integer(successes), y_denom = as.integer(trials),
        y_num_cont = rep(0, n), y_denom_cont = rep(0, n), X_list = list(X), cfg_list = list(family = "binomial"),
        init = rep(0, ncol(X) + 1L + n_times * 1L + 1L), n_iter = 800L, n_warmup = 400L, L = 10L, n_chains = 1L,
        seed = 7L, n_threads = 1L, verbose = FALSE, temporal_params = fixture_temporal_params(n, time_idx,
            type = "ar1", n_times = n_times), re_params = fixture_re_params(n), sigma_beta = fixture_prior_params()$sigma_beta,
        sigma_prior_scale = fixture_prior_params()$sigma_re_scale, phi_prior_shape = fixture_prior_params()$phi_shape,
        phi_prior_rate = fixture_prior_params()$phi_rate)
    draws <- result$samples[, 4]
    rho <- 2 * plogis(draws) - 1
    expect_true(all(rho > -1 & rho < 1))
    expect_lt(mean(rho), 0)
    expect_lt(quantile(rho, 0.9), 0)
})

test_that("the correlation prior reaches the density", {
    skip_on_cran()
    set.seed(20260825)
    n_times <- 6
    n_per_time <- 8
    n <- n_times * n_per_time
    time_idx <- rep(seq_len(n_times), each = n_per_time)
    x <- rnorm(n)
    p <- plogis(-0.2 + 0.3 * x)
    trials <- rpois(n, 8) + 5L
    successes <- rbinom(n, trials, p)
    X <- cbind(1, x)
    fit_at <- function(a, b) {
        tp <- fixture_temporal_params(n, time_idx, type = "ar1", n_times = n_times)
        tp$rho_prior_a <- a
        tp$rho_prior_b <- b
        res <- tulpaRatio:::cpp_tulpaRatio_sample(y_num = as.integer(successes), y_denom = as.integer(trials),
            y_num_cont = rep(0, n), y_denom_cont = rep(0, n), X_list = list(X), cfg_list = list(family = "binomial"),
            init = rep(0, ncol(X) + 1L + tp$n_params + as.integer(tp$type == "ar1")), n_iter = 800L,
            n_warmup = 400L, L = 10L, n_chains = 1L, seed = 11L, n_threads = 1L, verbose = FALSE, temporal_params = tp,
            re_params = fixture_re_params(n), sigma_beta = fixture_prior_params()$sigma_beta, sigma_prior_scale = fixture_prior_params()$sigma_re_scale,
            phi_prior_shape = fixture_prior_params()$phi_shape, phi_prior_rate = fixture_prior_params()$phi_rate)
        mean(2 * plogis(res$samples[, 4]) - 1)
    }
    up <- fit_at(9, 1)
    down <- fit_at(1, 9)
    expect_gt(up, down)
})
