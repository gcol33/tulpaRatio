test_that("HMC backend works for negbin_negbin model", {
    skip_on_cran()
    set.seed(456)
    N <- 80
    x <- rnorm(N)
    X <- cbind(1, x)
    true_beta_num <- c(2, 0.3)
    true_beta_denom <- c(2.5, -0.2)
    true_phi <- 5
    eta_num <- X %*% true_beta_num
    eta_denom <- X %*% true_beta_denom
    mu_num <- exp(eta_num)
    mu_denom <- exp(eta_denom)
    y_num <- rnbinom(N, size = true_phi, mu = mu_num)
    y_denom <- rnbinom(N, size = true_phi, mu = mu_denom)
    q_init <- c(rep(0, 4), log(5), log(5))
    fit <- tulpaRatio:::cpp_tulpaRatio_sample(y_num = as.integer(y_num), y_denom = as.integer(y_denom),
        y_num_cont = rep(0, N), y_denom_cont = rep(0, N), X_list = list(X, X), cfg_list = list(family = "negbin_negbin"),
        init = q_init, n_iter = 200L, n_warmup = 100L, L = 10L, n_chains = 1L, seed = 456L, n_threads = 1L,
        verbose = FALSE, temporal_params = fixture_temporal_params(N), re_params = fixture_re_params(N),
        sigma_beta = fixture_prior_params()$sigma_beta, sigma_prior_scale = fixture_prior_params()$sigma_re_scale,
        phi_prior_shape = fixture_prior_params()$phi_shape, phi_prior_rate = fixture_prior_params()$phi_rate)
    expect_true(all(is.finite(fit$samples)))
    expect_equal(nrow(fit$samples), 100)
    expect_equal(ncol(fit$samples), 6)
    beta_num_post <- colMeans(fit$samples[, 1:2])
    expect_equal(beta_num_post[1], true_beta_num[1], tolerance = 0.5)
})

test_that("HMC backend works for poisson_gamma model", {
    skip_on_cran()
    set.seed(789)
    N <- 60
    x <- rnorm(N)
    X <- cbind(1, x)
    true_beta_num <- c(2, 0.4)
    true_beta_denom <- c(1.5, 0.2)
    true_shape <- 3
    eta_num <- X %*% true_beta_num
    eta_denom <- X %*% true_beta_denom
    y_num <- rpois(N, exp(eta_num))
    y_denom <- rgamma(N, shape = true_shape, rate = true_shape/exp(eta_denom))
    q_init <- c(rep(0, 4), log(3))
    fit <- tulpaRatio:::cpp_tulpaRatio_sample(y_num = as.integer(y_num), y_denom = as.integer(rep(1,
        N)), y_num_cont = rep(0, N), y_denom_cont = y_denom, X_list = list(X, X), cfg_list = list(family = "poisson_gamma"),
        init = q_init, n_iter = 200L, n_warmup = 100L, L = 10L, n_chains = 1L, seed = 789L, n_threads = 1L,
        verbose = FALSE, temporal_params = fixture_temporal_params(N), re_params = fixture_re_params(N),
        sigma_beta = fixture_prior_params()$sigma_beta, sigma_prior_scale = fixture_prior_params()$sigma_re_scale,
        phi_prior_shape = fixture_prior_params()$phi_shape, phi_prior_rate = fixture_prior_params()$phi_rate)
    expect_true(all(is.finite(fit$samples)))
    expect_equal(nrow(fit$samples), 100)
    expect_equal(ncol(fit$samples), 5)
})

test_that("tratio() fits poisson_gamma model with HMC backend", {
    skip_on_cran()
    set.seed(123)
    n <- 30
    df <- data.frame(count = rpois(n, lambda = 10), effort = rgamma(n, shape = 5, rate = 1), x = rnorm(n))
    fit <- tratio(count | effort ~ x, data = df, family = ratiod_poisson_gamma(), mode = "hmc", control = list(iter = 100,
        warmup = 50, chains = 1, verbose = FALSE))
    expect_s3_class(fit, "ratiod_fit")
    expect_equal(fit$backend, "hmc")
    expect_true(is.array(fit$draws))
})

test_that("tratio() fits binomial model with HMC backend", {
    skip_on_cran()
    set.seed(456)
    n <- 30
    trials <- sample(10:20, n, replace = TRUE)
    df <- data.frame(successes = rbinom(n, trials, 0.4), trials = trials, x = rnorm(n))
    fit <- tratio(successes | trials ~ x, data = df, family = ratiod_binomial(), mode = "hmc", control = list(iter = 100,
        warmup = 50, chains = 1, verbose = FALSE))
    expect_s3_class(fit, "ratiod_fit")
    expect_equal(fit$backend, "hmc")
    expect_true(is.array(fit$draws))
})

test_that("tratio() fits negbin_negbin model with HMC backend", {
    skip_on_cran()
    set.seed(789)
    n <- 30
    df <- data.frame(y_num = rnbinom(n, size = 5, mu = 10), y_denom = rnbinom(n, size = 5, mu = 15),
        x = rnorm(n))
    fit <- tratio(y_num | y_denom ~ x, data = df, family = ratiod_negbin_negbin(), mode = "hmc", control = list(iter = 100,
        warmup = 50, chains = 1, verbose = FALSE))
    expect_s3_class(fit, "ratiod_fit")
    expect_equal(fit$backend, "hmc")
})

test_that("tratio() with random effects using HMC backend", {
    skip_on_cran()
    set.seed(111)
    n_groups <- 5
    n_per_group <- 6
    n <- n_groups * n_per_group
    df <- data.frame(count = rpois(n, lambda = 10), effort = rgamma(n, shape = 5, rate = 1), x = rnorm(n),
        site = factor(rep(1:n_groups, each = n_per_group)))
    fit <- tratio(count | effort ~ x + (1 | site), data = df, family = ratiod_poisson_gamma(), mode = "hmc",
        control = list(iter = 100, warmup = 50, chains = 1, verbose = FALSE))
    expect_s3_class(fit, "ratiod_fit")
    draws <- fit$draws
    if (is.array(draws) && length(dim(draws)) == 3) {
        param_names <- dimnames(draws)[[3]]
    }
    else if (is.matrix(draws)) {
        param_names <- colnames(draws)
    }
    else {
        param_names <- names(draws)
    }
    expect_true(any(grepl("sigma", param_names, ignore.case = TRUE)) || length(param_names) > 5)
})

test_that("summary.ratiod_fit returns correct structure", {
    skip_on_cran()
    set.seed(222)
    n <- 20
    df <- data.frame(count = rpois(n, lambda = 10), effort = rgamma(n, shape = 5, rate = 1), x = rnorm(n))
    fit <- tratio(count | effort ~ x, data = df, family = ratiod_poisson_gamma(), mode = "hmc", control = list(iter = 100,
        warmup = 50, chains = 1, verbose = FALSE))
    summ <- summary(fit)
    expect_true(is.data.frame(summ) || is.list(summ))
})

test_that("print.ratiod_fit works without error", {
    skip_on_cran()
    set.seed(333)
    n <- 20
    df <- data.frame(count = rpois(n, lambda = 10), effort = rgamma(n, shape = 5, rate = 1), x = rnorm(n))
    fit <- tratio(count | effort ~ x, data = df, family = ratiod_poisson_gamma(), mode = "hmc", control = list(iter = 100,
        warmup = 50, chains = 1, verbose = FALSE))
    output <- capture.output(print(fit))
    expect_true(length(output) > 0)
})

test_that("ratio() works with HMC fit", {
    skip_on_cran()
    set.seed(444)
    n <- 20
    df <- data.frame(count = rpois(n, lambda = 10), effort = rgamma(n, shape = 5, rate = 1), x = rnorm(n))
    fit <- tratio(count | effort ~ x, data = df, family = ratiod_poisson_gamma(), mode = "hmc", control = list(iter = 100,
        warmup = 50, chains = 1, verbose = FALSE))
    r <- ratio(fit)
    expect_s3_class(r, "ratiod_ratio")
    expect_true(is.matrix(r$draws))
    expect_equal(ncol(r$draws), n)
})

test_that("ratio_contrast() works with HMC fit", {
    skip_on_cran()
    set.seed(555)
    n <- 20
    df <- data.frame(count = rpois(n, lambda = 10), effort = rgamma(n, shape = 5, rate = 1), x = rnorm(n),
        group = factor(rep(c("A", "B"), each = n/2)))
    fit <- tratio(count | effort ~ x + group, data = df, family = ratiod_poisson_gamma(), mode = "hmc",
        control = list(iter = 100, warmup = 50, chains = 1, verbose = FALSE))
    contrast <- ratio_contrast(fit, contrast = ~group)
    expect_true(is.numeric(contrast) || is.list(contrast) || inherits(contrast, "ratiod_contrast"))
})
