test_that("specs/H == specs/A_r bit-exact for every ratio family", {
    skip_on_cran()
    skip_on_ci()
    set.seed(20260503)
    fit_at <- function(formula, data, family, mode) {
        fit <- tulpaRatio::tratio(formula = formula, data = data, family = family, mode = "hmc", control = list(iter = 1000L,
            warmup = 250L, chains = 1L, seed = 42L, verbose = FALSE, gradient_mode = mode))
        colMeans(fit$draws)
    }
    {
        n <- 100
        x1 <- rnorm(n)
        x2 <- rnorm(n)
        p <- plogis(0.4 + 0.8 * x1 - 0.5 * x2)
        n_trials <- sample(5:20, n, replace = TRUE)
        y <- rbinom(n, n_trials, p)
        dat <- data.frame(y = y, n_trials = n_trials, x1 = x1, x2 = x2)
        h <- fit_at(y | n_trials ~ x1 + x2, dat, tulpaRatio::ratiod_binomial(), "H")
        ar <- fit_at(y | n_trials ~ x1 + x2, dat, tulpaRatio::ratiod_binomial(), "A_r")
        expect_equal(h, ar, tolerance = 1e-10, info = "binomial")
    }
    {
        n <- 100
        x1 <- rnorm(n)
        x2 <- rnorm(n)
        mu_n <- exp(1 + 0.4 * x1)
        mu_d <- exp(0.5 + 0.3 * x2)
        y_num <- rpois(n, mu_n)
        y_denom <- rgamma(n, shape = 4, rate = 4/mu_d)
        dat <- data.frame(y_num = y_num, y_denom = y_denom, x1 = x1, x2 = x2)
        h <- fit_at(y_num | y_denom ~ x1 + x2, dat, tulpaRatio::ratiod_poisson_gamma(), "H")
        ar <- fit_at(y_num | y_denom ~ x1 + x2, dat, tulpaRatio::ratiod_poisson_gamma(), "A_r")
        expect_equal(h, ar, tolerance = 1e-10, info = "poisson_gamma")
    }
    {
        n <- 100
        x1 <- rnorm(n)
        x2 <- rnorm(n)
        mu_n <- exp(1 + 0.4 * x1)
        mu_d <- exp(0.5 + 0.3 * x2)
        y_num <- rnbinom(n, size = 5, mu = mu_n)
        y_denom <- rgamma(n, shape = 4, rate = 4/mu_d)
        dat <- data.frame(y_num = y_num, y_denom = y_denom, x1 = x1, x2 = x2)
        h <- fit_at(y_num | y_denom ~ x1 + x2, dat, tulpaRatio::ratiod_negbin_gamma(), "H")
        ar <- fit_at(y_num | y_denom ~ x1 + x2, dat, tulpaRatio::ratiod_negbin_gamma(), "A_r")
        expect_equal(h, ar, tolerance = 1e-10, info = "negbin_gamma")
    }
    {
        n <- 100
        x1 <- rnorm(n)
        x2 <- rnorm(n)
        mu_n <- exp(1 + 0.4 * x1)
        mu_d <- exp(0.7 + 0.3 * x2)
        y_num <- rnbinom(n, size = 5, mu = mu_n)
        y_denom <- rnbinom(n, size = 5, mu = mu_d)
        dat <- data.frame(y_num = y_num, y_denom = y_denom, x1 = x1, x2 = x2)
        h <- fit_at(y_num | y_denom ~ x1 + x2, dat, tulpaRatio::ratiod_negbin_negbin(), "H")
        ar <- fit_at(y_num | y_denom ~ x1 + x2, dat, tulpaRatio::ratiod_negbin_negbin(), "A_r")
        expect_equal(h, ar, tolerance = 1e-10, info = "negbin_negbin")
    }
    {
        n <- 100
        x1 <- rnorm(n)
        x2 <- rnorm(n)
        mu_n <- exp(1 + 0.4 * x1)
        mu_d <- exp(0.5 + 0.3 * x2)
        y_num <- rgamma(n, shape = 4, rate = 4/mu_n)
        y_denom <- rgamma(n, shape = 5, rate = 5/mu_d)
        dat <- data.frame(y_num = y_num, y_denom = y_denom, x1 = x1, x2 = x2)
        h <- fit_at(y_num | y_denom ~ x1 + x2, dat, tulpaRatio::ratiod_gamma_gamma(), "H")
        ar <- fit_at(y_num | y_denom ~ x1 + x2, dat, tulpaRatio::ratiod_gamma_gamma(), "A_r")
        expect_equal(h, ar, tolerance = 1e-10, info = "gamma_gamma")
    }
    {
        n <- 100
        x1 <- rnorm(n)
        x2 <- rnorm(n)
        y_num <- exp(rnorm(n, 1 + 0.4 * x1, 0.5))
        y_denom <- exp(rnorm(n, 0.5 + 0.3 * x2, 0.5))
        dat <- data.frame(y_num = y_num, y_denom = y_denom, x1 = x1, x2 = x2)
        h <- fit_at(y_num | y_denom ~ x1 + x2, dat, tulpaRatio::ratiod_lognormal(), "H")
        ar <- fit_at(y_num | y_denom ~ x1 + x2, dat, tulpaRatio::ratiod_lognormal(), "A_r")
        expect_equal(h, ar, tolerance = 1e-10, info = "lognormal")
    }
    {
        n <- 100
        x1 <- rnorm(n)
        x2 <- rnorm(n)
        p <- plogis(0.4 + 0.8 * x1 - 0.5 * x2)
        n_trials <- sample(5:20, n, replace = TRUE)
        phi <- 10
        a <- p * phi
        b <- (1 - p) * phi
        y <- vapply(seq_len(n), function(i) {
            rbinom(1, n_trials[i], rbeta(1, a[i], b[i]))
        }, integer(1))
        dat <- data.frame(y = y, n_trials = n_trials, x1 = x1, x2 = x2)
        h <- fit_at(y | n_trials ~ x1 + x2, dat, tulpaRatio::ratiod_beta_binomial(), "H")
        ar <- fit_at(y | n_trials ~ x1 + x2, dat, tulpaRatio::ratiod_beta_binomial(), "A_r")
        expect_equal(h, ar, tolerance = 1e-10, info = "beta_binomial")
    }
})
