test_that("VI variants recover a binomial posterior checked by quadrature", {
    skip_on_cran()
    set.seed(1701)
    d <- data.frame(n = rep(10L, 80), y = rbinom(80, 10, plogis(0.4)))
    y <- sum(d$y)
    n <- sum(d$n)
    log_density <- function(x) y * x - n * log1p(exp(x)) - x * x/8
    mode <- optimize(log_density, c(-5, 5), maximum = TRUE)$maximum
    mass <- function(x) exp(log_density(x) - log_density(mode))
    z <- integrate(mass, -8, 8, rel.tol = 1e-12)$value
    mean <- integrate(function(x) x * mass(x), -8, 8, rel.tol = 1e-12)$value/z
    variance <- integrate(function(x) (x - mean)^2 * mass(x), -8, 8, rel.tol = 1e-12)$value/z
    for (variant in c("meanfield", "fullrank", "lowrank")) {
        fit <- tratio(y | n ~ 1, data = d, family = ratiod_binomial(), mode = "vi", priors = ratiod_priors(beta = prior_normal(0,
            2)), control = list(vi_variant = variant, iter = 10000L, seed = 1701L, verbose = FALSE))
        expect_true(fit$vi_converged, info = variant)
        expect_lt(abs(fit$mu[1] - mean)/sqrt(variance), 0.3, label = variant)
        expect_lt(abs(fit$mu[1] - 0.4)/sqrt(variance), 3, label = variant)
        expect_lt(abs(sqrt(fit$Sigma[1, 1]/variance) - 1), 0.3, label = variant)
    }
})
