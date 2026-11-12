gamma_gamma_data <- function(n = 120, seed = 21) {
    set.seed(seed)
    d <- data.frame(x = rnorm(n), lon = runif(n), lat = runif(n))
    d$a <- rgamma(n, 6, 6/exp(1 + 0.5 * d$x))
    d$b <- rgamma(n, 6, 6/exp(2))
    d
}

test_that("an unknown family stops during engine construction", {
    expect_error(tulpaRatio:::cpp_tulpaRatio_sample(y_num = rep(0L, 20), y_denom = rep(1L, 20), y_num_cont = rep(1,
        20), y_denom_cont = rep(1, 20), X_list = list(matrix(1, 20, 1)), cfg_list = list(family = "gamma_gama"),
        init = 0, n_iter = 10L, n_warmup = 5L, method = "ess"), "Unknown likelihood family")
})

test_that("ESS, SGHMC and VI fit a gamma-gamma model", {
    skip_on_cran()
    d <- gamma_gamma_data()
    for (m in c("ess", "vi")) {
        fit <- suppressWarnings(tratio(a | b ~ x, data = d, family = ratiod_gamma_gamma(), mode = m,
            control = list(iter = 1000, chains = 1, seed = 1)))
        b <- coef(fit)
        expect_equal(unname(b[c("num:(Intercept)", "num:x", "denom:(Intercept)")]), c(1, 0.5, 2), tolerance = 0.15,
            label = m)
    }
    fit <- suppressWarnings(tratio(a | b ~ x, data = d, family = ratiod_gamma_gamma(), mode = "sghmc",
        control = list(iter = 2000, chains = 1, seed = 1)))
    expect_true(all(is.finite(coef(fit))))
    expect_equal(unname(coef(fit)["denom:(Intercept)"]), 2, tolerance = 0.5)
})

test_that("a gamma-gamma model with a GP field reads its continuous numerator", {
    skip_on_cran()
    d <- gamma_gamma_data(60)
    fit <- suppressWarnings(tratio(a | b ~ x, data = d, family = ratiod_gamma_gamma(), spatial = spatial_gp(~lon +
        lat), control = list(iter = 300, chains = 1, seed = 1)))
    expect_equal(unname(coef(fit)["num:x"]), 0.5, tolerance = 0.35)
})
