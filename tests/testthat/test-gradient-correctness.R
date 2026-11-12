FIELDS <- c("icar", "bym2", "rw1", "rw2", "temporal_ar1", "temporal_iid", "temporal_ar1_nc", "icar_rw1",
    "icar_disconnected", "bym2_disconnected", "icar_disconnected_rw1", "rw1_nc", "rw2_nc", "rw1_cyclic",
    "rw1_cyclic_nc", "rw1_panel", "rw1_nc_panel", "rw2_nc_panel", "rw1_cyclic_nc_panel")

MULTI_FIELDS <- c("icar_ms", "bym2_ms", "icar_st", "bym2_st")

STRUCTURE_FIELDS <- c("re", "re_crossed", "re_slopes", "re_slopes_corr", "hsgp", "tvc", "tvc_ar1", "tvc_iid")

COLLAPSED_FIELDS <- c("icar_collapsed", "bym2_collapsed")

COLLAPSED_TEMPORAL_FIELDS <- c("icar_collapsed_rw1", "bym2_collapsed_rw1")

CAR_PROPER_FIELDS <- c("car_proper", "car_proper_raw", "car_proper_ms", "car_proper_st")

ST_IV_FIELDS <- c("st4", "st4_nc")

ST_OTHER_FIELDS <- c("st2", "st2_rw2", "st2_ar1", "st3", "icar_st_nc", "st2_nc", "st2_rw2_nc", "st2_ar1_nc",
    "st2_cyclic", "st2_cyclic_nc", "st3_nc")

ST_IV_AR1_FIELDS <- c("st4_ar1", "st4_ar1_nc")

ST_GP_FIELDS <- c("stgp", "stgp_matern", "stgp_gneiting", "stgp_latent", "stgp_latent2", "stgp_nc", "stgp_gneiting_nc")

KERNEL_FIELDS <- c("gp", "gp_matern", "gp_gaussian", "gp_spherical", "gp_temporal", "gp_nc", "msgp_temporal",
    "msgp_hsgp", "svc", "svc_hsgp", "svc_nc", "temporal_gp", "ms_temporal", "latent", "latent2", "latent_num",
    "latent2_first_zero", "ms_temporal_nc", "ms_temporal_rw2", "ms_temporal_rw2_nc", "ms_temporal_seasonal",
    "ms_temporal_seasonal_nc", "temporal_gp_centered", "temporal_gp_matern32", "temporal_gp_matern52",
    "temporal_gp_gaussian", "temporal_gp_periodic", "temporal_gp_matern32_centered", "temporal_gp_periodic_centered")

KERNEL_COLLAPSED_FIELDS <- c("gp_collapsed", "stgp_collapsed", "stgp_gneiting_collapsed")

DISPATCH_MASK_FIELDS <- c("gp_st4", "gp_stgp", "gp_temporal_st4", "msgp_st4", "tvc_st4", "temporal_gp_st4",
    "gp_tgp", "svc_ms", "svc_ms_nc", "gp_slopes", "gp_slopes_corr", "gp_crossed", "gp_tgp_matern32",
    "gp_tgp_centered", "gp_tgp_gaussian_centered")

DISPATCH_MASK_COLLAPSED_FIELDS <- c("gp_collapsed_st4", "icar_collapsed_st4", "bym2_collapsed_st4")

RSR_FIELDS <- c("icar_rsr", "bym2_rsr", "hsgp_rsr", "gp_rsr")

RTR_FIELDS <- c("rw1_rtr", "temporal_ar1_rtr", "temporal_gp_rtr", "temporal_gp_matern32_rtr", "ms_temporal_rtr")

ALL_FIELDS <- c(FIELDS, MULTI_FIELDS, STRUCTURE_FIELDS, ST_IV_FIELDS, ST_OTHER_FIELDS, ST_IV_AR1_FIELDS,
    ST_GP_FIELDS, KERNEL_FIELDS, DISPATCH_MASK_FIELDS, RSR_FIELDS, RTR_FIELDS)

MODES <- c("handcoded", "arena")

test_that("gradient controls expose the supported scalar implementations", {
    expect_error(tulpaRatio:::.tratio_control(list(gradient_mode = "A_t")), "one of")
    expect_error(tulpaRatio:::cpp_gradient_check("none", mode = "tape"), "gradient mode")
})

AUTODIFF_MODES <- c("arena", "forward")

AUTODIFF_MODES_STR <- c("A_r", "A")

for (field in FIELDS) {
    for (mode in MODES) {
        test_that(sprintf("analytic gradient matches finite differences (%s, %s)", field, mode), {
            r <- tulpaRatio:::cpp_gradient_check(field, family = gradient_fixture_family(field), mode = mode)
            expect_gt(r$n_params, 0)
            dev <- grad_rel_dev(r)
            worst <- which.max(dev)
            expect_lt(max(dev), 1e-04, label = sprintf("%s/%s: worst parameter %d (block %s), analytic %.6f vs finite-diff %.6f",
                field, mode, worst, r$block[worst], r$analytic[worst], r$finite_diff[worst]))
        })
    }
}

for (field in COLLAPSED_TEMPORAL_FIELDS) {
    test_that(sprintf("analytic gradient matches finite differences (%s, handcoded)", field), {
        r <- tulpaRatio:::cpp_gradient_check(field, family = gradient_fixture_family(field), mode = "handcoded")
        expect_gt(r$n_params, 0)
        dev <- grad_rel_dev(r)
        worst <- which.max(dev)
        expect_lt(max(dev), 1e-04, label = sprintf("%s: worst parameter %d (block %s), analytic %.6f vs finite-diff %.6f",
            field, worst, r$block[worst], r$analytic[worst], r$finite_diff[worst]))
    })
}

for (field in DISPATCH_MASK_COLLAPSED_FIELDS) {
    test_that(sprintf("analytic gradient matches finite differences (%s, handcoded)", field), {
        r <- tulpaRatio:::cpp_gradient_check(field, family = gradient_fixture_family(field), mode = "handcoded")
        expect_gt(r$n_params, 0)
        dev <- grad_rel_dev(r)
        worst <- which.max(dev)
        expect_lt(max(dev), 1e-04, label = sprintf("%s: worst parameter %d (block %s), analytic %.6f vs finite-diff %.6f",
            field, worst, r$block[worst], r$analytic[worst], r$finite_diff[worst]))
    })
}

for (field in CAR_PROPER_FIELDS) {
    test_that(sprintf("analytic gradient matches finite differences (%s, handcoded)", field), {
        r <- tulpaRatio:::cpp_gradient_check(field, family = gradient_fixture_family(field), mode = "handcoded")
        expect_gt(r$n_params, 0)
        dev <- grad_rel_dev(r)
        worst <- which.max(dev)
        expect_lt(max(dev), 1e-04, label = sprintf("%s: worst parameter %d (block %s), analytic %.6f vs finite-diff %.6f",
            field, worst, r$block[worst], r$analytic[worst], r$finite_diff[worst]))
    })
}

for (field in c(MULTI_FIELDS, STRUCTURE_FIELDS, ST_IV_FIELDS, ST_OTHER_FIELDS, ST_IV_AR1_FIELDS, ST_GP_FIELDS,
    DISPATCH_MASK_FIELDS, RSR_FIELDS, RTR_FIELDS)) {
    for (mode in MODES) {
        test_that(sprintf("analytic gradient matches finite differences (%s, %s)", field, mode), {
            r <- tulpaRatio:::cpp_gradient_check(field, family = gradient_fixture_family(field), mode = mode)
            expect_gt(r$n_params, 0)
            dev <- grad_rel_dev(r)
            worst <- which.max(dev)
            expect_lt(max(dev), 1e-04, label = sprintf("%s/%s: worst parameter %d (block %s), analytic %.6f vs finite-diff %.6f",
                field, mode, worst, r$block[worst], r$analytic[worst], r$finite_diff[worst]))
        })
    }
}

for (field in ALL_FIELDS) {
    for (mode in AUTODIFF_MODES) {
        test_that(sprintf("autodiff carries every block (%s, %s)", field, mode), {
            r <- tulpaRatio:::cpp_gradient_check(field, family = gradient_fixture_family(field), mode = mode)
            zero <- r$analytic == 0
            expect_equal(sum(zero), 0L, label = sprintf("%s/%s: zero-gradient blocks %s", field, mode,
                paste(unique(r$block[zero]), collapse = ", ")))
            expect_lt(max(grad_rel_dev(r)), 1e-04)
        })
    }
}

for (field in c("rw1", "rw2", "icar_rw1")) {
    for (mode in MODES) {
        for (shared in c(TRUE, FALSE)) {
            test_that(sprintf("analytic gradient matches finite differences (%s, %s, poisson_gamma, shared=%s)",
                field, mode, shared), {
                r <- tulpaRatio:::cpp_gradient_check(field, mode = mode, family = "poisson_gamma", temporal_shared = shared)
                dev <- grad_rel_dev(r)
                worst <- which.max(dev)
                expect_lt(max(dev), 1e-04, label = sprintf("%s/%s/shared=%s: worst parameter %d (block %s), analytic %.6f vs finite-diff %.6f",
                  field, mode, shared, worst, r$block[worst], r$analytic[worst], r$finite_diff[worst]))
            })
        }
    }
}

ZI_COUNT_CASES <- list(list(family = "poisson_gamma", zi = "zi_poisson"), list(family = "poisson_gamma",
    zi = "hurdle_poisson"), list(family = "negbin_gamma", zi = "zi_negbin"), list(family = "negbin_gamma",
    zi = "hurdle_negbin"), list(family = "negbin_negbin", zi = "zi_negbin"), list(family = "negbin_negbin",
    zi = "hurdle_negbin"))

for (case in ZI_COUNT_CASES) {
    for (mode in MODES) {
        test_that(sprintf("analytic gradient matches finite differences (rw1, %s, %s, %s)", mode, case$family,
            case$zi), {
            r <- tulpaRatio:::cpp_gradient_check("rw1", mode = mode, family = case$family, zi = case$zi)
            expect_true("beta_zi" %in% r$block)
            expect_true(all(r$analytic[r$block == "beta_zi"] != 0))
            dev <- grad_rel_dev(r)
            worst <- which.max(dev)
            expect_lt(max(dev), 1e-04, label = sprintf("%s/%s/%s: worst parameter %d (block %s), analytic %.6f vs finite-diff %.6f",
                case$family, case$zi, mode, worst, r$block[worst], r$analytic[worst], r$finite_diff[worst]))
        })
    }
}

PER_OBS_FIELDS <- c("gp_st4", "tvc_st4", "latent", "latent2")

HANDCODED_FAMILIES <- c("binomial", "beta_binomial", "poisson_gamma", "negbin_gamma", "negbin_negbin",
    "gamma_gamma", "lognormal")

PER_OBS_ZI_CASES <- c(ZI_COUNT_CASES, list(list(family = "binomial", zi = "zi_binomial"), list(family = "binomial",
    zi = "zoib")))

check_per_obs <- function(field, family, zi = "none") {
    if (field == "latent2" && family %in% c("binomial", "beta_binomial")) {
        expect_error(tulpaRatio:::cpp_gradient_check(field, family = family, zi = zi),
                     "Bilinear factor dimensions")
        return(invisible(NULL))
    }
    r <- tulpaRatio:::cpp_gradient_check(field, mode = "handcoded", family = family, zi = zi)
    dev <- grad_rel_dev(r)
    worst <- which.max(dev)
    expect_lt(max(dev), 1e-04, label = sprintf("%s/%s/%s: worst parameter %d (block %s), analytic %.6f vs finite-diff %.6f",
        field, family, zi, worst, r$block[worst], r$analytic[worst], r$finite_diff[worst]))
}

for (field in PER_OBS_FIELDS) {
    test_that(sprintf("the %s gradient carries every hand-coded family", field), {
        for (family in HANDCODED_FAMILIES) check_per_obs(field, family)
    })
    test_that(sprintf("the %s gradient carries zero and one inflation", field), {
        for (case in PER_OBS_ZI_CASES) check_per_obs(field, case$family, case$zi)
    })
}

test_that("the hand-coded gradient reports the density's own log posterior", {
    cases <- c(lapply(HANDCODED_FAMILIES, function(f) list(family = f, zi = "none")), PER_OBS_ZI_CASES,
        list(list(family = "binomial", zi = "oi_binomial"), list(family = "binomial", zi = "hurdle_binomial")))
    for (field in c("rw1", "icar")) {
        for (case in cases) {
            r <- tulpaRatio:::cpp_gradient_check(field, mode = "handcoded", family = case$family, zi = case$zi)
            info <- sprintf("%s / %s / %s", field, case$family, case$zi)
            expect_equal(r$log_post_mode, r$log_post, tolerance = 1e-10, info = info)
            expect_lt(max(grad_rel_dev(r)), 1e-04, label = info)
        }
    }
})

test_that("an inflated model never reaches a gradient written without inflation", {
    cases <- list(c("binomial", "zi_binomial"), c("binomial", "oi_binomial"), c("binomial", "zoib"),
        c("binomial", "hurdle_binomial"), c("poisson_gamma", "zi_poisson"), c("negbin_negbin", "hurdle_negbin"))
    for (field in c("gp", "hsgp", "svc", "tvc", "temporal_gp", "ms_temporal", "latent", "st2")) {
        for (cs in cases) {
            r <- tulpaRatio:::cpp_gradient_check(field, mode = "handcoded", family = cs[1], zi = cs[2])
            expect_lt(max(grad_rel_dev(r)), 1e-04, label = sprintf("%s / %s / %s", field, cs[1], cs[2]))
        }
    }
})

test_that("the collapsed GP interaction carries the log-concave families", {
    for (field in c("stgp_collapsed", "stgp_gneiting_collapsed")) {
        for (family in setdiff(HANDCODED_FAMILIES, "beta_binomial")) {
            check_per_obs(field, family)
        }
    }
})

test_that("a count-response ZI or hurdle term reaches both instantiations", {
    for (case in ZI_COUNT_CASES) {
        r <- tulpaRatio:::cpp_gradient_check("rw1", mode = "arena", family = case$family, zi = case$zi)
        info <- sprintf("%s / %s", case$family, case$zi)
        expect_equal(r$log_post, r$log_post_mode, tolerance = 1e-08, info = info)
    }
})

test_that("each count ZI structure is a density of its own", {
    for (family in c("poisson_gamma", "negbin_gamma", "negbin_negbin")) {
        poisson <- identical(family, "poisson_gamma")
        lp <- function(zi) tulpaRatio:::cpp_logpost_at("rw1", family = family, zi = zi)
        none <- lp("none")
        inflated <- lp(if (poisson)
            "zi_poisson"
        else "zi_negbin")
        hurdle <- lp(if (poisson)
            "hurdle_poisson"
        else "hurdle_negbin")
        expect_gt(abs(inflated - none), 1, label = family)
        expect_gt(abs(hurdle - none), 1, label = family)
        expect_gt(abs(inflated - hurdle), 1, label = family)
    }
})

test_that("an unshared temporal effect changes the density it is meant to change", {
    for (field in c("rw1", "icar_rw1")) {
        lp_shared <- tulpaRatio:::cpp_logpost_at(field, family = "poisson_gamma", temporal_shared = TRUE)
        lp_unshared <- tulpaRatio:::cpp_logpost_at(field, family = "poisson_gamma", temporal_shared = FALSE)
        expect_gt(abs(lp_shared - lp_unshared), 1, label = sprintf("field = %s", field))
    }
    expect_equal(tulpaRatio:::cpp_logpost_at("rw1", family = "binomial", temporal_shared = TRUE), tulpaRatio:::cpp_logpost_at("rw1",
        family = "binomial", temporal_shared = FALSE))
})

test_that("the density does not depend on how many threads sum it", {
    for (field in ALL_FIELDS) {
        one <- tulpaRatio:::cpp_gradient_check(field, family = gradient_fixture_family(field), n_threads = 1L)
        many <- tulpaRatio:::cpp_gradient_check(field, family = gradient_fixture_family(field), n_threads = 4L)
        expect_equal(one$log_post, many$log_post, tolerance = 1e-08, info = sprintf("field = %s", field))
    }
})

test_that("the value each mode reports is the density its gradient describes", {
    for (mode in c("arena", "handcoded")) {
        for (field in ALL_FIELDS) {
            r <- tulpaRatio:::cpp_gradient_check(field, family = gradient_fixture_family(field), mode = mode)
            expect_equal(r$log_post, r$log_post_mode, tolerance = 1e-08, info = sprintf("%s/%s", field,
                mode))
        }
    }
})

for (field in KERNEL_FIELDS) {
    for (mode in MODES) {
        test_that(sprintf("analytic gradient matches finite differences (%s, %s)", field, mode), {
            r <- tulpaRatio:::cpp_gradient_check(field, family = gradient_fixture_family(field), mode = mode)
            expect_gt(r$n_params, 0)
            dev <- grad_rel_dev(r)
            worst <- which.max(dev)
            expect_lt(max(dev), 1e-04, label = sprintf("%s/%s: worst parameter %d (block %s), analytic %.6f vs finite-diff %.6f",
                field, mode, worst, r$block[worst], r$analytic[worst], r$finite_diff[worst]))
        })
    }
}

for (field in KERNEL_COLLAPSED_FIELDS) {
    test_that(sprintf("analytic gradient matches finite differences (%s, handcoded)", field), {
        r <- tulpaRatio:::cpp_gradient_check(field, family = gradient_fixture_family(field), mode = "handcoded")
        expect_gt(r$n_params, 0)
        dev <- grad_rel_dev(r)
        worst <- which.max(dev)
        expect_lt(max(dev), 1e-04, label = sprintf("%s: worst parameter %d (block %s), analytic %.6f vs finite-diff %.6f",
            field, worst, r$block[worst], r$analytic[worst], r$finite_diff[worst]))
    })
}

for (field in c("temporal_ar1", "tvc_ar1", "ms_temporal", "st4_ar1", "st4_ar1_nc", "st2_ar1")) {
    test_that(sprintf("the densities agree where the 1 - rho^2 floor binds (%s)", field), {
        r <- tulpaRatio:::cpp_gradient_check(field, family = gradient_fixture_family(field), mode = "handcoded", near_unit_rho = TRUE)
        expect_true(is.finite(r$log_post), label = sprintf("%s: log_post", field))
        expect_true(all(is.finite(r$analytic)), label = sprintf("%s: analytic gradient", field))
        a <- tulpaRatio:::cpp_gradient_check(field, family = gradient_fixture_family(field), mode = "arena", near_unit_rho = TRUE)
        expect_equal(r$log_post, a$log_post_mode, tolerance = 1e-08, info = sprintf("field = %s", field))
    })
}

test_that("a field the harness builds no structure for is an error, not a pass", {
    expect_error(tulpaRatio:::cpp_gradient_check("no_such_field"), "unknown field")
    expect_error(tulpaRatio:::cpp_logpost_at("no_such_field"), "unknown field")
})

test_that("the engine differentiates proper CAR fields", {
    skip_on_cran()
    set.seed(1)
    N <- 200L
    S <- 16L
    side <- ceiling(sqrt(S))
    g <- expand.grid(lon = seq_len(side), lat = seq_len(side))[seq_len(S), ]
    A <- matrix(0, S, S)
    for (i in seq_len(S)) for (j in seq_len(S)) {
        if (i != j && sqrt((g$lon[i] - g$lon[j])^2 + (g$lat[i] - g$lat[j])^2) <= 1.5)
            A[i, j] <- 1
    }
    dimnames(A) <- list(as.character(seq_len(S)), as.character(seq_len(S)))
    x <- rnorm(N)
    trials <- sample(10:40, N, replace = TRUE)
    df <- data.frame(y = rbinom(N, trials, plogis(0.5 + 0.3 * x)), trials = trials, x = x, site = factor(rep(seq_len(S),
        length.out = N)))
    fit_with <- function(mode) {
        tratio(y | trials ~ x, data = df, family = ratiod_binomial(), spatial = spatial_car(A, spatial_level = "group",
            group_var = "site", proper = TRUE), control = list(iter = 30, warmup = 15, chains = 1, verbose = FALSE,
            gradient_mode = mode))
    }
    for (mode in AUTODIFF_MODES_STR) {
        {
            fit <- suppressWarnings(fit_with(mode))
            expect_identical(fit$backend, "hmc")
            expect_true(all(is.finite(as.matrix(fit$draws))))
        }
    }
    expect_no_warning(fit_h <- suppressWarnings(fit_with("H"), classes = "tulpaRatio_divergence"))
    expect_identical(fit_h$backend, "hmc")
})

test_that("collapsed ICAR requests use differentiable joint engine fields", {
    skip_on_cran()
    set.seed(1)
    N <- 200L
    S <- 16L
    side <- ceiling(sqrt(S))
    g <- expand.grid(lon = seq_len(side), lat = seq_len(side))[seq_len(S), ]
    A <- matrix(0, S, S)
    for (i in seq_len(S)) for (j in seq_len(S)) {
        if (i != j && sqrt((g$lon[i] - g$lon[j])^2 + (g$lat[i] - g$lat[j])^2) <= 1.5)
            A[i, j] <- 1
    }
    dimnames(A) <- list(as.character(seq_len(S)), as.character(seq_len(S)))
    x <- rnorm(N)
    trials <- sample(10:40, N, replace = TRUE)
    df <- data.frame(y = rbinom(N, trials, plogis(0.5 + 0.3 * x)), trials = trials, x = x, site = factor(rep(seq_len(S),
        length.out = N)), time = factor(rep(seq_len(8), length.out = N)))
    fit_with <- function(mode) {
        tratio(y | trials ~ x, data = df, family = ratiod_binomial(), spatial = spatial_car(A, spatial_level = "group",
            group_var = "site", parameterization = "collapsed"), temporal = temporal_rw1("time"), control = list(iter = 30,
            warmup = 15, chains = 1, verbose = FALSE, gradient_mode = mode))
    }
    for (mode in AUTODIFF_MODES_STR) {
        {
            fit <- suppressWarnings(fit_with(mode))
            expect_identical(fit$backend, "hmc")
            expect_true(all(is.finite(as.matrix(fit$draws))))
        }
    }
    expect_no_warning(fit_h <- suppressWarnings(fit_with("H"), classes = "tulpaRatio_divergence"))
    expect_identical(fit_h$backend, "hmc")
})

test_that("centring makes the spatial field's mean invisible to the likelihood", {
    for (field in c("icar", "bym2", "icar_rw1", MULTI_FIELDS)) {
        raw <- tulpaRatio:::cpp_gradient_check(field, family = gradient_fixture_family(field), mode = "handcoded", precenter = FALSE)
        cen <- tulpaRatio:::cpp_gradient_check(field, family = gradient_fixture_family(field), mode = "handcoded", precenter = TRUE)
        beta <- which(raw$block == "beta_num")
        expect_equal(raw$analytic[beta], cen$analytic[beta], tolerance = 1e-08, info = sprintf("field = %s",
            field))
    }
})

test_that("the model sampler uses its handcoded gradient for non-centred GP fields", {
    skip_on_cran()
    set.seed(11)
    n_loc <- 30L
    reps <- 4L
    co <- cbind(runif(n_loc), runif(n_loc))
    S <- 0.8 * exp(-as.matrix(dist(co))/0.25)
    w <- as.vector(t(chol(S + diag(1e-08, n_loc))) %*% rnorm(n_loc))
    loc <- rep(seq_len(n_loc), each = reps)
    x <- rnorm(n_loc * reps)
    n <- rep(25L, n_loc * reps)
    df <- data.frame(y = rbinom(n_loc * reps, n, plogis(-0.3 + 0.6 * x + w[loc])), n = n, x = x, lon = co[loc,
        1], lat = co[loc, 2])
    for (par in c("centered", "noncentered")) {
        expect_no_warning(suppressWarnings(tratio(y | n ~ x, data = df, family = ratiod_binomial(), mode = "hmc",
            spatial = spatial_gp(~lon + lat, parameterization = par, nn = 5), control = list(iter = 60,
                warmup = 30, chains = 1, verbose = FALSE, seed = 7)), classes = "tulpaRatio_divergence"),
            message = "finite difference")
    }
})

test_that("the multi-scale arms' non-centred coordinate carries its density", {
    for (field in c("ms_temporal_nc", "ms_temporal_rw2_nc", "ms_temporal_seasonal_nc")) {
        for (sigma2 in c(0.01, 0.25, 4)) {
            for (n in c(7L, 12L, 25L)) {
                spread <- tulpaRatio:::cpp_ms_temporal_nc_coordinate_spread(field, n, sigma2, n_draws = 8L)
                expect_lt(spread, 1e-09)
            }
        }
    }
})

test_that("the non-centred coordinate carries its density over groups", {
    for (field in c("ms_temporal_nc", "ms_temporal_rw2_nc", "ms_temporal_seasonal_nc")) {
        for (sigma2 in c(0.01, 0.25, 4)) {
            for (n in c(7L, 12L)) {
                for (n_groups in c(2L, 3L, 5L)) {
                  spread <- tulpaRatio:::cpp_ms_temporal_nc_coordinate_spread(field, n, sigma2, n_draws = 8L,
                    n_groups = n_groups)
                  expect_lt(spread, 1e-09)
                }
            }
        }
    }
})

test_that("every gradient fixture builds the structure its name claims", {
    fields <- tulpaRatio:::cpp_gradient_fields()
    bare <- tulpaRatio:::cpp_gradient_check("none", n_obs = 200L, n_units = 8L, n_times = 10L)$n_params
    structureless <- character()
    for (field in setdiff(fields, "none")) {
        n <- tulpaRatio:::cpp_gradient_check(field, family = gradient_fixture_family(field), n_obs = 200L, n_units = 8L, n_times = 10L)$n_params
        if (n <= bare)
            structureless <- c(structureless, field)
    }
    expect_equal(structureless, character())
})

test_that("every binomial inflation type reports the density its gradient describes", {
    for (zi in c("zi_binomial", "hurdle_binomial", "oi_binomial", "zoib")) {
        for (field in c("none", "re", "icar")) {
            for (mode in c("handcoded", "arena")) {
                r <- tulpaRatio:::cpp_gradient_check(field, mode = mode, family = "binomial", zi = zi)
                info <- sprintf("%s/%s/%s", zi, field, mode)
                expect_equal(r$log_post, r$log_post_mode, tolerance = 1e-08, info = info)
                expect_lt(max(abs(r$analytic - r$finite_diff)/pmax(1, abs(r$finite_diff))), 1e-04, label = info)
            }
        }
    }
})

test_that("collapsed areal targets have finite engine gradients", {
      for(field in COLLAPSED_FIELDS) {
        r<-tulpaRatio:::cpp_gradient_check(field, family = gradient_fixture_family(field),n_obs=40L,n_units=8L,mode="H",seed=17L)
        expect_true(is.finite(r$log_post),info=field)
        expect_lt(max(grad_rel_dev(r)),1e-4,label=field)
      }
    })

test_that("proper CAR fields have automatic-differentiation engine gradients", {
      for(field in CAR_PROPER_FIELDS) for(mode in c("A_r","A")) {
        r<-tulpaRatio:::cpp_gradient_check(field, family = gradient_fixture_family(field),n_obs=40L,n_units=8L,mode=mode,seed=17L)
        expect_lt(max(grad_rel_dev(r)),1e-4,label=paste(field,mode))
      }
    })
