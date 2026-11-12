test_that("HMC backend runs for binomial model", {
    skip_on_cran()
    set.seed(123)
    n <- 200
    x <- rnorm(n)
    p <- plogis(-0.5 + 0.8 * x)
    trials <- rpois(n, 30) + 15
    successes <- rbinom(n, trials, p)
    df <- data.frame(successes = successes, trials = trials, x = x)
    X <- cbind(1, x)
    result <- tulpaRatio:::cpp_tulpaRatio_sample(y_num = as.integer(successes), y_denom = as.integer(trials),
        y_num_cont = rep(0, n), y_denom_cont = rep(0, n), X_list = list(X), cfg_list = list(family = "binomial"),
        init = rep(0, ncol(X)), n_iter = 200L, n_warmup = 100L, L = 10L, n_chains = 1L, seed = 123L,
        n_threads = 1L, verbose = FALSE, temporal_params = fixture_temporal_params(n), re_params = fixture_re_params(n),
        sigma_beta = fixture_prior_params()$sigma_beta, sigma_prior_scale = fixture_prior_params()$sigma_re_scale,
        phi_prior_shape = fixture_prior_params()$phi_shape, phi_prior_rate = fixture_prior_params()$phi_rate)
    expect_true(is.matrix(result$samples))
    expect_equal(nrow(result$samples), 100)
    expect_equal(ncol(result$samples), 3 - 1L)
    beta0_mean <- mean(result$samples[, 1])
    expect_true(abs(beta0_mean - (-0.5)) < 1)
    beta1_mean <- mean(result$samples[, 2])
    expect_true(abs(beta1_mean - 0.8) < 1)
})

test_that("HMC backend runs for negbin model", {
    skip_on_cran()
    set.seed(456)
    n <- 100
    x <- rnorm(n)
    mu_num <- exp(2 + 0.5 * x)
    mu_denom <- exp(3 - 0.3 * x)
    phi <- 5
    y_num <- rnbinom(n, mu = mu_num, size = phi)
    y_denom <- rnbinom(n, mu = mu_denom, size = phi)
    X <- cbind(1, x)
    result <- tulpaRatio:::cpp_tulpaRatio_sample(y_num = as.integer(y_num), y_denom = as.integer(y_denom),
        y_num_cont = rep(0, n), y_denom_cont = rep(0, n), X_list = list(X, X), cfg_list = list(family = "negbin_negbin"),
        init = rep(0, 6), n_iter = 200L, n_warmup = 100L, L = 10L, n_chains = 1L, seed = 456L, n_threads = 1L,
        verbose = FALSE, temporal_params = fixture_temporal_params(n), re_params = fixture_re_params(n),
        sigma_beta = fixture_prior_params()$sigma_beta, sigma_prior_scale = fixture_prior_params()$sigma_re_scale,
        phi_prior_shape = fixture_prior_params()$phi_shape, phi_prior_rate = fixture_prior_params()$phi_rate)
    expect_true(is.matrix(result$samples))
    expect_equal(ncol(result$samples), 6)
})

test_that("HMC backend runs with random effects", {
    skip_on_cran()
    set.seed(789)
    n_groups <- 10
    n_per_group <- 30
    n <- n_groups * n_per_group
    group <- rep(1:n_groups, each = n_per_group)
    re_true <- rnorm(n_groups, 0, 0.5)
    x <- rnorm(n)
    p <- plogis(-0.5 + 0.8 * x + re_true[group])
    trials <- rpois(n, 30) + 15
    successes <- rbinom(n, trials, p)
    X <- cbind(1, x)
    result <- tulpaRatio:::cpp_tulpaRatio_sample(y_num = as.integer(successes), y_denom = as.integer(trials),
        y_num_cont = rep(0, n), y_denom_cont = rep(0, n), X_list = list(X), cfg_list = list(family = "binomial"),
        init = rep(0, ncol(X) + 1L + n_groups), n_iter = 200L, n_warmup = 100L, L = 10L, n_chains = 1L,
        seed = 789L, n_threads = 1L, verbose = FALSE, temporal_params = fixture_temporal_params(n), re_params = fixture_re_params(n,
            group = group, n_groups = n_groups), sigma_beta = fixture_prior_params()$sigma_beta, sigma_prior_scale = fixture_prior_params()$sigma_re_scale,
        phi_prior_shape = fixture_prior_params()$phi_shape, phi_prior_rate = fixture_prior_params()$phi_rate)
    expect_equal(ncol(result$samples), 14 - 1L)
    log_sigma_idx <- 4 - 1L
    sigma_re_est <- mean(exp(result$samples[, log_sigma_idx]))
    expect_true(sigma_re_est > 0)
    expect_true(sigma_re_est < 5)
})

test_that("HMC backend runs with ICAR spatial effects", {
    skip_on_cran()
    set.seed(111)
    n_spatial <- 16
    n_per_spatial <- 10
    n <- n_spatial * n_per_spatial
    adj_row_ptr <- integer(n_spatial + 1)
    adj_col_idx <- integer(0)
    n_neighbors <- integer(n_spatial)
    for (i in 1:n_spatial) {
        neighbors <- integer(0)
        row <- (i - 1)%/%4 + 1
        col <- (i - 1)%%4 + 1
        if (col > 1)
            neighbors <- c(neighbors, (i - 1) - 1L)
        if (col < 4)
            neighbors <- c(neighbors, (i + 1) - 1L)
        if (row > 1)
            neighbors <- c(neighbors, (i - 4) - 1L)
        if (row < 4)
            neighbors <- c(neighbors, (i + 4) - 1L)
        n_neighbors[i] <- length(neighbors)
        adj_col_idx <- c(adj_col_idx, neighbors)
        adj_row_ptr[i + 1] <- adj_row_ptr[i] + length(neighbors)
    }
    spatial_group <- rep(1:n_spatial, each = n_per_spatial)
    x <- rnorm(n)
    spatial_effect <- rnorm(n_spatial, 0, 0.3)
    p <- plogis(-0.5 + 0.5 * x + spatial_effect[spatial_group])
    trials <- rpois(n, 15) + 5
    successes <- rbinom(n, trials, p)
    X <- cbind(1, x)
    spatial_params <- list(type = "icar", group = as.integer(spatial_group), n_units = as.integer(n_spatial),
        adj_row_ptr = as.integer(adj_row_ptr), adj_col_idx = as.integer(adj_col_idx), n_neighbors = as.integer(n_neighbors),
        bym2_scale = 1)
    result <- tulpaRatio:::cpp_tulpaRatio_sample(y_num = as.integer(successes), y_denom = as.integer(trials),
        y_num_cont = rep(0, n), y_denom_cont = rep(0, n), X_list = list(X), cfg_list = list(family = "binomial"),
        init = rep(0, ncol(X) + 1L + n_spatial), n_iter = 200L, n_warmup = 100L, L = 10L, n_chains = 1L,
        seed = 111L, n_threads = 1L, verbose = FALSE, temporal_params = fixture_temporal_params(n), re_params = fixture_re_params(n),
        spatial_params = modifyList(list(parameterization = "standard", shared = TRUE, rsr = list(active = FALSE),
            sigma_prior_U = 3, sigma_prior_alpha = 0.01, Q_inv = NULL, L_Q = NULL, node_prec = NULL),
            spatial_params), sigma_beta = fixture_prior_params()$sigma_beta, sigma_prior_scale = fixture_prior_params()$sigma_re_scale,
        phi_prior_shape = fixture_prior_params()$phi_shape, phi_prior_rate = fixture_prior_params()$phi_rate)
    expect_equal(ncol(result$samples), 20 - 1L)
    expect_equal(nrow(result$samples), 100)
    log_tau_idx <- 4 - 1L
    tau_est <- mean(exp(result$samples[, log_tau_idx]))
    expect_true(tau_est > 0)
})

test_that("HMC backend runs multiple chains in parallel", {
    skip_on_cran()
    set.seed(222)
    n <- 50
    x <- rnorm(n)
    p <- plogis(-0.3 + 0.6 * x)
    trials <- rpois(n, 15) + 5
    successes <- rbinom(n, trials, p)
    X <- cbind(1, x)
    result <- tulpaRatio:::cpp_tulpaRatio_sample(y_num = as.integer(successes), y_denom = as.integer(trials),
        y_num_cont = rep(0, n), y_denom_cont = rep(0, n), X_list = list(X), cfg_list = list(family = "binomial"),
        init = rep(0, ncol(X)), n_iter = 200L, n_warmup = 100L, L = 10L, n_chains = 2L, seed = 222L,
        n_threads = 1L, verbose = FALSE, temporal_params = fixture_temporal_params(n), re_params = fixture_re_params(n),
        sigma_beta = fixture_prior_params()$sigma_beta, sigma_prior_scale = fixture_prior_params()$sigma_re_scale,
        phi_prior_shape = fixture_prior_params()$phi_shape, phi_prior_rate = fixture_prior_params()$phi_rate)
    expect_true(is.list(result$samples))
    expect_equal(length(result$samples), 2)
    expect_equal(nrow(result$samples[[1]]), 100)
    expect_equal(nrow(result$samples[[2]]), 100)
})

test_that("cpp_get_max_threads returns positive integer", {
    n_threads <- tulpaRatio:::cpp_get_max_threads()
    expect_true(is.numeric(n_threads))
    expect_true(n_threads >= 1)
})

test_that("a non-finite energy counts as a divergence, not as acceptance", {
    skip_on_cran()
    G <- 12
    set.seed(5)
    n <- 48
    site <- rep(letters[1:G], length.out = n)
    u <- rnorm(G, sd = 0.5)[match(site, letters[1:G])]
    depth <- rnorm(n)
    df <- data.frame(count = rpois(n, exp(3 + 0.3 * depth + u)), effort = rgamma(n, shape = 10, rate = exp(-u)),
        depth = depth, site = site)
    fit <- tratio(count | effort ~ depth + (1 | site), data = df, family = ratiod_poisson_gamma(), mode = "hmc",
        control = list(iter = 1000, chains = 1, seed = 1))
    expect_equal(sum(fit$diagnostics$divergent), 0)
    expect_lt(fit$epsilon, 1)
})

test_that("negbin_negbin with RW2 temporal runs correctly (row 42)", {
    skip_on_cran()
    set.seed(4201)
    n_times <- 8
    n_per_time <- 15
    n <- n_times * n_per_time
    temporal_effect <- numeric(n_times)
    temporal_effect[1:2] <- rnorm(2, 0, 0.1)
    for (t in 3:n_times) {
        temporal_effect[t] <- 2 * temporal_effect[t - 1] - temporal_effect[t - 2] + rnorm(1, 0, 0.1)
    }
    temporal_effect <- temporal_effect - mean(temporal_effect)
    time_idx <- rep(1:n_times, each = n_per_time)
    x <- rnorm(n)
    mu_num <- exp(2 + 0.3 * x + temporal_effect[time_idx])
    mu_denom <- exp(3 + 0.2 * x + 0.5 * temporal_effect[time_idx])
    y_num <- rnbinom(n, size = 5, mu = mu_num)
    y_denom <- rnbinom(n, size = 10, mu = mu_denom)
    X <- cbind(1, x)
    n_params <- 2 + 2 + 2 + 1 + n_times
    result <- tulpaRatio:::cpp_tulpaRatio_sample(y_num = as.integer(y_num), y_denom = as.integer(y_denom),
        y_num_cont = rep(0, n), y_denom_cont = rep(0, n), X_list = list(X, X), cfg_list = list(family = "negbin_negbin"),
        init = rep(0, n_params), n_iter = 200L, n_warmup = 100L, L = 10L, n_chains = 1L, seed = 4201L,
        n_threads = 1L, verbose = FALSE, temporal_params = fixture_temporal_params(n, time_idx, type = "rw2",
            n_times = n_times), re_params = fixture_re_params(n), sigma_beta = fixture_prior_params()$sigma_beta,
        sigma_prior_scale = fixture_prior_params()$sigma_re_scale, phi_prior_shape = fixture_prior_params()$phi_shape,
        phi_prior_rate = fixture_prior_params()$phi_rate)
    expect_equal(ncol(result$samples), n_params)
    expect_equal(nrow(result$samples), 100)
    temporal_cols <- 5L + seq_len(n_times)
    temporal_means <- colMeans(result$samples[, temporal_cols])
    expect_true(all(is.finite(temporal_means)))
    expect_true(mean(result$accept_prob) > 0.1)
})

test_that("binomial with RW2 temporal runs correctly (row 43)", {
    skip_on_cran()
    set.seed(4301)
    n_times <- 10
    n_per_time <- 12
    n <- n_times * n_per_time
    temporal_effect <- numeric(n_times)
    temporal_effect[1:2] <- rnorm(2, 0, 0.15)
    for (t in 3:n_times) {
        temporal_effect[t] <- 2 * temporal_effect[t - 1] - temporal_effect[t - 2] + rnorm(1, 0, 0.1)
    }
    temporal_effect <- temporal_effect - mean(temporal_effect)
    time_idx <- rep(1:n_times, each = n_per_time)
    x <- rnorm(n)
    p <- plogis(-0.3 + 0.4 * x + temporal_effect[time_idx])
    trials <- rpois(n, 20) + 10
    successes <- rbinom(n, trials, p)
    X <- cbind(1, x)
    n_params <- 2 + 1 + 1 + n_times
    result <- tulpaRatio:::cpp_tulpaRatio_sample(y_num = as.integer(successes), y_denom = as.integer(trials),
        y_num_cont = rep(0, n), y_denom_cont = rep(0, n), X_list = list(X), cfg_list = list(family = "binomial"),
        init = rep(0, ncol(X) + 1L + n_times * 1L), n_iter = 200L, n_warmup = 100L, L = 10L, n_chains = 1L,
        seed = 4301L, n_threads = 1L, verbose = FALSE, temporal_params = fixture_temporal_params(n, time_idx,
            type = "rw2", n_times = n_times), re_params = fixture_re_params(n), sigma_beta = fixture_prior_params()$sigma_beta,
        sigma_prior_scale = fixture_prior_params()$sigma_re_scale, phi_prior_shape = fixture_prior_params()$phi_shape,
        phi_prior_rate = fixture_prior_params()$phi_rate)
    expect_equal(ncol(result$samples), n_params - 1L)
    expect_equal(nrow(result$samples), 100)
    temporal_cols <- 3L + seq_len(n_times)
    temporal_means <- colMeans(result$samples[, temporal_cols])
    expect_true(all(is.finite(temporal_means)))
    expect_true(mean(result$accept_prob) > 0.1)
})

test_that("negbin_negbin with MSGP runs correctly (row 44)", {
    skip_on_cran()
    skip_unless_slow()
    df <- generate_msgp_data_family(n = 40, seed = 4401, family = "negbin_negbin")
    ms <- spatial_multiscale(~x + y, m = 6)
    fit <- suppressWarnings(tratio(y_num | y_denom ~ 1, data = as.data.frame(df), spatial = ms, family = ratiod_negbin_negbin(),
        mode = "hmc", control = list(chains = 1, iter = 150, warmup = 75, verbose = FALSE)))
    expect_s3_class(fit, "ratiod_fit")
    expect_false(is.null(fit$diagnostics))
})

test_that("binomial with MSGP runs correctly (row 45)", {
    skip_on_cran()
    skip_unless_slow()
    df <- generate_msgp_data_family(n = 40, seed = 4501, family = "binomial")
    ms <- spatial_multiscale(~x + y, m = 6)
    fit <- suppressWarnings(tratio(y_num | y_denom ~ 1, data = as.data.frame(df), spatial = ms, family = ratiod_binomial(),
        mode = "hmc", control = list(chains = 1, iter = 150, warmup = 75, verbose = FALSE)))
    expect_s3_class(fit, "ratiod_fit")
})

test_that("negbin_negbin with GP + RW1 runs correctly (row 46)", {
    skip_on_cran()
    skip_unless_slow()
    df <- generate_gp_temporal_data(n_spatial = 25, n_times = 5, seed = 4601, family = "negbin_negbin")
    fit <- suppressWarnings(tratio(y_num | y_denom ~ 1, data = as.data.frame(df), spatial = spatial_gp(~x +
        y, nn = 8), temporal = temporal_rw1("time"), family = ratiod_negbin_negbin(), mode = "hmc", control = list(chains = 1,
        iter = 150, warmup = 75, verbose = FALSE)))
    expect_s3_class(fit, "ratiod_fit")
})

test_that("binomial with GP + RW1 runs correctly (row 47)", {
    skip_on_cran()
    skip_unless_slow()
    df <- generate_gp_temporal_data(n_spatial = 25, n_times = 5, seed = 4701, family = "binomial")
    fit <- suppressWarnings(tratio(y_num | y_denom ~ 1, data = as.data.frame(df), spatial = spatial_gp(~x +
        y, nn = 8), temporal = temporal_rw1("time"), family = ratiod_binomial(), mode = "hmc", control = list(chains = 1,
        iter = 150, warmup = 75, verbose = FALSE)))
    expect_s3_class(fit, "ratiod_fit")
})

test_that("poisson_gamma with GP + RW2 runs correctly (row 48)", {
    skip_on_cran()
    skip_unless_slow()
    df <- generate_gp_temporal_data(n_spatial = 25, n_times = 6, seed = 4801, family = "poisson_gamma")
    fit <- suppressWarnings(tratio(y_num | y_denom_cont ~ 1, data = as.data.frame(df), spatial = spatial_gp(~x +
        y, nn = 8), temporal = temporal_rw2("time"), family = ratiod_poisson_gamma(), mode = "hmc", control = list(chains = 1,
        iter = 150, warmup = 75, verbose = FALSE)))
    expect_s3_class(fit, "ratiod_fit")
})

test_that("poisson_gamma with GP + AR1 runs correctly (row 49)", {
    skip_on_cran()
    skip_unless_slow()
    df <- generate_gp_temporal_data(n_spatial = 25, n_times = 6, seed = 4901, family = "poisson_gamma")
    fit <- suppressWarnings(tratio(y_num | y_denom_cont ~ 1, data = as.data.frame(df), spatial = spatial_gp(~x +
        y, nn = 8), temporal = temporal_ar1("time"), family = ratiod_poisson_gamma(), mode = "hmc", control = list(chains = 1,
        iter = 150, warmup = 75, verbose = FALSE)))
    expect_s3_class(fit, "ratiod_fit")
})

test_that("negbin_negbin with GP + AR1 runs correctly (row 50)", {
    skip_on_cran()
    skip_unless_slow()
    df <- generate_gp_temporal_data(n_spatial = 25, n_times = 6, seed = 5001, family = "negbin_negbin")
    fit <- suppressWarnings(tratio(y_num | y_denom ~ 1, data = as.data.frame(df), spatial = spatial_gp(~x +
        y, nn = 8), temporal = temporal_ar1("time"), family = ratiod_negbin_negbin(), mode = "hmc", control = list(chains = 1,
        iter = 150, warmup = 75, verbose = FALSE)))
    expect_s3_class(fit, "ratiod_fit")
})

test_that("binomial with GP + AR1 runs correctly (row 51)", {
    skip_on_cran()
    skip_unless_slow()
    df <- generate_gp_temporal_data(n_spatial = 25, n_times = 6, seed = 5101, family = "binomial")
    fit <- suppressWarnings(tratio(y_num | y_denom ~ 1, data = as.data.frame(df), spatial = spatial_gp(~x +
        y, nn = 8), temporal = temporal_ar1("time"), family = ratiod_binomial(), mode = "hmc", control = list(chains = 1,
        iter = 150, warmup = 75, verbose = FALSE)))
    expect_s3_class(fit, "ratiod_fit")
})

test_that("RW2 temporal basic functionality works", {
    temp <- temporal_rw2("time_var")
    expect_s3_class(temp, "ratiod_temporal")
    expect_equal(temp$type, "rw2")
})

test_that("poisson_gamma with BYM2 + ZI runs correctly (row 52)", {
    skip_on_cran()
    skip_unless_slow()
    df <- generate_spatial_zi_data(n_sites = 25, seed = 5201, spatial_type = "bym2", family = "poisson_gamma")
    adj_mat <- matrix(0, 25, 25)
    for (i in 1:25) {
        for (j in df$adj_list[[i]]) {
            adj_mat[i, j] <- 1
        }
    }
    fit <- suppressWarnings(tratio(y_num | y_denom_cont ~ 1 + (1 | site), data = as.data.frame(df), spatial = spatial_bym2(adj_mat,
        spatial_level = "group", group_var = "site"), family = ratiod_zi_poisson(denom_family = "gamma",
        zi_formula = ~1), mode = "hmc", control = list(chains = 1, iter = 150, warmup = 75, verbose = FALSE)))
    expect_s3_class(fit, "ratiod_fit")
})

test_that("poisson_gamma with GP + ZI runs correctly (row 53)", {
    skip_on_cran()
    skip_unless_slow()
    df <- generate_spatial_zi_data(n_sites = 30, seed = 5301, family = "poisson_gamma")
    fit <- suppressWarnings(tratio(y_num | y_denom_cont ~ 1, data = as.data.frame(df), spatial = spatial_gp(~x +
        y, nn = 8), family = ratiod_zi_poisson(denom_family = "gamma", zi_formula = ~1), mode = "hmc",
        control = list(chains = 1, iter = 150, warmup = 75, verbose = FALSE)))
    expect_s3_class(fit, "ratiod_fit")
})

test_that("negbin_negbin with ICAR + ZI runs correctly (row 54)", {
    skip_on_cran()
    skip_unless_slow()
    df <- generate_spatial_zi_data(n_sites = 25, seed = 5401, family = "negbin_negbin")
    adj_mat <- matrix(0, 25, 25)
    for (i in 1:25) {
        for (j in df$adj_list[[i]]) {
            adj_mat[i, j] <- 1
        }
    }
    fit <- suppressWarnings(tratio(y_num | y_denom ~ 1 + (1 | site), data = as.data.frame(df), spatial = spatial_car(adj_mat,
        spatial_level = "group", group_var = "site"), family = ratiod_zi_negbin(denom_family = "negbin",
        zi_formula = ~1), mode = "hmc", control = list(chains = 1, iter = 150, warmup = 75, verbose = FALSE)))
    expect_s3_class(fit, "ratiod_fit")
})

test_that("poisson_gamma with crossed RE + ICAR runs correctly (row 57)", {
    skip_on_cran()
    skip_unless_slow()
    set.seed(5701)
    n_sites <- 16
    n_species <- 5
    n <- n_sites * n_species
    site <- rep(1:n_sites, each = n_species)
    species <- rep(1:n_species, n_sites)
    side <- 4
    adj_mat <- matrix(0, n_sites, n_sites)
    for (i in 1:n_sites) {
        row_i <- (i - 1)%/%side + 1
        col_i <- (i - 1)%%side + 1
        for (j in 1:n_sites) {
            row_j <- (j - 1)%/%side + 1
            col_j <- (j - 1)%%side + 1
            if (i != j && abs(row_i - row_j) + abs(col_i - col_j) == 1) {
                adj_mat[i, j] <- 1
            }
        }
    }
    w_site <- rnorm(n_sites, 0, 0.5)
    re_species <- rnorm(n_species, 0, 0.3)
    mu <- exp(2 + w_site[site] + re_species[species])
    y_num <- rpois(n, mu)
    y_denom_cont <- rgamma(n, shape = 5, rate = 5/exp(2))
    df <- data.frame(y_num = y_num, y_denom_cont = y_denom_cont, site = factor(site), species = factor(species))
    fit <- suppressWarnings(tratio(y_num | y_denom_cont ~ 1 + (1 | site) + (1 | species), data = df,
        spatial = spatial_car(adj_mat, spatial_level = "group", group_var = "site"), family = ratiod_poisson_gamma(),
        mode = "hmc", control = list(chains = 1, iter = 150, warmup = 75, verbose = FALSE)))
    expect_s3_class(fit, "ratiod_fit")
})

test_that("poisson_gamma with random slopes + RW1 runs correctly (row 60)", {
    skip_on_cran()
    skip_unless_slow()
    set.seed(6001)
    n_sites <- 10
    n_times <- 6
    n_per <- 5
    n <- n_sites * n_times * n_per
    site <- rep(1:n_sites, each = n_times * n_per)
    time <- rep(rep(1:n_times, each = n_per), n_sites)
    re_intercept <- rnorm(n_sites, 0, 0.3)
    re_slope <- rnorm(n_sites, 0, 0.2)
    temporal_effect <- cumsum(rnorm(n_times, 0, 0.15))
    temporal_effect <- temporal_effect - mean(temporal_effect)
    x <- rnorm(n)
    mu <- exp(2 + re_intercept[site] + (0.3 + re_slope[site]) * x + temporal_effect[time])
    y_num <- rpois(n, mu)
    y_denom_cont <- rgamma(n, shape = 5, rate = 5/exp(2))
    df <- data.frame(y_num = y_num, y_denom_cont = y_denom_cont, x = x, site = factor(site), time = time)
    fit <- suppressWarnings(tratio(y_num | y_denom_cont ~ x + (x | site), data = df, temporal = temporal_rw1("time"),
        family = ratiod_poisson_gamma(), mode = "hmc", control = list(chains = 1, iter = 150, warmup = 75,
            verbose = FALSE)))
    expect_s3_class(fit, "ratiod_fit")
})

test_that("poisson_gamma with ICAR + RW1 + ZI runs correctly (row 64)", {
    skip_on_cran()
    skip_unless_slow()
    set.seed(6401)
    n_sites <- 16
    n_times <- 5
    n <- n_sites * n_times
    side <- 4
    adj_mat <- matrix(0, n_sites, n_sites)
    for (i in 1:n_sites) {
        row_i <- (i - 1)%/%side + 1
        col_i <- (i - 1)%%side + 1
        for (j in 1:n_sites) {
            row_j <- (j - 1)%/%side + 1
            col_j <- (j - 1)%%side + 1
            if (i != j && abs(row_i - row_j) + abs(col_i - col_j) == 1) {
                adj_mat[i, j] <- 1
            }
        }
    }
    site <- rep(1:n_sites, n_times)
    time <- rep(1:n_times, each = n_sites)
    w_site <- rnorm(n_sites, 0, 0.4)
    temporal_effect <- cumsum(rnorm(n_times, 0, 0.2))
    temporal_effect <- temporal_effect - mean(temporal_effect)
    mu <- exp(2 + w_site[site] + temporal_effect[time])
    y_num <- rpois(n, mu)
    pi_zi <- 0.2
    zi_mask <- rbinom(n, 1, pi_zi)
    y_num[zi_mask == 1] <- 0
    y_denom_cont <- rgamma(n, shape = 5, rate = 5/exp(2))
    df <- data.frame(y_num = y_num, y_denom_cont = y_denom_cont, site = factor(site), time = time)
    fit <- suppressWarnings(tratio(y_num | y_denom_cont ~ 1 + (1 | site), data = df, spatial = spatial_car(adj_mat,
        spatial_level = "group", group_var = "site"), temporal = temporal_rw1("time"), family = ratiod_zi_poisson(denom_family = "gamma",
        zi_formula = ~1), mode = "hmc", control = list(chains = 1, iter = 150, warmup = 75, verbose = FALSE)))
    expect_s3_class(fit, "ratiod_fit")
})

test_that("negbin_negbin with BYM2 + ZI runs correctly (row 55)", {
    skip_on_cran()
    skip_unless_slow()
    df <- generate_spatial_zi_data(n_sites = 25, seed = 5501, spatial_type = "bym2", family = "negbin_negbin")
    adj_mat <- matrix(0, 25, 25)
    for (i in 1:25) {
        for (j in df$adj_list[[i]]) {
            adj_mat[i, j] <- 1
        }
    }
    fit <- suppressWarnings(tratio(y_num | y_denom ~ 1 + (1 | site), data = as.data.frame(df), spatial = spatial_bym2(adj_mat,
        spatial_level = "group", group_var = "site"), family = ratiod_zi_negbin(denom_family = "negbin",
        zi_formula = ~1), mode = "hmc", control = list(chains = 1, iter = 150, warmup = 75, verbose = FALSE)))
    expect_s3_class(fit, "ratiod_fit")
})

test_that("binomial with ICAR + ZI runs correctly (row 56)", {
    skip_on_cran()
    skip_unless_slow()
    df <- generate_spatial_zi_data(n_sites = 25, seed = 5601, family = "binomial")
    adj_mat <- matrix(0, 25, 25)
    for (i in 1:25) {
        for (j in df$adj_list[[i]]) {
            adj_mat[i, j] <- 1
        }
    }
    fit <- suppressWarnings(tratio(y_num | y_denom ~ 1 + (1 | site), data = as.data.frame(df), spatial = spatial_car(adj_mat,
        spatial_level = "group", group_var = "site"), family = ratiod_zi_binomial(zi_formula = ~1), mode = "hmc",
        control = list(chains = 1, iter = 150, warmup = 75, verbose = FALSE)))
    expect_s3_class(fit, "ratiod_fit")
})

test_that("poisson_gamma with crossed RE + BYM2 runs correctly (row 58)", {
    skip_on_cran()
    skip_unless_slow()
    set.seed(5801)
    n_sites <- 16
    n_species <- 5
    n <- n_sites * n_species
    site <- rep(1:n_sites, each = n_species)
    species <- rep(1:n_species, n_sites)
    side <- 4
    adj_mat <- matrix(0, n_sites, n_sites)
    for (i in 1:n_sites) {
        row_i <- (i - 1)%/%side + 1
        col_i <- (i - 1)%%side + 1
        for (j in 1:n_sites) {
            row_j <- (j - 1)%/%side + 1
            col_j <- (j - 1)%%side + 1
            if (i != j && abs(row_i - row_j) + abs(col_i - col_j) == 1) {
                adj_mat[i, j] <- 1
            }
        }
    }
    w_site <- rnorm(n_sites, 0, 0.5)
    re_species <- rnorm(n_species, 0, 0.3)
    mu <- exp(2 + w_site[site] + re_species[species])
    y_num <- rpois(n, mu)
    y_denom_cont <- rgamma(n, shape = 5, rate = 5/exp(2))
    df <- data.frame(y_num = y_num, y_denom_cont = y_denom_cont, site = factor(site), species = factor(species))
    fit <- suppressWarnings(tratio(y_num | y_denom_cont ~ 1 + (1 | site) + (1 | species), data = df,
        spatial = spatial_bym2(adj_mat, spatial_level = "group", group_var = "site"), family = ratiod_poisson_gamma(),
        mode = "hmc", control = list(chains = 1, iter = 150, warmup = 75, verbose = FALSE)))
    expect_s3_class(fit, "ratiod_fit")
})

test_that("negbin_negbin with crossed RE + ICAR runs correctly (row 59)", {
    skip_on_cran()
    skip_unless_slow()
    set.seed(5901)
    n_sites <- 16
    n_species <- 5
    n <- n_sites * n_species
    site <- rep(1:n_sites, each = n_species)
    species <- rep(1:n_species, n_sites)
    side <- 4
    adj_mat <- matrix(0, n_sites, n_sites)
    for (i in 1:n_sites) {
        row_i <- (i - 1)%/%side + 1
        col_i <- (i - 1)%%side + 1
        for (j in 1:n_sites) {
            row_j <- (j - 1)%/%side + 1
            col_j <- (j - 1)%%side + 1
            if (i != j && abs(row_i - row_j) + abs(col_i - col_j) == 1) {
                adj_mat[i, j] <- 1
            }
        }
    }
    w_site <- rnorm(n_sites, 0, 0.5)
    re_species <- rnorm(n_species, 0, 0.3)
    mu_num <- exp(2 + w_site[site] + re_species[species])
    mu_denom <- exp(2.5 + 0.5 * w_site[site])
    y_num <- rnbinom(n, size = 5, mu = mu_num)
    y_denom <- rnbinom(n, size = 8, mu = mu_denom)
    df <- data.frame(y_num = y_num, y_denom = y_denom, site = factor(site), species = factor(species))
    fit <- suppressWarnings(tratio(y_num | y_denom ~ 1 + (1 | site) + (1 | species), data = df, spatial = spatial_car(adj_mat,
        spatial_level = "group", group_var = "site"), family = ratiod_negbin_negbin(), mode = "hmc",
        control = list(chains = 1, iter = 150, warmup = 75, verbose = FALSE)))
    expect_s3_class(fit, "ratiod_fit")
})

test_that("poisson_gamma with random slopes + AR1 runs correctly (row 61)", {
    skip_on_cran()
    skip_unless_slow()
    set.seed(6101)
    n_sites <- 10
    n_times <- 6
    n_per <- 5
    n <- n_sites * n_times * n_per
    site <- rep(1:n_sites, each = n_times * n_per)
    time <- rep(rep(1:n_times, each = n_per), n_sites)
    re_intercept <- rnorm(n_sites, 0, 0.3)
    re_slope <- rnorm(n_sites, 0, 0.2)
    rho <- 0.7
    temporal_effect <- numeric(n_times)
    temporal_effect[1] <- rnorm(1, 0, 0.3)
    for (t in 2:n_times) {
        temporal_effect[t] <- rho * temporal_effect[t - 1] + rnorm(1, 0, 0.2)
    }
    x <- rnorm(n)
    mu <- exp(2 + re_intercept[site] + (0.3 + re_slope[site]) * x + temporal_effect[time])
    y_num <- rpois(n, mu)
    y_denom_cont <- rgamma(n, shape = 5, rate = 5/exp(2))
    df <- data.frame(y_num = y_num, y_denom_cont = y_denom_cont, x = x, site = factor(site), time = time)
    fit <- suppressWarnings(tratio(y_num | y_denom_cont ~ x + (x | site), data = df, temporal = temporal_ar1("time"),
        family = ratiod_poisson_gamma(), mode = "hmc", control = list(chains = 1, iter = 150, warmup = 75,
            verbose = FALSE)))
    expect_s3_class(fit, "ratiod_fit")
})

test_that("negbin_negbin with random slopes + RW1 runs correctly (row 62)", {
    skip_on_cran()
    skip_unless_slow()
    set.seed(6201)
    n_sites <- 10
    n_times <- 6
    n_per <- 5
    n <- n_sites * n_times * n_per
    site <- rep(1:n_sites, each = n_times * n_per)
    time <- rep(rep(1:n_times, each = n_per), n_sites)
    re_intercept <- rnorm(n_sites, 0, 0.3)
    re_slope <- rnorm(n_sites, 0, 0.2)
    temporal_effect <- cumsum(rnorm(n_times, 0, 0.15))
    temporal_effect <- temporal_effect - mean(temporal_effect)
    x <- rnorm(n)
    mu_num <- exp(2 + re_intercept[site] + (0.3 + re_slope[site]) * x + temporal_effect[time])
    mu_denom <- exp(2.5 + 0.5 * temporal_effect[time])
    y_num <- rnbinom(n, size = 5, mu = mu_num)
    y_denom <- rnbinom(n, size = 8, mu = mu_denom)
    df <- data.frame(y_num = y_num, y_denom = y_denom, x = x, site = factor(site), time = time)
    fit <- suppressWarnings(tratio(y_num | y_denom ~ x + (x | site), data = df, temporal = temporal_rw1("time"),
        family = ratiod_negbin_negbin(), mode = "hmc", control = list(chains = 1, iter = 150, warmup = 75,
            verbose = FALSE)))
    expect_s3_class(fit, "ratiod_fit")
})

test_that("binomial with random slopes + AR1 runs correctly (row 63)", {
    skip_on_cran()
    skip_unless_slow()
    set.seed(6301)
    n_sites <- 10
    n_times <- 6
    n_per <- 5
    n <- n_sites * n_times * n_per
    site <- rep(1:n_sites, each = n_times * n_per)
    time <- rep(rep(1:n_times, each = n_per), n_sites)
    re_intercept <- rnorm(n_sites, 0, 0.3)
    re_slope <- rnorm(n_sites, 0, 0.2)
    rho <- 0.7
    temporal_effect <- numeric(n_times)
    temporal_effect[1] <- rnorm(1, 0, 0.3)
    for (t in 2:n_times) {
        temporal_effect[t] <- rho * temporal_effect[t - 1] + rnorm(1, 0, 0.2)
    }
    x <- rnorm(n)
    p <- plogis(re_intercept[site] + (0.3 + re_slope[site]) * x + temporal_effect[time])
    trials <- rpois(n, 25) + 10
    y_num <- rbinom(n, trials, p)
    df <- data.frame(y_num = y_num, y_denom = trials, x = x, site = factor(site), time = time)
    fit <- suppressWarnings(tratio(y_num | y_denom ~ x + (x | site), data = df, temporal = temporal_ar1("time"),
        family = ratiod_binomial(), mode = "hmc", control = list(chains = 1, iter = 150, warmup = 75,
            verbose = FALSE)))
    expect_s3_class(fit, "ratiod_fit")
})

test_that("poisson_gamma with BYM2 + RW1 + ZI runs correctly (row 65)", {
    skip_on_cran()
    skip_unless_slow()
    set.seed(6501)
    n_sites <- 16
    n_times <- 5
    n <- n_sites * n_times
    side <- 4
    adj_mat <- matrix(0, n_sites, n_sites)
    for (i in 1:n_sites) {
        row_i <- (i - 1)%/%side + 1
        col_i <- (i - 1)%%side + 1
        for (j in 1:n_sites) {
            row_j <- (j - 1)%/%side + 1
            col_j <- (j - 1)%%side + 1
            if (i != j && abs(row_i - row_j) + abs(col_i - col_j) == 1) {
                adj_mat[i, j] <- 1
            }
        }
    }
    site <- rep(1:n_sites, n_times)
    time <- rep(1:n_times, each = n_sites)
    w_site <- rnorm(n_sites, 0, 0.4)
    temporal_effect <- cumsum(rnorm(n_times, 0, 0.2))
    temporal_effect <- temporal_effect - mean(temporal_effect)
    mu <- exp(2 + w_site[site] + temporal_effect[time])
    y_num <- rpois(n, mu)
    pi_zi <- 0.2
    zi_mask <- rbinom(n, 1, pi_zi)
    y_num[zi_mask == 1] <- 0
    y_denom_cont <- rgamma(n, shape = 5, rate = 5/exp(2))
    df <- data.frame(y_num = y_num, y_denom_cont = y_denom_cont, site = factor(site), time = time)
    fit <- suppressWarnings(tratio(y_num | y_denom_cont ~ 1 + (1 | site), data = df, spatial = spatial_bym2(adj_mat,
        spatial_level = "group", group_var = "site"), temporal = temporal_rw1("time"), family = ratiod_zi_poisson(denom_family = "gamma",
        zi_formula = ~1), mode = "hmc", control = list(chains = 1, iter = 150, warmup = 75, verbose = FALSE)))
    expect_s3_class(fit, "ratiod_fit")
})

test_that("negbin_negbin with ICAR + RW1 + ZI runs correctly (row 66)", {
    skip_on_cran()
    skip_unless_slow()
    set.seed(6601)
    n_sites <- 16
    n_times <- 5
    n <- n_sites * n_times
    side <- 4
    adj_mat <- matrix(0, n_sites, n_sites)
    for (i in 1:n_sites) {
        row_i <- (i - 1)%/%side + 1
        col_i <- (i - 1)%%side + 1
        for (j in 1:n_sites) {
            row_j <- (j - 1)%/%side + 1
            col_j <- (j - 1)%%side + 1
            if (i != j && abs(row_i - row_j) + abs(col_i - col_j) == 1) {
                adj_mat[i, j] <- 1
            }
        }
    }
    site <- rep(1:n_sites, n_times)
    time <- rep(1:n_times, each = n_sites)
    w_site <- rnorm(n_sites, 0, 0.4)
    temporal_effect <- cumsum(rnorm(n_times, 0, 0.2))
    temporal_effect <- temporal_effect - mean(temporal_effect)
    mu_num <- exp(2 + w_site[site] + temporal_effect[time])
    mu_denom <- exp(2.5 + 0.5 * w_site[site])
    y_num <- rnbinom(n, size = 5, mu = mu_num)
    y_denom <- rnbinom(n, size = 8, mu = mu_denom)
    pi_zi <- 0.2
    zi_mask <- rbinom(n, 1, pi_zi)
    y_num[zi_mask == 1] <- 0
    df <- data.frame(y_num = y_num, y_denom = y_denom, site = factor(site), time = time)
    fit <- suppressWarnings(tratio(y_num | y_denom ~ 1 + (1 | site), data = df, spatial = spatial_car(adj_mat,
        spatial_level = "group", group_var = "site"), temporal = temporal_rw1("time"), family = ratiod_zi_negbin(denom_family = "negbin",
        zi_formula = ~1), mode = "hmc", control = list(chains = 1, iter = 150, warmup = 75, verbose = FALSE)))
    expect_s3_class(fit, "ratiod_fit")
})
