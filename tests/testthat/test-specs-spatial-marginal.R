test_that("engine marginal and joint spatial posteriors agree across three seeds", {
    skip_on_cran()
    fixture <- spatial_bridge_fixture(1919L)
    d <- fixture$data
    fields <- list(icar = function() spatial_car(fixture$adjacency, group_var = "unit"), bym2 = function() spatial_bym2(fixture$adjacency,
        group_var = "unit"), gp = function() spatial_gp(~lon + lat, scale_coords = FALSE, nn = 5L, parameterization = "collapsed"))
    fn <- "cpp_tulpaRatio_sample"
    ns <- asNamespace("tulpaRatio")
    on.exit(if (methods::is(get0(fn, envir = ns, inherits = FALSE), "functionWithTrace")) suppressMessages(untrace(fn,
        where = ns)), add = TRUE)
    for (seed in c(18L, 181L, 182L)) for (name in names(fields)) {
        capture <- new.env(parent = emptyenv())
        suppressMessages(trace(fn, where = ns, print = FALSE, tracer = substitute({
            assign("args", mget(names(formals(tulpaRatio:::cpp_tulpaRatio_sample)), envir = environment()),
                envir = CAP)
            assign("start", proc.time()[["elapsed"]], envir = CAP)
        }, list(CAP = capture)), exit = substitute({
            assign("seconds", proc.time()[["elapsed"]] - get("start", envir = CAP), envir = CAP)
            assign("result", returnValue(), envir = CAP)
        }, list(CAP = capture))))
        fit <- suppressWarnings(tratio(y | n ~ x, data = d, family = ratiod_binomial(), spatial = fields[[name]](),
            mode = "hmc", control = list(iter = 8000L, warmup = 2500L, chains = 1L, cores = 1L, seed = seed,
                verbose = FALSE, adapt_delta = 0.99, gradient_mode = "A_r")))
        suppressMessages(untrace(fn, where = ns))
        expect_true(is.list(capture$args))
        if (name == "gp")
            expect_true(isTRUE(capture$args$spatial_params$gp$conditional_coordinates))
        outer <- if (name == "icar")
            3L
        else 4L
        args <- capture$args
        args$spatial_params$parameterization <- "collapsed"
        args$init <- args$init[seq_len(outer)]
        args$adapt_delta <- 0.995
        timing <- system.time(marginal <- do.call(tulpaRatio:::cpp_tulpaRatio_sample, args))
        joint <- capture$result
        expect_equal(ncol(marginal$samples), outer)
        expect_equal(dim(marginal[[if (name == "gp") "gp_w_star" else "icar_phi_star"]]), c(5500L, fixture$S))
        a <- marginal$samples
        b <- joint$samples[, seq_len(outer), drop = FALSE]
        compare <- function(x, y) {
            scale <- apply(x, 2, sd)
            c(mean = max(abs(colMeans(x) - colMeans(y))/scale), quantile = max(abs(sweep(apply(x, 2,
                quantile, c(0.1, 0.5, 0.9)) - apply(y, 2, quantile, c(0.1, 0.5, 0.9)), 2, scale, "/"))))
        }
        gap <- compare(a, b)
        phi <- marginal[[if (name == "gp")
            "gp_w_star"
        else "icar_phi_star"]]
        if (name != "gp")
            phi <- phi - rowMeans(phi)
        if (name == "bym2") {
            rho <- plogis(a[, 4])
            sigma <- exp(a[, 3])
            phi <- phi * (sigma * sqrt(rho) * args$spatial_params$bym2_scale) + marginal$bym2_theta_star *
                (sigma * sqrt(1 - rho))
        }
        eta_a <- a[, 1:2] %*% t(args$X_list[[1]]) + phi[, d$unit]
        eta_b <- qlogis(fitted(fit, summary = FALSE)$numerator)
        eta_gap <- compare(eta_a, eta_b)
        ess <- function(x) min(vapply(seq_len(ncol(x)), function(j) posterior::ess_bulk(x[, j]), numeric(1)))
        summary <- data.frame(field = name, seed = seed, marginal_seconds = timing[["elapsed"]], joint_seconds = capture$seconds,
            marginal_ess_sec = ess(a)/timing[["elapsed"]], joint_ess_sec = ess(b)/capture$seconds, mean_gap = gap[["mean"]],
            quantile_gap = gap[["quantile"]], eta_mean_gap = eta_gap[["mean"]], eta_quantile_gap = eta_gap[["quantile"]],
            marginal_divergences = sum(marginal$divergent), joint_divergences = sum(joint$divergent))
        message(paste(capture.output(print(summary, row.names = FALSE)), collapse = "\n"))
        evidence <- Sys.getenv("TULPA_BRIDGE_EVIDENCE", "")
        if (nzchar(evidence)) {
            saveRDS(list(summary = summary, marginal = marginal, joint = joint, args = args, eta_marginal = eta_a,
                eta_joint = eta_b), file.path(evidence, paste0("selected-geometry-", name, "-", seed,
                ".rds")))
        }
        expect_lt(gap[["mean"]], 0.35, label = paste(name, seed))
        expect_lt(gap[["quantile"]], 0.65, label = paste(name, seed))
        expect_lt(eta_gap[["mean"]], 0.35, label = paste(name, seed))
        expect_lt(eta_gap[["quantile"]], 0.65, label = paste(name, seed))
        expect_lt(sqrt(mean((colMeans(eta_a) - fixture$eta)^2)), 0.2)
        expect_equal(sum(marginal$divergent), 0L)
        expect_equal(sum(joint$divergent), 0L)
    }
})
