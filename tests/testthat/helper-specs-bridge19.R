bridge19_fixture <- function(seed, case) {
  set.seed(seed)
  groups <- if (isTRUE(case$grouped)) 2L else 1L
  d <- data.frame(time = rep(rep(seq_len(6L), each = 24L), groups),
    group = factor(rep(seq_len(groups), each = 144L)), n = 30L)
  d$x <- rnorm(nrow(d))
  if (isTRUE(case$linear_restriction)) d$x <- (d$time - 3.5) / 2
  field <- c(-.6, -.3, -.1, .1, .3, .6)[d$time]
  if (isTRUE(case$restricted)) {
    Q <- qr.Q(qr(model.matrix(~ x, d)))
    field <- field - drop(Q %*% crossprod(Q, field))
  }
  eta <- .2 + .5 * d$x + if (isTRUE(case$tvc)) field * d$x else field
  inverse <- switch(case$link %||% "logit", logit = plogis, probit = pnorm,
    cloglog = function(x) -expm1(-exp(x)))
  d$truth <- inverse(eta)
  d$y <- rbinom(nrow(d), d$n, d$truth)
  d
}

bridge19_cases <- function() {
  cases <- list(
    rw1_nc = list(temporal = temporal_rw1("time", parameterization = "noncentered")),
    rw2_nc = list(temporal = temporal_rw2("time", parameterization = "noncentered")),
    ar1_nc = list(temporal = temporal_ar1("time", parameterization = "noncentered")),
    cyclic_rw1_nc = list(temporal = temporal_rw1("time", cyclic = TRUE, parameterization = "noncentered")),
    cyclic_rw2_nc = list(temporal = temporal_rw2("time", cyclic = TRUE, parameterization = "noncentered")),
    grouped_rw1 = list(grouped = TRUE, temporal = temporal_rw1("time", "group")),
    grouped_rw2 = list(grouped = TRUE, temporal = temporal_rw2("time", "group")),
    grouped_rw1_nc = list(grouped = TRUE, temporal = temporal_rw1("time", "group", parameterization = "noncentered")),
    grouped_rw2_nc = list(grouped = TRUE, temporal = temporal_rw2("time", "group", parameterization = "noncentered")),
    grouped_ar1_nc = list(grouped = TRUE, temporal = temporal_ar1("time", "group", parameterization = "noncentered")),
    iid = list(temporal = temporal_iid("time")),
    grouped_iid = list(grouped = TRUE, temporal = temporal_iid("time", "group")),
    restricted_rw1 = list(restricted = TRUE, temporal = temporal_rtr(temporal_rw1("time"), ~ x)),
    restricted_rw1_nc = list(restricted = TRUE, temporal = temporal_rtr(temporal_rw1("time", parameterization = "noncentered"), ~ x)),
    restricted_rw2 = list(restricted = TRUE, linear_restriction = TRUE, temporal = temporal_rtr(temporal_rw2("time"), ~ x)),
    restricted_rw2_nc = list(restricted = TRUE, linear_restriction = TRUE, temporal = temporal_rtr(temporal_rw2("time", parameterization = "noncentered"), ~ x)),
    restricted_grouped_rw2 = list(grouped = TRUE, restricted = TRUE, linear_restriction = TRUE, temporal = temporal_rtr(temporal_rw2("time", "group", parameterization = "noncentered"), ~ x)),
    temporal_gp_ou = list(temporal = temporal_gp("time")),
    temporal_gp_matern = list(temporal = temporal_gp("time", cov = "matern", nu = 1.5)),
    temporal_gp_gaussian = list(temporal = temporal_gp("time", cov = "gaussian")),
    temporal_gp_periodic = list(temporal = temporal_gp("time", cov = "periodic", period = 4)),
    temporal_gp_centered = list(temporal = temporal_gp("time", parameterization = "centered")),
    temporal_gp_grouped = list(grouped = TRUE, temporal = temporal_gp("time", group_var = "group")),
    restricted_gp = list(restricted = TRUE, temporal = temporal_rtr(temporal_gp("time"), ~ x)),
    ms_rw1_ar1 = list(temporal = temporal_multiscale("time", trend = "rw1", seasonal = 3L, short_term = "ar1")),
    ms_rw2_iid = list(temporal = temporal_multiscale("time", trend = "rw2", seasonal = 3L, short_term = "iid")),
    ms_centered = list(temporal = temporal_multiscale("time", trend = "rw1", seasonal = 3L, short_term = "iid", parameterization = "centered")),
    ms_noncentered = list(temporal = temporal_multiscale("time", trend = "rw1", seasonal = 3L, short_term = "ar1", parameterization = "noncentered")),
    restricted_ms = list(restricted = TRUE, temporal = temporal_rtr(temporal_multiscale("time", trend = "rw1", short_term = "iid"), ~ x)),
    restricted_ms_rw2 = list(restricted = TRUE, linear_restriction = TRUE, temporal = temporal_rtr(temporal_multiscale("time", trend = "rw2", short_term = "iid"), ~ x)),
    tvc_rw1 = list(tvc = TRUE, temporal = temporal_tvc("time", ~ x, structure = "rw1")),
    tvc_rw2 = list(tvc = TRUE, temporal = temporal_tvc("time", ~ x, structure = "rw2")),
    tvc_ar1 = list(tvc = TRUE, temporal = temporal_tvc("time", ~ x, structure = "ar1")),
    tvc_iid = list(tvc = TRUE, temporal = temporal_tvc("time", ~ x, structure = "iid")),
    tvc_grouped = list(tvc = TRUE, grouped = TRUE, temporal = temporal_tvc("time", ~ x, structure = "rw1", group_var = "group")),
    probit = list(link = "probit", temporal = temporal_rw1("time")),
    cloglog = list(link = "cloglog", temporal = temporal_rw1("time")))
  cases
}

bridge19_pair <- function(args, name, seed) {
  work <- Sys.getenv("TULPA_BRIDGE_WORK", file.path(tempdir(), "bridge19"))
  dir.create(work, recursive = TRUE, showWarnings = FALSE)
  binaries <- c(getLoadedDLLs()[["tulpa"]][["path"]],
    getLoadedDLLs()[["tulpaRatio"]][["path"]],
    system.file("R/tulpaRatio.rdb", package = "tulpaRatio"))
  binary_tag <- paste(substr(unname(tools::md5sum(binaries)), 1L, 8L), collapse = "")
  canonical <- function(x) {
    if (inherits(x, "formula")) {
      environment(x) <- baseenv()
      return(x)
    }
    if (is.function(x)) return(list(formals = formals(x), body = body(x),
      environment = environmentName(environment(x))))
    if (is.list(x)) {
      attributes_ <- attributes(x)
      x <- lapply(x, canonical)
      attributes(x) <- attributes_
    }
    x
  }
  key_path <- file.path(work, paste0("cache-key-", Sys.getpid(), ".rds"))
  saveRDS(canonical(args), key_path, version = 3L)
  argument_tag <- unname(tools::md5sum(key_path))
  unlink(key_path)
  directory <- file.path(work, "comparisons", binary_tag, argument_tag)
  dir.create(directory, recursive = TRUE, showWarnings = FALSE)
  op <- options()
  on.exit(options(op))
  reached_engine <- FALSE
  trace("cpp_tulpaRatio_run_nuts_specs", where = asNamespace("tulpaRatio"),
    tracer = function() reached_engine <<- TRUE, print = FALSE)
  on.exit(untrace("cpp_tulpaRatio_run_nuts_specs", where = asNamespace("tulpaRatio")), add = TRUE)
  results <- list()
  for (specs in c(FALSE, TRUE)) {
    options(tulpaRatio.use_specs = specs)
    label <- if (specs) "bridge" else "legacy"
    path <- file.path(directory, paste(name, seed, label, "rds", sep = "."))
    if (file.exists(path) && identical(Sys.getenv("TULPA_BRIDGE_REUSE"), "true")) {
      results[[label]] <- readRDS(path)
    } else {
      reached_engine <- FALSE
      timing <- system.time(fit <- do.call(tratio, args))
      expect_identical(reached_engine, specs, info = paste(name, seed, label, "dispatch"))
      results[[label]] <- list(fit = fit, seconds = timing[["elapsed"]], reached_engine = reached_engine)
      saveRDS(results[[label]], path)
    }
  }
  expect_true(isTRUE(results$bridge$reached_engine), info = paste(name, seed, "cached dispatch"))
  comparison_draws <- function(fit) {
    out <- as.matrix(fit$draws)
    fields <- hmc_structure_contributions(hmc_fit_unpack(fit), hmc_fit_design(fit))
    for (field in fields) for (margin in c("eta", "eta_denom")) {
      values <- field[[margin]]
      if (is.null(values)) next
      colnames(values) <- paste0(field$name, ":", margin, "[", seq_len(ncol(values)), "]")
      out <- cbind(out, values)
    }
    out
  }
  a <- comparison_draws(results$legacy$fit)
  b <- comparison_draws(results$bridge$fit)
  expect_identical(colnames(a), colnames(b), info = paste(name, seed))
  precision <- grepl("^tau_", colnames(a))
  a[, precision] <- log(a[, precision, drop = FALSE])
  b[, precision] <- log(b[, precision, drop = FALSE])
  sd <- apply(a, 2L, stats::sd)
  varying <- sd > 1e-10
  expect_lt(max(abs(colMeans(a)[!varying] - colMeans(b)[!varying]), 0), 1e-9)
  a <- a[, varying, drop = FALSE]
  b <- b[, varying, drop = FALSE]
  sd <- sd[varying]
  mean_gap <- max(abs(colMeans(a) - colMeans(b)) / sd)
  quantile_gap <- max(abs(sweep(apply(a, 2L, quantile, c(.1, .5, .9)) -
    apply(b, 2L, quantile, c(.1, .5, .9)), 2L, sd, "/")))
  ess <- function(x) min(apply(x, 2L, posterior::ess_bulk))
  row <- data.frame(binary_tag = binary_tag, argument_tag = argument_tag,
    precision_coordinate = "log", route = name, seed = seed, mean_sd = mean_gap,
    quantile_sd = quantile_gap, legacy_seconds = results$legacy$seconds,
    bridge_seconds = results$bridge$seconds,
    legacy_ess_sec = ess(a) / results$legacy$seconds,
    bridge_ess_sec = ess(b) / results$bridge$seconds,
    legacy_divergences = results$legacy$fit$diagnostics$n_divergent,
    bridge_divergences = results$bridge$fit$diagnostics$n_divergent)
  output <- file.path(work, "route-comparisons-final.tsv")
  write.table(row, output, sep = "\t", row.names = FALSE,
    col.names = !file.exists(output), append = TRUE)
  message(sprintf("BRIDGE19 route=%s seed=%d mean_sd=%.6f quantile_sd=%.6f", name, seed, mean_gap, quantile_gap))
  expect_lt(mean_gap, .35, label = paste(name, seed, "means"))
  expect_lt(quantile_gap, .65, label = paste(name, seed, "quantiles"))
  results$bridge$fit
}
