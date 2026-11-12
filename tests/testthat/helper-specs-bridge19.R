bridge19_write_evidence <- function(row, filename) {
  work <- Sys.getenv("TULPA_BRIDGE_WORK", file.path(tempdir(), "bridge19"))
  dir.create(work, recursive = TRUE, showWarnings = FALSE)
  path <- file.path(work, filename)
  append <- file.exists(path)
  write.table(row, path, sep = "\t", row.names = FALSE, append = append,
    col.names = !append)
}

bridge19_control <- function(name, seed) {
  control <- list(iter = 6000L, warmup = 2000L, chains = 1L, cores = 1L,
    seed = seed, verbose = FALSE, adapt_delta = .99, gradient_mode = "A_r",
    metric = "auto")
  recipes <- list(st_i_iid = list(chains = 4L, adapt_delta = .999))
  recipe <- recipes[[name]]
  control[names(recipe)] <- recipe
  control
}

bridge19_comparison_draws <- function(fit) {
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

bridge19_chain_stat <- function(draws, fit, statistic) {
  ids <- fit$diagnostics$energy_chain
  stopifnot(length(ids) == nrow(draws))
  indices <- lapply(sort(unique(ids)), function(id) which(ids == id))
  stopifnot(length(unique(lengths(indices))) == 1L)
  out <- vapply(seq_len(ncol(draws)), function(column)
    statistic(do.call(cbind, lapply(indices, function(rows) draws[rows, column]))),
    numeric(1L))
  names(out) <- colnames(draws)
  out
}

bridge19_inventory <- function(group, routes) {
  bridge19_write_evidence(data.frame(group = group, route = routes,
    seeds = "18,181,182"), "route-inventory.tsv")
}

bridge19_select_cases <- function(cases, prefix = "") {
  selected <- Sys.getenv("TULPA_BRIDGE_ROUTE")
  if (!nzchar(selected)) return(cases)
  wanted <- strsplit(selected, ",", fixed = TRUE)[[1L]]
  indices <- match(wanted, paste0(prefix, names(cases)))
  stopifnot(!anyNA(indices))
  cases[indices]
}

bridge19_record_recovery <- function(fit, beta_error, numerator_error,
                                     denominator_error = NA_real_, scale) {
  row <- attr(fit, "bridge19")
  row$beta_max_abs <- beta_error
  row$numerator_error <- numerator_error
  row$denominator_error <- denominator_error
  row$error_scale <- scale
  bridge19_write_evidence(row, "route-recovery-final.tsv")
}

bridge19_link <- function(name) {
  links <- list(
    logit = list(link = stats::qlogis, inverse = stats::plogis),
    probit = list(link = stats::qnorm, inverse = stats::pnorm),
    cloglog = list(link = function(p) log(-log1p(-p)),
      inverse = function(x) -expm1(-exp(x))))
  links[[name]]
}

bridge19_temporal_recovery <- function(fit, data, link) {
  b <- as.matrix(fit$draws)
  design <- hmc_fit_design(fit)
  stopifnot(identical(unname(design$X_num[, 2L]), data$x))
  level <- b[, "beta_num[1]"]
  fields <- hmc_structure_contributions(hmc_fit_unpack(fit), design)
  for (field in fields) {
    stopifnot(!is.null(field$eta), ncol(field$eta) == nrow(data))
    level <- level + rowMeans(field$eta)
  }
  truth <- mean(bridge19_link(link)$link(data$truth) - .5 * data$x)
  data.frame(raw_intercept_mean = mean(b[, "beta_num[1]"]),
    raw_intercept_sd = sd(b[, "beta_num[1]"]),
    raw_coefficient_error = max(abs(colMeans(b)[1:2] - c(.2, .5))),
    predictor_level_mean = mean(level), predictor_level_sd = sd(level),
    predictor_level_truth = truth,
    predictor_coefficient_error = max(abs(c(mean(level) - truth,
      mean(b[, "beta_num[2]"]) - .5))))
}

bridge19_fixture <- function(seed, case) {
  set.seed(seed)
  groups <- if (isTRUE(case$grouped)) 2L else 1L
  d <- data.frame(time = rep(rep(seq_len(6L), each = 24L), groups),
    group = factor(rep(seq_len(groups), each = 144L)), n = 30L)
  d$x <- rnorm(nrow(d))
  if (isTRUE(case$linear_restriction)) d$x <- (d$time - 3.5) / 2
  field <- c(-.6, -.3, -.1, .1, .3, .6)[d$time]
  if (!is.null(case$period)) {
    field <- .5 * sin(2 * pi * (d$time - 1) / case$period)
    field <- field - mean(field)
  }
  if (isTRUE(case$restricted)) {
    Q <- qr.Q(qr(model.matrix(~ x, d)))
    field <- field - drop(Q %*% crossprod(Q, field))
  }
  eta <- .2 + .5 * d$x + if (isTRUE(case$tvc)) field * d$x else field
  d$truth <- bridge19_link(case$link %||% "logit")$inverse(eta)
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
    temporal_gp_periodic = list(period = 4L, temporal = temporal_gp("time", cov = "periodic", period = 4)),
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

engine_recovery_fit <- function(args, name, seed, control = bridge19_control(name, seed)) {
  args$control <- control
  reached_engine <- FALSE
  trace("cpp_tulpaRatio_run_nuts_specs", where = asNamespace("tulpaRatio"),
    tracer = function() reached_engine <<- TRUE, print = FALSE)
  on.exit(untrace("cpp_tulpaRatio_run_nuts_specs", where = asNamespace("tulpaRatio")), add = TRUE)
  timing <- system.time(fit <- do.call(tratio, args))
  expect_true(reached_engine, info = paste(name, seed, "dispatch"))
  expect_length(unique(fit$diagnostics$energy_chain), control$chains)
  draws <- bridge19_comparison_draws(fit)
  precision <- startsWith(colnames(draws), "tau_")
  draws[, precision] <- log(draws[, precision, drop = FALSE])
  varying <- apply(draws, 2L, sd) > 1e-10
  draws <- draws[, varying, drop = FALSE]
  row <- data.frame(route = name, seed = seed, seconds = timing[["elapsed"]],
    ess_sec = min(bridge19_chain_stat(draws, fit, posterior::ess_bulk)) / timing[["elapsed"]],
    divergences = fit$diagnostics$n_divergent,
    rhat = if (control$chains > 1L) max(bridge19_chain_stat(draws, fit, posterior::rhat)) else NA_real_)
  bridge19_write_evidence(row, "engine-route-diagnostics.tsv")
  work <- Sys.getenv("TULPA_BRIDGE_WORK", file.path(tempdir(), "bridge19"))
  saveRDS(list(fit = fit, args = args, diagnostics = row),
    file.path(work, paste0("engine-", name, "-", seed, ".rds")))
  attr(fit, "bridge19") <- row[c("route", "seed")]
  fit
}
