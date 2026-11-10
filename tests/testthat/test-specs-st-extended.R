test_that("Knorr-Held interaction bridges agree across seeds and recover planted fields", {
  skip_on_cran()
  nt <- 4L
  cases <- list(st_i = c("I", "rw1", "centered"),
    st_ii = c("II", "rw1", "centered"),
    st_iii = c("III", "rw1", "centered"),
    st_iv = c("IV", "rw1", "centered"),
    st_i_nc = c("I", "rw1", "noncentered"),
    st_ii_nc = c("II", "rw1", "noncentered"),
    st_iii_nc = c("III", "rw1", "noncentered"),
    st_iv_nc = c("IV", "rw1", "noncentered"),
    st_iv_ar1 = c("IV", "ar1", "noncentered"),
    st_iv_rw2 = c("IV", "rw2", "noncentered"),
    st_ii_rw2_cyclic = c("II", "rw2", "centered", "cyclic"),
    st_iv_rw2_cyclic = c("IV", "rw2", "noncentered", "cyclic"),
    st_gp = c("separable", "rw1", "centered"),
    st_gp_nc = c("separable", "rw1", "noncentered"),
    st_gneiting = c("nonsep_gp", "rw1", "noncentered"),
    st_hsgp = c("HSGP", "rw1", "noncentered"),
    st_hsgp_centered = c("HSGP", "rw1", "centered"),
    st_hsgp_ar1 = c("HSGP", "ar1", "noncentered"),
    st_iii_disconnected = c("III", "rw1", "centered", "disconnected"),
    st_iii_disconnected_nc = c("III", "rw1", "noncentered", "disconnected"),
    st_iv_disconnected = c("IV", "rw1", "noncentered", "disconnected"),
    st_iv_disconnected_ar1 = c("IV", "ar1", "noncentered", "disconnected"),
    st_iv_disconnected_rw2 = c("IV", "rw2", "noncentered", "disconnected"),
    st_shared = c("IV", "rw1", "noncentered", "shared"),
    st_unshared = c("IV", "rw1", "noncentered", "unshared"))
  bridge19_inventory("spatiotemporal", names(cases))
  selected <- Sys.getenv("TULPA_BRIDGE_ROUTE")
  if (nzchar(selected)) {
    wanted <- strsplit(selected, ",", fixed = TRUE)[[1L]]
    stopifnot(all(wanted %in% names(cases)))
    cases <- cases[wanted]
  }
  for (seed in c(18L, 181L, 182L)) for (name in names(cases)) {
    case <- cases[[name]]
    disconnected <- length(case) > 3L && case[4L] == "disconnected"
    paired <- length(case) > 3L && case[4L] %in% c("shared", "unshared")
    S <- if (disconnected) 5L else 3L
    adjacency <- matrix(0, S, S)
    edges <- if (disconnected) matrix(c(1L, 3L, 2L, 4L), ncol = 2L, byrow = TRUE) else
      matrix(c(1L, 2L, 2L, 3L), ncol = 2L, byrow = TRUE)
    adjacency[edges] <- 1
    adjacency <- adjacency + t(adjacency)
    set.seed(seed)
    d <- data.frame(unit = rep(seq_len(S), each = nt * 12L),
      time = rep(rep(seq_len(nt), each = 12L), S), n = 40L)
    d$x <- rnorm(nrow(d))
    d$lon <- c(.1, .5, .9, .2, .7)[d$unit]
    d$lat <- c(.2, -.3, .6, -.2, .8)[d$unit]
    spatial_field <- if (disconnected) c(-.5, -.25, .5, .25, 0) else c(-.5, 0, .5)
    field <- spatial_field[d$unit] * c(-.6, -.2, .2, .6)[d$time]
    if (paired) {
      beta <- c(3, .4)
      truth <- exp(beta[1L] + beta[2L] * d$x + field)
      denominator <- exp(1 + .2 * d$x + if (case[4L] == "shared") field else 0)
      d$y <- rpois(nrow(d), truth)
      d$denom <- rgamma(nrow(d), 30, rate = 30 / denominator)
    } else {
      beta <- c(.2, .5)
      truth <- plogis(beta[1L] + beta[2L] * d$x + field)
      d$y <- rbinom(nrow(d), d$n, truth)
    }
    constructor <- switch(case[2L], rw1 = temporal_rw1, rw2 = temporal_rw2, ar1 = temporal_ar1)
    time_margin <- if ("cyclic" %in% case) constructor("time", cyclic = TRUE) else constructor("time")
    main <- spatial_car(adjacency, group_var = "unit")
    st <- if (case[1L] == "HSGP") {
      spatiotemporal(spatial_hsgp(~ lon + lat, m = 3L), time_margin, type = "IV",
        parameterization = case[3L])
    } else if (case[1L] %in% c("separable", "nonsep_gp")) {
      spatiotemporal(main, time_margin, type = case[1L],
        parameterization = case[3L], coords = ~ lon + lat, nn = 4L,
        nonsep_type = if (case[1L] == "nonsep_gp") "gneiting" else "product")
    } else spatiotemporal(main, time_margin, type = case[1L], parameterization = case[3L],
      shared = !paired || case[4L] == "shared")
    args <- list(formula = if (paired) y | denom ~ x else y | n ~ x, data = d,
      family = if (paired) ratiod_poisson(denom_family = "gamma") else ratiod_binomial(),
      spatial = main, spatiotemporal = st, mode = "hmc", control = list(iter = 6000L, warmup = 2000L,
        chains = 1L, cores = 1L, seed = seed, verbose = FALSE,
        adapt_delta = .99, gradient_mode = "A_r"))
    fit <- bridge19_pair(args, name, seed)
    b <- as.matrix(fit$draws)
    expected_beta <- if (paired) c(beta, 1, .2) else beta
    expect_true(all(abs(colMeans(b)[seq_along(expected_beta)] - expected_beta) < .2), info = paste(name, seed))
    prediction <- fitted(fit, summary = FALSE)
    recovered <- colMeans(prediction$numerator)
    bridge19_record_recovery(fit, max(abs(colMeans(b)[seq_along(expected_beta)] - expected_beta)),
      if (paired) sqrt(mean(((recovered - truth) / truth)^2)) else max(abs(recovered - truth)),
      if (paired) sqrt(mean(((colMeans(prediction$denominator) - denominator) / denominator)^2)) else NA_real_,
      scale = if (paired) "relative_rmse" else "probability")
    expect_lt(if (paired) sqrt(mean(((recovered - truth) / truth)^2)) else max(abs(recovered - truth)),
      if (paired) .3 else .1, label = paste(name, seed, "recovery"))
    if (paired) expect_lt(sqrt(mean(((colMeans(prediction$denominator) - denominator) / denominator)^2)),
      .3, label = paste(name, seed, "denominator recovery"))
  }
})
