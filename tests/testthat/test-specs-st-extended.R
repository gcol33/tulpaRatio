test_that("Knorr-Held interaction bridges agree across seeds and recover planted fields", {
  skip_on_cran()
  S <- 3L
  nt <- 4L
  adjacency <- matrix(c(0, 1, 0, 1, 0, 1, 0, 1, 0), S)
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
    st_gp = c("separable", "rw1", "centered"),
    st_gp_nc = c("separable", "rw1", "noncentered"),
    st_gneiting = c("nonsep_gp", "rw1", "noncentered"),
    st_hsgp = c("HSGP", "rw1", "noncentered"),
    st_hsgp_centered = c("HSGP", "rw1", "centered"),
    st_hsgp_ar1 = c("HSGP", "ar1", "noncentered"))
  selected <- Sys.getenv("TULPA_BRIDGE_ROUTE")
  if (nzchar(selected)) {
    wanted <- strsplit(selected, ",", fixed = TRUE)[[1L]]
    stopifnot(all(wanted %in% names(cases)))
    cases <- cases[wanted]
  }
  for (seed in c(18L, 181L, 182L)) for (name in names(cases)) {
    case <- cases[[name]]
    set.seed(seed)
    d <- data.frame(unit = rep(seq_len(S), each = nt * 12L),
      time = rep(rep(seq_len(nt), each = 12L), S), n = 40L)
    d$x <- rnorm(nrow(d))
    d$lon <- c(.1, .5, .9)[d$unit]
    d$lat <- c(.2, -.3, .6)[d$unit]
    field <- c(-.5, 0, .5)[d$unit] * c(-.6, -.2, .2, .6)[d$time]
    truth <- plogis(.2 + .5 * d$x + field)
    d$y <- rbinom(nrow(d), d$n, truth)
    constructor <- switch(case[2L], rw1 = temporal_rw1, rw2 = temporal_rw2, ar1 = temporal_ar1)
    main <- spatial_car(adjacency, group_var = "unit")
    st <- if (case[1L] == "HSGP") {
      spatiotemporal(spatial_hsgp(~ lon + lat, m = 3L), constructor("time"), type = "IV",
        parameterization = case[3L])
    } else if (case[1L] %in% c("separable", "nonsep_gp")) {
      spatiotemporal(main, constructor("time"), type = case[1L],
        parameterization = case[3L], coords = ~ lon + lat, nn = 4L,
        nonsep_type = if (case[1L] == "nonsep_gp") "gneiting" else "product")
    } else spatiotemporal(main, constructor("time"), type = case[1L], parameterization = case[3L])
    args <- list(formula = y | n ~ x, data = d, family = ratiod_binomial(),
      spatial = main, spatiotemporal = st, mode = "hmc", control = list(iter = 6000L, warmup = 2000L,
        chains = 1L, cores = 1L, seed = seed, verbose = FALSE,
        adapt_delta = .99, gradient_mode = "A_r"))
    fit <- bridge19_pair(args, name, seed)
    b <- as.matrix(fit$draws)
    expect_true(all(abs(colMeans(b)[1:2] - c(.2, .5)) < .2), info = paste(name, seed))
    recovered <- colMeans(fitted(fit, summary = FALSE)$numerator)
    expect_lt(max(abs(recovered - truth)), .1, label = paste(name, seed, "recovery"))
  }
})
