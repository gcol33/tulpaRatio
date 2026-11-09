tvc_plot_fixture <- function() {
  structure(list(draws = array(seq_len(120) / 10, c(10, 6, 2)),
                 term_names = c("a", "b"), time_levels = 2011:2016,
                 n_times = 6), class = "ratiod_tvc_posterior")
}

test_that("TVC plot type selects ribbon or line geometry", {
  skip_if_not_installed("ggplot2")
  object <- tvc_plot_fixture()
  ribbon <- plot(object, term = "b", type = "ribbon")
  line <- plot(object, term = "b", type = "line")
  geoms <- function(p) vapply(p$layers, function(layer) class(layer$geom)[1], "")
  expect_true("GeomRibbon" %in% geoms(ribbon))
  expect_false("GeomRibbon" %in% geoms(line))
  expect_true("GeomLine" %in% geoms(line))
  expect_identical(ribbon$data, line$data)
  expect_equal(line$data$mean, unname(colMeans(object$draws[, , 2])))
  expect_equal(ribbon$data$lower,
               unname(apply(object$draws[, , 2], 2, quantile, probs = 0.025)))
  expect_equal(ribbon$data$upper,
               unname(apply(object$draws[, , 2], 2, quantile, probs = 0.975)))
  expect_error(plot(object, type = "invalid"), "arg.*ribbon.*line")
})

test_that("TVC base graphics selects the same interval and mean", {
  plotter <- getS3method("plot", "ratiod_tvc_posterior")
  env <- new.env(parent = environment(plotter))
  environment(plotter) <- env
  env$requireNamespace <- function(...) FALSE
  env$plot <- function(...) invisible(NULL)
  env$abline <- function(...) invisible(NULL)
  polygons <- list()
  curves <- list()
  env$polygon <- function(x, y, ...) polygons[[length(polygons) + 1L]] <<- list(x, y)
  env$lines <- function(x, y, ...) curves[[length(curves) + 1L]] <<- list(x, y)
  object <- tvc_plot_fixture()
  plotter(object, type = "line")
  expect_length(polygons, 0)
  plotter(object, type = "ribbon")
  expect_length(polygons, 1)
  expect_identical(curves[[1]], curves[[2]])
  expect_equal(polygons[[1]][[2]],
               c(apply(object$draws[, , 1], 2, quantile, probs = 0.025),
                 rev(apply(object$draws[, , 1], 2, quantile, probs = 0.975))),
               ignore_attr = TRUE)
})
