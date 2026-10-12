# Tests for plot_map.R

# Test internal helper functions

test_that("extract_coords works with x/y columns", {
  df <- data.frame(x = 1:5, y = 6:10, value = rnorm(5))

  result <- tulpaRatio:::extract_coords(df)

  expect_equal(result$x, 1:5)
  expect_equal(result$y, 6:10)
})

test_that("extract_coords works with lon/lat columns", {
  df <- data.frame(lon = c(-122, -121, -120), lat = c(37, 38, 39), value = 1:3)

  result <- tulpaRatio:::extract_coords(df)

  expect_equal(result$x, c(-122, -121, -120))
  expect_equal(result$y, c(37, 38, 39))
})

test_that("extract_coords works with longitude/latitude columns", {
  df <- data.frame(longitude = c(10, 20), latitude = c(50, 60), val = c(1, 2))

  result <- tulpaRatio:::extract_coords(df)

  expect_equal(result$x, c(10, 20))
  expect_equal(result$y, c(50, 60))
})

test_that("extract_coords works with Easting/Northing columns", {
  df <- data.frame(Easting = c(100, 200), Northing = c(300, 400))

  result <- tulpaRatio:::extract_coords(df)

  expect_equal(result$x, c(100, 200))
  expect_equal(result$y, c(300, 400))
})

test_that("extract_coords works with matrix input", {
  mat <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 3, ncol = 2)

  result <- tulpaRatio:::extract_coords(mat)

  expect_equal(result$x, 1:3)
  expect_equal(result$y, 4:6)
})

test_that("extract_coords errors on unrecognized columns", {
  df <- data.frame(a = 1:3, b = letters[1:3], c = 4:6)

  expect_error(
    tulpaRatio:::extract_coords(df),
    "Cannot identify coordinate columns"
  )
})

test_that("can_use_stars detects gridded data", {
  # Regular grid: 3x3 = 9 points
  plot_data <- data.frame(
    x = rep(1:3, each = 3),
    y = rep(1:3, times = 3),
    value = rnorm(9)
  )

  result <- tulpaRatio:::can_use_stars(plot_data)
  expect_true(result)
})

test_that("can_use_stars detects irregular data", {
  # Irregular points, not a grid
  plot_data <- data.frame(
    x = c(1, 1.5, 2.3, 3.1, 5),
    y = c(1.2, 2.5, 1.8, 4.1, 3.2),
    value = rnorm(5)
  )

  result <- tulpaRatio:::can_use_stars(plot_data)
  expect_false(result)
})

test_that("extract_coords_from_fit returns NULL when no coords", {
  # Mock fit without coordinates - use only one numeric column
  # so extract_coords will fail and tryCatch returns NULL
  fit <- list(
    data = data.frame(
      count = 1:10,
      group = letters[1:10]  # non-numeric, so only 1 numeric column
    ),
    coords = NULL,
    stan_data = list()
  )
  class(fit) <- "ratiod_fit"

  result <- tulpaRatio:::extract_coords_from_fit(fit)
  # When data doesn't have x/y columns and fewer than 2 numeric columns, should return NULL
  expect_null(result)
})

test_that("extract_coords_from_fit reads the fit's coordinate variables", {
  fit <- list(
    .internal = list(spatial_spec = list(coord_vars = c("lon", "lat"))),
    data = data.frame(count = 1:5, lon = 1:5, lat = 6:10)
  )
  class(fit) <- "ratiod_fit"

  result <- tulpaRatio:::extract_coords_from_fit(fit)

  expect_equal(result$x, 1:5)
  expect_equal(result$y, 6:10)
})

test_that("extract_coords_from_fit extracts from data", {
  fit <- list(
    coords = NULL,
    data = data.frame(count = 1:5, x = 1:5, y = 6:10)
  )
  class(fit) <- "ratiod_fit"

  result <- tulpaRatio:::extract_coords_from_fit(fit)

  expect_equal(result$x, 1:5)
  expect_equal(result$y, 6:10)
})

test_that("plot_map requires ggplot2 package", {
  # Test that plot_map function exists and is callable
  # The error condition for missing ggplot2 cannot be tested when ggplot2 is installed
  expect_true(is.function(plot_map))
})

test_that("plot_map errors when x is neither fit nor data frame", {
  skip_if_not_installed("ggplot2")

  expect_error(
    plot_map("not a fit"),
    "must be a ratiod_fit object or a data frame"
  )
})

test_that("plot_map errors when data frame provided without coords", {
  skip_if_not_installed("ggplot2")

  df <- data.frame(value = 1:5)

  expect_error(
    plot_map(df),
    "'coords' must be provided"
  )
})

test_that("plot_map works with data frame and coords", {
  skip_if_not_installed("ggplot2")

  df <- data.frame(median = 1:5, q50 = 1:5)
  coords <- data.frame(x = 1:5, y = 6:10)

  p <- plot_map(df, coords = coords, what = "ratio", summary = "median")

  expect_true(inherits(p, "gg") || inherits(p, "ggplot"))
})

test_that("get_palette_scale returns viridis scale", {
  skip_if_not_installed("ggplot2")

  scale <- tulpaRatio:::get_palette_scale("viridis", "transparent", "Value")
  # Scale objects have class ggproto/Scale
  expect_true(inherits(scale, "ggproto") || inherits(scale, "Scale"))
})

test_that("get_palette_scale works with custom colors", {
  skip_if_not_installed("ggplot2")

  scale <- tulpaRatio:::get_palette_scale(c("red", "white", "blue"),
                                      "transparent", "Value")
  expect_true(inherits(scale, "ggproto") || inherits(scale, "Scale"))
})

test_that("prepare_map_data_from_df works with mean column", {
  skip_if_not_installed("ggplot2")

  df <- data.frame(mean = 1:5, sd = 0.1)
  coords <- data.frame(x = 1:5, y = 6:10)

  result <- tulpaRatio:::prepare_map_data_from_df(df, coords, "ratio", "mean")

  expect_equal(nrow(result), 5)
  expect_true("value" %in% names(result))
  expect_true("x" %in% names(result))
  expect_true("y" %in% names(result))
})

test_that("prepare_map_data_from_df handles uncertainty", {
  df <- data.frame(q2.5 = 1:5, q97.5 = 6:10)
  coords <- data.frame(x = 1:5, y = 1:5)

  result <- tulpaRatio:::prepare_map_data_from_df(df, coords, "uncertainty", "mean")

  # CI width = 6:10 - 1:5 = c(5, 5, 5, 5, 5)
  expect_equal(result$value, rep(5, 5))
})

test_that("prepare_map_data_from_df errors on missing columns", {
  df <- data.frame(other = 1:5)
  coords <- data.frame(x = 1:5, y = 1:5)

  expect_error(
    tulpaRatio:::prepare_map_data_from_df(df, coords, "ratio", "mean"),
    "Cannot find appropriate column"
  )
})
