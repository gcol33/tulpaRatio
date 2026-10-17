# An areal adjacency with row names is matched to the grouping variable by
# name; without, its rows follow the sorted levels of the grouping variable.

alignment_data <- function() {
  set.seed(11)
  ids <- as.character(1:12)
  adj <- matrix(0, 12, 12, dimnames = list(ids, ids))
  for (i in 1:11) adj[i, i + 1] <- adj[i + 1, i] <- 1
  field <- sin(seq_len(12) / 2)
  d <- data.frame(region = rep(ids, each = 6))
  d$x <- rnorm(nrow(d))
  d$total <- rpois(nrow(d), 40) + 1L
  d$count <- rbinom(nrow(d), d$total,
                    stats::plogis(-0.5 + 0.4 * d$x + field[match(d$region, ids)]))
  list(data = d, adj = adj)
}

test_that("observations index the rows their names give", {
  adj <- matrix(0, 3, 3, dimnames = list(c("c", "a", "b"), c("c", "a", "b")))
  expect_identical(areal_unit_index(adj, c("a", "b", "c", "a")), c(2L, 3L, 1L, 2L))
  expect_identical(areal_unit_index(unname(adj), c("a", "b", "c", "a")), c(1L, 2L, 3L, 1L))
  expect_identical(areal_unit_labels(unname(adj), c("10", "2", "1")), c("1", "10", "2"))
})

test_that("a named adjacency fits the same model whatever its row order", {
  skip_on_cran()
  a <- alignment_data()
  perm <- c(7, 3, 12, 1, 9, 5, 11, 2, 8, 4, 10, 6)
  fit_with <- function(adj) {
    tratio(count | total ~ x, data = a$data, family = ratiod_binomial(),
           spatial = spatial_car(adj, level = "group", group_var = "region"),
           control = list(iter = 600, chains = 1, seed = 3))
  }
  sorted <- fit_with(a$adj)
  shuffled <- fit_with(a$adj[perm, perm])
  expect_equal(coef(shuffled), coef(sorted), tolerance = 0.05)
  # Character ids "1".."12" sort as "1", "10", "11", "12", "2", ...; the names
  # keep each region on its own neighbours.
  expect_equal(unname(coef(sorted)["num:x"]), 0.4, tolerance = 0.15)

  # Prediction reads the units by name too
  nd <- data.frame(region = c("10", "2", "7"), x = 0, total = 100L)
  ratio_mean <- function(fit) {
    p <- as.data.frame(predict(fit, newdata = nd))
    p$mean[p$component == "ratio"]
  }
  p_sorted <- ratio_mean(sorted)
  expect_length(p_sorted, 3L)
  expect_equal(ratio_mean(shuffled), p_sorted, tolerance = 0.05)
  field <- sin(c(10, 2, 7) / 2)
  expect_identical(order(p_sorted), order(field))
})

test_that("a group the adjacency does not name is refused", {
  a <- alignment_data()
  adj <- a$adj
  rownames(adj)[1] <- colnames(adj)[1] <- "z"
  expect_error(
    tratio(count | total ~ x, data = a$data, family = ratiod_binomial(),
           spatial = spatial_car(adj, level = "group", group_var = "region"),
           control = list(iter = 50, chains = 1, seed = 1)),
    "not among the adjacency's row names")
  adj <- a$adj
  colnames(adj) <- rev(colnames(adj))
  expect_error(
    tratio(count | total ~ x, data = a$data, family = ratiod_binomial(),
           spatial = spatial_car(adj, level = "group", group_var = "region"),
           control = list(iter = 50, chains = 1, seed = 1)),
    "same units in the same order")
})
