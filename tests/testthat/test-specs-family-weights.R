test_that("registered likelihood weights and their arena derivatives match density differences", {
  entries <- cpp_ratio_family_entries()
  for (name in names(entries)) {
    entry <- entries[[name]]
    point <- c(c(-.35, .2)[seq_len(entry$n_processes)], log(c(2.4, 1.7))[seq_len(entry$n_extra)])
    evaluate <- function(q) cpp_test_family_weights(name, q)
    r <- evaluate(point)
    expect_equal(r$values, c(r$score, r$weights), tolerance = 1e-12, info = name)
    h <- 1e-4
    derivative <- vapply(seq_along(point), function(j) {
      plus <- minus <- point
      plus[j] <- plus[j] + h; minus[j] <- minus[j] - h
      (evaluate(plus)$values - evaluate(minus)$values) / (2 * h)
    }, numeric(length(r$values)))
    expect_equal(r$derivatives, derivative, tolerance = 2e-7, info = name)
    eta <- seq_len(entry$n_processes)
    score <- vapply(eta, function(j) {
      plus <- minus <- point
      plus[j] <- plus[j] + h; minus[j] <- minus[j] - h
      (evaluate(plus)$ll - evaluate(minus)$ll) / (2 * h)
    }, numeric(1))
    expect_equal(r$score, score, tolerance = 2e-7, info = name)
    expect_equal(r$derivatives[eta, eta, drop = FALSE],
      -matrix(r$weights, length(eta)), tolerance = 1e-10, info = name)
  }
})
