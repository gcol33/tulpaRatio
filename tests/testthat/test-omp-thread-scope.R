# The OpenMP thread count is process-wide. A backend that takes a per-fit count
# has to put the previous value back, or it is choosing the width of every
# region that runs after it, in fits it knows nothing about (#8, #16).

skip_if_no_omp <- function() {
  skip_if(tulpaRatio:::cpp_num_procs() < 2, "needs more than one core")
}

fixture <- function() {
  set.seed(1)
  m <- 300L
  list(
    y = as.numeric(rbinom(m, 20, 0.4)),
    n = rep(20, m),
    X = cbind(1, rnorm(m)),
    group = as.integer(rep(0:9, length.out = m))
  )
}

pg_binomial_fit <- function(d, n_threads) {
  invisible(tulpaRatio:::cpp_pg_binomial_gibbs(
    y = d$y, n = d$n, X = d$X, group = d$group, n_groups = 10L,
    n_iter = 60L, n_warmup = 30L, thin = 1L, verbose = FALSE,
    n_threads = as.integer(n_threads)
  ))
}

pg_negbin_fit <- function(d, n_threads) {
  invisible(tulpaRatio:::cpp_pg_negbin_gibbs(
    y = d$y, X = d$X, group = d$group, n_groups = 10L,
    n_iter = 60L, n_warmup = 30L, thin = 1L, prior_beta_sd = 10.0,
    prior_sigma_scale = 2.5, prior_r_shape = 2.0, prior_r_rate = 1.0,
    r_init = 5.0, store_eta = FALSE, verbose = FALSE,
    n_threads = as.integer(n_threads)
  ))
}

test_that("a fit puts the thread count back", {
  skip_if_no_omp()
  d <- fixture()
  before <- tulpaRatio:::cpp_get_max_threads()
  for (backend in list(pg_binomial_fit, pg_negbin_fit)) {
    backend(d, 1L)
    expect_equal(tulpaRatio:::cpp_get_max_threads(), before)
  }
})

test_that("the thread count survives a fit that asks for more than one thread", {
  skip_if_no_omp()
  d <- fixture()
  before <- tulpaRatio:::cpp_get_max_threads()
  expect_equal(tulpaRatio:::cpp_get_max_threads(), before)
  pg_binomial_fit(d, 2L)
  expect_equal(tulpaRatio:::cpp_get_max_threads(), before)
})

test_that("a fit asking for no particular count leaves the value alone", {
  skip_if_no_omp()
  d <- fixture()
  restore <- tulpaRatio:::cpp_get_max_threads()
  on.exit(tulpaRatio:::cpp_set_max_threads(restore), add = TRUE)

  tulpaRatio:::cpp_set_max_threads(3L)
  pg_binomial_fit(d, 0L)
  expect_equal(tulpaRatio:::cpp_get_max_threads(), 3L)
})

test_that("teams resizing across fits do not corrupt the heap", {
  skip_if_no_omp()
  # These backends still size their own regions per fit, so a fit narrower than
  # the one before it shrinks a team. #8 was a shrink of the chain team, whose
  # workers carry thread_local workspaces; nothing here does.
  d <- fixture()
  before <- tulpaRatio:::cpp_get_max_threads()
  for (w in c(8L, 1L, 4L, 2L, 8L, 1L)) {
    pg_binomial_fit(d, w)
  }
  expect_equal(tulpaRatio:::cpp_get_max_threads(), before)
})
