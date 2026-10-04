# Structural tests for tratio()'s control surface. These exercise the knob
# registry and its validation, not the statistical behaviour of any backend.

test_that("defaults reproduce the documented values", {
  d <- .tratio_control(list())
  expect_equal(d$chains, 4L)
  expect_equal(d$iter, 2000L)
  expect_equal(d$thin, 1L)
  expect_true(d$verbose)
  expect_equal(d$metric, "auto")
  expect_equal(d$gradient_mode, "auto")
  expect_equal(d$re_param, "noncentered")
  expect_equal(d$vi_variant, "auto")
})

test_that("knobs a backend defaults for itself stay unset", {
  d <- .tratio_control(list())
  expect_null(d$seed)
  expect_null(d$L)
  expect_null(d$adapt_delta)
  expect_null(d$batch_size)
})

test_that("warmup and cores derive from iter and chains", {
  expect_equal(.tratio_control(list())$warmup, 1000L)
  expect_equal(.tratio_control(list(iter = 500))$warmup, 250L)
  expect_equal(.tratio_control(list(iter = 500, warmup = 100))$warmup, 100L)
  expect_equal(.tratio_control(list(chains = 2))$cores,
               getOption("mc.cores", 2L))
})

test_that("a misspelled knob errors instead of silently doing nothing", {
  expect_error(.tratio_control(list(itr = 100)), "Unknown control knob")
  expect_error(.tratio_control(list(nope = 1)), "adapt_delta")
  expect_error(.tratio_control(list(100)), "must be named")
  expect_error(.tratio_control(c(iter = 100)), "must be a list")
})

test_that("refresh is gone rather than silently accepted", {
  # It was a signature argument no backend ever read.
  expect_error(.tratio_control(list(refresh = 10)), "Unknown control knob")
})

test_that("enumerated knobs are checked against their choices", {
  expect_equal(.tratio_control(list(metric = "dense"))$metric, "dense")
  expect_equal(.tratio_control(list(gradient_mode = "A_r"))$gradient_mode, "A_r")
  expect_error(.tratio_control(list(metric = "banana")), "should be one of")
})

test_that("scalar knobs are range- and type-checked", {
  expect_equal(.tratio_control(list(adapt_delta = 0.95))$adapt_delta, 0.95)
  expect_error(.tratio_control(list(adapt_delta = 1.5)), "between 0.5 and 0.99")
  expect_false(.tratio_control(list())$walnuts)
  expect_true(.tratio_control(list(walnuts = TRUE))$walnuts)
  expect_error(.tratio_control(list(walnuts = "yes")), "TRUE or FALSE")
  expect_error(.tratio_control(list(chains = 0)), "positive whole number")
  expect_error(.tratio_control(list(iter = 10.5)), "positive whole number")
  expect_error(.tratio_control(list(epsilon = -1)), "positive number")
})

test_that("a warmup that would leave no draws is rejected", {
  expect_error(.tratio_control(list(iter = 100, warmup = 100)), "must be below")
  expect_error(.tratio_control(list(iter = 100, warmup = 150)), "must be below")
})

test_that("stochastic-gradient backends receive only the knobs that were set", {
  ctl <- .tratio_control(list(epsilon = 0.05, alpha = 0.2))
  sg <- .tratio_sg_args(ctl, "sghmc")
  expect_equal(sort(names(sg)), c("alpha", "epsilon"))
  expect_equal(sg$epsilon, 0.05)
  expect_false("L" %in% names(sg))

  # alpha is friction, meaningful to SGHMC only
  expect_false("alpha" %in% names(.tratio_sg_args(ctl, "sgld")))
  expect_true("schedule_a" %in%
                names(.tratio_sg_args(.tratio_control(list(schedule_a = 0.02)),
                                      "sgld")))
  expect_length(.tratio_sg_args(.tratio_control(list()), "sghmc"), 0L)
})

test_that("tratio() rejects tuning knobs passed as bare arguments", {
  df <- data.frame(a = c(2, 3), b = c(5, 7))
  expect_error(tratio(a | b ~ 1, data = df, iter = 10), "unused argument")
})
