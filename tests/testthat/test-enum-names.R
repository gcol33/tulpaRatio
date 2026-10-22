# The choices R offers and the strings the sampler parses are one list: every
# name R can send is one C++ knows, and a name C++ does not know stops instead
# of reading as a default.

enum_names <- function(kind) tulpaRatio:::cpp_enum_names(kind)

test_that("the control choices are names the sampler parses", {
  ctrl <- tulpaRatio:::.TRATIO_CONTROL
  expect_setequal(ctrl$metric$choices, enum_names("metric"))
  expect_true(all(ctrl$gradient_mode$choices %in% enum_names("gradient_mode")))
})

test_that("structure and covariance choices are names the sampler parses", {
  expect_setequal(eval(formals(spatial_multiscale)$sampler), enum_names("msgp_sampler"))
  expect_setequal(setdiff(eval(formals(temporal_tvc)$structure), "gp"),
                  enum_names("tvc"))
  expect_setequal(eval(formals(spatial_gp)$cov), enum_names("cov"))
  for (spec in list(temporal_rw1("t"), temporal_rw2("t"), temporal_ar1("t"),
                    temporal_iid("t"), temporal_gp("t"))) {
    expect_true(spec$type %in% enum_names("temporal"), label = spec$type)
  }
})

test_that("an unknown name stops rather than reading as a default", {
  expect_error(tulpaRatio:::cpp_gradient_check("rw1", mode = "banana"), "gradient mode")
  expect_error(tulpaRatio:::cpp_gradient_check("rw1", zi = "zi_banana"), "inflation")
  expect_error(tulpaRatio:::cpp_test_temporal_log_prior(rnorm(5), "rw3", 1, 0, FALSE),
               "temporal structure")
  expect_error(tulpaRatio:::cpp_enum_names("banana"), "enum kind")
})
