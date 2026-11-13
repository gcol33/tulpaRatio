test_that("seeded chains retain their draws after other gradient families run", {
  skip_on_cran()
  fields <- c("none", "gp", "gp_temporal", "icar", "bym2",
              "svc", "svc_hsgp", "tvc", "latent", "hsgp", "ms_temporal",
              "temporal_gp", "st4", "gp_tgp", "icar_rsr")
  for (mode in c("N", "A_r", "H")) {
    first <- tulpaRatio:::cpp_gradient_chain_fixture("none", mode)
    for (field in fields) {
      tulpaRatio:::cpp_gradient_check(field, n_obs = 20L, n_units = 4L,
                                      n_times = 4L, mode = "H")
    }
    replay <- tulpaRatio:::cpp_gradient_chain_fixture("none", mode)
    expect_identical(replay, first, info = mode)
  }
})
