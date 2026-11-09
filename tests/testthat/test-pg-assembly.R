test_that("PG assembly preserves each ordered draw block and chain", {
  chain <- list(beta = matrix(1:6, 3, 2), re = matrix(7:12, 3, 2),
                eta = matrix(13:24, 3, 4), sigma_re = 1:3, tau = 4:6,
                sigma2_gp = 7:9, phi_gp = 10:12, sigma2_trend = 13:15,
                sigma2_seasonal = 16:18, sigma2_short = 19:21, rho_short = 22:24,
                r = 25:27, beta_num = matrix(28:33, 3, 2),
                beta_denom = matrix(34:36, 3, 1), r_num = 37:39, r_denom = 40:42)
  X <- matrix(1, 4, 2, dimnames = list(NULL, c("(Intercept)", "x")))
  beta <- tulpaRatio:::pg_beta_block(X)
  layouts <- list(
    c(beta, sigma_re = "sigma_re", tau_spatial = "tau"),
    c(beta, sigma2_gp = "sigma2_gp", phi_gp = "phi_gp", sigma_re = "sigma_re"),
    c(beta, sigma2_trend = "sigma2_trend", sigma2_seasonal = "sigma2_seasonal",
      sigma2_short = "sigma2_short", rho_short = "rho_short"),
    c(beta, tau_spatial = "tau"),
    c(beta, r = "r", sigma_re = "sigma_re"),
    c(num_a = "beta_num", num_b = "beta_num", denom_a = "beta_denom",
      r_num = "r_num", r_denom = "r_denom"),
    c(phi_num = "r_num", phi_denom = "r_denom"))
  for (blocks in layouts) {
    f <- tulpaRatio:::pg_assemble_fit(list(chain, chain), list(), data.frame(),
      list(), 8, 2, 2, blocks, internal_extra = list(X = X))
    expect_named(dimnames(f$draws), c("iteration", "chain", "variable"))
    expect_identical(dimnames(f$draws)[[3]], names(blocks))
    for (field in unique(unname(blocks))) {
      cols <- which(blocks == field)
      expect_equal(as.vector(f$draws[, 1, cols]), as.vector(chain[[field]]))
      expect_equal(as.vector(f$draws[, 2, cols]), as.vector(chain[[field]]))
    }
    expect_equal(f$.internal$beta, rbind(chain$beta, chain$beta))
    expect_equal(f$.internal$eta_array[, 2, ], chain$eta)
  }
})

test_that("PG chain runner preserves seeds and the caller RNG without a seed", {
  run <- function(i) rnorm(4)
  expected <- lapply(1:3, function(i) {set.seed(20 + i - 1); run(i)})
  expect_identical(tulpaRatio:::pg_run_chains(3, 20, FALSE, run), expected)
  set.seed(3)
  expected <- lapply(1:2, run)
  set.seed(3)
  expect_identical(tulpaRatio:::pg_run_chains(2, NULL, FALSE, run), expected)
})
