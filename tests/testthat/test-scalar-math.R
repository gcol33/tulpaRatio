test_that("scalar guards share values and finite autodiff derivatives", {
  cases <- list(
    list('exp',710,exp(700),exp(700)),
    list('exp',-710,exp(-700),exp(-700)),
    list('log',0,-1e10,0), list('log',-1,-1e10,0),
    list('log',1e-20,log(1e-20),1/1e-15),
    list('sqrt',0,0,0), list('sqrt',-1,0,0),
    list('pow',0,0,0), list('pow_dual',0,0,0), list('expm1',1e-16,expm1(1e-16),exp(1e-16)),
    list('lift',0,3,7))
  for (case in cases) {
    z <- tulpaRatio:::cpp_test_scalar_math(case[[1]],case[[2]])
    expect_equal(unname(z[,'value']),rep(case[[3]],3), tolerance=0)
    expect_equal(unname(z[c('forward','arena'),'derivative']),rep(case[[4]],2), tolerance=0)
  }
})

test_that("logit endpoints and empty log sums have shared finite pullbacks", {
  for (p in c(0,1,-1,2)) {
    z <- tulpaRatio:::cpp_test_scalar_boundaries(p)
    clipped <- min(max(p,1e-15),1-1e-15)
    expect_equal(unname(z[,1]),rep(log(clipped/(1-clipped)),3),tolerance=0)
    expect_equal(unname(z[2:3,2]),rep(1/(clipped*(1-clipped)),2),tolerance=0)
    expect_identical(unname(z[,3]),rep(-Inf,3))
    expect_identical(unname(z[2:3,4]),c(0,0))
  }
})
