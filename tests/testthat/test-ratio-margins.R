test_that("ratio margins match normalized densities and their derivatives", {
  densities <- list(
    poisson=function(y,n,e,p) dpois(y,exp(e),log=TRUE),
    negbin=function(y,n,e,p) dnbinom(y,size=p,mu=exp(e),log=TRUE),
    gamma=function(y,n,e,p) dgamma(y,shape=p,rate=p/exp(e),log=TRUE),
    lognormal=function(y,n,e,p) dlnorm(y,e,p,log=TRUE) + log(2*pi)/2,
    binomial=function(y,n,e,p) dbinom(y,n,plogis(e),log=TRUE),
    beta_binomial=function(y,n,e,p) lchoose(n,y) +
      lbeta(y+plogis(e)*p,n-y+plogis(-e)*p) - lbeta(plogis(e)*p,plogis(-e)*p))
  h <- 1e-5
  for (family in names(densities)) for (cached in c(FALSE,TRUE)) {
    f <- densities[[family]]
    y <- 3; n <- 8L; e <- -0.35; p <- 2.4
    r <- tulpaRatio:::cpp_test_ratio_margin(family,y,n,e,p,cached)
    expect_equal(unname(r['ll']), f(y,n,e,p), tolerance=1e-12, info=family)
    expect_equal(unname(r['residual']), (f(y,n,e+h,p)-f(y,n,e-h,p))/(2*h),
                 tolerance=1e-8,info=family)
    extra <- (f(y,n,e,p+h)-f(y,n,e,p-h))/(2*h)
    if (family == 'lognormal') extra <- extra*p
    if (family %in% c('poisson','binomial')) extra <- 0
    expect_equal(unname(r['extra']),extra,tolerance=1e-8,info=family)
  }
})

test_that("residual kernels accumulate every margin's extra score", {
  z <- tulpaRatio:::cpp_test_residual_accumulation()
  pairs <- list(c('binomial','none'),c('poisson','gamma'),c('negbin','gamma'),
                c('negbin','negbin'),c('gamma','gamma'),c('lognormal','lognormal'),
                c('beta_binomial','none'))
  for (i in seq_along(pairs)) {
    expected <- c(5,-7,13)
    for (k in 1:2) {
      family <- pairs[[i]][k]
      if (family == 'none') next
      p <- c(2.4,1.7)[k]
      r <- tulpaRatio:::cpp_test_ratio_margin(family,c(3,8)[k],8L,c(-.35,.2)[k],p,TRUE)
      score <- r['extra']
      if (family == 'lognormal') score <- score/p
      expected[k] <- expected[k] + score
      expected[3] <- expected[3] + r['ll']
    }
    expect_equal(unname(z[i,]),unname(expected),tolerance=1e-14)
  }
})
