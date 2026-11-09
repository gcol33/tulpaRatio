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
