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
    scalars <- tulpaRatio:::cpp_test_ratio_margin_scalars(family,y,n,e,p)
    expect_equal(unname(scalars[,1]),rep(f(y,n,e,p),3),tolerance=1e-12,info=family)
    expect_equal(unname(scalars[2:3,2]),rep(unname(r['residual']),2),tolerance=1e-12,info=family)
    scalar_extra <- if (family == 'lognormal') r['extra']/p else r['extra']
    expect_equal(unname(scalars[2:3,3]),rep(unname(scalar_extra),2),tolerance=1e-12,info=family)
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
      expected[k] <- expected[k] + score
      expected[3] <- expected[3] + r['ll']
    }
    expect_equal(unname(z[i,]),unname(expected),tolerance=1e-14)
  }
})

test_that("margin field selection preserves caches and observed curvature", {
  for (family in c('poisson','negbin','gamma','lognormal','binomial','beta_binomial')) {
    for (cached in c(FALSE,TRUE)) for (phi in c(.2,2.4,1e5)) {
      eta <- -.35
      all <- tulpaRatio:::cpp_test_margin_fields(family,eta,phi,15L,cached)
      expect_identical(unname(all[1:3]),unname(
        tulpaRatio:::cpp_test_ratio_margin(family,3.,8L,eta,phi,cached)))
      for (k in 1:4) {
        part <- tulpaRatio:::cpp_test_margin_fields(family,eta,phi,as.integer(2^(k-1L)),cached)
        expect_identical(part[k],all[k])
        expect_identical(part[-k],rep(0.,3))
      }
      h <- 1e-5
      plus <- tulpaRatio:::cpp_test_margin_fields(family,eta+h,phi,2L,cached)[2]
      minus <- tulpaRatio:::cpp_test_margin_fields(family,eta-h,phi,2L,cached)[2]
      expect_equal(all[4],(minus-plus)/(2*h),tolerance=1e-8)
    }
  }
})

test_that("curvature derivatives differentiate the shared margins", {
  for (family in c('poisson','negbin','gamma','lognormal','binomial','beta_binomial')) {
    for (eta in c(-2.,-.35,3.)) for (phi in c(.2,2.4,10.)) {
      d <- tulpaRatio:::cpp_test_margin_curvature(family,eta,phi)
      h <- 1e-5
      en <- tulpaRatio:::cpp_test_margin_curvature(family,eta-h,phi)
      ep <- tulpaRatio:::cpp_test_margin_curvature(family,eta+h,phi)
      pn <- tulpaRatio:::cpp_test_margin_curvature(family,eta,phi-h)
      pp <- tulpaRatio:::cpp_test_margin_curvature(family,eta,phi+h)
      expect_equal(d[2],(ep[1]-en[1])/(2*h),tolerance=1e-7)
      expect_equal(d[3],(pp[1]-pn[1])/(2*h),tolerance=1e-7)
      expect_equal(d[4],(pp[5]-pn[5])/(2*h),tolerance=1e-7)
    }
  }
})

test_that("Gibbs observation kernels read normalized uncapped margins", {
  pairs <- list(c('poisson','gamma'),c('negbin','negbin'),
                c('binomial','none'),c('negbin','gamma'))
  for (family in 0:3) for (eta in c(-40.,-.35,20.,21.,30.,40.)) {
    pair <- pairs[[family+1L]]
    a <- tulpaRatio:::cpp_test_margin_fields(pair[1],eta,2.4,15L,FALSE)
    b <- if (pair[2]=='none') rep(0.,4) else {
      e <- .2; p <- 1.7; y <- if (pair[2]=='negbin') 8. else 4.
      r <- tulpaRatio:::cpp_test_ratio_margin(pair[2],y,8L,e,p,FALSE)
      w <- if (pair[2]=='negbin') exp(e)/(exp(e)+p)*p/(exp(e)+p)*(y+p) else p*y/exp(e)
      c(unname(r),w)
    }
    yd <- if (pair[2] %in% c('none','negbin')) 8. else 4.
    z <- tulpaRatio:::cpp_test_gibbs_margin(family,3L,yd,eta,.2,2.4,1.7)
    expect_equal(unname(z),c(a[1]+b[1],a[2],b[2],a[4],b[4]),tolerance=1e-12)
  }
})
