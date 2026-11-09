test_that("transformed priors include their coordinate Jacobians", {
  lambda <- -log(.01)/1.7
  for (x in c(-70,-12,-.4,0,.6,12)) for (prior in c('gamma','half_cauchy','pc_variance','pc_precision')) {
    expected <- switch(prior,
      gamma=c(2.4*x-1.7*exp(x),2.4-1.7*exp(x)),
      half_cauchy=c(-log1p((exp(x)/1.7)^2)+x,1-2*(exp(x)/1.7)^2/(1+(exp(x)/1.7)^2)),
      pc_variance=c(log(lambda)-lambda*exp(x/2)+log(.5)+x/2,.5*(1-lambda*exp(x/2))),
      pc_precision=c(log(lambda)-lambda*exp(-x/2)+log(.5)-x/2,-.5*(1-lambda*exp(-x/2))))
    z <- tulpaRatio:::cpp_test_ratio_prior(prior,x)
    expect_equal(unname(z[,1]),rep(expected[1],3),tolerance=1e-14,info=prior)
    expect_equal(unname(z[,2]),rep(expected[2],3),tolerance=1e-14,info=prior)
  }
})

test_that("softplus and log probabilities retain both tails", {
  softplus <- function(x) if (x>0) x+log1p(exp(-x)) else log1p(exp(x))
  for (x in c(-100,-35.01,-10.01,0,10.01,35.01,100)) {
    expected <- list(softplus=c(softplus(x),plogis(x)),
      log_inv_logit=c(-softplus(-x),plogis(-x)),
      log1m_inv_logit=c(-softplus(x),-plogis(x)))
    for (name in names(expected)) {
      z <- tulpaRatio:::cpp_test_ratio_prior(name,x)
      expect_equal(unname(z[,1]),rep(expected[[name]][1],3),tolerance=0,info=name)
      expect_equal(unname(z[,2]),rep(expected[[name]][2],3),tolerance=1e-15,info=name)
    }
  }
})
