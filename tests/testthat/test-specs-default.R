test_that("every fitted HMC structure enters the engine", {
  set.seed(176)
  d<-data.frame(y=rbinom(48,10,.6),n=10L,x=rnorm(48),g=factor(rep(1:4,12)),time=rep(1:6,8))
  calls<-0L
  trace("cpp_tulpaRatio_sample",tracer=function() calls<<-calls+1L,
        print=FALSE,where=asNamespace("tulpaRatio"))
  on.exit(untrace("cpp_tulpaRatio_sample",where=asNamespace("tulpaRatio")),add=TRUE)
  for(case in list(list(),list(formula=y|n~x+(1|g)),
      list(temporal=temporal_rw1("time")),
      list(temporal=temporal_tvc("time",~x,structure="ar1")))) {
    before<-calls
    args<-modifyList(list(formula=y|n~x,data=d,family=ratiod_binomial(),mode="hmc",
      control=list(iter=30L,warmup=15L,chains=1L,seed=176L,cores=1L,verbose=FALSE)),case)
    fit<-suppressWarnings(do.call(tratio,args))
    expect_identical(calls,before+1L)
    expect_identical(fit$backend,"hmc")
    expect_equal(dim(fit$draws)[1],15L)
    expect_true(all(is.finite(as.matrix(fit$draws))))
  }
})
