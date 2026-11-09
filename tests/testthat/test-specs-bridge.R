bridge_fixture <- function() {
  set.seed(176)
  d <- expand.grid(g=1:12,h=1:8,rep=1:5)
  d$x <- rnorm(nrow(d))
  g <- rnorm(12,0,.4); g <- g-mean(g)
  h <- rnorm(8,0,.3); h <- h-mean(h)
  slope <- .35*g+rnorm(12,0,.2); slope <- slope-mean(slope)
  d$trials <- 12L
  d$y <- rbinom(nrow(d),12,plogis(.3+.6*d$x+g[d$g]+h[d$h]))
  d$ys <- rbinom(nrow(d),12,plogis(.3+.6*d$x+g[d$g]+slope[d$g]*d$x))
  d$exposure <- runif(nrow(d),.5,4)
  d$yp <- rpois(nrow(d),d$exposure*exp(.3+.6*d$x))
  d$yn <- rnbinom(nrow(d),mu=d$exposure*exp(.3+.6*d$x),size=4)
  d$yl <- exp(.3+.6*d$x+log(d$exposure)+rnorm(nrow(d),0,.4))
  d$yc <- rpois(nrow(d),exp(.3+.6*d$x))
  d$yd <- rnbinom(nrow(d),mu=exp(.4+.2*d$x),size=4)
  d
}

bridge_pair <- function(family,formula,d) {
  calls <- 0L
  trace('cpp_tulpaRatio_run_nuts_specs',where=asNamespace('tulpaRatio'),
    tracer=function() calls <<- calls+1L,print=FALSE)
  on.exit(untrace('cpp_tulpaRatio_run_nuts_specs',where=asNamespace('tulpaRatio')))
  legacy <- bridge_fit(family,formula,d,FALSE)
  spec <- bridge_fit(family,formula,d,TRUE)
  expect_equal(calls,1L)
  list(legacy=legacy,spec=spec)
}

bridge_fit <- function(family, formula, d, specs, chains=2L) {
  op <- options(tulpaRatio.use_specs=specs); on.exit(options(op))
  tratio(formula,data=d,family=family,mode='hmc',control=list(
    iter=2000L,warmup=750L,chains=chains,seed=176L,verbose=FALSE,gradient_mode='A_r'))
}

test_that("correlated and crossed effects reach the engine and recover fixed effects", {
  skip_on_cran()
  d <- bridge_fixture()
  for (formula in list(y | trials ~ x+(1|g)+(1|h),ys | trials ~ x+(1+x|g))) {
    pair <- bridge_pair(ratiod_binomial(),formula,d)
    legacy <- pair$legacy; spec <- pair$spec
    expect_length(spec$samples,2L)
    a <- as.matrix(legacy$draws); b <- as.matrix(spec$draws)
    expect_identical(colnames(a),colnames(b))
    expect_lt(max(abs(colMeans(a)-colMeans(b))/apply(a,2,sd)),.4)
    expect_true(all(abs(colMeans(b)[1:2]-c(.3,.6)) < c(.2,.2)))
  }
})

test_that("exposures and their inflation variants reach the spec bridge", {
  skip_on_cran()
  d <- bridge_fixture()
  fixtures <- list(
    list(f=yp | exposure ~ x,family=ratiod_poisson(denom_family='fixed')),
    list(f=yn | exposure ~ x,family=ratiod_negbin(denom_family='fixed')),
    list(f=yl | exposure ~ x,family=ratiod_lognormal(denom_fixed=TRUE)),
    list(f=yc | yd ~ x,family=ratiod_poisson(denom_family='negbin')))
  for (fixture in fixtures) {
    pair <- bridge_pair(fixture$family,fixture$f,d)
    legacy <- pair$legacy; spec <- pair$spec
    a <- as.matrix(legacy$draws); b <- as.matrix(spec$draws)
    expect_lt(max(abs(colMeans(a)-colMeans(b))/apply(a,2,sd)),.25)
    expect_true(all(abs(colMeans(b)[1:2]-c(.3,.6)) < .12))
    if (fixture$family$denominator$distribution=='fixed') {
      expect_equal(fitted(spec,summary=FALSE)$denominator[1,],d$exposure)
    } else {
      expect_true(all(abs(colMeans(b)[3:4]-c(.4,.2)) < .15))
    }
  }
  set.seed(177)
  d$yp[runif(nrow(d))<.3] <- 0L
  for (family in list(ratiod_zi_poisson(denom_family='fixed'),
                      ratiod_hurdle_poisson(denom_family='fixed'))) {
    if (grepl('hurdle',family$name)) {
      d$yp <- ifelse(d$yp==0,0L,pmax(1L,d$yp))
    }
    pair <- bridge_pair(family,yp | exposure ~ x,d)
    legacy <- pair$legacy; spec <- pair$spec
    a <- as.matrix(legacy$draws); b <- as.matrix(spec$draws)
    expect_lt(max(abs(colMeans(a)-colMeans(b))/apply(a,2,sd)),.3)
  }
})
