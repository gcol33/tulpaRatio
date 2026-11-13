test_that("collapsed unit and location margins retain constants beyond eta twenty", {
  for (e in c(-40, -.35, 0, 20, 21, 30, 40)) {
    for (family in 0:3) {
      y <- 3L; yd <- if (family == 0L) 8 else 4
      pn <- 2.4; pd <- 1.7; ed <- .2
      mu <- exp(e); md <- exp(ed)
      if (family == 0L) {
        p <- plogis(e); q <- plogis(-e)
        ll <- if (e > 0) dbinom(yd-y, yd, q, log=TRUE) else dbinom(y, yd, p, log=TRUE)
        score <- y-yd*p
        curvature <- yd*p*q
      } else {
        ll <- if (family == 2L) dpois(y, mu, log=TRUE) else dnbinom(y, size=pn, mu=mu, log=TRUE)
        score <- if (family == 2L) y-mu else y-mu*(y+pn)/(mu+pn)
        curvature <- if (family == 2L) mu else mu/(mu+pn)*pn/(mu+pn)*(y+pn)
        if (family == 1L) {
          ll <- ll + dnbinom(yd,size=pd,mu=md,log=TRUE)
          score <- score + yd-md*(yd+pd)/(md+pd)
          curvature <- curvature + md/(md+pd)*pd/(md+pd)*(yd+pd)
        } else {
          ll <- ll + dgamma(yd,shape=pd,rate=pd/md,log=TRUE)
          score <- score + pd*(yd/md-1)
          curvature <- curvature + pd*yd/md
        }
      }
      z <- tulpaRatio:::cpp_test_collapsed_margin(family,y,yd,e,ed,pn,pd,TRUE)
      expect_equal(unname(z[1,]), c(ll,score,curvature), tolerance=1e-12)
      expect_equal(unname(z[2,]), c(ll,score,curvature), tolerance=1e-12)
      h <- 1e-5
      plus <- tulpaRatio:::cpp_test_collapsed_margin(family,y,yd,e+h,ed+h,pn,pd,TRUE)
      minus <- tulpaRatio:::cpp_test_collapsed_margin(family,y,yd,e-h,ed-h,pn,pd,TRUE)
      expect_equal((plus[,1]-minus[,1])/(2*h), rep(score,2), tolerance=1e-8)
      expect_equal((minus[,2]-plus[,2])/(2*h), rep(curvature,2), tolerance=1e-5)
    }
  }
})

test_that("an unshared GP uses only the numerator in its inner likelihood", {
  for (family in 1:3) {
    z <- tulpaRatio:::cpp_test_collapsed_margin(family,3L,4,.4,.2,2.4,1.7,FALSE)
    mu <- exp(.4)
    ll <- if (family == 2L) dpois(3,mu,log=TRUE) else dnbinom(3,size=2.4,mu=mu,log=TRUE)
    expect_equal(z[2,1],ll,tolerance=1e-12)
  }
})

test_that("sampled ICAR posteriors match dense integration", {
  skip_on_cran()
  set.seed(224)
  n <- rep(250L,2)
  y <- rbinom(2,n,plogis(.4+c(.35,-.35)))
  d <- data.frame(y=y,n=n,site=1:2)
  beta <- seq(-.6,1.4,length.out=401)
  field <- seq(-2,2,length.out=801)
  sigma <- seq(.0025,3,by=.005)
  ll <- outer(beta,field,function(b,f)
    dbinom(y[1],n[1],plogis(b+f),log=TRUE) +
    dbinom(y[2],n[2],plogis(b-f),log=TRUE))
  likelihood <- exp(ll-max(ll))
  conditional <- outer(field,sigma,function(f,s) dnorm(f,0,s/2))
  joint <- (likelihood %*% conditional) * dnorm(beta,0,2.5)
  joint <- sweep(joint,2,dexp(sigma,rate=-log(.01)),'*')
  beta_weights <- rowSums(joint); beta_weights <- beta_weights/sum(beta_weights)
  ref_mean <- sum(beta*beta_weights)
  ref_sd <- sqrt(sum((beta-ref_mean)^2*beta_weights))
  ref_q <- approx(cumsum(beta_weights),beta,c(.1,.5,.9),ties='ordered')$y
  adj <- matrix(c(0,1,1,0),2)
  fit <- tratio(y | n ~ 1,data=d,family=ratiod_binomial(),
    spatial=spatial_car(adj,spatial_level='group',group_var='site',
                        parameterization='standard'), mode='hmc',
    control=list(iter=5000L,warmup=1500L,chains=1L,seed=224L,
                 gradient_mode='H',adapt_delta=.99))
  draws <- as.matrix(fit$draws)[,1L]
  expect_lt(abs(mean(draws)-ref_mean)/ref_sd,.15)
  expect_lt(max(abs(quantile(draws,c(.1,.5,.9))-ref_q))/ref_sd,.2)
})
