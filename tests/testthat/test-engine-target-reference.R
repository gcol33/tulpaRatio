test_that("engine gradient modes use the same independently defined target", {
  X<-cbind(1,c(-1,.2,1.4)); y<-c(2L,4L,7L); n<-rep(10L,3)
  beta<-c(.2,-.3);sigma<-2.5
  eta<-drop(X%*%beta);p<-plogis(eta)
  value<-sum(dbinom(y,n,p,log=TRUE))-.5*sum(beta^2)/sigma^2
  gradient<-drop(crossprod(X,y-n*p))-beta/sigma^2
  for(mode in c("H","A_r","N")) {
    result<-tulpaRatio:::cpp_tulpaRatio_sample(y_num=y,y_denom=n,
      y_num_cont=numeric(3),y_denom_cont=numeric(3),X_list=list(X),
      cfg_list=list(family="binomial"),init=beta,n_iter=1L,n_warmup=0L,
      gradient_mode=mode,sigma_beta=sigma,method="evaluate")
    expect_equal(result$log_post,value,tolerance=1e-10,info=mode)
    expect_equal(result$log_prior,-.5*sum(beta^2)/sigma^2,tolerance=1e-10,info=mode)
    expect_equal(result$gradient,gradient,tolerance=1e-6,info=mode)
  }
})

test_that("VI requests reach their Gaussian approximation through the engine API", {
  set.seed(176)
  d<-data.frame(x=seq(-1,1,length.out=40),n=30L)
  d$y<-rbinom(nrow(d),d$n,plogis(.3+.4*d$x))
  for(variant in c("meanfield","lowrank","fullrank","auto")) {
    fit<-suppressWarnings(tratio(y|n~x,data=d,family=ratiod_binomial(),mode="vi",
      control=list(iter=250L,chains=1L,seed=176L,vi_variant=variant,verbose=FALSE)))
    expected<-if(variant=="auto") "fullrank" else variant
    expect_identical(fit$vi_variant,expected,info=variant)
    expect_equal(dim(fit$Sigma),c(2L,2L),info=variant)
    expect_true(all(is.finite(fit$draws)),info=variant)
    expect_gt(min(eigen(fit$Sigma,symmetric=TRUE,only.values=TRUE)$values),0,label=variant)
    if(expected=="meanfield") expect_equal(fit$Sigma[row(fit$Sigma)!=col(fit$Sigma)],c(0,0))
    if(expected=="fullrank") expect_equal(tcrossprod(fit$L),fit$Sigma,tolerance=1e-10)
    if(expected=="lowrank") expect_equal(ncol(fit$L),fit$vi_rank)
    X<-cbind(1,d$x)
    reference<-plogis(fit$.internal$samples[,fit$.internal$layout$beta_num,drop=FALSE]%*%t(X))
    expect_equal(unname(fitted(fit,summary=FALSE)$numerator),unname(reference),tolerance=1e-12)
    predicted<-predict(fit,newdata=d[1:5,,drop=FALSE],component="numerator",summary=FALSE)
    expect_equal(unname(predicted$numerator),unname(reference[,1:5,drop=FALSE]),tolerance=1e-12)
  }
})

test_that("engine point-method fitted values retain their temporal fields", {
  set.seed(177)
  d<-data.frame(x=rnorm(30),time=rep(1:3,each=10),n=20L)
  d$y<-rbinom(nrow(d),d$n,plogis(.2+.4*d$x+c(-.5,0,.5)[d$time]))
  for(method in c("vi","sghmc","sgld")) {
    fit<-suppressWarnings(tratio(y|n~x,data=d,family=ratiod_binomial(),mode=method,
      temporal=temporal_rw1("time"),control=list(iter=150L,warmup=50L,chains=1L,
        seed=177L,epsilon=1e-4,vi_variant="meanfield",verbose=FALSE)))
    q<-fit$.internal$samples;layout<-fit$.internal$layout
    expect_true(all(is.finite(q)),info=method)
    field<-q[,layout$temporal$effects,drop=FALSE]
    field<-field-rowMeans(field)
    eta<-q[,layout$beta_num,drop=FALSE]%*%t(cbind(1,d$x))+field[,d$time,drop=FALSE]
    expect_equal(unname(fitted(fit,summary=FALSE)$numerator),unname(plogis(eta)),
      tolerance=1e-12,info=method)
  }
})

test_that("minibatch requests stop during model construction", {
  d<-data.frame(y=rep(4L,20),n=rep(10L,20),x=seq_len(20)/20)
  for(method in c("sghmc","sgld")) expect_error(
    tratio(y|n~x,data=d,family=ratiod_binomial(),mode=method,
      control=list(iter=20L,warmup=10L,chains=1L,seed=176L,batch_size=10L,verbose=FALSE)),
    "Minibatch stochastic-gradient sampling is unsupported")
})
