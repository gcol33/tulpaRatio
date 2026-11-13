test_that("the sampled binomial GP posterior matches dense integration", {
  skip_on_cran()
  y <- 170L; n <- 250L; sigma <- 2.5
  beta <- seq(-8,8,length.out=801)
  exact <- vapply(beta,function(b) integrate(function(w)
    dbinom(y,n,plogis(b+w))*dnorm(w,0,sigma),-20,20,rel.tol=1e-10)$value,0.)
  posterior <- exact*dnorm(beta,0,2.5); posterior <- posterior/sum(posterior)
  ref_mean <- sum(beta*posterior)
  ref_sd <- sqrt(sum((beta-ref_mean)^2*posterior))
  ref_q <- approx(cumsum(posterior),beta,c(.1,.5,.9),ties='ordered')$y
  fit <- tulpaRatio:::cpp_tulpaRatio_sample(
    y_num=y,y_denom=n,y_num_cont=numeric(),y_denom_cont=numeric(),
    X_list=list(matrix(c(1,1),1)),cfg_list=list(family='binomial'),
    init=c(0,0),n_iter=8000L,n_warmup=2000L,seed=224L,sigma_beta=sigma,
    metric="dense",re_params=list(),spatial_params=list(),temporal_params=list())
  sampled <- fit$samples[,1L]
  bridge19_write_evidence(data.frame(route="gaussian_integral",seed=224L,
    divergences=sum(fit$divergent),ess=posterior::ess_bulk(matrix(sampled)),
    mean_gap=abs(mean(sampled)-ref_mean)/ref_sd,
    quantile_gap=max(abs(quantile(sampled,c(.1,.5,.9))-ref_q))/ref_sd),
    "gaussian-integration-reference.tsv")
  expect_lt(abs(mean(sampled)-ref_mean)/ref_sd,.12)
  expect_lt(max(abs(quantile(sampled,c(.1,.5,.9))-ref_q))/ref_sd,.15)
})
