test_that("R reads compiled family shapes, links, inflation and extra counts", {
  entries <- tulpaRatio:::cpp_ratio_family_entries()
  for (name in names(entries)) {
    entry <- entries[[name]]
    model <- tulpaRatio:::MODEL_TYPES[[name]]
    for (key in names(entry)) expect_identical(model[[key]],entry[[key]])
    expect_equal(entry$n_extra,length(model$dispersion))
    expect_equal(entry$n_processes,if (model$responses[['denom']] %in% c('trials','exposure')) 1L else 2L)
  }
  for (name in c('poisson_fixed','negbin_fixed','lognormal_fixed','poisson_negbin')) {
    expect_false(entries[[name]]$handcoded)
    expect_true(entries[[name]]$n_processes > 0L)
  }
})

test_that("spec callbacks read each model's priors and shared margin scores", {
  pairs <- list(binomial=c('binomial','none'),poisson_gamma=c('poisson','gamma'),
    negbin_gamma=c('negbin','gamma'),negbin_negbin=c('negbin','negbin'),
    gamma_gamma=c('gamma','gamma'),lognormal=c('lognormal','lognormal'),
    beta_binomial=c('beta_binomial','none'))
  for (family in names(pairs)) {
    for (knobs in list(c(2,.3,1.2),c(5,1.1,3.4))) {
      shape <- knobs[1]; rate <- knobs[2]; scale <- knobs[3]
      r <- tulpaRatio:::cpp_test_family_likelihood(family,shape,rate,scale)
      ll <- 0; residual <- c(0,0); extra <- numeric(); e <- 0L
      for (k in 1:2) {
        margin <- pairs[[family]][k]
        if (margin=='none') next
        phi <- c(2.4,1.7)[e+1L]
        y <- if (k==1L) 3 else if (margin=='negbin') 8 else 4
        z <- tulpaRatio:::cpp_test_ratio_margin(margin,y,8L,c(-.35,.2)[k],phi,FALSE)
        ll <- ll+z[['ll']]; residual[k] <- z[['residual']]
        if (margin=='lognormal') {
          e <- e+1L
          ll <- ll-log1p((phi/scale)^2)+log(phi)
          extra <- c(extra,z[['extra']]+1-2*(phi/scale)^2/(1+(phi/scale)^2))
        } else if (margin %in% c('negbin','gamma','beta_binomial')) {
          e <- e+1L
          ll <- ll+shape*log(phi)-rate*phi
          extra <- c(extra,phi*z[['extra']]+shape-rate*phi)
        }
      }
      expect_equal(unname(r),c(ll,ll,residual,extra),tolerance=1e-12)
    }
    a <- tulpaRatio:::cpp_test_family_likelihood(family,2,.3,1.2)
    b <- tulpaRatio:::cpp_test_family_likelihood(family,5,1.1,3.4)
    expect_identical(tulpaRatio:::cpp_test_family_likelihood(family,2,.3,1.2),a)
  }
})
