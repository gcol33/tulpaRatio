# tulpaRatio 1.7.3

* **`spatial_rsr()` is carried, or refused, by every backend (#92).** HMC
  received the restriction only on its centred NNGP branch, where it projected
  the field and then evaluated the NNGP prior on the projected values, with an
  N x N projection applied to a field held at the unique locations; on an areal
  field the entry point cleared the flag, so `spatial_rsr(spatial_car(...))`
  gave draws bit-identical to the unrestricted fit. The log posterior now
  builds the spatial field's value at each observation once (areal, NNGP,
  HSGP or multi-scale GP), and under RSR the predictors read `(I - Q Q') f`
  with `Q` the orthonormal basis of the restricted covariates; every field
  prior reads `f`. The gradient of a restricted model is the arena gradient of
  that density, and `fitted()` applies the same projection. A collapsed
  parameterization is refused under RSR. The PG sampler drew the field from
  the unrestricted ICAR conditional and read a corner of the projection; it now
  draws it from its exact Gaussian conditional, with precision
  `tau Q + A' Omega A` for `A = (I - Q Q') S`, and its R wrapper no longer
  errors before sampling. ESS, Gibbs, SGHMC, SGLD and VI, which never read the
  restriction, refuse it, and PG refuses it on anything but an intrinsic CAR
  field. `predict()` refuses a restricted field at new rows.

* **`predict()` on the fitted rows reproduces `fitted()` (#93, #94, #95).**
  Three separate defects pulled the two apart:
  - The HMC areal lookup, the PG areal lookup and the PG random-effect lookup
    numbered a grouping variable's units in order of first appearance, where the
    fit numbers them by factor level, so whenever the two orders differed each
    row read another unit's effect (#93).
  - An HMC binomial prediction with a spatial field added the field to
    `log(p)` and took `plogis()` of that, which is not the predictor; the field
    now enters the linear predictors and the response is taken once through
    `hmc_response_draws()`, as `fitted()` takes it (#94).
  - Every Polya-Gamma sampler stored the predictor computed at the top of an
    iteration beside the parameters updated during it, so saved draw `s` paired
    the parameters of iteration `s` with the predictor of iteration `s - 1`; the
    predictor is refreshed at the save (#95). The field-free and
    random-effect-only fits run tulpa's own sampler, which has the same lag
    (gcol33/tulpa#941).

* **The log posterior no longer depends on the thread count on Windows arm64
  (#83).** On aarch64-w64-mingw (Rtools45 clang 19) an OpenMP
  `reduction(+:double)` over a 4-thread team returned the serial value in 0 of
  500 repeats of a plain Poisson sum on a 4-core runner, while 1 and 2 threads
  and a hand-rolled per-thread sum were exact. The observation sum in the
  density, the Laplace log-likelihood and the test kernels now go through
  `ratiod_omp::sum_range()` (`src/omp_sum.h`): each thread sums one chunk into
  its own slot and the slots are added in thread order. The HMC gradient's
  observation loop summed the zero- and one-inflation coefficients, the
  temporal and spatial fields and the random slopes with `omp atomic`, and on
  the same platform the analytic `beta_zi` gradient missed finite differences
  by up to a factor of four; those sums now share the loop's per-thread slots
  with the other blocks.

* **The Laplace tier runs on the nested-Laplace engine (#85, #84).** The
  in-tree mode-finder (`laplace_core.cpp`) picked one value per hyperparameter
  and conditioned the fixed effects on it, so intervals omitted hyperparameter
  uncertainty. The tier now hands the model to `tulpa::tulpa_nested_laplace_joint()`:
  one likelihood arm per process, one latent block per random-effect term,
  areal field (ICAR, BYM2, proper CAR), Gaussian-process field (single and two
  scale, on a Hilbert-space basis) and multi-scale temporal component. The
  engine integrates every block hyperparameter and every dispersion on its outer
  grid. `summary()` reads the fixed effects off the grid mixture (mean,
  covariance, skew-aware interval), so the rows no longer move with a draw seed;
  hyperparameters are reported as posteriors instead of constant columns; the
  stored draws are drawn from the mixture.

  Consequences of the move:
  - Two-process families (`poisson_gamma`, `negbin_gamma`, `negbin_negbin`,
    `gamma_gamma`) are fitted as two arms with their own fixed effects and
    dispersions, shared latent blocks entering both. The old tier fitted the
    numerator alone, so `ratio()` of such a fit was a numerator rate (#89).
  - Every random-effect term is fitted; only the first was (#88).
  - `predict()` works for fits without a spatial or temporal field (it always
    errored, #87), and `fitted()` reads each random effect at the group each row
    belongs to (#90).
  - A Gaussian-process field uses the Hilbert-space basis, not the nearest-
    neighbour approximation, and takes the squared-exponential kernel.
  - Spatial and multi-scale temporal terms can share a Laplace fit; proper CAR
    and `spatial_hsgp()` are fitted.
  - Restricted spatial regression reaches the engine as a projected areal block
    (the field enters each arm as `(I - Q Q') S z`) or as a projected
    Hilbert-space basis; the projected effect is orthogonal to the restricted
    covariates. Spatially varying coefficients have no block on the engine and
    are refused with `mode = "hmc"` as the way out.
  - A model with no random effect and no field is fitted on the dispersion axes
    alone.
  - `summary()` lists every fixed effect of a one-process fit; it listed the
    intercept only (#91).

* **The SoftAbs divergence retry is off by default (#86).** `control$riemannian`
  used to resolve to an automatic setting that switched the retry on for BYM2
  and ICAR fits with a dense mass matrix. The retry replaces a divergent NUTS
  trajectory with a second one under a Hessian-based metric, so which kernel
  moves the chain depends on the current state and no acceptance correction
  covers the choice. On the engine's copy of the scheme (gcol33/tulpa#937,
  Neal's funnel) a seed settled in the neck with 451 divergent transitions
  against 6 without it. `control = list(riemannian = TRUE)` still enables it;
  the automatic setting is removed.

* **`mcmc_diagnostics()` is gone; use `diagnostics()` (#82).** It was an
  exported `lifecycle::deprecate_warn()` shim forwarding to `diagnostics()`,
  and the engine's shim of the same name was deleted in gcol33/tulpa#773, so
  two packages exported the name and whichever attached second masked the
  other. The package is pre-release, where the convention is to delete the old
  name rather than carry a shim. `lifecycle` was reachable from nothing else
  and leaves `Imports` with it.

# tulpaRatio 1.7.2

* **The standalone intrinsic walks take the same coordinate (#79).**
  `temporal_rw1()` and `temporal_rw2()` take `parameterization`, reading the
  same transform the multi-scale arms do. `temporal_effects()` and
  `temporal_gmrf_prior_grad()` are the two places it lives -- the one chokepoint
  every eta assembly reads and the one all eight gradient functions reach -- so
  the coordinate is a branch in each rather than a change to any caller.

  Measured on a 500-observation fixture with no temporal signal, centred ->
  non-centred: an RW1's bulk ESS for `tau_temporal` goes 409 to 805 at 0 and 1
  divergences, and an RW2 picks up 23 divergences of 2000 for no ESS gain. Both
  return the same posterior, the intercept and slope agreeing to four decimals.
  Non-centred helps where the variance is shrunk toward zero and can cost where
  the data pin the field, which is why it is an argument rather than a default.

* **The non-centred walks take `group_var` (#81).** The prior augments ONE
  global constant whatever the number of groups, so G - 1 group contrasts stay
  improper: a per-group level shifts eta only for that group's observations and
  the likelihood identifies it. Running the single-group transform per group
  would augment each group's own constant and shrink those contrasts, which is
  a different model, so the transform re-mixes instead. Each group's coordinate
  splits into its increments and one sum coordinate; the G sum coordinates
  split in turn into the global direction the augmentation fills and G - 1
  contrasts, which stay unscaled because they are flat and scaling them would
  couple the group levels to `tau`. The cyclic branch reads its mean coordinate
  the same way, so one map covers RW1, RW2 and the seasonal walk.

  `cpp_ms_temporal_nc_coordinate_spread()` now runs over groups, and
  `cpp_ms_temporal_nc_group_contrast_flat()` moves the contrast directions
  alone: it asks that the prior stay where it is and that group g's field move
  by exactly the unscaled offset. A finite difference cannot separate a
  transform that augments per group -- that density is self-consistent and its
  gradient matches its own difference quotient -- and neither can a flatness
  check on its own, which a transform that dropped the contrasts would pass.

* **Every gradient fixture is checked to build the structure its name claims.**
  `make_model()` selects a structure by a chain of flags, each naming the fields
  it covers, so a name added to `KNOWN_FIELDS` but to no flag falls through every
  branch and yields a model with no structured field: the gradient check then
  passes on a two-parameter regression while reporting the structure's name.
  `rw1_nc` did exactly that, coming back with 2 parameters against the 13 its
  siblings carry. `test-gradient-correctness.R` now asserts every field builds
  more than the structureless model.

* **A non-centred coordinate for the multi-scale temporal arms (#79).**
  `temporal_multiscale()` takes `parameterization`, and `"noncentered"` samples
  the trend and seasonal arms as `z ~ N(0, I)`, reaching the effects through
  `sigma * A^-1 z`. `A` stacks the arm's difference operator on `1'/sqrt(n)`,
  which is exactly the augmented precision's factor `Q + 11'/n = A'A`, so the
  arm's quadratic form and the transform's Jacobian cancel term for term and
  what is left is `N(0, I)` with the variance reaching the linear predictor
  through the transform alone. `A^-1` is a cumulative sum, once for RW1 and
  twice for RW2, so the transform and its adjoint are O(n).

  A non-cyclic RW2 annihilates a linear ramp as well as the constant and only
  the constant is augmented, so one coordinate stays a free direction under the
  flat prior it already had. Any right inverse of `A` reproduces the density,
  but not any right inverse is well conditioned: the double integral's own
  particular solution carries a large ramp that the free coordinate can undo,
  which measured 406 divergences of 2000 before the ramp was projected out and
  3 after.

  Measured on the multiscale RW2 recovery fixture, whose true trend is zero so
  the whole posterior sits in the funnel's neck, centred -> non-centred: the
  divergence count goes 30 of 2000 to 3, and by decile of `log sigma2_trend` the
  neck goes from 12.5% divergent to 0%. The bulk ESS of `sigma2_trend` goes 21
  to 310 and its rhat 1.126 to 1.013, and the intercept's ESS 596 to 1776. The
  centred run reports a posterior median of 1.6e-04 against 4.2e-05, biased
  upward by the part of the neck it never reaches; only the non-centred run has
  converged on that parameter. The intercept and slope are unmoved.

  `test-gradient-correctness.R` carries the three walks in either coordinate
  across all four gradient modes, and
  `cpp_ms_temporal_nc_coordinate_spread()` checks the invariant a finite
  difference cannot see: a transform that builds the wrong field leaves the
  density self-consistent and its gradient still matching its own difference
  quotient, but at fixed hyperparameters the two coordinates' densities have to
  differ by one number for every `z`. Measured spread 1e-13.

* **A proper CAR field's mean is removed from the linear predictor (#80).**
  `Q(rho) = D - rho*W` is full rank at `rho < 1`, so the prior identifies the
  field's mean, and the field was left uncentred on that reasoning. The
  likelihood cannot see that mean apart from the intercept though: `eta` reads
  `beta_0 + phi_s`, and moving the intercept by `c` against every `phi` by `-c`
  leaves `eta` where it was. Measured on a 36-unit fixture, the two correlated
  at -0.977, each carried a posterior sd near 0.06 while their sum carried
  0.013, and the intercept's bulk ESS was 71 of 2000 at rhat 1.053. Longer
  warmup made it worse, not better: 1.053 -> 1.112 and ESS 71 -> 32 going from
  1500 to 3500, because adaptation tunes for the bulk and the ridge is not in
  it.

  `spatial_car()` now takes `center`, TRUE by default, and an SVC term's
  treatment is what it applies: `eta` reads the centred field and the prior
  still reads the raw one. Since `Q(rho)` does not annihilate the constant, the
  two are different quadratic forms, and the density and every gradient now
  take the prior's view of the field separately from `eta`'s.

  The identified posterior does not move. On that fixture the level comes back
  at 0.5200 against 0.5204, the slope at 0.3003 against 0.3004, with rho and tau
  inside their own uncertainty. What changes is that the level is the intercept:
  ESS 71 -> 1953 and rhat 1.053 -> 1.001.

  `center = FALSE` keeps the previous parameterization and requires
  `proper = TRUE`; an intrinsic field's mean is unidentified until it is
  removed, so there is no model on the other side of that switch.
  `car_proper_raw` covers it in `test-gradient-correctness.R`.

* **A proper CAR field reaches only gradients written for it.** `GF_AREAL`
  covered a proper CAR field as much as an intrinsic one, so
  `spatial_car(proper = TRUE)` alongside `temporal_multiscale()` or a
  spatiotemporal interaction selected
  `compute_gradient_ms_temporal_handcoded` or
  `compute_gradient_spatiotemporal_handcoded`. Neither writes proper CAR's
  `rho` gradient or its log-determinant, and both centred a field the log
  posterior leaves alone, so the gradient they returned was the gradient of a
  different density. Measured against central differences of the log posterior
  on a 400-observation, 25-unit fixture: 210 on the regression coefficients,
  where every field in the sweep agrees to better than 1e-04 relative.

  Proper CAR now carries `GF_AREAL_PROPER`, so a function whose `covers()` mask
  does not name it falls through to the composite -- the path
  `spatial_car(proper = TRUE)` on its own already took, and the one that
  carries the `rho` block. The same combinations now agree with finite
  differences to 2e-07 and 3e-06. `car_proper_ms` and `car_proper_st` join
  `test-gradient-correctness.R`, which is what nothing reached before.

* **One predicate decides whether the spatial block is centred.** The rule was
  written out at five sites, and three spellings had drifted from the log
  posterior's. `spatial_block_is_centred()` (`hmc_sampler.h`) is now the only
  one, so a gradient and the density it is evaluated against cannot disagree
  about which field the likelihood saw.

* **Recovery assertions read the quantity their model identifies.** The SVC
  recovery test compared a single 80-observation realization's posterior mean
  against the value the data were simulated from, where the maximum likelihood
  slope carries a standard error near 0.05: seed 11's sample implies 0.2533 and
  seed 22's implies 0.2062, so the assertion failed on sampling noise. It now
  runs the three seeds the RW1 test runs, asserts the posterior mean against the
  same sample's unpenalized MLE -- measured agreement within 0.002, an order of
  magnitude sharper than a tolerance against 0.3 can be -- and keeps recovery of
  the simulated values as interval coverage across seeds.

  The proper CAR test asserted `rhat < 1.05` on the intercept, which that model
  does not identify on its own: the intercept and the field's mean correlate at
  -0.977 and their sum carries a fifth of either one's posterior sd. It now
  asserts on the sum, which converges (ESS 1970 of 2000, rhat 1.002), and on the
  intercept's interval coverage.

* **The multiscale RW2 fixture's divergences are pinned where they belong.**
  They sit in the neck of `sigma2_trend`, whose true value on that fixture is
  zero: 12.5% of draws in the lowest decile of `log sigma2_trend` diverge and
  none above the third, and the divergent draws sit 1.96 posterior sd below the
  clean ones. `test-recovery-constrained-fields.R` asserts that location rather
  than a bare count, which is what separates the known centred-parameterisation
  funnel from a fault elsewhere. The trend arm's non-centred coordinate is #79.

* **The SVC range prior a spec declares is the one the fit runs (#78).**
  `prepare_svc_for_hmc()` wrote `phi_prior_lower = 0.3`, `phi_prior_upper = 30`
  as literals into each of its three branches, so every SVC fit ran on that
  interval and nothing a caller set could change it. `spatial_svc()` carried no
  field for the bounds at all, and the `%||% 0.01` / `%||% 10.0` fallbacks in
  `fit_hmc()` -- the numbers `?spatial_svc` and the `ModelData` defaults both
  named -- were unreachable, because `svc_info` always supplied a value for the
  `%||%` to pass through.

  `spatial_svc()` now takes `range = c(lower, upper)`, validated as
  `0 < lower < upper` the way `spatial_multiscale()` validates `range_local`
  and `range_regional`, and `prepare_svc_for_hmc()` reads it off the spec.
  `svc_default_range()` is the one place the default lives; the `ModelData`
  defaults carry the same pair, and the print method reports the interval in
  force. Fits that set nothing are unchanged: the default is the `[0.3, 30]`
  those fits already ran under, and the draws are bit-identical.

  The bounds apply to `approx = "nngp"`. The HSGP arm parameterises a
  lengthscale under LogNormal(0, 1), which this uniform does not set, so a
  `range` passed alongside it is an error rather than a value nothing reads.
  `ess_sampler.cpp` and `vi_sampler.cpp` each wrote a third and fourth pair of
  bounds into a block they declare unsupported (`has_svc = false`); both are
  gone.

* **Intrinsic fields identified by augment-and-centre (#22).** The temporal
  random walks and the multi-scale trend and seasonal arms were identified by a
  soft penalty on the field's sum, `-0.5 * lambda * (sum phi)^2` at
  `tulpa::s2z_precision(n)`. That pins the constant direction without removing
  it: the direction stays in the linear predictor at precision `1 / (kappa*n)^2
  * n`, five orders of magnitude above the `tau` the field's own directions
  carry, and a step size that holds it stable is the one the whole field then
  has to move at.

  These fields now use the engine's construction (`tulpa/sum_to_zero.h`,
  gcol33/tulpa#241): the precision is augmented, `Q -> Q + 11'/n`, so the
  constant carries the field's own `tau`, and the effects are centred on their
  way into `eta`, so the likelihood never sees it. The normalizer takes the
  augmented rank -- one more pinned direction, except that a non-cyclic RW2
  keeps its linear null direction and stays deficient by one. Measured on a
  500-observation binomial fixture, the adapted step size rises from 0.101 to
  0.351 for `temporal_rw1()` and from 0.105 to 0.255 for `temporal_rw2()`, and
  the leapfrog steps per iteration fall from 35 to 12 and from 42 to 20; the
  multi-scale arms move from 0.066 to 0.124 and from 0.040 to 0.056.

  The reported field changes with it. `fitted()`, `ratio()` and `predict()` add
  the stored field back into `eta`, so an intrinsic field's draws are now stored
  centred -- the series the likelihood saw, rather than one differing from it by
  a level. `temporal_ar1()`, `temporal_iid()` and the temporal GP are proper,
  identify their own level, and are unchanged.

* **`spatial_multiscale(parameterization = "noncentered")` (#75).** The
  multi-scale field was the one continuous field in the package with a single
  coordinate: `spatial_gp()` offers `"centered"`, `"noncentered"` and
  `"collapsed"`, and `temporal_gp()` and `temporal_ar1()` each offer a second
  coordinate, while the multi-scale block carried two variance parameters over
  the same locations with no alternative to the effects themselves. #36 removed
  the name that selected nothing; it now selects a coordinate.

  Non-centred, each scale's block holds `z ~ N(0, I)` and its field is
  `w = L(sigma2, phi) z` through the NNGP autoregression the single-scale field
  already uses, so all four hyperparameters reach the linear predictor through
  the transforms as well as through the priors. Stored draws are the effects in
  either coordinate. `approx = "hsgp"` ignores the argument: its basis
  coefficients are already that coordinate.

  `MultiscaleGPFieldT` is the reader and `MultiscaleGPView::accumulate` the
  adjoint, one of each rather than a branch at each of the six sites that read
  the field -- the templated density, the three gradient functions that write
  the block, and the sample-storage transform. Both scales enter the linear
  predictor through the same sum, so the likelihood's derivative with respect
  to either is the one vector, and the adjoint carries it onto whichever
  coordinate is being sampled. `multiscale_gp_log_lik_t()` built its own copy
  of the two `GPData` views beside `make_msgp_gp_views()`; it now calls it.

  `msgp_nc` joins the finite-difference sweep in all four gradient modes. A
  finite difference cannot see a transform that builds the wrong field -- the
  density would stay self-consistent and its gradient would still match its own
  difference quotient -- so `cpp_msgp_nc_coordinate_spread()` reads the
  invariant that does: at fixed hyperparameters the two densities differ by
  `log|det L|`, one number for every `z`. It measures 2e-13.

* **A collapsed workspace no longer reads a previous model's structure (#76).**
  The collapsed GP's per-thread workspace caches the NNGP coefficients for the
  last `(sigma2, phi)`, the location-to-observation map, and a Newton mode the
  next call warm-starts from. Its cache key was the number of GP locations, so
  a thread that had evaluated one model and was then handed another with the
  same number of locations kept the first model's coefficients and map. Two
  collapsed-GP fits in one session over a field of the same size was enough:
  the second fit's chain threads started from the first fit's structure, and
  the same call with the same seed depended on what had run before it.
  `CollapsedICARWorkspace` was keyed the same way on `(S, bym2, N)`.

  Both workspaces now carry the id of the `ModelData` their structure was built
  for and drop every cached quantity when handed a different one. The Newton
  warm start goes with the rest: the mode of the previous leapfrog step is
  worth starting from, another model's mode is not, and the search is capped at
  20 iterations, so a bad enough start is not guaranteed to recover.

  `cpp_gradient_race()` gains `prime_other_model`, which hands the calling
  thread a second model of the same dimensions at the same point before the
  race, and clears the Newton warm start before every evaluation so a collapsed
  gradient is a function of the point alone. The probe read 4.40e-01 for
  `gp_collapsed` and 8.49e-01 for `gp_collapsed_st4` against the old key and
  reads 0 against the new one; the eight collapsed fields are now swept by
  `test-omp-gradient-race.R` alongside every other field.

* **`compute_log_post` is `compute_log_post_impl<double>` (#28).** The two
  densities returned the same number on every structure the templated one
  expressed, so keeping them apart bought nothing and cost the standing risk
  every bug in #11, #14 and #18 came from: a term in one copy and not the
  other, invisible because the runtime gradient check differences whichever
  density the active mode differentiates. The analytic body is deleted; what
  used to separate the two now sits inside the template.

  The structures no autodiff scalar can express -- the collapsed ICAR/BYM2 and
  collapsed GP marginals, which locate a mode with Newton and add a Laplace
  correction, and proper CAR's dense log-determinant -- are `if constexpr`
  arms, so `log_post_impl_gap()` names exactly the blocks the other
  instantiations leave out, as before. The observation loop is one lambda,
  summed by an OpenMP reduction in the `double` instantiation and serially in
  the others, so the H path keeps its parallel likelihood without a second copy
  of the 250-line body. `skip_obs_loop` and the temporal GP's precomputed block
  prior become a `LogPostOptions` struct defaulted to the full density;
  `precomputed_st_log_prior` goes, having never had a caller.

  Writing the collapsed fields into the template found the guard they needed:
  the spatial hyperparameter priors were gated on the field having a parameter
  slot, so a marginalized field would have dropped the prior on its own scale.
  Nothing reached that -- the gap kept collapsed models out of the templated
  density entirely -- but the merge could not have kept it.

  The double twins the deleted body was the last caller of go with it:
  `gp_nngp_log_lik`, `multiscale_gp_log_lik`, `log_prior_sigma2_pc`,
  `log_prior_phi_uniform` and `gp_gradient_w` in `hmc_gp.h`, `hsgp_evaluate`
  and `hsgp_log_prior_beta` in `hmc_hsgp.h`, `ar1_nc_log_prior` in
  `hmc_temporal.h` and `log_prior_temporal_sigma2_pc` in `hmc_temporal_gp.h`,
  each of which had a templated counterpart that is now the only definition.
  So do `compute_gradient_gp_autodiff`, `compute_gradient_msgp_autodiff` and
  `compute_gradient_gp_temporal_autodiff`, three hand-written `ad::Var`
  transcriptions of the density that `resolve_gradient_fn` never returned and
  nothing else named -- a copy no test could reach and no assertion could hold
  to the original. 2260 lines net.

  What the harness asserts changes with it: comparing `compute_log_post` to
  `compute_log_post_impl<double>` is now comparing a function with itself, so
  that check is replaced by two that still have content. The density must not
  depend on how many threads sum it, which is what the new `n_threads` argument
  to `cpp_gradient_check` reaches; and the value each mode reports must be the
  density its gradient describes, which holds the fused `skip_obs_loop` path
  and the `arena::Var` instantiation to the `double` one.

  Across twenty-two structures every entry of the analytic gradient is
  bit-identical to before, and the density moves by at most 3.7e-16 relative --
  one to two units in the last place, from summing the same terms in the
  template's order. Wall-clock on a 4000-row binomial fit, 400 iterations,
  before and after: plain GLM 0.59 s / 0.60 s, ICAR + RW1 16.5 s / 15.7 s,
  collapsed ICAR + RW1 320 s / 281 s, crossed random effects 46.7 s / 44.1 s.

* **`test-temporal.R`'s AR1 fit asserts a correlation its fixture identifies.**
  It read `mean(rho_samples) > 0` on ten time points, where the posterior for
  rho is nearly its prior: sd 0.32 around a mean of 0.07, with a 90% interval
  of [-0.54, +0.69] even at 8000 iterations. The sign of the mean over 100
  retained draws sat inside its own Monte Carlo error, and across five seeds it
  came out anywhere from -0.03 to +0.40, so what the assertion turned on was
  which trajectory the sampler happened to take. The fixture now carries eighty
  time points, where the posterior mean is +0.47 or better across eight seeds
  against 0 for a chain that left rho at its prior, and the threshold says so.
  The strengthened fixture passes on the code either side of #28.

* **The templated log posterior expresses the HSGP multi-scale GP and the
  non-centred GP (#26).** These were the two structures `log_post_impl_gap()`
  named as expressible but unwritten, and both were reachable from R:
  `spatial_multiscale(approx = "hsgp")` and
  `spatial_gp(parameterization = "noncentered")`, the latter the default
  coordinate for a GP fit. Declared rather than written, they cost the
  `A_r` / `A` / `A_t` gradient modes those two models, since the front door
  refuses a mode whose density is missing a block. The multi-scale case was a
  missing block of the kind #18 fixed for the plain HSGP field -- the
  parameters would carry no prior and contribute nothing to the linear
  predictor -- and the non-centred case was worse than missing: the templated
  density read the parameter slot as `w` when it holds `z`, so it evaluated a
  different model rather than an incomplete one.

  Both are now written against the same primitives the analytic density uses,
  rather than transcribed beside them. `hsgp_evaluate_t()` is one spectral
  expansion for every scalar type, with the Eigen matvec as its `double` arm,
  and the plain HSGP field, the SVC terms and both multi-scale scales now read
  it instead of carrying three inline copies of `f = Phi (sqrt(S) . beta)`.
  `NNGPNCWorkspace` and `nngp_nc_forward()` are templated, so the `z -> w`
  recursion has one definition rather than one per scalar type, and the
  conditional-variance floor is `ratiod_cov::nngp_floor_cond_var` on both.
  Measured against central differences of `compute_log_post`, all three
  autodiff modes now agree to 5.7e-09 relative on `gp_nc` and 2.0e-09 on
  `msgp_hsgp`, with no identically-zero gradient entry, and the two densities
  agree to 5e-13. Both fields move from the gap list into the harness's kernel
  sweep, where they are checked under `handcoded` and `arena` and included in
  the density-equality and fused-value assertions.

* **`temporal_iid()` reaches the standalone temporal field, and
  `temporal_ar1(parameterization = "noncentered")` reaches the AR1 one (#40).**
  `TemporalType::IID` had a branch in `temporal_log_prior()` and in its
  templated twin, but no R constructor and no `"iid"` case in the C++ parser,
  so nothing could select it; the H-mode and A/A_r branches had no IID case at
  all and would have contributed zero. A complete non-centred AR1
  reparameterization -- `ar1_nc_forward`, `ar1_nc_log_prior`, `ar1_nc_gradient`
  -- sat in `hmc_temporal.h` with no callers anywhere. Both are now wired:
  `temporal_effects()` reconstructs the effects wherever a density or a
  gradient reads the block, and `temporal_gmrf_prior_grad()` writes the
  gradient in whichever coordinate the sampler moves in. Wiring
  `ar1_nc_gradient` up showed it was wrong: its `log_tau` and `logit_rho` sums
  paired the backward adjoint `adj[t]`, which carries the AR1 recursion
  backward, with a forward derivative `dphi[t]/dtheta` that already carries the
  same recursion, counting it twice. Against central differences of the log
  posterior the two hyperparameters were off by 12% and 8% while all ten field
  entries matched to 4e-10; the correct pairing is the direct partial
  `grad_phi_lik[t]`, which the function's own `T == 1` branch already used.
  All four gradient modes now agree with finite differences on `temporal_iid`
  and `temporal_ar1_nc`, both added to `test_gradient_check.cpp`.

* **`spatial_multiscale()` under `mode = "laplace"` fits instead of reporting a
  missing `group_var` (#39).** `cpp_laplace_fit_multiscale_gp` was a registered
  export with no caller: `fit_laplace_spatial()` had no `"multiscale"` branch,
  so the model fell through to the areal path and failed on a message about
  areal grouping. `spatial_gp()` failed the same way -- the areal preparation
  ran ahead of the GP dispatch -- and would have failed again on
  `result$hessian`, which `cpp_laplace_fit_gp` never returned. The two
  coordinate-based fields now dispatch before the areal preparation, and both
  run through one mode-finder: `laplace_mode_nngp()` takes a list of NNGP
  blocks, so a single-scale GP passes one and a multi-scale GP passes two,
  replacing two near-identical Newton loops. Three defects went with the
  duplication: the multi-scale copy ignored `cov_type` and hard-coded an
  exponential covariance, and wrote a conditional mean from raw covariances
  rather than the Kriging weights; both copies assembled only the diagonal of
  the NNGP precision, so Newton converged somewhere other than the mode the
  reported log-marginal describes; and both mapped observation `i` to location
  `i`, dropping the spatial term entirely for every observation past the last
  unique location. `obs_to_loc` is now passed in, the Hessian is assembled at
  the converged point and returned, and `hsgp` and `svc` name themselves in
  `BACKEND_STRUCTURE_SUPPORT` rather than reaching a dispatch that would fit
  the model without the field.

  The Laplace NNGP path also read R's 1-based `nn_order` as 0-based, writing one
  past the end of the field block on the last location. Nothing reached it
  before; it now converts at the call site, as the sampler entry does, and
  `add_nngp_block_laplace()` refuses an ordering outside `0..n-1` rather than
  corrupting the heap.

* **Multi-scale HSGP is reachable (#38).** `data.msgp_is_hsgp` selected a
  complete C++ branch that swapped the two neighbour sets for one shared
  Hilbert-space basis, and no R function could set it -- `spatial_multiscale()`
  had no `approx` argument, and the internal validator the dispatch resolved to
  raised an error saying so. `spatial_multiscale(approx = "hsgp", m =,
  c_boundary =)` now selects it, the neighbour computation is skipped on that
  path, and `msgp_hsgp` joins the finite-difference harness. On 90 observations
  with `m = 8` it fits in 3.2 s against 14.3 s for the neighbour sets. The
  templated log posterior still declares the gap it declared before, and now
  also skips the block rather than reading neighbour sets sized for the
  locations against a parameter block sized for the basis.

* **`spatial_multiscale(sampler =)` offers the two strategies that exist
  (#36).** Five of its seven names -- `"noncentered"`, `"centered"`,
  `"interweaved"`, `"adaptive"`, `"riemannian"` -- produced byte-identical
  sampling: the field is centred at every site that reads it for the density or
  the gradient, and the only enumerator read anywhere selects L-BFGS warmup
  adaptation. The argument now takes `"auto"` and `"lbfgs"`, which is what it
  selects between, and the enum's documented default no longer claims a
  parameterization the code does not carry. A non-centred coordinate for the
  multi-scale field is filed separately.

  `make_msgp_gp_views()` existed twice, and the copy feeding the gradients left
  `nn_neighbor_dist` empty. Its current callers do not read that field, but the
  non-centred NNGP helpers do, so the two are now one builder that fills every
  field `GPData` declares.

* **`spatial_gp(solver =, cg_tol =, cg_maxiter =)` are gone (#37).** The CG /
  PCG / GPU dispatch in `hmc_gp_cg.h` was complete and had no callers -- the
  header was not `#include`d anywhere, so it was never compiled. Routing the
  sampler through it would have made the package slower: the systems it solves
  are the `k x k` neighbour covariances, and CG needs about `k` iterations on a
  dense covariance rather than converging early. Measured against the Cholesky
  already running, on exponential-covariance neighbourhoods in the unit square,
  10,000 solves each: k=5 3.1x slower, k=10 2.3x, k=15 (the `nn` default) 3.1x,
  k=30 (the `nn_regional` default) 2.4x, k=50 2.8x, returning a 1e-6 solution
  rather than a machine-accurate one. The N-scaling of an NNGP is `O(N k^3)`
  and the inner solver does not touch N, so `"cg"`'s documented case -- "better
  for large N (> 2000)" -- does not hold. `"gpu"` had no path either: nothing in
  the package builds CUDA, and its own caller commented the GPU branch out. The
  three arguments and the dead header are removed; `gpu_available()` and
  `gpu_info()` are unaffected.


* **An areal spatial field paired with `temporal_multiscale()` now gets its
  multi-scale block (#74).** `cpp_hmc_fit` set `has_multiscale_temporal` from
  the temporal bundle and then cleared it about a hundred lines later, among
  the feature flags that belong to `cpp_hmc_fit_gp`. `compute_param_layout()`
  reads that field to allocate the trend, seasonal and short-term blocks, so it
  allocated none of them while R went on naming their columns. The combination
  is reachable because `use_gp_sampler` is
  `is_gp_spatial(spatial) || (is_multiscale_temporal(temporal) && !has_areal_spatial)`:
  a multi-scale temporal term with an areal spatial field is the one case that
  routes to `cpp_hmc_fit`. Measured on `num | denom ~ x` with
  `spatial_car(level = "group")` and
  `temporal_multiscale("year", trend = "rw1", short_term = "ar1")`, 6 units by
  12 times, poisson_gamma, 100 iterations: R named 39 columns, 27 of them the
  multi-scale block's, and those 27 were not a posterior -- `sigma2_trend` and
  `sigma2_short` spanned `[0, Inf]` with a standard deviation of `NaN`,
  `rho_short` reached exactly -1 and 1 though it is logit-mapped onto the open
  interval, and the field columns had standard deviations in the thousands.
  All 27 now respect their own priors. `test-temporal-multiscale.R` checks the
  block against its prior support the way `test-gp-sampler-blocks.R` does, and
  fails 29 assertions without the fix.

* **The GP sampler entry carries the random-effect block it was handed (#72).**
  `cpp_hmc_fit_gp` set `n_re_terms = 0` and `has_re_slopes = false`
  unconditionally, and the R backend handed it `re_group` / `n_re_groups`
  alone rather than the `re_params` bundle `cpp_hmc_fit` receives.
  `prepare_hmc_data()` and `hmc_param_layout()` above it parsed and named the
  slope and multi-term structure regardless, so the two layouts disagreed on
  the size of the RE block and every column after it was read at the wrong
  offset. Measured on `num | denom ~ x + (1 + x | g)` with
  `spatial_gp(~ lon + lat, nn = 3)`: R named 28 columns, the sampler wrote the
  legacy block instead, and the column labelled `phi_gp` ran to 2.37e+149
  against a uniform prior on `(0.01, 100)`. It now reports 4.08 to 98.10.
  Affects random slopes (correlated and uncorrelated) and crossed / nested
  multi-term random effects on `spatial_gp()`, `spatial_multiscale()`,
  `spatial_hsgp()`, and on a multiscale temporal model with no areal spatial
  term.

* **The AR1 temporal rho prior reaches the GP sampler entry (#73).** That
  entry's temporal block read `type`, `time_idx`, `group_idx`, `n_times`,
  `n_groups`, `n_params`, `cyclic`, `shared`, `tau_shape` and `tau_rate` and
  stopped there, so the `rho_prior_a` / `rho_prior_b` the R backend put on the
  list crossed the `.Call` boundary and were never assigned; the fit ran on
  `ModelData`'s default anchors. Measured on `num | denom ~ x` with
  `temporal_ar1("year")`, 6 groups x 10 times: the posterior mean of `rho_ar1`
  under `prior_beta(2, 2)` and `prior_beta(50, 1)` was bit-identical on a GP
  main effect and moved from -0.071 to 0.972 on an areal one. The GP arm now
  moves -0.103 to 0.933.

* Both bundles are assembled by one builder each entry point calls
  (`apply_re_params` / `apply_temporal_params`, `src/hmc_model_data_blocks.h`),
  the way #70 did for the spatiotemporal, latent, TVC and SVC bundles, and the
  R backend builds them once above the branch. A field read by one extraction
  and not the other is what both of these were.

* **The random-effect gradient is written once (#72).** With the block
  reaching the sampler, `resolve_gradient_fn()` has to keep such a model off
  the specialized functions, which read the legacy single-term block: crossed
  and nested terms now raise their own feature bit (`GF_RE_MULTI`) alongside
  `GF_RE_SLOPES`, and both fall through to `compute_gradient_composite`. That
  function carried a partial copy of the block -- no Cholesky factor, no LKJ
  prior, no `re = diag(sigma) L z` transform -- so a correlated slope beside a
  GP field reached it with the slope scale's gradient at zero against a
  finite difference of 0.409, and the runtime gradient check fell back to
  numerical. The prior phase and the chain rule are now
  `re_blocks_prior_grad()` / `re_blocks_writeback()`
  (`src/hmc_re_blocks_grad.h`), called by `compute_gradient_analytical()` and
  by the composite alike, with `re_blocks_eta()` / `re_blocks_scatter()` for
  the observation loop. Three new finite-difference fixtures cover what could
  not be built before: `gp_slopes_corr`, `gp_crossed`, and `gp_slopes` at the
  composite it now dispatches to.

# tulpaRatio 1.7.1

* **A GP spatial main effect carries the blocks beside it (#70).**
  `is_gp_spatial()` routes the whole model to `cpp_hmc_fit_gp_v2`, which was
  handed `gp_params`, `ms_gp_params`, `ms_temporal_params`, `rsr_params`,
  `temporal_params` and the ZI arguments and nothing else. A spatiotemporal
  interaction, a latent factor, a TVC or an SVC paired with `spatial_gp()`,
  `spatial_multiscale()` or `spatial_hsgp()` was dropped from the `ModelData`
  the sampler ran on, while `initialize_hmc_params_full()` and
  `hmc_param_layout()` above it still allocated and named that block's
  parameters: every column from the dropped block onward was read at the wrong
  offset and reported under the wrong name. Measured on 6 units x 5 times with
  `spatial_gp(~ lon + lat)` and `spatiotemporal(type = "separable")`,
  `phi_st_space` came back on `(22.7, 43380.7)` against a uniform prior on
  `(0.01, 10)` whose density is `-Inf` outside those bounds. The four bundles
  are built once, above the sampler branch, and both entry points assemble
  their blocks through one set of builders (`src/hmc_model_data_blocks.h`)
  rather than each writing its own extraction. `spatiotemporal()` no longer
  refuses a GP main effect.

* **A specialized gradient declares the blocks it writes (#71).**
  `resolve_gradient_fn()` selected a hand-coded gradient by a chain of
  negations, one per feature per function, and several of them omitted
  `has_spatiotemporal`, so a model carrying an interaction reached a function
  that never writes it. Each entry now names the blocks it handles and the
  model's own blocks are one mask, so a model carrying anything outside that
  set falls through to the composite; adding a block to `GradFeature` is a
  block every entry that does not name it falls through on. Three combinations
  the negations could not express fall out of it: a GP margin and a GMRF margin
  are different temporal blocks, a multiscale temporal term is not an SVC term,
  and no specialized function writes the random-slope block.

* **The composite gradient is a catch-all for every sampled field.** It had no
  NNGP arm at all -- neither the GP main effect, the multi-scale GP, nor the
  NNGP approximation of an SVC -- so a model routed to it left those blocks at
  zero while `compute_log_post()` carried them into eta. All three are written
  from the same `ratiod_gp` / `ratiod_svc` entry points the dedicated kernels
  use. A collapsed field alongside a second block routes to the numerical
  gradient of the density, its marginal being an inner Laplace at a mode that
  moves with every other block in eta.

* Twelve `cpp_gradient_check()` fields cover the combinations, each held to
  central differences at 1e-8 to 1e-7, and `cpp_gradient_dispatch()` reports the
  function a model resolves to so the mask is asserted directly rather than
  inferred from whether a gradient happens to come out right.
  `test-gp-sampler-blocks.R` fits three GP-plus-block models end to end and
  reads the draws against the prior support the densities refuse outside of.

# tulpaRatio 1.7.0

* **The GP spatiotemporal interaction exists (#68, #69).** `type = "separable"`
  reached the sampler and segfaulted on the first log-posterior evaluation, in
  every gradient mode. `SpatiotemporalData`'s NNGP members -- `nn`, `nn_idx`,
  `nn_order`, `nn_dist_space`, `nn_dist_time`, `coords`, `time_values`,
  `nonsep_type`, `cov_space`, `cov_time` -- were never assigned by anything:
  `st_params` did not carry them and the sampler entry did not write them, so
  `st_gp_nngp_log_lik()` read `nn_order[0]` on an empty vector, and the four
  scalars among them were indeterminate rather than unset. Both types are now
  built end to end. The interaction carries its own `coords`, so an areal main
  effect pairs with a continuous interaction over the same units; `nonsep_type`,
  `cov_space`, `cov_time` and `nn` are its own too, and `type = "nonsep_gp"` is
  reachable from the front door for the first time.

* **That interaction now has a gradient (#68).** `compute_gradient_composite`
  and `compute_gradient_spatiotemporal_handcoded` both skipped the GP types
  entirely, eta included, so a fit would have moved the block under nothing but
  its sum-to-zero penalty and computed every other block's gradient at a linear
  predictor missing the interaction. `st_gp_nngp_grad()` (`src/st_prior_grad.h`)
  writes the field, its precision and both ranges, sharing the conditional
  decomposition with the density through `st_gp_nngp_scan()` so the two cannot
  disagree about what the prior is. `sigma2` factors out of the neighbour block,
  which collapses its gradient to a closed form with no extra solve; the two
  ranges take one further pair of triangular solves against the factor the scan
  already built. Checked against central differences at `1e-7` to `1e-5` on four
  new `cpp_gradient_check()` fields (`stgp`, `stgp_matern`, `stgp_gneiting`,
  `stgp_latent`), on both the analytic and the templated density.

* **The interaction's covariance is one definition (#68).** The GP branch
  carried its own Cholesky, its own two triangular solves and its own
  conditional-variance floor, a fifth copy of what `ratiod_cov` (`src/hmc_cov.h`)
  holds for every other NNGP path; it now reads that one. Each arm is written as
  a correlation with `r(0, 0) = 1` and scaled by `sigma2` once, so the marginal
  variance is what the PC prior is a prior on and the neighbour block's diagonal
  is the kernel's own value rather than a constant written beside it.
  `cpp_test_st_gp_log_lik()` drives the shipped density against a dense
  multivariate normal built from the kernel formulas, at `nn = n - 1` where the
  NNGP is exact.

* **Two non-separability names are gone, for different reasons (#68).**
  Cressie-Huang (1999) had no branch of its own and was evaluated as the product
  covariance. The additive kernel `C_s(h) + C_t(u)` cannot be an interaction
  covariance on a complete grid: its rank there is `S + T - 1` of `S * T`, and
  every direction it drops is an interaction -- measured on a 6 x 5 grid, rank
  10 of 30 and smallest eigenvalue -1.6e-15, against condition numbers of 8.4
  and 10.8 for the product and Gneiting arms.

* **`spatiotemporal_gp()` is withdrawn (#68).** It emitted `type = "st_gp"`, a
  string no other R file read and the sampler entry rejected, and its
  observation-level index set had no relationship to the `S x T` sum-to-zero
  penalty the density applies. `spatiotemporal(type = "nonsep_gp",
  nonsep_type =, coords =)` is the spelling that works, and carries the same
  knobs.

* **A spatial GP indexed an interaction by row (#68).** `build_st_index()` set
  `s_idx <- seq_len(N)` for a GP spatial component while `n_spatial` counted
  unique locations, so `st_flat` ran past the `S x T` grid it indexes the moment
  two observations shared a location. It reads `obs_to_loc`.

* **`compute_gradient_latent_handcoded` was selected for models it cannot write
  (#71).** Its guard checked for no spatial and no temporal main effect and not
  for an interaction, so a latent factor alongside one returned the latent
  factors at -2.34 against a finite difference of 0.114. The remaining
  specialized guards omitting the same check are listed in #71.

# tulpaRatio 1.6.0

* **The spatiotemporal interaction's AR1 time margin exists (#66).**
  `spatiotemporal(temporal = temporal_ar1(...))` allocated a correlation and
  put a Beta prior on it, and nothing else read it:
  `spatiotemporal_log_prior()` opened with `(void)rho`, `type_ii_log_prior()`
  carried a `// AR1 would need rho parameter` comment and no branch, and
  `type_iv_log_prior()` fell through both stencils, so its Kronecker quadratic
  form was exactly zero and its `rank_time` read `rw2_rank`. The field had no
  temporal prior curvature at all, and the density and its gradient disagreed
  on the normalizer. The AR1 precision `R(rho)` is now the interaction's time
  margin on Type II, Type IV and HSGP-ST -- the quadratic form and the
  rho-dependent `log|R| = log(1 - rho^2)`, differentiated together -- and
  `ratiod_ar1::ar1_precision_apply()` / `ar1_quadratic_form()` /
  `ar1_log_prior()` (`src/ar1_shared.h`) are the one matrix the scalar AR1
  density and the Kronecker margin both read. Measured at the gradient check's
  own draw, `logit_rho_st` went from an analytic `0.000000` against a
  finite-difference `-0.299363` -- the prior alone, since the density had no
  other dependence -- to `-9.133676` against `-9.133675`.

* **One spatiotemporal interaction gradient (#66).** The two hand-coded
  gradients that reach an interaction each carried their own copy of every
  type. They had drifted: `compute_gradient_composite()` had no Type II RW2
  branch at all and normalized that arm at rank `T` instead of `T - 2`, no
  AR1 branch anywhere, and neither wrote `logit_rho_st`.
  `st_interaction_prior_grad()` (`src/st_prior_grad.h`) is the single
  implementation both now call, so a term added to the density reaches both
  gradients or neither. `st_time_rank()` is the matching single rank, read by
  the density, the non-centered rank correction and both gradients.

* **Type I and Type III no longer allocate a correlation.** Neither reads a
  time margin -- Type I is iid over the whole grid, Type III's time margin is
  unstructured -- so the parameter the layout used to allocate for them moved
  under its own prior and nothing else. A time margin an interaction reads but
  cannot express (a GP margin under Type IV) is now refused at the door rather
  than silently run as RW1.

* **`temporal_ar1()`'s correlation lives on (-1, 1) (#67).** It was mapped onto
  `(0, 1)`, so a negative autocorrelation was not reachable from that block at
  all, while the TVC arm, the multi-scale short-term arm and the spatiotemporal
  interaction all used `rho = 2u - 1`. `?temporal_ar1` documented
  Uniform(-1, 1) and `?priors_default` documented Beta(2, 2), and the code did
  neither. The mapping is now `2u - 1` with the shared Beta(2, 2) on
  `u = (rho + 1) / 2`, which **changes the posterior of every existing
  `temporal_ar1()` fit**. `ratiod_ar1::rho_from_logit()` and `drho_dlogit()`
  are the one map and the one chain rule; the eight hand-written copies of each
  are gone.

* **`temporal_ar1(rho_prior = )` is wired (#67).** It was stored on the spec
  and never read again, as was `ratiod_priors(rho_temporal = )`. Both now reach
  the density and every gradient as the Beta anchors on `u = (rho + 1) / 2`,
  the block's own prior taking precedence; a `spatiotemporal()` interaction's
  AR1 margin reads its own temporal spec's prior the same way. A `rho_prior`
  that is not a `prior_beta()` is refused at the door -- which includes
  `temporal_ar1(rho = 0.8)`, where `rho` used to partial-match `rho_prior` and
  store a bare number nothing read.

# tulpaRatio 1.5.3

* **One AR1 correlation prior, read by every density and every gradient
  (#58).** The TVC rho carried a Uniform(-1, 1) prior in `compute_log_post` and
  a Beta(2, 2) one in `compute_log_post_impl`, exactly one `log u + log(1 - u)`
  apart, so the autodiff modes sampled a different posterior from the handcoded
  one -- 1.408954 in the density and the whole gap in the gradient at
  `logit_rho_tvc`. The prior is Beta(2, 2) on `u = (rho + 1) / 2` everywhere it
  applies now, which is what `priors_default()` documents for a temporal
  correlation and what the multi-scale short-term arm already used; the
  spatiotemporal interaction's rho moves onto it from Uniform(-1, 1) as well, so
  a fit carrying one shifts by that term. `ratiod_ar1::log_prior_logit_rho()`
  and its gradient (`src/ar1_shared.h`) are the single expression all four
  correlations read -- the two densities, both handcoded gradient functions and
  the multi-scale arm -- with each site naming its own `(a, b)`, so the
  Uniform(0, 1) the plain `temporal_ar1()` block carries is the same helper at
  `(1, 1)` rather than a second spelling. Four unreachable copies of the prior
  in `hmc_tvc.h`, `hmc_tvc_autodiff.h` and `hmc_temporal_multiscale.h` are gone.
  `tvc_ar1` joins the structures in `test-gradient-correctness.R`, whose
  density-equality assertion it now answers.

* **One floored `1 - rho^2` (#59).** It was clamped at `1e-10` in five places,
  clamped at `1e-12` in three, added as `+ 1e-10` in five more, and left
  unfloored in eight -- including the AR1 marginal variance, its templated twin,
  the temporal GP's two conditional variances and the Laplace path. `+ 1e-10`
  and `max(., 1e-10)` are different functions, so paths that must agree parted
  company wherever either bound: at a correlation near the edge of its range the
  two densities were 3.14 apart on `temporal_ar1`, 27.2 on `tvc_ar1` and 1.9e-03
  on `ms_temporal`, and one step further out `compute_log_post` returned `-Inf`
  with a non-finite handcoded gradient while the templated density stayed
  finite. `ratiod_ar1::one_minus_rho2()` is the one clamp every site reads.
  `cpp_gradient_check(near_unit_rho = TRUE)` places each sampled correlation
  where the floor binds, which nothing in the sweep reached before.

* Two defects found while testing the above are filed rather than folded in:
  no gradient function writes `logit_rho_st`, so a Type IV interaction with an
  AR1 time margin moves its rho under a flat gradient and fuses a log posterior
  missing the prior (#66); and `temporal_ar1()`'s rho lives on (0, 1) while its
  documentation describes (-1, 1), with `rho_prior` inert (#67).

# tulpaRatio 1.5.2

* **PG GP fits carry their spatial field, so they can predict one (#63).**
  `convert_pg_gp_to_ratiod_fit()` read the field draws off `"w_gp"` while the
  sampler returns them under `"gp"`, and `[[` on a missing name is `NULL`, so
  `.internal$w_gp` was `NULL` on every such fit and `predict_spatial_gp_pg()`
  returned nothing without saying so. `fitted()` was unaffected -- it reads the
  linear predictor the sampler stores directly. The `(sigma2, phi)` draws that
  go with the field were also read from the first chain alone while the field
  spanned every chain, so a multi-chain fit paired half its draws with nothing;
  both are stacked in the same chain order now and the kriging entry refuses a
  pair that is not row-aligned.

* **A GP prediction is made in the units its fit was made in.**
  `spatial_gp(scale_coords = TRUE)`, the default, centres and scales the
  coordinates before the neighbour structure is built, and `scale()` puts the
  centre and scale on an attribute that subsetting the matrix drops. Nothing
  recorded them, so `coords.0` was read raw against a scaled training set on
  every backend: kriging the training locations of a fit back onto themselves
  recovered its own field at correlation -0.01. `gp_scale_new_coords()` is the
  conversion, and the same fixture now reads 0.95.

* **One kriging implementation, and one covariance family behind it.**
  `kriging_predict()` and `cov_function()` were an R triple loop over
  (draw, location, neighbour pair) and an R copy of the covariance family that
  knew three of the four kernels, resolving `"spherical"` to an exponential one
  -- the same defect fixed in the compiled family at 267f2cb, in the other copy.
  `cpp_kriging_predict()` replaces both: it reads `hmc_cov.h`, the family every
  fitting path reads, and takes the whole draw matrix at once, so the neighbour
  sets are found once rather than per draw. The HMC and PG prediction paths both
  go through it.

* **`cov_type_code()` is the one name-to-code mapping.** Three copies of it in R
  each ended in a bare `0L`, so a covariance name none of them knew ran
  exponential; one of the three was missing `"spherical"` outright. The C++ side
  meets it at `ratiod_cov::cov_type_from_int()` in `cov_type_code.h`, which the
  PG entries now share, and both ends refuse a value they do not know.

# tulpaRatio 1.5.1

* **The PG binomial GP sampler now pairs observations with locations by
  coordinate rather than by row position (#60).** `spatial_gp()` reduces the
  data to unique locations and builds the NNGP on those, but
  `cpp_pg_binomial_gibbs_gp()` was never given the map back: it aggregated the
  per-location Polya-Gamma likelihood over row position and discarded every row
  past `n_spatial`, and it wrote the field into only the first `n_spatial`
  linear predictors. On the four-observations-per-location design used to
  measure it, 75 of 100 observations reached no location at all, the 25 that did
  were credited to locations by row order, and the field never entered three
  quarters of eta. `obs_to_loc` is now passed and read on both sides. The
  coordinates handed to the sampler were also the observation-order matrix
  rather than the unique locations the neighbour structure indexes, so the
  neighbour blocks were assembled from the wrong points; `unique_coords` is what
  goes in now. Recovery of the true field over three seeds: correlation 0.95,
  0.98, 0.98.

* **`sigma2_gp` and `phi_gp` are drawn from the NNGP density the field was
  sampled under (#62).** The `sigma2` acceptance ratio was built from an
  independent `N(0, sigma2)` quadratic form carrying no log-determinant, so
  raising `sigma2` only ever reduced the penalty and the ratio pointed one way
  at every value; `phi`'s ratio was the log-scale Jacobian alone, so `phi` was
  drawn from its prior rather than its posterior. Both now difference
  `sum_k log N(w_k; mu_k, d_k)` over the sampler's own conditioning order,
  evaluated by the same conditional the field sweep draws from. The `sigma2`
  prior's change of variables was also carrying a full `log(sigma2)` where the
  PC prior on `sigma = sqrt(sigma2)` gives `+0.5 log(sigma2)` once the walk's
  Jacobian is included. Against a true `sigma2` of 0.7, posterior medians over
  three seeds are 0.36, 0.60 and 0.65, where the issue measured `[94, 1746]` and
  `[24.9, 11568]`.

* **`spatial_multiscale(cov = )` is read by the multiscale PG sampler (#61).**
  `cov_type` was accepted by `cpp_pg_binomial_gibbs_multiscale_gp()` and never
  mentioned again; an exponential kernel was written out inline at six sites, so
  three of the four documented choices silently ran the fourth. Both scales now
  go through the same `pg_nngp_conditional()` the single-scale entry uses. That
  also replaces what stood beside the kernel: the conditional mean was a raw
  covariance-weighted sum of the neighbours in place of the NNGP weights
  `C^-1 c`, the conditional precision was `1 / sigma2` in place of the
  conditional variance, `nn_idx` was indexed by location and its values read as
  locations where both are positions in the conditioning order, and each
  scale's `sigma2` was a conjugate draw under an independent `N(0, sigma2)`
  form that read the PC prior's upper bound as an inverse-gamma rate and its
  tail probability not at all.

* **One convention, checked at the entry.** `PgNngpGraph` carries the neighbour
  structure the two entries share, `pg_check_nngp_graph()` refuses a `nn_order`
  that is not a 0-based permutation of the locations, a `nn_idx` entry that is
  not an earlier position in the order, and a coordinate matrix that is not one
  row per location, and `pg_check_obs_to_loc()` does the same for the
  observation map. The 1-based `nn_order` fixed in 267f2cb was a silent
  out-of-bounds read; it is now an error naming the argument.

# tulpaRatio 1.5.0

* **One walk of the sampler's parameter vector now feeds `fitted()`,
  `predict()`, `ratio()`, the draws matrix and the per-structure extractors
  (#44).** There were four copies of that walk. Three of them were thinner than
  the fourth: on the default non-centred parameterization they added the raw
  `z` to the linear predictor instead of `sigma_re * z`, so every fitted value
  and every prediction on a model with random effects was off by the posterior
  draw of `sigma_re`; they recognised one intercept-only random-effect term, so
  a crossed or slope-carrying model read parameters that were not random
  effects at all; they recognised `icar` and `bym2` among the spatial types and
  dropped every other field from the predictor without saying so; and they
  applied the logistic link to every family, reporting a number in (0, 1) where
  a count ratio belongs. `fitted(fit, component = "ratio")` and `ratio(fit)`
  now return the same draws, and so does `predict(fit, newdata = fit$data)`.

* **The block order the R side reads is the order the sampler writes (#44).**
  `hmc_param_layout()` (`R/hmc_unpack.R`) mirrors `compute_param_layout()` in
  `src/hmc_sampler.cpp` block for block, and is the only place an offset is
  computed. The previous walk placed SVC before the areal field, the GP before
  the temporal block, TVC before zero-inflation and HSGP among the spatial
  types, where the sampler lays them out in a different order; a fit combining
  two of those structures had its columns named after the wrong parameters from
  the first mismatch onward. It also walked a temporal block for a multiscale
  temporal model, which the sampler allocates none for. Every fit now checks
  that its layout accounts for exactly the columns the sampler returned, and
  errors rather than mislabelling them.

* **`svc()`, `tvc()`, `temporal()` and `spatiotemporal_effects()` return their
  draws instead of erroring on every fit (#46).** No backend stored the draws
  they read. The converter now stores each structure's own draws, shaped the
  way its extractor indexes them, and the `svc()` / `tvc()` error text names the
  argument that routes them (`spatial = spatial_svc(...)`,
  `temporal = temporal_tvc(...)`) rather than an argument `tratio()` does not
  have. The four `\dontrun{}` examples that showed the wrong spelling are
  fixed. `tvc()` gained a `group` argument, since the draws carry one
  trajectory per grouping level.

* **`pp_check()` plots a fit (#48).** It read `object$draws$y_num_rep`, and
  `object$draws` is a matrix on every backend, so the call raised
  `$ operator is invalid for atomic vectors` on its second executable line and
  never reached the guard written to explain the missing draws. The replicates
  are now drawn rather than looked up: `posterior_predict()` simulates a
  numerator and a denominator response from each posterior draw's fitted means
  and that draw's dispersion, through the same family machinery
  `prior_predict()` uses.

* **`as_draws()`, `pp_check()` and `posterior_predict()` are methods on the
  generics that own them, not new generics (#52).** They were bare
  `UseMethod()` calls, so attaching tulpaRatio after posterior or bayesplot
  masked their generic and a user's `as_draws()` on a `draws_matrix` or
  `pp_check(y, yrep)` on plain vectors dispatched into this package and found
  no method. `posterior::as_draws`, `bayesplot::pp_check` and
  `tulpa::posterior_predict` are imported and re-exported instead; bayesplot
  moves from Suggests to Imports.

* **`spread_draws()`, `gather_draws()` and `point_interval()` are renamed to
  `draws_wide()`, `draws_long()` and `draws_interval()` (#52).** tidybayes owns
  those three names and takes different arguments under them (`variable[index]`
  specifications rather than bare parameter names), so a method on its generic
  would not have honoured the contract. The old names are removed.

* **The offline gradient check can build a model for every kernel the
  dispatcher can select (#45).** `cpp_gradient_check` covered the areal fields,
  the random-effect layouts, HSGP and TVC; it could not build a GP, multi-scale
  GP, SVC, temporal-GP, multiscale-temporal or latent-factor model, so the
  handcoded gradients those run on under the default mode had no
  finite-difference case. `make_model()` now builds `gp`, `gp_collapsed`,
  `gp_temporal`, `msgp`, `msgp_temporal`, `svc`, `svc_hsgp`, `temporal_gp`,
  `ms_temporal`, `latent`, `tvc_ar1` and `temporal_ar1`, and an unrecognised
  field name is an error rather than a model with no structured field in it,
  which would pass while testing nothing. The temporal block's AR1 arm carries
  a rho that its RW1 and RW2 siblings have no counterpart for, so the two cases
  that reached it before covered none of the terms that rho enters.

* **The runtime gradient check differences away from the origin (#45).** Every
  structured block starts at exactly zero, where a quadratic-form prior
  contributes nothing to any gradient: a wrong precision matrix, a sign error
  on it and an absent sum-to-zero penalty are all invisible there. The check
  now moves each field block off zero by a small deterministic offset first.
  Both call sites share one function, so a mismatch raises an R-level warning
  from the multi-chain path as well as the single-chain one.

* **The PG binomial GP sampler indexed one past the end of every per-location
  array, and could take R down with it.** `backend_pg.R` passed `nn_order`
  1-based while the C++ uses its values directly as indices into `w`, `coords`
  and the per-location likelihood accumulators, all sized `n_spatial`, so each
  could reach `n_spatial` itself. `backend_hmc.R` converts the same vector with
  `- 1L`; the PG path did not. The symptom was `STATUS_HEAP_CORRUPTION`
  (`0xC0000374`) killing the R process, reached at some parameter states and not
  others -- 20 locations crashed at `nn = 5` and `nn = 6` while 25 locations
  survived -- and, where it did not crash, the GP effects were attached to the
  wrong locations.

* **`spatial_gp(cov = )` runs the kernel you asked for under `mode = "pg"`
  (#35).** The selector tested `cov_type` 0 and 1 and sent everything else to an
  unconditional `else` computing Matern 2.5, so `"gaussian"` and `"spherical"`
  -- both documented and `match.arg`-validated -- ran a kernel the user did not
  choose. It now routes through the shared `ratiod_cov` family introduced above,
  and an unrecognised code is an error rather than whichever kernel a
  fallthrough reaches. The neighbour block goes through the shared
  factorization at the same time, replacing a seventh distinct regularization
  (a `1e-10` value substituted into the pivot). Measured on a 12-location
  fixture, against an NNGP conditional written out independently in R: all four
  kernels now agree to machine precision, and at the probed location the
  conditional mean moves by 0.11 for `"gaussian"` and 0.34 for `"spherical"`
  against the Matern 2.5 both used to return.

* **`tratio(mode = "pg")` honours `control$seed` and `control$chains`.**
  `fit_pg_binomial()` accepts both and threads `seed` down to a per-chain
  `set.seed(seed + chain - 1)`, but `tratio()` called it without either, so PG
  fits ran off whatever RNG state they inherited -- two runs of the same call
  differed by 1.23 on the intercept and 3.19 on `sigma_re` -- and always used
  the backend default of 4 chains whatever `chains` said. Runs at a fixed seed
  are now bit-identical.

* **`spatial_gp(parameterization = "noncentered")` gets its handcoded gradient
  back (#57).** Two defects made it disagree with the density it reports, and
  `resolve_gradient_fn` sends every GP model at the default gradient mode to
  `compute_gradient_gp_handcoded`, so this was the ordinary path for such a fit
  rather than an opt-in kernel. `nngp_nc_backward` read a `C_mat` it never
  filled -- the comment beside it said it deliberately did not rebuild one --
  which made every `dC/dphi` entry zero and so dropped both the `-dC * alpha`
  term from `dalpha` and the `alpha' dC alpha` term from `dd_dphi`; and the
  caller subtracted `z` after `nngp_nc_backward` had already seeded `grad_z`
  with `-z`, applying the `N(0, 1)` prior twice. Measured on the harness: worst
  relative deviation `4.610e-01` at `log_phi_gp`, flat across difference steps
  from 1e-3 to 1e-7, and the residual on every one of the 25 `z` entries equal
  to `-z` to 1.1e-07 with a fitted slope of 0.99999997. Both now measure
  `0.000e+00` with a V-shaped sweep.

  The fits themselves were slow rather than wrong: the runtime gradient check
  introduced in this release catches the `log_phi` mismatch at its probe point
  and falls back to numerical gradients, which is what a non-centred GP fit had
  been running on. That fallback no longer fires. Before this release the check
  differenced at the origin, where `z = 0` makes `w = 0` and both defects
  contribute exactly nothing, so it saw neither.

* **One covariance family and one neighbour-block factorization behind every
  NNGP path (#42, #55, #56).** The GP path, the SVC path and their two
  templated twins each carried a copy of the four kernels and of the small
  dense Cholesky over a neighbour set, and the copies had drifted apart on four
  separate numbers: the Gaussian kernel was written with and without the 0.5 in
  its exponent, Matern 3/2 carried `sqrt(3)` exactly in two of them and rounded
  to `1.732050808` in the third, one had no spherical case and fell through to
  exponential, and the neighbour block was regularized four different ways
  (`1e-8` always, `1e-8` only once a pivot had already gone non-positive, a
  `1e-6` floor substituted into the pivot, `1e-4` always) with the conditional
  variance floored three more (`1e-10`, `1e-6`, and a `1e-4` blend keeping 1%
  of the gradient). The analytic gradient paths factorized the block a third
  time again, with no ridge and a pivot replaced by `1e-5`. So the value a fit
  reported and the gradient it moved on came from different copies. Measured
  before: the two densities were 30 nats apart on the NNGP SVC and 0.086 apart
  on a Matern GP, with the deviation flat across difference steps from 1e-3 to
  1e-7. Everything now routes through `src/hmc_cov.h`, templated once for the
  double evaluation and the `ad::Var` / `fwd::Dual` / `arena::Var` gradients
  alike; across `gp`, `gp_matern`, `gp_gaussian`, `gp_spherical`, `gp_temporal`,
  `msgp`, `msgp_temporal`, `svc` and `svc_hsgp` the two densities now agree to
  the printed digit and every gradient matches central differences of both to
  2e-06 or better.

* **`cov = "gaussian"` is `exp(-0.5 (d / phi)^2)` on every path.** The analytic
  path evaluated `exp(-(d / phi)^2)` while taking its `phi` derivative from the
  other form, so it was running on half the derivative of the density it
  reported; the templated path already used the 0.5. The 0.5 is what makes
  `phi` a lengthscale here as it is for the other three kernels -- it is the
  standard squared-exponential and the `nu -> infinity` limit of Matern at a
  fixed `phi` -- so the shared kernel keeps it. A `phi` estimated by a previous
  `cov = "gaussian"` fit under a handcoded gradient mode corresponds to
  `phi / sqrt(2)` now; a fit under `A`, `A_r` or `A_t` is unchanged.

* **`cov = "spherical"` is differentiated rather than silently run as
  exponential (#42).** Neither analytic derivative had a spherical case and the
  templated dispatcher had no spherical kernel, so both fell through to the
  exponential arm. `dk/dphi = 1.5 sigma2 (r - r^3) / phi` is written out, which
  is why `dcov_dphi` now takes `sigma2` -- it is the one case not expressible
  through the covariance value. An unrecognised covariance name is an error at
  both string parsers, where the GP one used to map anything it did not
  recognise to spherical and the SVC one to exponential.

* **A TVC AR1 field's `log tau` gradient had the wrong sign on its stationary
  normalizer (#43).** The AR1 prior parameterizes the stationary variance as
  `1 / (tau (1 - rho^2))`, so `-0.5 log(2 pi var)` differentiates to `+0.5` in
  `log tau`; `ar1_grad_log_tau` seeded the sum at `-0.5`, leaving the reported
  gradient short by exactly 1 at every value of every parameter (4.46185
  against a finite difference of 5.46185 on the harness fixture). The three
  AR1 gradient functions, `rw1_grad_w` and `rw2_grad_w` also read past their
  field on a block too short to carry an increment or a second difference;
  those priors are flat there, and they now return that instead.

* **The fused log posterior of the multi-scale GP and latent-factor gradients
  carries the binomial coefficient.** `compute_obs_ll()` dropped it while
  `compute_log_post()` adds it back, so those two kernels reported a value
  7156 nats away from the density their gradient describes on a binomial fit.
  NUTS consumes that value as `log_prob`.

* Prediction maps a new data set onto the fit's own grouping levels through the
  same construction the formula parser used (`as.factor()`, or `interaction()`
  for a nested term). It previously matched against `unique()` order, which
  differs from the factor's level order whenever the levels are not in order of
  first appearance, and attached the wrong group's effect.

# tulpaRatio 1.4.3

* **Zero-inflation and hurdle terms on count responses now reach the templated
  log posterior, so `mode = "ess"` and the `A`/`A_r`/`A_t` gradient modes stop
  silently fitting a plain Poisson/NegBin (#34).**
  `compute_log_post_impl` read `data.zi_type` only inside its
  `ModelType::BINOMIAL` arm. The three count-response arms --
  `NEGBIN_NEGBIN`, `POISSON_GAMMA` and `NEGBIN_GAMMA` -- called the plain
  `log_lik_negbin`/`log_lik_poisson` whatever `zi_type` said, while still
  allocating and sampling a ZI coefficient against a likelihood that did not
  contain it. That density is the one ESS evaluates and the one the three
  autodiff gradient modes differentiate, so
  `tratio(..., family = ratiod_zinegbin(), mode = "ess")` (and
  `ratiod_zipois()`, `ratiod_hurdle_negbin()`, `ratiod_hurdle_pois()`, or an
  explicit `zi = zi_negbin()` and friends) completed with no error and
  returned a zero-inflation logit drawn from its prior. Measured on the
  finite-difference harness, the templated density returned the same number
  for `zi_poisson`, `hurdle_poisson` and no zero-inflation at all, and
  disagreed with the analytic density by between 1.7 and 46 log units across
  the six family/structure combinations; both now agree to 1e-11, and the
  analytic ZI gradient -- which nothing had ever checked on a count response
  -- matches central differences to machine precision in the `handcoded` and
  `arena` modes alike. `hmc_zi.h`'s count-response densities are templated on
  the scalar type rather than duplicated, so `compute_log_post` and
  `compute_log_post_impl<T>` evaluate one implementation. `make_model()` in
  `test_gradient_check.cpp` gained a `zi` argument and the `negbin_negbin` and
  `negbin_gamma` families, since it previously hardcoded `ZIType::NONE` for
  every call site and no test combined `mode = "ess"` with any ZI family.

* **`temporal_tvc(structure = ...)`: `"iid"` unblocked, `"gp"` now errors
  instead of silently downgrading to RW1, and the Gibbs backend now rejects
  anything but `"rw1"` instead of silently ignoring the argument (#32).**
  `TemporalType::IID` had a complete log-prior and gradient in both the H and
  autodiff paths (`tvc_term_log_prior`, `tvc_prior_gradients_ws`, and the
  templated density via `hmc_tvc_autodiff.h`'s `using` declarations), and the
  C++ string parser already recognized `"iid"` -- but `temporal_tvc()`'s
  `match.arg` only offered `"rw1"/"rw2"/"ar1"/"gp"`, so no R call could ever
  reach it. `structure = "iid"` is now accepted, verified against
  `test_gradient_check.cpp`'s finite-difference harness in both H and A_r
  modes (0 deviation), and recovers a simulated intercept and slope in
  `test-recovery-constrained-fields.R`. Separately, `"gp"` was accepted and
  documented as a first-class option, but `TVCData` has no GP fields at all
  (no lengthscale, no covariance branch) -- the C++ parser silently mapped
  any unrecognized string, `"gp"` included, to RW1, so
  `temporal_tvc(..., structure = "gp")` ran to completion and silently fit an
  RW1 varying coefficient. `temporal_tvc()` now errors immediately if
  `structure = "gp"` is requested, rather than reaching the sampler at all.
  Independently, the Gibbs backend's TVC update
  (`# ---- 5. Update TVC coefficients via univariate MH with RW1 conditional
  proposal ----`) is hardcoded to RW1 and never read the `tvc_structure` field
  R was already sending it, so `mode = "gibbs"` silently ran RW1 regardless of
  what `structure` asked for; the Gibbs backend now rejects any TVC structure
  but `"rw1"` with a clear error pointing at `mode = "hmc"`. The C++ side's
  own duplicated string-to-enum parser in `hmc_sampler.cpp` was replaced with
  a call to the existing `ratiod_tvc::parse_tvc_structure()`, which had no
  callers before this.

* **`spatial_car(..., proper = TRUE)` now fits a proper CAR instead of
  silently downgrading to ICAR (#31).** The R object already computed rho
  bounds and printed "Proper CAR"; a complete C++ implementation
  (`hmc_car_proper.h`) already existed too. Neither was wired to the other --
  every backend collapsed `type = "car_proper"` to `"icar"` before it reached
  C++ (`# Handle all ICAR-type models (car, car_proper both use ICAR
  parameterization)`), so `rho` stayed fixed at 1 and the roxygen example's
  claim that `summary(fit_car)` shows a `rho_spatial` parameter was false. The
  HMC backend now threads `rho` through as its own parameter (`Q(rho) = D -
  rho*W`, `Beta(1,1)` prior by default, logit-transformed), added to
  `SpatialType`, `ParamLayout`, and the shared `spatial_gmrf_prior_grad`/
  `compute_log_post` machinery that `compute_gradient_composite` and every H
  gradient mode already read from. `phi`'s gradient is exact; `rho`'s
  log-determinant term has no cheap closed form (it needs `trace(Q^-1 W)`), so
  it is central-differenced on that one scalar and chain-ruled to logit scale,
  verified against `test_gradient_check.cpp`'s finite-difference harness (0
  deviation across all 29 parameters on the `car_proper` field). Proper CAR is
  full rank for `rho < 1`, so unlike ICAR/BYM2 it is not hard-centered.
  Autodiff gradient modes (`A`, `A_r`, `A_t`) refuse at the front door with a
  clear error rather than differentiate a density missing the field's prior
  entirely, since the log-determinant is not yet expressed in the templated
  density; `mode = "hmc"` with the default `gradient_mode = "auto"` (or `"H"`)
  is unaffected. Gibbs already rejected `car_proper` (its type whitelist never
  included it); Laplace, PG, VI, SGHMC, and ESS now reject it explicitly too,
  instead of silently fitting an ICAR model or -- for the three backends that
  call `compute_log_post_impl<double>` directly -- silently dropping the
  spatial prior's contribution to the posterior altogether. Fixing this also
  surfaced a second, independent bug in `predict()`'s areal-prediction path,
  which read `phi_spatial` starting one column early for any type it treated
  as "ICAR-shaped" once `car_proper` genuinely had an extra `rho` parameter
  in front of it.

* **A spatiotemporal Type IV interaction has a non-centered parameterization,
  which fixes chains freezing on data with little or no true interaction
  (#24).** `spatiotemporal(..., type = "IV")` reached `rhat` 1.14 at
  iter=1000/warmup=500 on data simulated with no space-time signal, and a 4x
  longer run moved that to 2.05 while every chain's own posterior sd shrank.
  The interaction's precision `tau` has no likelihood signal pulling it away
  from the boundary in that case, so its posterior drifts to very large
  values (tau ~ 4.3e8 was observed at warmup's end); under the centered
  parameterization the field's conditional scale shrinks as `1/sqrt(tau)`,
  so exploring both regions needs step sizes that differ by orders of
  magnitude, and NUTS answered by pinning the trajectory length at
  `max_treedepth` on essentially every iteration and collapsing the step size
  toward its floor -- the signature of a funnel between a variance component
  and the field it scales, not of a chain that merely needs more draws. A
  finite-difference check of `compute_gradient_spatiotemporal_handcoded`'s
  Type IV (Kronecker) block, absent until now, confirmed the analytic
  gradient was not the cause. `spatiotemporal(..., parameterization =
  "noncentered")` samples a tau-free `z` and reconstructs the interaction as
  `z / sqrt(tau)`, decoupling the two: `rhat` on the same repro drops to 1.00
  at iter=1000 and stays at 1.00 at iter=4000, with stable per-chain sd and
  a step size about 10x larger. The non-centered path already existed in the
  gradient code but was never reachable from R and was itself unchecked
  against finite differences; the warmup-end precision-informed mass-matrix
  warm-start specific to Type IV (which reads the raw parameter as the
  centered field directly) is now skipped under the non-centered
  parameterization, which it does not describe. `parameterization` still
  defaults to `"centered"`, since data that do carry real space-time
  structure move `tau` away from the boundary and may not need it.

* **The Gibbs BYM2 sampler no longer leaks the spatial field's level into the
  intercept unchecked (#15).** The per-site update held the structured field
  (`phi`) to sum zero by moving its mean into the intercept every sweep,
  unconditionally: no acceptance test weighed that move against the
  intercept's own prior. `dev_notes/gibbs_prior_recovery.R` (now
  `tests/testthat/test-gibbs-prior-recovery.R`) fits with zero binomial trials,
  where the likelihood is flat and the posterior must equal the prior exactly;
  under a `N(0, 0.5^2)` intercept prior the sampled sd came out at 1049 instead
  of 0.5, a ~2000x inflation. The move now competes for acceptance like every
  other update that touches the intercept, and gained a dedicated block-level
  move (mirroring the one the unstructured component already had) so the
  now-resisted level direction still mixes: BYM2 intercept ESS is 749 at
  iter=2000/warmup=1000 where it used to need the same bar ICAR clears at
  1000/500. The Gibbs priors already matched `compute_log_post`'s (Half-Cauchy
  `sigma_total`, Uniform(0,1) `rho`, Gamma `tau`, threaded through `fit_gibbs()`
  via `priors`) from prior work on this issue; this closes out the remaining
  checklist items, including removing the unused `pc_lambda` left over from an
  earlier tau update.

* **A per-fit `cores` budget can now reach within-chain work (#17).** For a
  single-chain HMC fit, `cores` used to be capped at `chains` before the user
  ever saw it (`min(chains, cpp_get_max_threads())`), so a one-chain fit always
  got exactly one thread for its gradient and likelihood loops regardless of
  machine size, and the regions that do carry a `num_threads()` clause
  (`compute_gradient_analytical`'s fallback path, the `linalg_fast.h` kernels)
  had no way to reach more than one. `cores` is now a genuine budget split two
  ways: `concurrent_chains = min(cores, chains)` chains run at once, each with
  `cores %/% concurrent_chains` threads of its own. Reaching that width when
  chains also run concurrently needed OpenMP nesting enabled
  (`omp_set_max_active_levels(2)`): a region inside a chain body is a nested
  one, and OpenMP pins nested teams to one thread by default. Most common model
  configurations (plain Poisson-gamma/binomial/negbin with simple random
  effects) take a vectorized Eigen gradient path that never reaches these OMP
  regions at all; the scalar fallback that does carry them serves zero-
  inflation, correlated random slopes, and GP/multiscale spatial models.

* **The two log posteriors are now the same function (#18).** `compute_log_post`
  and the templated `compute_log_post_impl<double>` that the `A_r` / `A` / `A_t`
  gradient modes differentiate returned different numbers on every model and
  omitted different structures. For those modes the runtime gradient check
  differences that same templated density, so a structure missing from it was
  invisible to both the gradient and its own numerical reference: they agreed on
  the same wrong value. Differencing the two densities against each other
  instead, over every structure the parameter layout allocates for, is what
  found the following.

  | structure | before | after |
  | --- | --- | --- |
  | HSGP spatial field | absent: all 11 parameters gradient exactly 0, worst relative deviation 1.00 | 1.3e-7 |
  | random slopes, uncorrelated | slope variance absent, 1.00 | 1.2e-9 |
  | random slopes, correlated | Cholesky factor and variance absent, 1.08 | 1.0e-8 |
  | crossed random intercepts | second term absent: 5 effects and its variance 0, 1.33 | 1.2e-9 |
  | TVC precision | a different prior on each side, 0.376 | 4.2e-10 |
  | collapsed ICAR / BYM2 | read `params[-1]`, access violation | refused |

  The residual constant that remained on every model was the binomial
  coefficient: the eta-form likelihood drops it as constant in eta, which is
  right for a Laplace inner solve and wrong for a log posterior that is reported
  and compared. `lp - lp_impl` was a flat -7156.27 across all fields and is now
  0 to floating-point noise.

* **A collapsed spatial field refuses an autodiff gradient mode instead of
  crashing.** The collapsed parameterizations marginalize the field out by
  locating its mode with Newton and adding a Laplace correction, which is not a
  closed-form function of the parameters, so the templated density cannot
  express them and the layout allocates no slot for the field. Reading that
  absent slot was an access violation. `gradient_mode` in `"A_r"`, `"A"`, `"A_t"`
  now errors at the front door naming the parameterization; `"H"` and `"N"` carry
  the analytic density and are unaffected. The same declaration covers the HSGP
  multi-scale GP and the non-centred GP, which are expressible but not written
  yet (#26).

* **The TVC precision takes the prior it is configured with.** `tvc_tau_shape`
  and `tvc_tau_rate` reach the sampler from `priors` and defaulted to
  `Gamma(2, 0.5)`, but two of the three implementations ignored them for a
  hard-coded PC prior with `P(sigma > 1) = 0.01`. The density, the specialized
  gradient and the composite gradient now all read the configured
  hyperparameters, through one derivative helper.

* **The random-effect prior is one implementation.** `hmc_re.h` computes the
  prior and the effective effects for every layout the sampler allocates -- one
  or more crossed terms, intercept-only or with slopes, correlated or not,
  centred or non-centred -- and both densities call it, so a term cannot be added
  to one and not the other. The TVC quadratic forms, per-term prior and eta
  assembly move the same way: templated in `hmc_tvc.h` and re-exported to
  `ratiod_tvc_ad`, replacing copies that had already drifted on the cyclic
  wrap-around edge. The RW1/RW2/AR1 templates take a pointer, so a caller holding
  a strided slice passes an offset rather than copying into a vector.

* `cpp_gradient_check()` reaches the structures it could not build before
  (`hsgp`, `tvc`, `re`, `re_crossed`, `re_slopes`, `re_slopes_corr`,
  `icar_collapsed`, `bym2_collapsed`), differences either density via
  `reference =`, and reports `log_post_impl` and `impl_gap` alongside the
  gradient. `test-gradient-correctness.R` asserts across every field that the two
  densities return the same number, that no block has an identically-zero
  gradient under any autodiff mode, and that a collapsed model refuses one.
  Its tolerance now allows an absolute `1e-6` before the relative comparison: a
  central difference of a log posterior of magnitude 1e3-1e4 resolves a gradient
  entry to about 1e-7, and the HSGP basis coefficients run to 6e-4, where 1e-4
  relative asked for more than the difference can deliver.

# tulpaRatio 1.4.2

* **Every intrinsic field is now identified at the engine's constant (#12).**
  `s2z_precision(n)` returns `1 / (kappa * n)^2` from `sd(sum phi) = kappa * n`,
  so callers pass a PRECISION. The spatiotemporal and TVC penalties took that
  constant as an argument and their call sites filled it with a bare `0.001` --
  which is `kappa` itself, weaker by ~2.5e6 at `T = 20`, pinning the field's sum
  at sd 31.6 instead of 0.02. Each penalty now derives the constant from the
  length of the sum it pins, so no call site can supply one. The interaction
  margins take their own constants rather than sharing one: a space margin sums
  `S` terms and a time margin `T`.

* **SVC keeps its own constant, deliberately.** Its penalty is rewritten on the
  sum rather than the mean -- the two forms are the same penalty, related by a
  factor of `n_obs` -- and both it and its gradient now read one
  `svc_sum_to_zero_ridge()`. That constant is NOT `s2z_precision(n_obs)`: an
  NNGP field's prior is proper, so its mean is identified and what is wanted is
  a ridge, not the pin a rank-deficient field needs. Substituting the intrinsic
  constant was tried and reverted -- at `N = 80` it is 156.25 against the
  ridge's 0.0125, and at that stiffness 4 chains x 1000 iterations return in 9
  seconds with per-chain posterior SD 0, Rhat 15.1 and the slope at 0.095
  against a truth of 0.3, where the ridge samples for minutes. Deriving the
  right constant for a proper field on its own terms is #25.

* **The multiscale trend and seasonal arms are pinned (#12).** Neither carried a
  sum-to-zero term on any path. Both are intrinsic and both enter the same
  linear predictor, so each had a constant null direction aliased with the
  intercept and with the other -- a two-dimensional ridge the fitted series
  hides, because only the level moves. Both are now pinned in the log posterior
  and in the analytic gradient; the proper short-term arm (AR1/IID) identifies
  its own level and is left alone. On a 500-observation binomial fit against a
  known intercept of 0.5, over 4 chains:

  | trend + seasonal | intercept | posterior SD | Rhat | slope (truth 0.3) |
  |---|---|---|---|---|
  | RW1 + cyclic seasonal, before | 2.6101 | 7.1840 | 1.589 | 0.2886 |
  | RW1 + cyclic seasonal, after | 0.5054 | 0.0170 | 1.012 | 0.2878 |
  | RW2, before | 3.7200 | 1.1098 | 2.163 | 0.2926 |
  | RW2, after | 0.4868 | 0.0167 | 1.000 | 0.2907 |

  The slope is right either way, which is why nothing downstream flagged it.

* **The spatiotemporal interaction's margins are identified (#12).** Same fit
  with a Knorr-Held Type IV interaction over 9 units x 5 times: the intercept
  posterior SD moves from 0.1106 to 0.0109 and Rhat from 3.43 to 1.14. It does
  not reach 1.05, and a 4x longer run makes it worse rather than better; that
  residual is tracked separately as #24.

* **The cyclic RW rank is no longer overcounted (#12).** RW1 and RW2 ranks were
  computed inline as `cyclic ? T : T-1` / `T-2` at 16 sites. A cycle-graph
  Laplacian still annihilates the constant, so a cyclic RW1 has rank `T-1`, and
  a cyclic RW2 `T-1` as well (a linear ramp is not periodic). Every site now
  calls `tulpa::rw1_rank()` / `rw2_rank()` from the engine, which cannot drift
  from the engine's own normalizers.

* **The spatiotemporal analytic gradient matches the penalty it differentiates.**
  Three further `0.001` literals sat in the ST gradient paths, including the one
  feeding the fused log-posterior reconstruction. With the old constant the
  resulting mismatch was too small to see; at the correct precision
  `test-gradient-correctness.R` fails on `icar_st` and `bym2_st` until the
  gradient carries the same per-margin constants. `hmc_sampler.h`'s
  `build_and_factorize()` keeps its `0.001`, renamed `diag_ridge`: it is a
  Tikhonov ridge `lambda*I` that makes a Kronecker of two intrinsic precisions
  factorize, not the rank-1-per-margin sum-to-zero penalty.

* **The multiscale, TVC and SVC sum-to-zero terms are written once.** The
  double and autodiff twins of the multiscale kernels, `tvc_sum_to_zero_penalty`
  and `svc_sum_to_zero_penalty` are collapsed into one body templated over the
  scalar type. The missing multiscale pin survived a previous migration pass
  precisely because it had to be found in six places; there is now one.
  `src/soft_sum_to_zero.h` is gone in favour of the engine's
  `tulpa/soft_sum_to_zero.h`.

* **Recovery tests cover the fields that were unconstrained.**
  `test-recovery-constrained-fields.R` gains multiscale (RW1 trend + cyclic
  seasonal, and RW2 trend), TVC and spatiotemporal cases. Shape and class
  assertions pass just as happily when a field's level is free, which is how
  all three items reached a release. The TVC and spatiotemporal cases assert
  recovery but not `Rhat`: neither model mixes to 1.05, which reproduces on the
  released build for TVC (#23) and is a separate residual for the interaction
  (#24). Both comments say so and name the issue to restore them under.

# tulpaRatio 1.4.1

* **The spec path's ICAR / BYM2 field is identified (#19).** The bridge that
  builds the engine's `ModelData` assigned the CSR adjacency without the
  component partition the sum-to-zero augmentation iterates over, so the field's
  constant direction carried no precision and its level random-walked:
  `mean(phi_spatial)` reached +200.9, -342.8 and +423.1 at seeds 42, 43 and 44
  where the legacy backend holds it within 0.02 of zero, and `tau_spatial` came
  out at 16.89 against the legacy 11.90. The field's shape and the fixed effects
  were unaffected, which is why it read as a healthy fit. The bridge now calls
  `ModelData::set_spatial_adjacency()`, which derives the partition and the
  component count from the adjacency (tulpa 0.0.108). The two B1d spec-vs-legacy
  parity tests run again, having been skipped since the defect was filed.

# tulpaRatio 1.4.0

* `tratio()` replaces `ratiod()` as the package's front door, joining the
  `tulpa*` family of fitting verbs alongside `tulpa::tulpa()` and
  `tulpaObs::tobs()`. The `ratiod_*` family constructors, the `ratiod_fit`
  class, and every other `ratiod_`-prefixed name are unchanged.

* `tratio()` carries statistical arguments in its signature and every perf,
  numerical, and tuning knob in a single `control = list()`, matching the
  convention the engine and tulpaObs already follow. Moved into `control`:
  `chains`, `iter`, `warmup`, `thin`, `cores`, `seed`, `verbose`,
  `adapt_delta`, `max_treedepth`, `metric`, `riemannian`, `gradient_mode`,
  `re_param`, `vi_variant`, and the stochastic-gradient knobs (`batch_size`,
  `epsilon`, `alpha`, `schedule_*`, `use_schedule`) that previously arrived
  through `...`. An unrecognised knob is now an error rather than a silent
  no-op, and a `warmup` that would leave no post-warmup draws is rejected.

* `refresh` is removed. It was a documented argument that no backend ever
  read, so passing it changed nothing.

## Bug fixes

* Spatiotemporal models fitted under an autodiff gradient mode now sample the
  density they are meant to (#14). The log posterior is written twice, once as
  `compute_log_post` and once as the templated `compute_log_post_impl` that the
  `arena`, `forward`, and `tape` modes differentiate, and the spatiotemporal
  interaction was present only in the first. The interaction parameters
  therefore carried no prior and contributed nothing to either linear
  predictor, so their gradient was exactly zero and every other parameter was
  differentiated against a linear predictor short one effect. The
  interaction now enters the templated log posterior as well, covering the
  Knorr-Held types I to IV, the separable and non-separable GP forms, the HSGP
  basis form, and the non-centred Type IV parameterization. The interaction
  math in `src/hmc_spatiotemporal.h` is templated in place rather than copied
  into a second autodiff header, so both paths call one implementation.
  `gradient_mode = "handcoded"` was correct throughout and is unchanged, to the
  last bit.

* The samplers no longer hold their per-thread scratch in the shape that
  corrupts the heap on Windows (#16). Every workspace in `hmc_sampler.cpp` was
  a `static thread_local` object owning heap buffers. Under the mingw toolchain
  such an object gets emutls-backed storage and a destructor registered through
  `__cxa_thread_atexit`, and the two are released by thread-exit hooks whose
  relative order is unspecified, so the destructor can free through storage that
  is already gone and the process dies at some later unrelated free. A worker
  thread exits only when libgomp narrows a team, which is why the fault appeared
  when a fit asked for fewer chains than the one before it. The 42 declarations
  now go through `RATIOD_TLS_WORKSPACE` (`src/tls_workspace.h`), which keeps the
  workspace behind a constant-initialized pointer: no initialization guard, no
  destructor registration, nothing for a dying worker to free. No object in the
  package registers a thread-exit destructor any more.

* `tratio()` no longer fails against a tulpa version the requirement allows. It
  calls `tulpa::tulpa_check_control()`, exported from tulpa 0.0.99, while
  DESCRIPTION asked only for `tulpa (>= 0.0.95)`.

# tulpaRatio 1.3.1

* `diagnostics()` replaces `mcmc_diagnostics()`, following the engine rename in
  tulpa 0.0.95. It delegates to `tulpa::diagnostics()`, which selects chain or
  approximation diagnostics from the fit's draws provenance rather than from the
  function's name. `mcmc_diagnostics()` is deprecated and still returns the same
  value.

## Bug fixes

* A fit no longer decides how many threads later fits get (#16). The Laplace
  and Polya-Gamma backends take a per-fit `cores`, which they applied by moving
  the process-wide OpenMP thread count and leaving it moved, and it defaults to
  one thread. Every region in the session that read that value afterwards was
  therefore sized by whichever of those fits ran last, in fits it knew nothing
  about. Each backend now restores the previous value when it returns, on the
  interrupt path as well as the normal one.

* Successive fits in one session no longer corrupt the heap on Windows (#8).
  The OpenMP team for chain-parallel work was sized per fit from `cores`, so a
  fit asking for fewer cores than the one before it shrank the team; libgomp
  then destroyed the surplus workers and ran their `thread_local` destructors
  while later work was still in flight, and the damage surfaced as
  `STATUS_HEAP_CORRUPTION` at an unrelated `free()`. Four chains followed by
  two was enough to trigger it. The team now only ever grows, sized by the
  widest core budget any fit has asked for, and `cores` bounds how many chains
  are in flight instead, so it keeps its meaning. Growing a team was always
  safe; only shrinking faulted. Its size never comes from
  `omp_get_max_threads()`: the Laplace and Polya-Gamma backends move that value
  through `omp_set_num_threads()` and leave it moved, so reading it let a
  Laplace fit ahead of the first chain fit pin the team at one thread and
  serialize the chains of every fit after it.

* The Gibbs spatial backend runs its chains in parallel under OpenMP (#7). It
  looped over chains in R, so a 4-chain fit cost close to four times a 1-chain
  fit; on a 40-site binomial ICAR model that ratio drops from 3.7x to 0.94x.
  `cores` now reaches this backend, bounded by the chain count and the machine.
  Chain seeds are still derived in R, so a fit with a given seed returns the
  same draws as before, and the parallel and serial paths agree bitwise.

* The handcoded gradient no longer disagrees with its own log posterior when a
  temporal effect is not shared between numerator and denominator. The
  vectorized and fused kernels added the temporal effect to the denominator's
  linear predictor and fed the denominator residual back into the temporal
  gradient, both unconditionally, while `compute_log_post()` applied the effect
  to the numerator alone. NUTS therefore sampled a density whose gradient
  pointed elsewhere, silently, for every family with a real denominator
  (`poisson_gamma`, `negbin_negbin`, `negbin_gamma`, `gamma_gamma`,
  `lognormal`). Binomial models were unaffected. The gradient checks now cover
  a continuous-denominator family with `shared = FALSE`, which is the only
  configuration that can see this.

* `ModelData`'s scalar members are initialised at declaration. The constructor
  set only `unique_id`, so roughly thirty members held indeterminate values
  until assigned; the GP entry point never assigned `re_parameterization` and
  read stack residue to choose between the centred and non-centred
  parameterisation.

* The scalar gradient fallback reduces its per-thread partial sums in
  thread-index order rather than in a critical section, whose summation order
  varies from run to run.

* `mcmc_diagnostics()` now reports per-parameter Rhat and ESS (#4). It handed a
  multi-parameter draws array to the single-variable `posterior::rhat()` /
  `ess_bulk()` / `ess_tail()`, collapsing every parameter to one scalar that was
  then recycled across the returned rows: a well-mixed four-chain fit reported
  `rhat = 2.124`, `ess_bulk = 1`. Diagnostics now delegate to
  `tulpa::mcmc_diagnostics()`, the engine's per-parameter implementation, which
  removes the local re-derivation (`compute_diagnostics_basic()`,
  `compute_split_rhat()`, `compute_ess_basic()`). `summary()`, `plot_rhat()`,
  `plot_ess()`, `diagnostic_summary()` and `check_diagnostics()` all inherit the
  correction, and a converged fit no longer prints a false non-convergence
  warning.

* `print()` on a `ratiod_diagnostic_summary` no longer errors when a fit has a
  parameter with Rhat > 1.01 or ESS < 400. The worst-Rhat and worst-ESS tables
  are column extracts of the diagnostics table and dispatched on
  `print.ratiod_diagnostics()`, which rounded columns the extract does not
  carry.

## Internal

* B1a PoC: plain binomial fits can route through tulpa's `LikelihoodSpec`
  path (`tulpa::get_nuts_fn()`) behind a feature flag
  (`options(tulpaRatio.use_specs = TRUE)` or `TULPARATIO_USE_SPECS=1`).
  Off by default; legacy backend remains the production path until full
  parity ships in B5. Posterior means agree with the legacy backend within
  Monte Carlo noise.

* B1b: the same feature flag now covers all 7 ratio families (binomial,
  poisson_gamma, negbin_gamma, negbin_negbin, gamma_gamma, lognormal,
  beta_binomial) for the simplest non-ZI / no-spatial / no-RE / single-chain
  configuration. Autodiff only — H-kernel port is deferred to B2.

* B1c: zero-inflation, hurdle, and one-inflation variants now route through
  the `LikelihoodSpec` path. Each family's templated likelihood dispatches on
  `data.zi_type` and calls a shared mixture helper in
  `src/lik_specs/lik_helpers.h`; per-family allowlist lives in
  `src/lik_dispatch.cpp` and `R/backend_hmc.R::SPEC_ZI_COMPAT`. Posterior
  parity with the legacy backend matches within Monte Carlo noise for every
  binomial × {ZI, hurdle, OI, ZOIB} combination; for count families
  (poisson_gamma, negbin_*) the legacy A_r path was already silently
  ignoring ZI, so the spec path here gives the mathematically correct
  posterior — recorded as a behaviour change, not a parity mismatch (see
  `dev_notes/B1c_zi_surface.md`). Autodiff only; ZI variants for the
  hand-coded H-kernel remain deferred.

* The tulpa ABI test no longer pins a hardcoded version number. It compared
  `cpp_tulpa_abi_version()` against a literal `32L`, a second copy of a
  constant tulpa already owns, so every tulpa ABI bump broke the test with a
  stale-literal failure that said nothing about compatibility. Two unguarded
  accessors (`cpp_tulpa_compiled_abi_version()`, the value baked in from the
  `LinkingTo` headers, and `cpp_tulpa_runtime_abi_version()`, the value in the
  loaded tulpa DLL) let the test assert the invariant that matters — that the
  two agree — and report both numbers when they do not. tulpa remains the
  single source of truth for the version.

* Documentation is regenerated under roxygen2 8.0.0, matching the rest of the
  packages that link against tulpa. `RoxygenNote` migrates to
  `Config/roxygen2/version`. Two external links now resolve to a topic alias
  rather than an Rd filename (`posterior::as_draws_df()`,
  `loo::stacking_weights()`), which clears the "Non-topic package-anchored
  link(s)" note from `R CMD check`; `INFERENCE_TIERS` drops the `\docType{data}`
  and `\format{}` entries roxygen2 8.0.0 no longer emits for documented values;
  and the package page lists authors alongside the maintainer.

# numdenom 1.3.0

## New Inference Backends

* **Tiered inference mode system**: Inference backends are now organized into
  three tiers based on epistemic guarantees:
  - Tier 1 (Exact): HMC, ESS, Polya-Gamma - diagnosable convergence
  - Tier 2 (Structured): Laplace - controlled approximation
  - Tier 3 (Optimized): VI, SGHMC, SGLD - no convergence guarantee (explicit opt-in)

* **Elliptical Slice Sampling (ESS)**: New Tier 1 backend for models with
  Gaussian priors. Use `mode = "ess"`.

* **Variational Inference (VI)**: New Tier 3 backend with mean-field, full-rank,
  and low-rank variants. Use `mode = "vi"`.

* **Stochastic Gradient MCMC**: New Tier 3 backends for very large data:
  - SGHMC: `mode = "sghmc"`
  - SGLD: `mode = "sgld"`

## Performance Improvements

* **Thread-safe autodiff**: A_t (tape autodiff) mode now uses RAII-based
  TapeScope for thread isolation. Parallel chains work correctly with all
  gradient modes.

* **Hand-coded gradients**: Added O(N) analytical gradients for:
  - GP spatial models (5.4x speedup)
  - Spatially varying coefficients (SVC)
  - Time-varying coefficients (TVC)
  - Correlated/uncorrelated random slopes
  - Crossed random effects
  - ZI/Hurdle binomial families
  - Latent factor models

* **Spatial prediction**: New `predict()` method for `ratiod_fit` objects
  supports prediction at new spatial locations.

## Internal Changes

* Forward-mode autodiff (`fwd::Dual`) for thread-safe gradient computation
* GPU backend stubs (preparation for future CUDA support)
* CRT-based negative binomial Polya-Gamma sampler

---

# numdenom 1.2.0

## New Features

### Spatiotemporal Interaction

* **Spatiotemporal interaction effects**: New `spatiotemporal()` function specifies
  space-time interactions using Knorr-Held (2000) Type I-IV models:
  - Type I: Unstructured (IID) interaction
  - Type II: Structured time at each location
  - Type III: Structured space at each time point
  - Type IV: Fully structured (Kronecker product)
  - Separable: For GP-based spatial/temporal effects

* **Non-separable spatiotemporal GP**: New `spatiotemporal_gp()` function for continuous
  space-time GP models with Gneiting or Cressie-Huang non-separable covariance.

* **Spatiotemporal effect extraction**: New `spatiotemporal_effects()` extracts posterior
  distributions in array, long, or summary format with visualization support.

### Zero-Inflation Variants

* **Zero-inflated binomial**: New `ratiod_zibinomial()` for proportions with excess zeros.

* **One-inflated binomial**: New `ratiod_oibinomial()` for proportions with excess ones
  (100% success/detection).

* **Zero-and-one inflated binomial (ZOIB)**: New `ratiod_zoibinomial()` for proportions
  with excess at both boundaries.

* **Hurdle binomial**: New `ratiod_hurdle_binomial()` for binomial data with hurdle
  at zero.

## Documentation

* New `vignette("spatial-temporal")` with examples of space-time modeling.
* New `vignette("zero-inflation")` covering ZI, OI, ZOIB, and hurdle binomial models.

---

# numdenom 1.1.0

## New Features

* **Latent factors for unmeasured confounders**: New `latent_factor()` function allows
  specifying latent factors that capture shared unmeasured variation between numerator
  and denominator processes. Latent factors are observation-level random effects with
  sum-to-zero constraints for identifiability. Use `latent = latent_factor(n_factors = 1)`
  in the `ratiod()` call.

* **Latent factor extraction**: New `latent_factors()` function extracts posterior
  summaries or full draws for latent factor scores from fitted models.

## Documentation

* Updated `vignette("random-effects")` with section on latent factors.

---

# numdenom 1.0.0

Initial release with:

* Three model families: `ratiod_negbin_negbin()`, `ratiod_binomial()`, `ratiod_poisson_gamma()`
* Native HMC/NUTS backend (no Stan dependency)
* Shared random effects (default) between numerator and denominator
* Spatial structure: CAR, BYM2, GP/NNGP, RSR
* Temporal structure: AR(1), RW1, cyclic
* Random slopes (correlated and uncorrelated)
* Nested and crossed random effects
* LOO/WAIC model comparison
* Zero-inflation support
