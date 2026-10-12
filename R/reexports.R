# Each verb is defined elsewhere and re-exported here so that as_draws(fit),
# pp_check(fit) and posterior_predict(fit) reach the ratiod_fit methods through
# the same generic every other package uses.

#' @importFrom posterior as_draws
#' @export
posterior::as_draws

#' @importFrom bayesplot pp_check
#' @export
bayesplot::pp_check

#' @importFrom tulpa posterior_predict
#' @export
tulpa::posterior_predict

#' @importFrom tulpa latent_factors
#' @export
tulpa::latent_factors

#' @importFrom tulpa spatiotemporal_effects
#' @export
tulpa::spatiotemporal_effects

#' @importFrom loo loo
#' @export
loo::loo

#' @importFrom loo waic
#' @export
loo::waic
