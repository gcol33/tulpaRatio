#' Zero-Inflated Model Families
#'
#' @description
#' Distribution families for handling excess zeros in count data.
#' Zero-inflation occurs when there are more zeros than expected
#' under the standard count distribution.
#'
#' Two model types are available:
#' - **Zero-inflated (ZI)**: Mixture of point mass at zero and count distribution
#' - **Hurdle**: Separate processes for zero vs positive counts
#'
#' @examples
#' ratiod_zi_poisson()
#' ratiod_hurdle_negbin()
#' @name ratiod_zi_families
NULL


#' Zero-Inflated Negative Binomial Family
#'
#' @description
#' Zero-inflated negative binomial for the numerator process.
#' Models excess zeros as a mixture:
#'
#' \deqn{P(Y = 0) = \pi + (1 - \pi) \cdot P_{NB}(0)}
#' \deqn{P(Y = y) = (1 - \pi) \cdot P_{NB}(y), \quad y > 0}
#'
#' where \eqn{\pi} is the zero-inflation probability and
#' \eqn{P_{NB}} is the negative binomial PMF.
#'
#' @param link_num Link function for the count mean: "log".
#' @param link_denom Link function for the denominator mean: "log" (default);
#'   a gamma denominator also takes "identity" or "inverse".
#' @param link_zi Link function for the zero-inflation probability: "logit" (default), "probit" or "cloglog".
#' @param denom_family Denominator distribution: "negbin" (default),
#'   "gamma" for continuous effort, or "fixed" for a known exposure. A known
#'   exposure is data: it scales the count mean, which is then the rate times
#'   the exposure, and the ratio is the rate.
#'
#' @return A `ratiod_family` object with zero-inflation
#'
#' @details
#' The `zi_formula` argument specifies the linear predictor for the
#' zero-inflation probability. Its default estimates a single intercept.
#'
#' For ecological data, zero-inflation often represents:
#' - Structural zeros (species truly absent vs not detected)
#' - Sampling zeros (inadequate effort)
#' - False negatives
#'
#' @examples
#' # Create family object
#' fam <- ratiod_zi_negbin()
#' print(fam)
#'
#' # Simulate zero-inflated count data
#' set.seed(123)
#' n <- 60
#' zi_prob <- 0.3
#' habitat <- factor(rep(c("forest", "grassland"), each = n/2))
#' site <- factor(rep(1:10, each = n/10))
#' u <- rnorm(10, sd = 0.5)[site]
#' df <- data.frame(
#'   count = ifelse(runif(n) < zi_prob, 0,
#'                  rnbinom(n, size = 3, mu = exp(2 + 0.5 * (habitat == "grassland") + u))),
#'   total = rnbinom(n, size = 5, mu = 50 * exp(u)),
#'   habitat = habitat,
#'   site = site
#' )
#'
#' \donttest{
#' # Fit model
#' fit <- tratio(
#'   count | total ~ habitat + (1 | site),
#'   data = df,
#'   family = ratiod_zi_negbin(),
#'   control = list(iter = 400, chains = 1, seed = 1)
#' )
#' summary(fit)
#' }
#'
#' @seealso [ratiod_hurdle_negbin()] for hurdle model alternative
#'
#' @param zi_formula Optional one-sided formula for the zero-inflation or hurdle probability; NULL uses an intercept.
#' @export
ratiod_zi_negbin <- function(link_num = "log", link_denom = "log",
                           link_zi = "logit", denom_family = "negbin", zi_formula = NULL) {
  new_ratiod_family(count_model_type("neg_binomial_2", denom_family),
                    list(num = link_num, denom = link_denom, zi = link_zi),
                    list(type = "mixture", zi_formula = zi_formula))
}

#' Zero-Inflated Poisson Family
#'
#' @description
#' Zero-inflated Poisson for the numerator process.
#' Useful when overdispersion is primarily due to excess zeros.
#'
#' @inheritParams ratiod_zi_negbin
#' @param denom_family Denominator distribution: "gamma" (default),
#'   "negbin" for a count denominator, or "fixed" for a known exposure.
#'
#' @return A `ratiod_family` object with zero-inflation
#'
#' @details
#' Use `ratiod_zi_poisson()` when:
#' - Excess zeros are the main source of overdispersion
#' - The non-zero counts follow Poisson (no additional overdispersion)
#'
#' Use `ratiod_zi_negbin()` when:
#' - Both excess zeros AND overdispersion in positive counts
#'
#' @examples
#' # Create family object
#' fam <- ratiod_zi_poisson()
#' print(fam)
#'
#' # Simulate zero-inflated CPUE data
#' set.seed(123)
#' n <- 60
#' zi_prob <- 0.25
#' df <- data.frame(
#'   catch = ifelse(runif(n) < zi_prob, 0, rpois(n, lambda = 5)),
#'   effort = rgamma(n, shape = 4, rate = 1),
#'   depth = rnorm(n),
#'   vessel = factor(rep(1:6, each = n/6))
#' )
#'
#' \donttest{
#' # Fit model
#' fit <- tratio(
#'   catch | effort ~ depth + (1 | vessel),
#'   data = df,
#'   family = ratiod_zi_poisson(denom_family = "gamma"),
#'   control = list(iter = 400, chains = 1, seed = 1)
#' )
#' summary(fit)
#' }
#'
#' @param zi_formula Optional one-sided formula for the zero-inflation or hurdle probability; NULL uses an intercept.
#' @export
ratiod_zi_poisson <- function(link_num = "log", link_denom = "log",
                         link_zi = "logit", denom_family = "gamma", zi_formula = NULL) {
  new_ratiod_family(count_model_type("poisson", denom_family),
                    list(num = link_num, denom = link_denom, zi = link_zi),
                    list(type = "mixture", zi_formula = zi_formula))
}

#' Hurdle Negative Binomial Family
#'
#' @description
#' Hurdle model where zero vs positive counts are modelled as separate processes.
#'
#' \deqn{P(Y = 0) = 1 - \theta}
#' \deqn{P(Y = y | Y > 0) = P_{NB}^+(y), \quad y > 0}
#'
#' where \eqn{\theta} is the probability of a positive count and
#' \eqn{P_{NB}^+} is the truncated-at-zero negative binomial.
#'
#' @inheritParams ratiod_zi_negbin
#' @param link_hurdle Link function for the hurdle probability: "logit" (default), "probit" or "cloglog".
#'
#' @return A `ratiod_family` object with hurdle structure
#'
#' @details
#' The hurdle model differs from zero-inflation in interpretation:
#'
#' **Zero-inflated**: Zeros come from two sources (structural + sampling)
#' - \eqn{\pi}: probability of structural zero
#' - Some zeros also come from the count process
#'
#' **Hurdle**: Zeros are all from one source (the "hurdle" process)
#' - \eqn{1 - \theta}: probability of zero
#' - All positive counts from truncated count process
#'
#' Choose hurdle when zeros represent a distinct biological/sampling state.
#'
#' @examples
#' # Create family object
#' fam <- ratiod_hurdle_negbin()
#' print(fam)
#'
#' # Simulate hurdle data (presence/absence + abundance)
#' set.seed(123)
#' n <- 60
#' presence_prob <- 0.7
#' df <- data.frame(
#'   count = ifelse(runif(n) > presence_prob, 0,
#'                  rnbinom(n, size = 3, mu = 10) + 1),
#'   total = rnbinom(n, size = 5, mu = 50),
#'   habitat = factor(rep(c("forest", "grassland"), each = n/2)),
#'   site = factor(rep(1:10, each = n/10))
#' )
#'
#' \donttest{
#' # Fit model
#' fit <- tratio(
#'   count | total ~ habitat + (1 | site),
#'   data = df,
#'   family = ratiod_hurdle_negbin(),
#'   control = list(iter = 400, chains = 1, seed = 1)
#' )
#' summary(fit)
#' }
#'
#' @param zi_formula Optional one-sided formula for the zero-inflation or hurdle probability; NULL uses an intercept.
#' @export
ratiod_hurdle_negbin <- function(link_num = "log", link_denom = "log",
                                link_hurdle = "logit", denom_family = "negbin", zi_formula = NULL) {
  new_ratiod_family(count_model_type("neg_binomial_2", denom_family),
                    list(num = link_num, denom = link_denom, hurdle = link_hurdle),
                    list(type = "hurdle", zi_formula = zi_formula))
}

#' Hurdle Poisson Family
#'
#' @description
#' Hurdle model with truncated Poisson for positive counts.
#'
#' @inheritParams ratiod_hurdle_negbin
#' @param denom_family Denominator distribution: "gamma" (default),
#'   "negbin" for a count denominator, or "fixed" for a known exposure.
#'
#' @return A `ratiod_family` object with hurdle structure
#'
#' @examples
#' ratiod_hurdle_poisson()
#'
#' @param zi_formula Optional one-sided formula for the zero-inflation or hurdle probability; NULL uses an intercept.
#' @export
ratiod_hurdle_poisson <- function(link_num = "log", link_denom = "log",
                              link_hurdle = "logit", denom_family = "gamma", zi_formula = NULL) {
  new_ratiod_family(count_model_type("poisson", denom_family),
                    list(num = link_num, denom = link_denom, hurdle = link_hurdle),
                    list(type = "hurdle", zi_formula = zi_formula))
}

#' Check if family is zero-inflated
#'
#' @param family A ratiod_family object
#' @return Logical
#' @keywords internal
is_zi_family <- function(family) {
  isTRUE(family$zero_inflated)
}


#' @rdname ratiod_zi_families
#' @param x A ratiod_family_zi object
#' @param ... Ignored
#'
#' @export
print.ratiod_family_zi <- function(x, ...) {
  zi_type <- switch(x$zi_type, hurdle = "Hurdle", mixture = "Zero-inflated",
                     one_inflated = "One-inflated", zoib = "Zero-and-one-inflated")
  cat("tulpaRatio family:", x$name, "\n")
  cat(sprintf("[%s model]\n", zi_type))
  cat(x$description, "\n\n")

  cat("Numerator:  ", x$numerator$base_distribution,
      "(", x$numerator$link, ")\n", sep = "")

  if (x$zi_type == "hurdle") {
    cat("Hurdle:     ", "bernoulli",
        "(", x$numerator$link_hurdle, ")\n", sep = "")
  } else if (isTRUE(x$zero_inflated)) {
    cat("ZI prob:    ", "bernoulli",
        "(", x$numerator$link_zi, ")\n", sep = "")
  }
  if (isTRUE(x$one_inflated)) {
    cat("OI prob:    ", "bernoulli",
        "(", x$numerator$link_oi, ")\n", sep = "")
  }

  print_family_denominator(x)

  invisible(x)
}


#' Zero-Inflated Binomial Family
#'
#' @description
#' Zero-inflated binomial for the numerator when modeling proportions
#' with excess zeros (e.g., species detection/non-detection).
#'
#' \deqn{P(Y = 0) = \pi + (1 - \pi) \cdot (1-p)^n}
#' \deqn{P(Y = y) = (1 - \pi) \cdot \binom{n}{y} p^y (1-p)^{n-y}, \quad y > 0}
#'
#' @param link_num Link function for the success probability: "logit" (default), "probit" or "cloglog".
#' @param link_zi Link function for the zero-inflation probability: "logit" (default), "probit" or "cloglog".
#'
#' @return A `ratiod_family` object with zero-inflation
#'
#' @details
#' Use this family when modeling proportions (successes/trials) where
#' excess zeros occur beyond what binomial predicts. Common applications:
#' - Species occupancy with false negatives
#' - Survey data with non-response
#' - Epidemiological data with under-reporting
#'
#' @examples
#' # Create family object
#' fam <- ratiod_zi_binomial()
#' print(fam)
#'
#' @seealso [ratiod_binomial()], [ratiod_zi_negbin()]
#'
#' @param zi_formula Optional one-sided formula for the zero-inflation or hurdle probability; NULL uses an intercept.
#' @export
ratiod_zi_binomial <- function(link_num = "logit", link_zi = "logit", zi_formula = NULL) {
  new_ratiod_family("binomial", list(num = link_num, zi = link_zi),
                    list(type = "mixture", zi_formula = zi_formula))
}

#' One-Inflated Binomial Family
#'
#' @description
#' One-inflated binomial for proportions with excess ones (100% success).
#' Useful when perfect detection/success has a structural component.
#'
#' \deqn{P(Y = n) = \psi + (1 - \psi) \cdot p^n}
#' \deqn{P(Y = y) = (1 - \psi) \cdot \binom{n}{y} p^y (1-p)^{n-y}, \quad y < n}
#'
#' @param link_num Link function for the success probability: "logit" (default), "probit" or "cloglog".
#' @param link_oi Link function for the one-inflation probability: "logit" (default), "probit" or "cloglog".
#'
#' @return A `ratiod_family` object with one-inflation
#'
#' @details
#' Use this when excess 100% success rates occur due to:
#' - Perfect detection in highly suitable habitat
#' - Saturation effects
#' - Structural constraints ensuring full success
#'
#' @examples
#' # Create family object
#' fam <- ratiod_oi_binomial()
#' print(fam)
#'
#' @param oi_formula Optional one-sided formula for the one-inflation probability; NULL uses an intercept.
#' @export
ratiod_oi_binomial <- function(link_num = "logit", link_oi = "logit", oi_formula = NULL) {
  new_ratiod_family("binomial", list(num = link_num, oi = link_oi),
                    list(type = "one_inflated", oi_formula = oi_formula))
}

#' Zero-and-One-Inflated Binomial Family
#'
#' @description
#' Zero-and-one inflated binomial for proportions with excess zeros AND ones.
#' Commonly needed in ecological applications where both absence (structural)
#' and perfect detection (saturation) have distinct processes.
#'
#' @param link_num Link function for the success probability: "logit" (default), "probit" or "cloglog".
#' @param link_zi Link function for the zero-inflation probability: "logit" (default), "probit" or "cloglog".
#' @param link_oi Link function for the one-inflation probability: "logit" (default), "probit" or "cloglog".
#'
#' @return A `ratiod_family` object with zero-and-one inflation
#'
#' @details
#' The ZOIB (Zero-and-One Inflated Binomial) model has three components:
#' 1. Zero process: P(structural zero) = pi_0
#' 2. One process: P(structural one | not zero) = pi_1
#' 3. Binomial process: P(Y=y | non-structural) follows binomial
#'
#' Use when both boundaries (0 and n) have excess observations.
#'
#' @examples
#' # Create family object
#' fam <- ratiod_zoi_binomial()
#' print(fam)
#'
#' @param zi_formula Optional one-sided formula for the zero-inflation or hurdle probability; NULL uses an intercept.
#' @param oi_formula Optional one-sided formula for the one-inflation probability; NULL uses an intercept.
#' @export
ratiod_zoi_binomial <- function(link_num = "logit", link_zi = "logit",
                               link_oi = "logit", zi_formula = NULL, oi_formula = NULL) {
  new_ratiod_family("binomial", list(num = link_num, zi = link_zi, oi = link_oi),
                    list(type = "zoib", zi_formula = zi_formula, oi_formula = oi_formula))
}

#' Hurdle Binomial Family
#'
#' @description
#' Hurdle model for binomial data where zero counts are modeled separately
#' from positive counts using a truncated binomial.
#'
#' @param link_num Link function for the success probability: "logit" (default), "probit" or "cloglog".
#' @param link_hurdle Link function for the hurdle probability P(Y > 0): "logit" (default), "probit" or "cloglog".
#'
#' @return A `ratiod_family` object with hurdle structure
#'
#' @examples
#' fam <- ratiod_hurdle_binomial()
#' print(fam)
#'
#' @param zi_formula Optional one-sided formula for the zero-inflation or hurdle probability; NULL uses an intercept.
#' @export
ratiod_hurdle_binomial <- function(link_num = "logit", link_hurdle = "logit", zi_formula = NULL) {
  new_ratiod_family("binomial", list(num = link_num, hurdle = link_hurdle),
                    list(type = "hurdle", zi_formula = zi_formula))
}
