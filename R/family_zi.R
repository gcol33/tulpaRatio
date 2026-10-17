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
#' ratiod_zipois()
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
#' The zero-inflation probability can have its own linear predictor via the
#' `zi` argument in `tratio()`. If not specified, a single intercept is estimated.
#'
#' For ecological data, zero-inflation often represents:
#' - Structural zeros (species truly absent vs not detected)
#' - Sampling zeros (inadequate effort)
#' - False negatives
#'
#' @examples
#' # Create family object
#' fam <- ratiod_zinegbin()
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
#'   family = ratiod_zinegbin(),
#'   control = list(iter = 400, chains = 1, seed = 1)
#' )
#' summary(fit)
#' }
#'
#' @seealso [ratiod_hurdle_negbin()] for hurdle model alternative
#'
#' @export
ratiod_zinegbin <- function(link_num = "log", link_denom = "log",
                           link_zi = "logit", denom_family = "negbin") {
  zi_count_family("neg_binomial_2", "mixture", denom_family,
                  link_num, link_denom, link_zi)
}


#' Zero-Inflated Poisson Family
#'
#' @description
#' Zero-inflated Poisson for the numerator process.
#' Useful when overdispersion is primarily due to excess zeros.
#'
#' @inheritParams ratiod_zinegbin
#' @param denom_family Denominator distribution: "gamma" (default),
#'   "negbin" for a count denominator, or "fixed" for a known exposure.
#'
#' @return A `ratiod_family` object with zero-inflation
#'
#' @details
#' Use `ratiod_zipois()` when:
#' - Excess zeros are the main source of overdispersion
#' - The non-zero counts follow Poisson (no additional overdispersion)
#'
#' Use `ratiod_zinegbin()` when:
#' - Both excess zeros AND overdispersion in positive counts
#'
#' @examples
#' # Create family object
#' fam <- ratiod_zipois()
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
#'   family = ratiod_zipois(denom_family = "gamma"),
#'   control = list(iter = 400, chains = 1, seed = 1)
#' )
#' summary(fit)
#' }
#'
#' @export
ratiod_zipois <- function(link_num = "log", link_denom = "log",
                         link_zi = "logit", denom_family = "gamma") {
  zi_count_family("poisson", "mixture", denom_family,
                  link_num, link_denom, link_zi)
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
#' @inheritParams ratiod_zinegbin
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
#' @export
ratiod_hurdle_negbin <- function(link_num = "log", link_denom = "log",
                                link_hurdle = "logit", denom_family = "negbin") {
  zi_count_family("neg_binomial_2", "hurdle", denom_family,
                  link_num, link_denom, link_hurdle)
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
#' ratiod_hurdle_pois()
#'
#' @export
ratiod_hurdle_pois <- function(link_num = "log", link_denom = "log",
                              link_hurdle = "logit", denom_family = "gamma") {
  zi_count_family("poisson", "hurdle", denom_family,
                  link_num, link_denom, link_hurdle)
}


# Denominators each count numerator is fitted with, inflated or not; every
# pair is one likelihood of the samplers (get_hmc_model_type()).
count_denominators <- list(
  poisson = c(gamma = "gamma", negbin = "neg_binomial_2", fixed = "fixed"),
  neg_binomial_2 = c(negbin = "neg_binomial_2", gamma = "gamma", fixed = "fixed")
)

#' Build a zero-inflated or hurdle count family
#'
#' @param base Count distribution: "poisson" or "neg_binomial_2".
#' @param type "mixture" (zero-inflated) or "hurdle".
#' @param denom_family Name of the denominator distribution.
#' @param link_num,link_denom,link_inflation Links of the count mean, the
#'   denominator mean and the zero or hurdle probability.
#' @return A `ratiod_family_zi` object
#' @keywords internal
zi_count_family <- function(base, type, denom_family, link_num, link_denom,
                            link_inflation) {
  validate_link(link_num, "log")
  validate_link(link_inflation, LINKS_PROBABILITY)
  denominator <- count_denominator(base, denom_family, link_denom)
  hurdle <- identical(type, "hurdle")
  short <- c(poisson = "pois", neg_binomial_2 = "negbin")[[base]]
  label <- c(poisson = "Poisson", neg_binomial_2 = "negative binomial")[[base]]
  denom_label <- c(gamma = "Gamma", negbin = "negative binomial",
                   fixed = "known exposure")[[denom_family]]

  numerator <- list(
    distribution = paste0(if (hurdle) "hurdle_" else "zero_inflated_",
                          if (base == "poisson") "poisson" else "neg_binomial"),
    base_distribution = base,
    link = link_num
  )
  numerator[[if (hurdle) "link_hurdle" else "link_zi"]] <- link_inflation

  structure(
    list(
      name = paste0(if (hurdle) "hurdle_" else "zi", short, "_", denom_family),
      numerator = numerator,
      denominator = denominator,
      zero_inflated = TRUE,
      zi_type = type,
      description = sprintf("%s %s numerator, %s denominator",
                            if (hurdle) "Hurdle" else "Zero-inflated",
                            label, denom_label)
    ),
    class = c("ratiod_family_zi", "ratiod_family", "list")
  )
}


#' The denominator of a count family
#'
#' @param base Count distribution of the numerator: "poisson" or
#'   "neg_binomial_2".
#' @param denom_family One of the names of `count_denominators[[base]]`.
#' @param link_denom Link of the denominator mean; unread for a known exposure.
#' @return The `denominator` element of a `ratiod_family`.
#' @keywords internal
count_denominator <- function(base, denom_family, link_denom) {
  allowed <- count_denominators[[base]]
  if (!(length(denom_family) == 1L && denom_family %in% names(allowed))) {
    stop(sprintf("`denom_family` must be one of %s.",
                 paste0('"', names(allowed), '"', collapse = ", ")),
         call. = FALSE)
  }
  fixed <- identical(denom_family, "fixed")
  if (!fixed) {
    validate_link(link_denom,
                  if (identical(allowed[[denom_family]], "gamma")) LINKS_GAMMA else "log")
  }
  list(distribution = allowed[[denom_family]], link = if (fixed) NULL else link_denom)
}


#' Check if family is zero-inflated
#'
#' @param family A ratiod_family object
#' @return Logical
#' @keywords internal
is_zi_family <- function(family) {
  isTRUE(family$zero_inflated)
}


#' Check if family is hurdle model
#'
#' @param family A ratiod_family object
#' @return Logical
#' @keywords internal
is_hurdle_family <- function(family) {
  isTRUE(family$zi_type == "hurdle")
}


#' @rdname ratiod_zi_families
#' @param x A ratiod_family_zi object
#' @param ... Ignored
#'
#' @export
print.ratiod_family_zi <- function(x, ...) {
  zi_type <- if (x$zi_type == "hurdle") "Hurdle" else "Zero-inflated"
  cat("tulpaRatio family:", x$name, "\n")
  cat(sprintf("[%s model]\n", zi_type))
  cat(x$description, "\n\n")

  cat("Numerator:  ", x$numerator$base_distribution,
      "(", x$numerator$link, ")\n", sep = "")

  if (x$zi_type == "hurdle") {
    cat("Hurdle:     ", "bernoulli",
        "(", x$numerator$link_hurdle, ")\n", sep = "")
  } else {
    cat("ZI prob:    ", "bernoulli",
        "(", x$numerator$link_zi, ")\n", sep = "")
  }

  cat("Denominator:", x$denominator$distribution,
      if (!is.null(x$denominator$link)) paste0("(", x$denominator$link, ")") else "(fixed)",
      "\n")

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
#' fam <- ratiod_zibinomial()
#' print(fam)
#'
#' @seealso [ratiod_binomial()], [ratiod_zinegbin()]
#'
#' @export
ratiod_zibinomial <- function(link_num = "logit", link_zi = "logit") {

  validate_link(link_num, LINKS_PROBABILITY)
  validate_link(link_zi, LINKS_PROBABILITY)

  structure(
    list(
      name = "zibinomial",
      numerator = list(
        distribution = "zero_inflated_binomial",
        base_distribution = "binomial",
        link = link_num,
        link_zi = link_zi
      ),
      denominator = list(
        distribution = "fixed",
        link = NULL
      ),
      zero_inflated = TRUE,
      zi_type = "mixture",
      description = "Zero-inflated binomial numerator, fixed trials denominator"
    ),
    class = c("ratiod_family_zi", "ratiod_family", "list")
  )
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
#' fam <- ratiod_oibinomial()
#' print(fam)
#'
#' @export
ratiod_oibinomial <- function(link_num = "logit", link_oi = "logit") {

  validate_link(link_num, LINKS_PROBABILITY)
  validate_link(link_oi, LINKS_PROBABILITY)

  structure(
    list(
      name = "oibinomial",
      numerator = list(
        distribution = "one_inflated_binomial",
        base_distribution = "binomial",
        link = link_num,
        link_oi = link_oi
      ),
      denominator = list(
        distribution = "fixed",
        link = NULL
      ),
      zero_inflated = FALSE,  # One-inflated, not zero-inflated
      one_inflated = TRUE,
      zi_type = "one_inflated",
      description = "One-inflated binomial numerator, fixed trials denominator"
    ),
    class = c("ratiod_family_zi", "ratiod_family", "list")
  )
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
#' fam <- ratiod_zoibinomial()
#' print(fam)
#'
#' @export
ratiod_zoibinomial <- function(link_num = "logit", link_zi = "logit",
                               link_oi = "logit") {

  validate_link(link_num, LINKS_PROBABILITY)
  validate_link(link_zi, LINKS_PROBABILITY)
  validate_link(link_oi, LINKS_PROBABILITY)

  structure(
    list(
      name = "zoibinomial",
      numerator = list(
        distribution = "zero_one_inflated_binomial",
        base_distribution = "binomial",
        link = link_num,
        link_zi = link_zi,
        link_oi = link_oi
      ),
      denominator = list(
        distribution = "fixed",
        link = NULL
      ),
      zero_inflated = TRUE,
      one_inflated = TRUE,
      zi_type = "zoib",
      description = "Zero-and-one inflated binomial, fixed trials denominator"
    ),
    class = c("ratiod_family_zi", "ratiod_family", "list")
  )
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
#' @export
ratiod_hurdle_binomial <- function(link_num = "logit", link_hurdle = "logit") {

  validate_link(link_num, LINKS_PROBABILITY)
  validate_link(link_hurdle, LINKS_PROBABILITY)

  structure(
    list(
      name = "hurdle_binomial",
      numerator = list(
        distribution = "hurdle_binomial",
        base_distribution = "binomial",
        link = link_num,
        link_hurdle = link_hurdle
      ),
      denominator = list(
        distribution = "fixed",
        link = NULL
      ),
      zero_inflated = TRUE,
      zi_type = "hurdle",
      description = "Hurdle binomial numerator, fixed trials denominator"
    ),
    class = c("ratiod_family_zi", "ratiod_family", "list")
  )
}


#' Check if family is one-inflated
#'
#' @param family A ratiod_family object
#' @return Logical
#' @keywords internal
is_oi_family <- function(family) {
  isTRUE(family$one_inflated)
}


#' Check if family is ZOIB (zero-and-one inflated)
#'
#' @param family A ratiod_family object
#' @return Logical
#' @keywords internal
is_zoib_family <- function(family) {
  isTRUE(family$zero_inflated) && isTRUE(family$one_inflated)
}
