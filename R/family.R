#' Model Families
#'
#' @description
#' Distribution families for the numerator and denominator processes.
#' Each family specifies the likelihood structure for joint modelling.
#'
#' @examples
#' ratiod_poisson_gamma()
#' ratiod_binomial()
#' @name ratiod_families
NULL

#' Negative Binomial-Negative Binomial Family
#'
#' @description
#' Two-process model where both numerator and denominator follow negative
#' binomial distributions. This is the core family for count-count ratios
#' where both processes exhibit overdispersion.
#'
#' Use cases:
#' - Species counts / total counts (relative abundance)
#' - Events / opportunities (both counts)
#' - Any ratio where numerator and denominator are overdispersed counts
#'
#' @param link_num Link function for the numerator mean: "log".
#' @param link_denom Link function for the denominator mean: "log".
#'
#' @return A `ratiod_family` object
#'
#' @examples
#' # Create family object
#' fam <- ratiod_negbin_negbin()
#' print(fam)
#'
#' # Simulate data for relative abundance
#' set.seed(123)
#' n <- 60
#' df <- data.frame(
#'   sp_count = rnbinom(n, size = 5, mu = 15),
#'   total_count = rnbinom(n, size = 8, mu = 100),
#'   habitat = factor(rep(c("forest", "grassland"), each = n/2)),
#'   site = factor(rep(1:10, each = n/10))
#' )
#'
#' \donttest{
#' # Fit model
#' fit <- tratio(
#'   sp_count | total_count ~ habitat + (1 | site),
#'   data = df,
#'   family = ratiod_negbin_negbin(),
#'   control = list(iter = 400, chains = 1, seed = 1)
#' )
#' summary(fit)
#' }
#'
#' @export
ratiod_negbin_negbin <- function(link_num = "log", link_denom = "log") {
  new_ratiod_family("negbin_negbin", list(num = link_num, denom = link_denom))
}
#' Binomial Family for Trial-Based Ratios
#'
#' @description
#' Trial-based model where the numerator represents successes out of a
#' known number of trials, the denominator.
#'
#' Use cases:
#' - Detection / availability (occupancy-style)
#' - Successes / trials (hierarchical binomial)
#' - Proportions with known denominators
#'
#' @param link Link function for the success probability: "logit" (default), "probit" or "cloglog".
#'
#' @return A `ratiod_family` object
#'
#' @examples
#' # Create family object
#' fam <- ratiod_binomial()
#' print(fam)
#'
#' # Simulate detection/availability data
#' set.seed(123)
#' n <- 50
#' df <- data.frame(
#'   detections = rbinom(n, size = 10, prob = 0.4),
#'   trials = rep(10L, n),
#'   effort = rnorm(n),
#'   site = factor(rep(1:10, each = 5))
#' )
#'
#' \donttest{
#' # Fit model
#' fit <- tratio(
#'   detections | trials ~ effort + (1 | site),
#'   data = df,
#'   family = ratiod_binomial(),
#'   control = list(iter = 400, chains = 1, seed = 1)
#' )
#' summary(fit)
#' }
#'
#' @export
ratiod_binomial <- function(link = "logit") {
  new_ratiod_family("binomial", list(num = link))
}
#' Poisson-Gamma Family for Count/Effort Ratios
#'
#' @description
#' Two-process model where the numerator follows a Poisson distribution
#' and the denominator (effort/exposure) follows a Gamma distribution.
#' This is the natural family for CPUE-type data.
#'
#' Use cases:
#' - Catch per unit effort (CPUE)
#' - Observations per hour
#' - Events per continuous exposure
#'
#' @param link_num Link function for the count rate: "log".
#' @param link_denom Link function for the effort mean: "log" (default), "identity" or "inverse".
#'
#' @return A `ratiod_family` object
#'
#' @examples
#' # Create family object
#' fam <- ratiod_poisson_gamma()
#' print(fam)
#'
#' # Simulate CPUE data
#' set.seed(123)
#' n <- 60
#' season <- factor(rep(c("spring", "summer", "fall"), each = n/3))
#' vessel <- factor(rep(1:10, each = n/10))
#' u <- rnorm(10, sd = 0.5)[vessel]
#' depth <- rnorm(n)
#' df <- data.frame(
#'   catch = rpois(n, exp(1.5 + 0.3 * depth + 0.4 * (season == "summer") + u)),
#'   effort_hours = rgamma(n, shape = 4, rate = exp(-u)),
#'   depth = depth,
#'   season = season,
#'   vessel = vessel
#' )
#'
#' \donttest{
#' # Fit model
#' fit <- tratio(
#'   catch | effort_hours ~ depth + season + (1 | vessel),
#'   data = df,
#'   family = ratiod_poisson_gamma(),
#'   control = list(iter = 400, chains = 1, seed = 1)
#' )
#' summary(fit)
#' }
#'
#' @export
ratiod_poisson_gamma <- function(link_num = "log", link_denom = "log") {
  new_ratiod_family("poisson_gamma", list(num = link_num, denom = link_denom))
}
#' Negative Binomial-Gamma Family for Overdispersed Count/Effort Ratios
#'
#' @description
#' Two-process model where the numerator follows a negative binomial distribution
#' and the denominator (effort/exposure) follows a Gamma distribution. This fills
#' the gap between Poisson-Gamma (equidispersed counts) and NegBin-NegBin
#' (integer denominator): use it when counts are overdispersed and effort is
#' continuous.
#'
#' Use cases:
#' - Overdispersed catch per unit effort (CPUE)
#' - Overdispersed observations per continuous exposure
#' - Any ratio of overdispersed counts over continuous positive effort
#'
#' @param link_num Link function for the count rate: "log".
#' @param link_denom Link function for the effort mean: "log" (default), "identity" or "inverse".
#'
#' @return A `ratiod_family` object
#'
#' @examples
#' # Create family object
#' fam <- ratiod_negbin_gamma()
#' print(fam)
#'
#' # Simulate overdispersed CPUE data
#' set.seed(123)
#' n <- 60
#' df <- data.frame(
#'   catch = rnbinom(n, size = 5, mu = 8),
#'   effort_hours = rgamma(n, shape = 4, rate = 1),
#'   depth = rnorm(n),
#'   vessel = factor(rep(1:6, each = n/6))
#' )
#'
#' \donttest{
#' # Fit model
#' fit <- tratio(
#'   catch | effort_hours ~ depth + (1 | vessel),
#'   data = df,
#'   family = ratiod_negbin_gamma(),
#'   control = list(iter = 400, chains = 1, seed = 1)
#' )
#' summary(fit)
#' }
#'
#' @export
ratiod_negbin_gamma <- function(link_num = "log", link_denom = "log") {
  new_ratiod_family("negbin_gamma", list(num = link_num, denom = link_denom))
}
#' Count Families by Denominator
#'
#' @description
#' A Poisson or negative binomial count over any denominator the samplers fit:
#' a known exposure (`"fixed"`), a count (`"negbin"`) or a continuous effort
#' (`"gamma"`). A known exposure is data: it scales the count mean, which is
#' then the rate times the exposure, and the ratio is the rate. This is the
#' disease-mapping model of observed cases over a population or an expected
#' count. The zero-inflated and hurdle counterparts are [ratiod_zi_poisson()],
#' [ratiod_zi_negbin()], [ratiod_hurdle_poisson()] and [ratiod_hurdle_negbin()].
#'
#' `ratiod_poisson(denom_family = "gamma")` is [ratiod_poisson_gamma()],
#' `ratiod_negbin(denom_family = "negbin")` is [ratiod_negbin_negbin()] and
#' `ratiod_negbin(denom_family = "gamma")` is [ratiod_negbin_gamma()].
#'
#' @param link_num Link function for the count mean: "log".
#' @param link_denom Link function for the denominator mean: "log" (default);
#'   a gamma denominator also takes "identity" or "inverse". Unread for a
#'   known exposure.
#' @param denom_family Denominator distribution: "fixed" (default) for a known
#'   exposure, "negbin" for a count denominator, or "gamma" for continuous
#'   effort.
#'
#' @return A `ratiod_family` object
#'
#' @examples
#' fam <- ratiod_poisson()
#' print(fam)
#'
#' # Cases over a known population at risk
#' set.seed(1)
#' n <- 60
#' df <- data.frame(population = round(runif(n, 500, 5000)), x = rnorm(n))
#' df$cases <- rpois(n, df$population * exp(-6 + 0.3 * df$x))
#'
#' \donttest{
#' fit <- tratio(cases | population ~ x, data = df,
#'               family = ratiod_poisson(),
#'               control = list(iter = 400, chains = 1, seed = 1))
#' coef(fit)
#' }
#' @export
ratiod_poisson <- function(link_num = "log", link_denom = "log",
                           denom_family = "fixed") {
  new_ratiod_family(count_model_type("poisson", denom_family),
                    list(num = link_num, denom = link_denom))
}
#' @rdname ratiod_poisson
#' @export
ratiod_negbin <- function(link_num = "log", link_denom = "log",
                          denom_family = "fixed") {
  new_ratiod_family(count_model_type("neg_binomial_2", denom_family),
                    list(num = link_num, denom = link_denom))
}
#' Links of probabilities and of gamma means
#'
#' A success, zero-inflation, one-inflation or hurdle probability takes any
#' link of `LINKS_PROBABILITY`; a gamma mean any link of `LINKS_GAMMA`. Count
#' means and the log-normal location take the log link.
#'
#' @keywords internal
LINKS_PROBABILITY <- c("logit", "probit", "cloglog")

#' @rdname LINKS_PROBABILITY
#' @keywords internal
LINKS_GAMMA <- c("log", "identity", "inverse")

#' Canonical links: the link of each domain the Gibbs, Polya-Gamma and
#' Laplace samplers and the collapsed fields are written for
#' @rdname LINKS_PROBABILITY
#' @keywords internal
LINKS_CANONICAL <- c("log", "logit")


# Labels and links of each response distribution.
FAMILY_DISTRIBUTIONS <- list(
  poisson = list(short = "poisson", label = "Poisson", links = "log"),
  neg_binomial_2 = list(short = "negbin", label = "Negative binomial", links = "log",
                        inflated = "neg_binomial"),
  binomial = list(short = "binomial", label = "Binomial", links = LINKS_PROBABILITY),
  beta_binomial = list(short = "beta_binomial", label = "Beta-binomial", links = LINKS_PROBABILITY),
  gamma = list(short = "gamma", label = "Gamma", links = LINKS_GAMMA),
  lognormal = list(short = "lognormal", label = "Log-normal", links = "log"),
  fixed = list(short = "fixed", label = "Fixed", links = NULL)
)

model_type_spec <- function(num, denom, responses, dispersion = NULL,
                            samplers = character(), phi_rate = 0.1, name = NULL,
                            spec_inflation = NULL, handcoded = FALSE) {
  list(distributions = c(num = num, denom = denom), responses = responses,
       dispersion = dispersion,
       samplers = c("hmc", "ess", "sghmc", "sgld", "vi", samplers),
       phi_prior = c(shape = 2, rate = phi_rate), name = name,
       labels = vapply(FAMILY_DISTRIBUTIONS[c(num, denom)], `[[`, "", "label"),
       handcoded = handcoded,
       spec_inflation = spec_inflation)
}

# Each model's distributions, response domains, draw names and inference support.
MODEL_TYPES <- list(
  binomial = model_type_spec("binomial", "fixed", NULL,
    samplers = c("laplace", "gibbs", "pg"), name = "binomial_fixed"),
  beta_binomial = model_type_spec("beta_binomial", "fixed", NULL,
    c(num = "phi"), name = "beta_binomial_fixed"),
  negbin_negbin = model_type_spec("neg_binomial_2", "neg_binomial_2", NULL,
    c(num = "phi_num", denom = "phi_denom"), c("laplace", "gibbs", "pg")),
  poisson_negbin = model_type_spec("poisson", "neg_binomial_2", c(num = "count", denom = "count"),
    c(denom = "phi_denom")),
  poisson_gamma = model_type_spec("poisson", "gamma", NULL,
    c(denom = "shape"), c("laplace", "gibbs"), phi_rate = 0.5),
  negbin_gamma = model_type_spec("neg_binomial_2", "gamma", NULL,
    c(num = "phi_num", denom = "phi_denom"), c("laplace", "gibbs")),
  gamma_gamma = model_type_spec("gamma", "gamma", NULL,
    c(num = "shape_num", denom = "shape_denom"), "laplace", phi_rate = 0.5),
  lognormal = model_type_spec("lognormal", "lognormal", NULL,
    c(num = "sigma_num", denom = "sigma_denom"), phi_rate = 2,
    name = "lognormal_lognormal"),
  poisson_fixed = model_type_spec("poisson", "fixed", c(num = "count", denom = "exposure")),
  negbin_fixed = model_type_spec("neg_binomial_2", "fixed", c(num = "count", denom = "exposure"),
    c(num = "phi_num")),
  lognormal_fixed = model_type_spec("lognormal", "fixed", c(num = "positive", denom = "exposure"),
    c(num = "sigma_num"))
)

model_type_for_distributions <- function(num, denom) {
  matches <- vapply(MODEL_TYPES, function(spec) {
    identical(unname(spec$distributions), c(num, denom))
  }, logical(1))
  if (!any(matches)) {
    stop(sprintf("No sampler fits a %s numerator with a %s denominator.", num, denom),
         call. = FALSE)
  }
  names(MODEL_TYPES)[matches][[1L]]
}

count_model_type <- function(base, denom_family) {
  candidates <- Filter(function(spec) identical(spec$distributions[["num"]], base), MODEL_TYPES)
  denominators <- vapply(candidates, function(spec) spec$distributions[["denom"]], "")
  aliases <- vapply(FAMILY_DISTRIBUTIONS[denominators], `[[`, "", "short")
  if (!(length(denom_family) == 1L && denom_family %in% aliases)) {
    stop(sprintf("`denom_family` must be one of %s.",
                 paste0('"', aliases, '"', collapse = ", ")), call. = FALSE)
  }
  names(candidates)[match(denom_family, aliases)]
}

new_ratiod_family <- function(model_type, links, inflation = NULL) {
  spec <- MODEL_TYPES[[model_type]]
  if (is.null(spec)) stop(sprintf("Unknown model type: '%s'", model_type), call. = FALSE)
  distributions <- spec$distributions
  num <- FAMILY_DISTRIBUTIONS[[distributions[["num"]]]]
  denom <- FAMILY_DISTRIBUTIONS[[distributions[["denom"]]]]
  validate_link(links$num, num$links)
  if (!identical(distributions[["denom"]], "fixed")) validate_link(links$denom, denom$links)
  family <- list(name = spec$name %||% model_type,
                 numerator = list(distribution = distributions[["num"]], link = links$num),
                 denominator = list(distribution = distributions[["denom"]],
                   link = if (identical(distributions[["denom"]], "fixed")) NULL else links$denom))
  if (identical(model_type, "beta_binomial")) family$overdispersed <- TRUE
  description <- sprintf("%s numerator, %s denominator", spec$labels[[1L]], spec$labels[[2L]])
  classes <- c("ratiod_family", "list")
  if (!is.null(inflation)) {
    types <- list(
      mixture = list(prefix = "zi", distribution = "zero_inflated", label = "Zero-inflated", links = "zi"),
      hurdle = list(prefix = "hurdle", distribution = "hurdle", label = "Hurdle", links = "hurdle"),
      one_inflated = list(prefix = "oi", distribution = "one_inflated", label = "One-inflated", links = "oi"),
      zoib = list(prefix = "zoi", distribution = "zero_one_inflated", label = "Zero-and-one inflated", links = c("zi", "oi")))
    inf <- types[[inflation$type]]
    if (is.null(inf)) stop("Unknown inflation type.", call. = FALSE)
    if (!distributions[["num"]] %in% c("binomial", "poisson", "neg_binomial_2") ||
        (inflation$type %in% c("one_inflated", "zoib") && model_type != "binomial")) {
      stop("This model type does not support the requested inflation.", call. = FALSE)
    }
    for (nm in inf$links) {
      validate_link(links[[nm]], LINKS_PROBABILITY)
      family$numerator[[paste0("link_", nm)]] <- links[[nm]]
    }
    family$name <- paste(c(inf$prefix, num$short,
                           if (model_type != "binomial") denom$short), collapse = "_")
    family$numerator$base_distribution <- distributions[["num"]]
    family$numerator$distribution <- paste(inf$distribution, num$inflated %||% num$short, sep = "_")
    family <- c(family, inflation[intersect(c("zi_formula", "oi_formula"), names(inflation))])
    family$zero_inflated <- inflation$type != "one_inflated"
    if (inflation$type %in% c("one_inflated", "zoib")) family$one_inflated <- TRUE
    family$zi_type <- inflation$type
    description <- paste(inf$label, description)
    classes <- c("ratiod_family_zi", classes)
  }
  family$description <- description
  structure(family, class = classes)
}

print_family_denominator <- function(x) {
  cat("Denominator:", x$denominator$distribution,
      if (!is.null(x$denominator$link)) paste0("(", x$denominator$link, ")") else "(fixed)",
      "\n")
}

#' Links of a fitted family
#'
#' @param family A `ratiod_family` object
#' @param model_type Model type string
#' @return A list with the links of the numerator (`num`), the denominator
#'   (`denom`, `NULL` for a fixed denominator) and the zero- or hurdle (`zi`)
#'   and one-inflation (`oi`) probabilities, plus `probability`, whether the
#'   numerator link maps to a success probability, and `exposure`, whether the
#'   denominator is a known exposure
#' @keywords internal
family_links <- function(family, model_type) {
  probability <- identical(MODEL_TYPES[[model_type]]$responses[["denom"]], "trials")
  exposure <- identical(MODEL_TYPES[[model_type]]$responses[["denom"]], "exposure")
  num <- family$numerator
  list(
    probability = probability,
    exposure = exposure,
    num = num$link %||% if (probability) "logit" else "log",
    denom = if (probability || exposure) NULL else family$denominator$link %||% "log",
    zi = num$link_zi %||% num$link_hurdle %||% "logit",
    oi = num$link_oi %||% "logit"
  )
}


#' Whether every link of a family is canonical
#'
#' @inheritParams family_links
#' @return `TRUE` or `FALSE`
#' @keywords internal
family_canonical_links <- function(family, model_type) {
  links <- family_links(family, model_type)
  all(unlist(links[c("num", "denom", "zi", "oi")]) %in% LINKS_CANONICAL)
}


#' Whether a family's likelihood is one the hand-written samplers carry
#'
#' Its model type has a hand-coded likelihood and every link is canonical.
#'
#' @inheritParams family_links
#' @return `TRUE` or `FALSE`
#' @keywords internal
family_handcoded <- function(family, model_type) {
  isTRUE(MODEL_TYPES[[model_type]]$handcoded) && family_canonical_links(family, model_type)
}


#' Model type and links of a family, as the samplers read them
#'
#' @inheritParams family_links
#' @return The list every sampler entry point takes as `family`
#' @keywords internal
family_params <- function(family, model_type) {
  links <- family_links(family, model_type)
  list(model_type = model_type, link_num = links$num, link_denom = links$denom,
       link_zi = links$zi, link_oi = links$oi)
}


#' Validate link function
#'
#' @param link Link function name
#' @param allowed Character vector of allowed links
#' @keywords internal
validate_link <- function(link, allowed) {
  if (!(length(link) == 1L && link %in% allowed)) {
    stop(
      sprintf("Link function '%s' is not supported here; the samplers fit: %s.",
              paste(link, collapse = ", "), paste(allowed, collapse = ", ")),
      call. = FALSE
    )
  }
}

#' Beta-Binomial Family for Overdispersed Proportions
#'
#' @description
#' Two-process model where the numerator follows a beta-binomial distribution,
#' allowing for overdispersion in binomial data. The denominator is the known
#' number of trials.
#'
#' Use cases:
#' - Overdispersed proportions (e.g., infection rates across sites)
#' - Hierarchical binomial with extra-binomial variation
#' - Proportions with correlation within clusters
#'
#' @param link Link function for the success probability: "logit" (default), "probit" or "cloglog".
#'
#' @return A `ratiod_family` object
#'
#' @details
#' The beta-binomial distribution arises when the success probability
#' itself follows a beta distribution. This creates overdispersion
#' (variance > binomial variance) and is useful when:
#' - There is unmodelled heterogeneity in success probability
#' - Observations within clusters are correlated
#' - The binomial assumption of independent trials is violated
#'
#' The overdispersion parameter \eqn{\rho} (intra-class correlation)
#' ranges from 0 (binomial) to 1 (maximum overdispersion).
#'
#' @examples
#' # Create family object
#' fam <- ratiod_beta_binomial()
#' print(fam)
#'
#' # Simulate overdispersed proportion data
#' set.seed(123)
#' n <- 40
#' df <- data.frame(
#'   infected = rbinom(n, size = 20, prob = rbeta(n, 2, 5)),
#'   tested = rep(20L, n),
#'   treatment = factor(rep(c("control", "treated"), each = n/2)),
#'   site = factor(rep(1:8, each = n/8))
#' )
#'
#' \donttest{
#' # Fit model
#' fit <- tratio(
#'   infected | tested ~ treatment + (1 | site),
#'   data = df,
#'   family = ratiod_beta_binomial(),
#'   control = list(iter = 400, chains = 1, seed = 1)
#' )
#' summary(fit)
#' }
#'
#' @export
ratiod_beta_binomial <- function(link = "logit") {
  new_ratiod_family("beta_binomial", list(num = link))
}

#' Gamma-Gamma Family for Continuous Ratio Data
#'
#' @description
#' Two-process model where both numerator and denominator follow gamma
#' distributions. This is the natural family for ratios of positive
#' continuous quantities.
#'
#' Use cases:
#' - Biomass ratios (species biomass / total biomass)
#' - Concentration ratios (analyte / reference)
#' - Duration ratios (time in activity / total time)
#'
#' @param link_num Link function for the numerator mean: "log" (default), "identity" or "inverse".
#' @param link_denom Link function for the denominator mean: "log" (default), "identity" or "inverse".
#'
#' @return A `ratiod_family` object
#'
#' @details
#' The gamma distribution is appropriate for positive continuous data
#' with right skew. The ratio of two gamma random variables does not
#' have a simple closed form, which is why ratio models them jointly.
#'
#' Shape parameters are estimated for both numerator and denominator,
#' allowing different amounts of variability in each process.
#'
#' @examples
#' # Create family object
#' fam <- ratiod_gamma_gamma()
#' print(fam)
#'
#' # Simulate biomass ratio data
#' set.seed(123)
#' n <- 50
#' df <- data.frame(
#'   species_biomass = rgamma(n, shape = 3, rate = 0.5),
#'   total_biomass = rgamma(n, shape = 10, rate = 0.3),
#'   habitat = factor(rep(c("forest", "grassland"), each = n/2)),
#'   site = factor(rep(1:10, each = n/10))
#' )
#'
#' \donttest{
#' # Fit model
#' fit <- tratio(
#'   species_biomass | total_biomass ~ habitat + (1 | site),
#'   data = df,
#'   family = ratiod_gamma_gamma(),
#'   control = list(iter = 400, chains = 1, seed = 1)
#' )
#' summary(fit)
#' }
#'
#' @export
ratiod_gamma_gamma <- function(link_num = "log", link_denom = "log") {
  new_ratiod_family("gamma_gamma", list(num = link_num, denom = link_denom))
}

#' Log-Normal Family for Multiplicative Processes
#'
#' @description
#' Two-process model where both numerator and denominator follow log-normal
#' distributions. Appropriate when the underlying process is multiplicative
#' and residuals are log-normally distributed.
#'
#' Use cases:
#' - Abundance indices (CPUE with continuous effort)
#' - Economic ratios (costs, prices)
#' - Body condition indices
#'
#' @param link_num Link function for the numerator median: "log".
#' @param link_denom Link function for the denominator median: "log".
#' @param denom_fixed If `TRUE`, the denominator is a known exposure rather
#'   than a modelled response: the numerator's median is the exposure times
#'   `exp(eta)`, and the ratio is `exp(eta)`.
#'
#' @return A `ratiod_family` object
#'
#' @details
#' The log-normal distribution arises when a variable is the product of
#' many independent positive factors. If \eqn{Y = e^X} where \eqn{X} is
#' normal, then \eqn{Y} is log-normal.
#'
#' For ratios, if both numerator and denominator are log-normal, the
#' ratio is also log-normal (difference of normals on log scale).
#' This can simplify interpretation when working on the log scale.
#'
#' @examples
#' # Create family object
#' fam <- ratiod_lognormal()
#' print(fam)
#'
#' # Simulate body condition data
#' set.seed(123)
#' n <- 60
#' df <- data.frame(
#'   weight = rlnorm(n, meanlog = 3, sdlog = 0.3),
#'   length_cubed = rlnorm(n, meanlog = 2, sdlog = 0.2),
#'   age = sample(1:5, n, replace = TRUE),
#'   sex = factor(rep(c("M", "F"), each = n/2)),
#'   cohort = factor(rep(1:6, each = n/6))
#' )
#'
#' \donttest{
#' # Fit model
#' fit <- tratio(
#'   weight | length_cubed ~ age + sex + (1 | cohort),
#'   data = df,
#'   family = ratiod_lognormal(),
#'   control = list(iter = 400, chains = 1, seed = 1)
#' )
#' summary(fit)
#' }
#'
#' @export
ratiod_lognormal <- function(link_num = "log", link_denom = "log",
                             denom_fixed = FALSE) {
  new_ratiod_family(if (denom_fixed) "lognormal_fixed" else "lognormal",
                    list(num = link_num, denom = link_denom))
}

#' @rdname ratiod_families
#' @param x A ratiod_family object
#' @param ... Ignored
#'
#' @export
print.ratiod_family <- function(x, ...) {
  cat("tulpaRatio family:", x$name, "\n")
  cat(x$description, "\n\n")
  cat("Numerator:  ", x$numerator$distribution,
      "(", x$numerator$link, ")\n", sep = "")
  print_family_denominator(x)
  invisible(x)
}
