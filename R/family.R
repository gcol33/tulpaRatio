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
  validate_link(link_num, c("log"))
  validate_link(link_denom, c("log"))

  structure(
    list(
      name = "negbin_negbin",
      numerator = list(
        distribution = "neg_binomial_2",
        link = link_num
      ),
      denominator = list(
        distribution = "neg_binomial_2",
        link = link_denom
      ),
      description = "Negative binomial numerator, negative binomial denominator"
    ),
    class = c("ratiod_family", "list")
  )
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
#' @param link Link function for the success probability: "logit".
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
  validate_link(link, "logit")

  structure(
    list(
      name = "binomial_fixed",
      numerator = list(
        distribution = "binomial",
        link = link
      ),
      denominator = list(
        distribution = "fixed",
        link = NULL
      ),
      description = "Binomial numerator with fixed denominator"
    ),
    class = c("ratiod_family", "list")
  )
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
#' @param link_denom Link function for the effort mean: "log".
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
  validate_link(link_num, c("log"))
  validate_link(link_denom, c("log"))

  structure(
    list(
      name = "poisson_gamma",
      numerator = list(
        distribution = "poisson",
        link = link_num
      ),
      denominator = list(
        distribution = "gamma",
        link = link_denom
      ),
      description = "Poisson numerator, Gamma denominator (CPUE-type)"
    ),
    class = c("ratiod_family", "list")
  )
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
#' @param link_denom Link function for the effort mean: "log".
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
  validate_link(link_num, c("log"))
  validate_link(link_denom, c("log"))

  structure(
    list(
      name = "negbin_gamma",
      numerator = list(
        distribution = "neg_binomial_2",
        link = link_num
      ),
      denominator = list(
        distribution = "gamma",
        link = link_denom
      ),
      description = "Negative binomial numerator, Gamma denominator (overdispersed CPUE-type)"
    ),
    class = c("ratiod_family", "list")
  )
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
#' @param link Link function for the success probability: "logit".
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
  validate_link(link, "logit")

  structure(
    list(
      name = "beta_binomial_fixed",
      numerator = list(
        distribution = "beta_binomial",
        link = link
      ),
      denominator = list(
        distribution = "fixed",
        link = NULL
      ),
      overdispersed = TRUE,
      description = "Beta-binomial numerator with fixed denominator (overdispersed proportions)"
    ),
    class = c("ratiod_family", "list")
  )
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
#' @param link_num Link function for the numerator mean: "log".
#' @param link_denom Link function for the denominator mean: "log".
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
  validate_link(link_num, "log")
  validate_link(link_denom, "log")

  structure(
    list(
      name = "gamma_gamma",
      numerator = list(
        distribution = "gamma",
        link = link_num
      ),
      denominator = list(
        distribution = "gamma",
        link = link_denom
      ),
      description = "Gamma numerator, Gamma denominator (continuous ratios)"
    ),
    class = c("ratiod_family", "list")
  )
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
ratiod_lognormal <- function(link_num = "log", link_denom = "log") {
  validate_link(link_num, "log")
  validate_link(link_denom, "log")

  structure(
    list(
      name = "lognormal_lognormal",
      numerator = list(
        distribution = "lognormal",
        link = link_num
      ),
      denominator = list(
        distribution = "lognormal",
        link = link_denom
      ),
      description = "Log-normal numerator and denominator (multiplicative processes)"
    ),
    class = c("ratiod_family", "list")
  )
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
  cat("Denominator:", x$denominator$distribution,
      if (!is.null(x$denominator$link)) paste0("(", x$denominator$link, ")") else "(fixed)",
      "\n")
  invisible(x)
}
