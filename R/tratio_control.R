# Control surface for tratio(): the perf / numerical / tuning knobs kept out of
# the front-door signature, which carries statistical arguments only. One entry
# per knob, so adding a knob is a line in the registry plus its use site rather
# than a new front-door argument.
#
# `default = NULL` marks a knob whose backend supplies its own default. Those
# are forwarded only when the caller sets one, leaving the backend default in
# place otherwise.

.tratio_scalar <- function(x, nm, pred, what) {
  if (length(x) != 1L || is.na(x) || !pred(x)) {
    stop(sprintf("control$%s must be %s, got %s.", nm, what,
                 paste(deparse(x), collapse = " ")), call. = FALSE)
  }
  x
}

.tratio_count <- function(x, nm) {
  .tratio_scalar(x, nm, function(v) is.numeric(v) && v >= 1 && v == round(v),
                 "a single positive whole number")
  as.integer(x)
}

.tratio_nonneg_int <- function(x, nm) {
  .tratio_scalar(x, nm, function(v) is.numeric(v) && v >= 0 && v == round(v),
                 "a single non-negative whole number")
  as.integer(x)
}

.tratio_positive <- function(x, nm) {
  .tratio_scalar(x, nm, function(v) is.numeric(v) && v > 0,
                 "a single positive number")
  as.numeric(x)
}

.tratio_flag <- function(x, nm) {
  .tratio_scalar(x, nm, is.logical, "TRUE or FALSE")
  as.logical(x)
}

.tratio_adapt_delta <- function(x, nm) {
  .tratio_scalar(x, nm, function(v) is.numeric(v) && v >= 0.5 && v <= 0.99,
                 "a single number between 0.5 and 0.99")
  as.numeric(x)
}

.tratio_seed <- function(x, nm) {
  .tratio_scalar(x, nm, is.numeric, "a single number")
  as.integer(x)
}

.TRATIO_CONTROL <- list(
  # Sampling budget, read by every backend.
  chains        = list(default = 4L,   validate = .tratio_count),
  iter          = list(default = 2000L, validate = .tratio_count),
  warmup        = list(default = NULL, validate = .tratio_nonneg_int),
  thin          = list(default = 1L,   validate = .tratio_count),
  cores         = list(default = NULL, validate = .tratio_count),
  seed          = list(default = NULL, validate = .tratio_seed),
  verbose       = list(default = TRUE, validate = .tratio_flag),

  # NUTS / HMC.
  adapt_delta   = list(default = NULL, validate = .tratio_adapt_delta),
  max_treedepth = list(default = NULL, validate = .tratio_count),
  riemannian    = list(default = FALSE, validate = .tratio_flag),
  L             = list(default = NULL, validate = .tratio_nonneg_int),
  metric        = list(default = "auto",
                       choices = c("auto", "dense", "diag", "block_diag")),
  gradient_mode = list(default = "auto",
                       choices = c("auto", "N", "A", "A_r", "A_t", "H")),
  re_param      = list(default = "noncentered",
                       choices = c("noncentered", "centered")),

  # Variational inference.
  vi_variant    = list(default = "auto",
                       choices = c("auto", "meanfield", "lowrank", "fullrank")),

  # Stochastic gradient backends.
  batch_size     = list(default = NULL, validate = .tratio_count),
  epsilon        = list(default = NULL, validate = .tratio_positive),
  alpha          = list(default = NULL, validate = .tratio_positive),
  schedule_a     = list(default = NULL, validate = .tratio_positive),
  schedule_b     = list(default = NULL, validate = .tratio_positive),
  schedule_gamma = list(default = NULL, validate = .tratio_positive),
  use_schedule   = list(default = NULL, validate = .tratio_flag)
)

# Resolve a user `control` list against the registry: reject unknown names,
# validate the supplied ones, fill defaults, then settle the two knobs whose
# default is derived from another (`warmup` from `iter`, `cores` from `chains`).
.tratio_control <- function(control) {
  tulpa::tulpa_check_control(control, names(.TRATIO_CONTROL), "tratio")

  given <- names(control) %||% character(0)
  ctrl <- lapply(names(.TRATIO_CONTROL), function(nm) {
    spec <- .TRATIO_CONTROL[[nm]]
    if (!(nm %in% given)) return(spec$default)
    v <- control[[nm]]
    if (is.null(v)) return(NULL)
    if (!is.null(spec$choices)) {
      return(match.arg(as.character(v), spec$choices))
    }
    spec$validate(v, nm)
  })
  names(ctrl) <- names(.TRATIO_CONTROL)

  if (is.null(ctrl$warmup)) ctrl$warmup <- as.integer(floor(ctrl$iter / 2))
  if (is.null(ctrl$cores))  ctrl$cores  <- .tratio_count(
    getOption("mc.cores", ctrl$chains), "cores")

  if (ctrl$warmup >= ctrl$iter) {
    stop(sprintf(
      "control$warmup (%d) must be below control$iter (%d) -- the fit would ",
      ctrl$warmup, ctrl$iter), "keep no post-warmup draws.", call. = FALSE)
  }

  ctrl
}

# The knobs each stochastic-gradient backend accepts beyond the shared budget.
# Only those the caller actually set are forwarded, so an unset knob leaves the
# backend's own default standing.
.TRATIO_SG_KNOBS <- list(
  sghmc = c("epsilon", "alpha", "L"),
  sgld  = c("epsilon", "schedule_a", "schedule_b", "schedule_gamma",
            "use_schedule")
)

.tratio_sg_args <- function(ctrl, backend) {
  knobs <- .TRATIO_SG_KNOBS[[backend]]
  args <- ctrl[knobs]
  args[!vapply(args, is.null, logical(1))]
}
