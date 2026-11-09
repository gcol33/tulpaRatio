# =============================================================================
# Fair benchmark driver: subprocess isolation + cooldown
# Avoids CPU thermal throttling that inflates in-loop timings by 1.5-2.7x
#
# Model mode (named models, uses bench_subprocess.R):
#   Rscript benchmarks/bench_fair.R                          # all 18 models
#   Rscript benchmarks/bench_fair.R NB_ICAR PG_GPt           # specific models
#   Rscript benchmarks/bench_fair.R --seeds 3 NB_HSGP        # 3 seeds (default 5)
#   Rscript benchmarks/bench_fair.R --cooldown 10 PG_ST_IV   # 10s cooldown (default 5)
#
# Row mode (gradient_methods.md rows, uses bench_single_row.R):
#   Rscript benchmarks/bench_fair.R --row 35 38 14 74        # specific rows
#   Rscript benchmarks/bench_fair.R --row all                # all 137 rows
#   Rscript benchmarks/bench_fair.R --row 1-30               # range (PG family)
#   Rscript benchmarks/bench_fair.R --row 31-60 --seeds 3    # NB family, 3 seeds
#   Rscript benchmarks/bench_fair.R --row all --cooldown 10  # all rows, 10s cooldown
# =============================================================================

args <- commandArgs(trailingOnly = TRUE)

# Parse flags
n_seeds <- 5L
cooldown <- 5L
row_mode <- FALSE
row_args <- character(0)
models_requested <- character(0)
grad_mode <- "H"
i <- 1
while (i <= length(args)) {
  if (args[i] == "--seeds") {
    n_seeds <- as.integer(args[i + 1]); i <- i + 2
  } else if (args[i] == "--cooldown") {
    cooldown <- as.integer(args[i + 1]); i <- i + 2
  } else if (args[i] == "--grad") {
    grad_mode <- args[i + 1]; i <- i + 2
  } else if (args[i] == "--row") {
    row_mode <- TRUE; i <- i + 1
    # Collect all remaining non-flag args as row specs
    while (i <= length(args) && !startsWith(args[i], "--")) {
      row_args <- c(row_args, args[i]); i <- i + 1
    }
  } else {
    models_requested <- c(models_requested, args[i]); i <- i + 1
  }
}

# ---- Shared helpers ----

RSCRIPT <- file.path(R.home("bin"), "Rscript.exe")

run_subprocess <- function(script, script_args, cooldown_sec) {
  Sys.sleep(cooldown_sec)
  system2(RSCRIPT, args = c(script, script_args), stdout = TRUE, stderr = TRUE)
}

parse_time <- function(out) {
  m <- regmatches(out, regexpr("[0-9]+\\.[0-9]+s", out))
  if (length(m) > 0) as.numeric(sub("s", "", m[1])) else NA_real_
}

print_summary <- function(results, stan_refs = NULL) {
  cat("\n\n========== RESULTS ==========\n")
  cat(sprintf("%-20s %8s %8s %8s %6s\n", "Model", "Median", "Stan", "Ratio", ""))
  cat(paste0(rep("-", 55), collapse = ""), "\n")
  for (nm in names(results)) {
    med <- results[[nm]]
    stan <- if (!is.null(stan_refs) && nm %in% names(stan_refs)) stan_refs[[nm]] else NA
    if (is.na(med)) {
      cat(sprintf("%-20s %8s\n", nm, "FAIL"))
    } else if (is.na(stan)) {
      cat(sprintf("%-20s %7.1fs %8s\n", nm, med, "N/A"))
    } else {
      ratio <- med / stan
      v <- if (ratio <= 1) "WIN" else if (ratio <= 1.2) "~" else "LOSS"
      cat(sprintf("%-20s %7.1fs %7.1fs %7.2fx %s\n", nm, med, stan, ratio, v))
    }
  }
}

# =============================================================================
# ROW MODE: benchmark gradient_methods.md rows via bench_single_row.R
# =============================================================================
if (row_mode) {

  # Parse row specs: "all", "35", "1-30", etc.
  expand_rows <- function(specs) {
    rows <- integer(0)
    for (s in specs) {
      if (s == "all") return(c(1:137, 138:149, 150:170))
      if (grepl("-", s)) {
        parts <- as.integer(strsplit(s, "-")[[1]])
        rows <- c(rows, seq(parts[1], parts[2]))
      } else {
        rows <- c(rows, as.integer(s))
      }
    }
    sort(unique(rows))
  }

  if (length(row_args) == 0) stop("--row requires row numbers, ranges, or 'all'")
  rows <- expand_rows(row_args)
  cat(sprintf("Benchmarking %d rows (%s), %d seeds, %ds cooldown, grad=%s\n",
              length(rows), paste(range(rows), collapse = "-"), n_seeds, cooldown, grad_mode))

  results <- list()
  seeds <- seq_len(n_seeds)

  for (row in rows) {
    label <- sprintf("Row %d", row)
    cat(sprintf("\n=== %s ===\n", label))
    times <- numeric(length(seeds))
    all_failed <- TRUE
    for (s in seq_along(seeds)) {
      out <- run_subprocess("benchmarks/bench_single_row.R",
                            c(as.character(row), grad_mode, "600", as.character(seeds[s])),
                            cooldown)
      # bench_single_row.R outputs RESULT:<row>:<mode>:<time>
      result_line <- grep("^RESULT:", out, value = TRUE)
      if (length(result_line) > 0) {
        parts <- strsplit(result_line[1], ":")[[1]]
        time_str <- parts[4]
        if (!grepl("ERROR|TIMEOUT", time_str)) {
          times[s] <- as.numeric(time_str)
          all_failed <- FALSE
          cat(sprintf("  seed %d: %.1fs\n", s, times[s]))
        } else {
          times[s] <- NA
          cat(sprintf("  seed %d: %s\n", s, time_str))
        }
      } else {
        # Fallback: parse from stderr/stdout
        t <- parse_time(out)
        times[s] <- t
        if (!is.na(t)) all_failed <- FALSE
        status_lines <- grep("STATS|seed=|Row|RESULT", out, value = TRUE)
        if (length(status_lines) > 0) cat(paste(status_lines, collapse = "\n"), "\n")
        else if (is.na(t)) cat(paste(tail(out, 3), collapse = "\n"), "\n")
      }
    }
    med <- if (all_failed) NA_real_ else median(times, na.rm = TRUE)
    results[[label]] <- med
    if (!is.na(med)) cat(sprintf("  MEDIAN: %.1fs\n", med))
    else cat("  MEDIAN: FAIL\n")
  }

  print_summary(results)
  quit(save = "no")
}

# =============================================================================
# MODEL MODE: benchmark named models via bench_subprocess.R
# =============================================================================

all_models <- c(
  "PG_base", "PG_RE", "PG_ICAR",
  "NB_base", "NB_RE", "NB_ICAR", "NB_HSGP",
  "Bin_base", "Bin_RE",
  "PG_GPt", "Bin_GPt", "NB_GPt",
  "PG_BYM2", "NB_BYM2", "Bin_BYM2",
  "PG_ST_IV", "Bin_ST_IV", "NB_ST_IV"
)
stan_ref <- c(
  PG_base = 1.2, PG_RE = 2.5, PG_ICAR = 3.0,
  NB_base = 1.5, NB_RE = 2.9, NB_ICAR = 3.5, NB_HSGP = 21.0,
  Bin_base = 9.4, Bin_RE = NA,
  PG_GPt = 10.0, Bin_GPt = 3.3, NB_GPt = 21.3,
  PG_BYM2 = 13.2, NB_BYM2 = 86.0, Bin_BYM2 = 12.4,
  PG_ST_IV = 82.5, Bin_ST_IV = 56.0, NB_ST_IV = 133.5
)

if (length(models_requested) == 0) models_requested <- all_models
bad <- setdiff(models_requested, all_models)
if (length(bad) > 0) stop("Unknown models: ", paste(bad, collapse = ", "),
                          "\nAvailable: ", paste(all_models, collapse = ", "))

cat(sprintf("Benchmarking %d models, %d seeds, %ds cooldown\n",
            length(models_requested), n_seeds, cooldown))

seeds <- seq_len(n_seeds)
results <- list()

for (model in models_requested) {
  cat(sprintf("\n=== %s ===\n", model))
  times <- numeric(length(seeds))
  for (s in seq_along(seeds)) {
    out <- run_subprocess("benchmarks/bench_subprocess.R",
                          c(model, as.character(seeds[s])), cooldown)
    cat(paste(out, collapse = "\n"), "\n")
    times[s] <- parse_time(out)
    if (is.na(times[s])) cat("  WARNING: failed to parse time\n")
  }
  med <- median(times, na.rm = TRUE)
  results[[model]] <- med
  cat(sprintf("  MEDIAN: %.1fs\n", med))
}

print_summary(results, stan_ref)
