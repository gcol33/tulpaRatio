args <- commandArgs(trailingOnly = TRUE)
if (length(setdiff(args, "--check"))) stop("usage: Rscript tools/unity.R [--check]")
check_only <- "--check" %in% args
src <- "src"

groups <- list(core = c("array_utils.cpp", "crt_rng.cpp", "enum_names.cpp", "precision_density.cpp"
), exports = c("RcppExports.cpp", "thread_count.cpp"), likelihood = c("lik_dispatch.cpp", "tulpa_bridge.cpp"
), pg = c("pg_binomial.cpp", "pg_negbin.cpp", "pg_rng.cpp", "pg_spatial.cpp"), samplers = "gibbs_spatial.cpp",
    spatial = c("hsgp_spectral.cpp", "kriging.cpp"), tests = c("test_chain_concurrency.cpp", "test_gradient_check.cpp",
    "test_helpers.cpp", "test_negbin_spec.cpp", "test_precision_jet.cpp", "test_precision_spec.cpp",
    "test_ratio_margins.cpp", "test_ratio_math.cpp", "test_scalar_math.cpp", "test_tls_workspace.cpp",
    "test_engine_margins.cpp", "test_precision_scalar.cpp", "test_spec17.cpp"))

unity <- paste0("unity_", names(groups), ".cpp")
sources <- list.files(src, "\\.cpp$")
tus <- sources[!startsWith(sources, "unity_")]
members <- unlist(groups, use.names = FALSE)
dup <- unique(members[duplicated(members)])
missing <- setdiff(tus, members)
unknown <- setdiff(members, tus)
extra <- setdiff(sources[startsWith(sources, "unity_")], unity)
if (length(dup)) stop("in more than one unity group: ", paste(dup, collapse = ", "))
if (length(missing)) stop("in no unity group: ", paste(missing, collapse = ", "))
if (length(unknown)) stop("named by a unity group but absent from src/: ", paste(unknown, collapse = ", "))
if (length(extra)) stop("undeclared unity files: ", paste(extra, collapse = ", "))

src_abs <- paste0(normalizePath(src, winslash = "/"), "/")
prerequisites <- function(files) {
  seen <- character()
  pending <- files
  while (length(pending)) {
    current <- pending[1L]
    pending <- pending[-1L]
    if (current %in% seen) next
    seen <- c(seen, current)
    lines <- readLines(file.path(src, current), warn = FALSE)
    includes <- regmatches(lines,
      regexec('^[[:space:]]*#[[:space:]]*include[[:space:]]*"([^"]+)"', lines))
    includes <- vapply(Filter(length, includes), `[[`, "", 2L)
    paths <- file.path(src, dirname(current), includes)
    paths <- normalizePath(paths[file.exists(paths)], winslash = "/")
    paths <- paths[startsWith(paths, src_abs)]
    pending <- c(pending, substring(paths, nchar(src_abs) + 1L))
  }
  seen
}
wrap <- function(head, items) {
  c(paste(head, "\\"), paste0("  ", items, c(rep(" \\", length(items) - 1L), "")))
}
objects <- sub("\\.cpp$", ".o", unity)
block <- c(
  "# unity-build: begin",
  wrap("OBJECTS =", objects),
  "all: $(SHLIB)",
  unlist(lapply(seq_along(unity), function(i) {
    wrap(paste0(objects[i], ":"), c(unity[i], prerequisites(groups[[i]])))
  })),
  "# unity-build: end"
)

outputs <- setNames(lapply(groups, function(files) {
  c("#include <RcppEigen.h>", "", paste0('#include "', files, '"'))
}), file.path(src, unity))
for (mk in file.path(src, c("Makevars", "Makevars.win"))) {
  x <- readLines(mk)
  b <- grep("^# unity-build: begin", x)
  e <- grep("^# unity-build: end", x)
  outputs[[mk]] <- if (length(b) == 1L && length(e) == 1L && e > b) {
    c(x[seq_len(b - 1L)], block, x[-seq_len(e)])
  } else if (!length(b) && !length(e)) {
    c(x, "", block)
  } else stop(mk, ": unbalanced unity-build markers")
}

stale <- character()
for (path in names(outputs)) {
  x <- if (file.exists(path)) readLines(path, warn = FALSE) else NULL
  if (identical(outputs[[path]], x)) next
  stale <- c(stale, path)
  if (!check_only) {
    con <- file(path, "wb")
    writeLines(outputs[[path]], con, sep = "\n")
    close(con)
  }
}
cat(sprintf("%d translation units in %d groups\n", length(tus), length(groups)))
if (length(stale)) {
  cat(if (check_only) "stale:" else "rewrote:", paste(stale, collapse = ", "), "\n")
  if (check_only) quit(status = 1L)
}
