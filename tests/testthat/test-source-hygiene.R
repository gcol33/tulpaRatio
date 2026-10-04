# Locate the package source tree. Present when the suite runs from a checkout
# (devtools::test(), R CMD check on the tarball); absent when the tests run
# against an installed package, where these lints have nothing to read.
hygiene_src_dir <- function() {
  dir <- normalizePath(getwd(), winslash = "/", mustWork = FALSE)
  for (i in 1:5) {
    if (file.exists(file.path(dir, "DESCRIPTION")) &&
        dir.exists(file.path(dir, "src"))) {
      return(file.path(dir, "src"))
    }
    parent <- dirname(dir)
    if (identical(parent, dir)) break
    dir <- parent
  }
  NULL
}

hygiene_sources <- function(dir) {
  files <- list.files(dir, pattern = "\\.(h|hpp|cpp)$", full.names = TRUE,
                      recursive = TRUE)
  # src/tests/catch.hpp is vendored third-party code and is not ours to style.
  files[!grepl("/tests/catch\\.hpp$", files)]
}

test_that("no issue-tracker references in source comments", {
  src <- hygiene_src_dir()
  skip_if(is.null(src), "source tree not available")

  offenders <- character(0)
  for (f in hygiene_sources(src)) {
    lines <- readLines(f, warn = FALSE)
    hits <- grep("gcol33/[A-Za-z]+#[0-9]|(^|[^A-Za-z0-9_])#[0-9]{1,4}([^0-9]|$)",
                 lines)
    # A preprocessor directive is not an issue reference.
    hits <- hits[!grepl("^\\s*#\\s*[A-Za-z]", lines[hits])]
    if (length(hits)) {
      offenders <- c(offenders, sprintf("%s:%d: %s", basename(f), hits,
                                        trimws(lines[hits])))
    }
  }

  expect_equal(offenders, character(0))
})

test_that("gradient kernels do not print from inside the chain region", {
  src <- hygiene_src_dir()
  skip_if(is.null(src), "source tree not available")

  # run_hmc_chain_cpp runs on an OpenMP worker whenever chains > 1, so every
  # kernel it reaches is off the main thread and cannot call into R. These
  # files hold only such kernels; the sampler drivers, the progress bar and
  # the serial VI / ESS / Gibbs loops print by design and are not listed.
  kernels <- c("hmc_svc.h", "hmc_gp.h", "hmc_gp_collapsed.h",
               "hmc_gp_autodiff.h", "hmc_tvc.h", "hmc_icar_collapsed.h",
               "log_post_impl.h")

  offenders <- character(0)
  for (nm in kernels) {
    f <- file.path(src, nm)
    if (!file.exists(f)) next
    lines <- readLines(f, warn = FALSE)

    # Drop regions guarded by a compile-time debug macro; those are off in a
    # released build, which is the other way this is allowed to be satisfied.
    depth <- 0L
    guarded <- logical(length(lines))
    for (i in seq_along(lines)) {
      if (grepl("^\\s*#\\s*if.*DEBUG", lines[i])) depth <- depth + 1L
      guarded[i] <- depth > 0L
      if (depth > 0L && grepl("^\\s*#\\s*endif", lines[i])) depth <- depth - 1L
    }

    hits <- which(grepl("Rcpp::Rcout|\\bRprintf\\s*\\(|\\bREprintf\\s*\\(", lines) &
                    !guarded)
    if (length(hits)) {
      offenders <- c(offenders, sprintf("%s:%d: %s", nm, hits,
                                        trimws(lines[hits])))
    }
  }

  expect_equal(offenders, character(0))
})

test_that("no session-scoped debug counters in kernels", {
  src <- hygiene_src_dir()
  skip_if(is.null(src), "source tree not available")

  offenders <- character(0)
  for (f in hygiene_sources(src)) {
    lines <- readLines(f, warn = FALSE)
    hits <- grep("static\\s+(thread_local\\s+)?int\\s+\\w*(debug|DEBUG)\\w*",
                 lines)
    if (length(hits)) {
      offenders <- c(offenders, sprintf("%s:%d: %s", basename(f), hits,
                                        trimws(lines[hits])))
    }
  }

  expect_equal(offenders, character(0))
})

test_that("floating-point sums do not use the OpenMP reduction clause", {
  src <- hygiene_src_dir()
  skip_if(is.null(src), "source tree not available")

  # libomp's tree reduction drops partial sums on aarch64-w64-mingw once the
  # team is as wide as the machine. ratiod_omp::sum_range (omp_sum.h) combines
  # per-thread slots itself.
  offenders <- character(0)
  for (f in hygiene_sources(src)) {
    lines <- readLines(f, warn = FALSE)
    code <- sub("//.*$", "", lines)
    hits <- grep("#\\s*pragma\\s+omp.*reduction\\s*\\(|^\\s*reduction\\s*\\(", code)
    if (length(hits)) {
      offenders <- c(offenders, sprintf("%s:%d: %s", basename(f), hits,
                                        trimws(code[hits])))
    }
  }

  expect_equal(offenders, character(0))
})
