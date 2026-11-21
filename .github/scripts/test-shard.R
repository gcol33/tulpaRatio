# Test-file sharding shared by the workflows that split the NOT_CRAN suite
# into parallel jobs. Files are placed by measured cost: `seconds` lists each
# file's seconds from a coverage run (the slower of the two workflows), a
# file it does not list costs the median, and files are taken from the most
# expensive down and each given to the shard with the least cost so far
# (ties go to the lower shard and the earlier name, so every job computes the
# same plan). The filter matches exactly one shard's files in
# testthat::test_dir(filter =).

shard_files <- function(tests, shard, n, seconds = ".github/test-seconds.csv") {
  files <- sort(list.files(tests, "^test-.*\\.R$"))
  measured <- utils::read.csv(seconds)
  cost <- measured$seconds[match(files, measured$file)]
  cost[is.na(cost)] <- stats::median(measured$seconds)
  load <- numeric(n)
  owner <- integer(length(files))
  for (i in order(-cost, files)) {
    k <- which.min(load)
    owner[i] <- k
    load[k] <- load[k] + cost[i]
  }
  files[owner == shard]
}

# Test file stems hold letters, digits, '-', '_' and '.'; only the dot is a
# regex metacharacter among them.
shard_filter <- function(files) {
  stem <- gsub(".", "\\.", sub("\\.R$", "", sub("^test-", "", files)),
               fixed = TRUE)
  paste0("^(", paste(stem, collapse = "|"), ")$")
}

# Results of a test_dir() run as a data frame with each test's failure and
# error messages, written to `out`.
write_shard_results <- function(res, out) {
  r <- as.data.frame(res)
  r$message <- vapply(res, function(t) paste(vapply(
    Filter(function(e) inherits(e, c("expectation_failure", "expectation_error")),
           t$results),
    conditionMessage, ""), collapse = " | "), "")
  utils::write.csv(r[, c("file", "test", "failed", "error", "skipped", "real",
                         "message")], out, row.names = FALSE)
}

# Print the failures and per-file seconds of a results file; exit 1 on any
# failure or error.
report_shard_results <- function(out) {
  r <- utils::read.csv(out)
  bad <- r[r$failed > 0 | r$error, ]
  for (i in seq_len(nrow(bad))) {
    cat(bad$file[i], "::", bad$test[i], "\n  ", bad$message[i], "\n")
  }
  cat(nrow(r), "tests,", sum(r$failed), "failed,", sum(r$error), "errors\n")
  print(round(sort(tapply(r$real, r$file, sum), decreasing = TRUE)))
  if (nrow(bad) > 0) quit(status = 1L)
}
