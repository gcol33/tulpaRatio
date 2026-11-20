# Test-file sharding shared by the workflows that split the NOT_CRAN suite
# into parallel jobs. Shard k of n takes every n-th test file in sorted order
# from the k-th, and the filter matches exactly those files in
# testthat::test_dir(filter =).

shard_files <- function(tests, shard, n) {
  files <- sort(list.files(tests, "^test-.*\\.R$"))
  files[(seq_along(files) - 1L) %% n == shard - 1L]
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
