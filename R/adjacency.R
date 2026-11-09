adjacency_csr <- function(adj) {
  n <- nrow(adj)
  if (inherits(adj, "Matrix")) {
    rows <- methods::as(methods::as(adj, "generalMatrix"), "RsparseMatrix")
    keep <- if (inherits(rows, "nMatrix")) seq_along(rows@j) else which(rows@x > 0)
    row_idx <- rep.int(seq_len(n), diff(rows@p))[keep]
    col_idx <- rows@j[keep] + 1L
  } else {
    edges <- which(t(adj) > 0, arr.ind = TRUE)
    row_idx <- edges[, 2L]
    col_idx <- edges[, 1L]
  }
  n_neighbors <- tabulate(row_idx, nbins = n)
  list(
    adj_row_ptr = as.integer(c(0L, cumsum(n_neighbors))),
    adj_col_idx = as.integer(col_idx),
    n_neighbors = n_neighbors,
    adj_list = unname(split(as.integer(col_idx), factor(row_idx, levels = seq_len(n))))
  )
}
