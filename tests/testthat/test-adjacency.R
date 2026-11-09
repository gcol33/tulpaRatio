test_that("CSR and neighbor lists retain directed edges and isolated rows", {
  adj <- matrix(0, 5, 5)
  adj[cbind(c(1, 1, 3, 4), c(2, 4, 1, 3))] <- c(1, 2, 1, 1)
  expected <- list(
    adj_row_ptr = c(0L, 2L, 2L, 3L, 4L, 4L),
    adj_col_idx = c(2L, 4L, 1L, 3L),
    n_neighbors = c(2L, 0L, 1L, 1L, 0L),
    adj_list = list(c(2L, 4L), integer(0), 1L, 3L, integer(0)))
  expect_identical(adjacency_csr(adj), expected)
  expect_identical(adjacency_csr(adj > 0), expected)
  expect_identical(adjacency_csr(matrix(0, 0, 0)), list(
    adj_row_ptr = 0L, adj_col_idx = integer(0), n_neighbors = integer(0),
    adj_list = list()))
})

test_that("sparse storage preserves CSR ordering and expands symmetric triangles", {
  skip_if_not_installed("Matrix")
  adj <- matrix(0, 5, 5)
  adj[cbind(c(1, 2, 2, 3), c(2, 1, 3, 2))] <- 1
  expected <- adjacency_csr(adj)
  sparse <- Matrix::Matrix(adj, sparse = TRUE)
  representations <- list(
    sparse, methods::as(sparse, "generalMatrix"),
    methods::as(methods::as(sparse, "generalMatrix"), "RsparseMatrix"),
    methods::as(sparse, "TsparseMatrix"), sparse > 0,
    methods::as(sparse, "nMatrix"), Matrix::Matrix(adj, sparse = FALSE))
  for (x in representations) expect_identical(adjacency_csr(x), expected)
  expect_identical(adjacency_csr(Matrix::Diagonal(5)), adjacency_csr(diag(5)))
  explicit_zero <- Matrix::sparseMatrix(i = c(1, 2, 3), j = c(2, 1, 4),
                                       x = c(1, 1, 0), dims = c(5, 5))
  expect_identical(adjacency_csr(explicit_zero), adjacency_csr(as.matrix(explicit_zero)))
})

test_that("every spatial interface uses its required index base", {
  adj <- matrix(0, 4, 4)
  adj[cbind(c(1, 2, 2, 3), c(2, 1, 3, 2))] <- 1
  spatial <- spatial_car(adj, group_var = "g")
  d <- data.frame(g = factor(1:4), a = 1:4, b = 10L)
  f <- ratiod_formula(a | b ~ 1 + (1 | g), data = d)
  csr <- adjacency_csr(adj)
  hmc <- prepare_spatial_for_hmc(spatial, d, 4L)
  laplace <- prepare_spatial_for_laplace(spatial, d, f)
  pg <- prepare_spatial_for_pg(spatial, d, f)
  st <- prepare_spatial_precision(spatial)
  for (x in list(hmc, laplace, st)) {
    expect_identical(x$adj_row_ptr, csr$adj_row_ptr)
    expect_identical(x$n_neighbors, csr$n_neighbors)
  }
  expect_identical(hmc$adj_col_idx, csr$adj_col_idx - 1L)
  expect_identical(laplace$adj_col_idx, csr$adj_col_idx - 1L)
  expect_identical(st$adj_col_idx, csr$adj_col_idx)
  expect_identical(pg$adj_list, csr$adj_list)
  expect_identical(pg$n_neighbors, csr$n_neighbors)
})

test_that("single and crossed random intercepts share group ordering and offsets", {
  d <- data.frame(a = 1:6, b = 10L, g = factor(rep(c("b", "a"), 3)),
                  h = factor(rep(1:3, each = 2)))
  for (rhs in c("1", "1 + (1 | g)", "1 + (1 | g) + (1 | h)")) {
    f <- ratiod_formula(as.formula(paste("a | b ~", rhs)), data = d)
    re <- extract_re_from_data(f, d)
    terms <- f$numerator$random_effects
    expect_equal(re$n_re_terms, length(terms))
    offset <- 0L
    for (i in seq_along(terms)) {
      expect_identical(re$re_terms[[i]]$group_idx, as.integer(terms[[i]]$group))
      expect_identical(re$re_terms[[i]]$offset, offset)
      offset <- offset + terms[[i]]$n_groups
    }
    if (length(terms)) expect_equal(re$total_groups, offset)
  }
})
