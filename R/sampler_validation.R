validate_legacy_collapsed <- function(spatial_info, temporal_info, spatial, family) {
  if (!identical(spatial_info$parameterization, "collapsed")) return(invisible(NULL))
  assert_legacy_collapsed_likelihood(family)
  if (!is.null(spatial$component)) {
    stop("The legacy collapsed spatial sampler requires a connected graph; ",
         "this adjacency has several components or an island.", call. = FALSE)
  }
  if (!is.null(spatial_info$rsr_basis)) {
    stop("A restricted spatial field (`spatial_rsr()`) cannot be collapsed: ",
         "the legacy marginal integrates the unrestricted field. ",
         "Use the default parameterization.", call. = FALSE)
  }
  if (!is.null(temporal_info$rtr_basis)) {
    stop("A restricted temporal effect (`temporal_rtr()`) cannot sit beside a ",
         "collapsed spatial field: the legacy marginal reads the temporal ",
         "effect unprojected. Use the default spatial parameterization.", call. = FALSE)
  }
  invisible(NULL)
}
