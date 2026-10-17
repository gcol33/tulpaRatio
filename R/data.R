#' Pennsylvania Lung Cancer, 2002
#'
#' Lung cancer cases and population at risk in the 67 counties of
#' Pennsylvania in 2002, by race, sex and age group, with each county's
#' smoking rate. The rate cases / population varies with the strata, with
#' smoking, and between neighbouring counties; [pennlc_adj] gives the
#' counties' contiguity.
#'
#' @format A data frame with 1072 rows (67 counties x 2 races x 2 sexes x 4
#'   age groups) and 9 columns:
#' \describe{
#'   \item{county}{County name, lower case; the row names of [pennlc_adj].}
#'   \item{race}{`"white"` or `"other"`.}
#'   \item{sex}{`"female"` or `"male"`.}
#'   \item{age}{Age group: `"<40"`, `"40-59"`, `"60-69"` or `"70+"`.}
#'   \item{cases}{Lung cancer cases in 2002.}
#'   \item{population}{Population of the stratum. One stratum (Cameron
#'     county, other race, female, 70+) has none, and no cases.}
#'   \item{lon, lat}{Longitude and latitude of the county centroid.}
#'   \item{smoking}{Proportion of smokers in the county.}
#' }
#'
#' @source The `pennLC` data of the SpatialEpi package (version 1.2.8), from
#'   the Pennsylvania Department of Health (cases and population) and the
#'   Behavioral Risk Factor Surveillance System (smoking). The build script is
#'   `data-raw/pennlc.R` in the package's source repository.
#'
#' @examples
#' head(pennlc)
#' # Crude rate per 100,000 by age group
#' with(pennlc, tapply(cases, age, sum) / tapply(population, age, sum) * 1e5)
"pennlc"

#' Contiguity of the Pennsylvania Counties
#'
#' The queen contiguity of Pennsylvania's 67 counties: entry (i, j) is 1 when
#' counties i and j share a border or a corner. Rows and columns are named
#' by county, so `spatial_car()` and `spatial_bym2()` match them to the
#' `county` column of [pennlc] by name.
#'
#' @format A symmetric 67 x 67 integer matrix with county names as dimnames.
#'   Every county has between 2 and 9 neighbours.
#'
#' @source Computed with `spdep::poly2nb()` from the county polygons of the
#'   `pennLC` data of the SpatialEpi package (version 1.2.8); see
#'   `data-raw/pennlc.R` in the package's source repository.
#'
#' @examples
#' dim(pennlc_adj)
#' rowSums(pennlc_adj)["allegheny"]
"pennlc_adj"
