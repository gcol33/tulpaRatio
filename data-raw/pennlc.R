# Pennsylvania lung cancer, 2002: cases and population by county and stratum,
# with county smoking rates and the counties' contiguity.
#
# Source: the pennLC data of the SpatialEpi package (version 1.2.8), from the
# Pennsylvania Department of Health (cases, population) and the Behavioral
# Risk Factor Surveillance System (smoking).

data(pennLC, package = "SpatialEpi")

strata <- pennLC$data
counties <- data.frame(
  county = as.character(pennLC$geo$county),
  lon = pennLC$geo$x,
  lat = pennLC$geo$y
)
counties <- merge(counties,
                  data.frame(county = as.character(pennLC$smoking$county),
                             smoking = pennLC$smoking$smoking),
                  by = "county")

pennlc <- data.frame(
  county = as.character(strata$county),
  race = factor(strata$race, levels = c("w", "o"), labels = c("white", "other")),
  sex = factor(strata$gender, levels = c("f", "m"), labels = c("female", "male")),
  age = factor(strata$age, levels = c("Under.40", "40.59", "60.69", "70+"),
               labels = c("<40", "40-59", "60-69", "70+")),
  cases = as.integer(strata$cases),
  population = as.integer(strata$population)
)
pennlc <- merge(pennlc, counties, by = "county", sort = FALSE)
pennlc <- pennlc[order(pennlc$county, pennlc$race, pennlc$sex, pennlc$age), ]
rownames(pennlc) <- NULL
stopifnot(nrow(pennlc) == 1072L, !anyNA(pennlc))

# Queen contiguity of the county polygons, named by county
polygons <- sf::st_as_sf(pennLC$spatial.polygon)
ids <- names(pennLC$spatial.polygon)
nb <- spdep::poly2nb(polygons, queen = TRUE)
pennlc_adj <- spdep::nb2mat(nb, style = "B", zero.policy = TRUE)
dimnames(pennlc_adj) <- list(ids, ids)
storage.mode(pennlc_adj) <- "integer"
ord <- order(ids)
pennlc_adj <- pennlc_adj[ord, ord]
stopifnot(isSymmetric(pennlc_adj), all(rowSums(pennlc_adj) > 0),
          setequal(rownames(pennlc_adj), pennlc$county))

save(pennlc, file = "data/pennlc.rda", compress = "xz")
save(pennlc_adj, file = "data/pennlc_adj.rda", compress = "xz")
