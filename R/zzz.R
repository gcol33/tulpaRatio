.onLoad <- function(libname, pkgname) {
  # Load tulpa so its R_RegisterCCallable engine symbols are available before
  # any compiled backend resolves them with R_GetCCallable.
  requireNamespace("tulpa")
  models <- MODEL_TYPES
  entries <- cpp_ratio_family_entries()
  for (name in names(entries)) {
    entry <- entries[[name]]
    models[[name]][names(entry)] <- entry
  }
  assign("MODEL_TYPES", models, envir = asNamespace(pkgname))
}
