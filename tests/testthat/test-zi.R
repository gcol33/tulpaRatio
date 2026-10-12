# test-zi.R
# Tests for zero-inflation specification and helper functions

test_that("zi_poisson creates correct structure", {
  zi <- zi_poisson()

  expect_s3_class(zi, "ratiod_zi")
  expect_equal(zi$type, "zi_poisson")
  expect_null(zi$formula)
  expect_equal(zi$distribution, "poisson")
})


test_that("zi_poisson with formula", {
  zi <- zi_poisson(~ habitat)

  expect_s3_class(zi, "ratiod_zi")
  expect_equal(zi$type, "zi_poisson")
  expect_true(inherits(zi$formula, "formula"))
})


test_that("zi_negbin creates correct structure", {
  zi <- zi_negbin()

  expect_s3_class(zi, "ratiod_zi")
  expect_equal(zi$type, "zi_negbin")
  expect_null(zi$formula)
  expect_equal(zi$distribution, "negbin")
})


test_that("hurdle_poisson creates correct structure", {
  zi <- hurdle_poisson()

  expect_s3_class(zi, "ratiod_zi")
  expect_equal(zi$type, "hurdle_poisson")
  expect_null(zi$formula)
  expect_equal(zi$distribution, "poisson")
})


test_that("hurdle_negbin creates correct structure", {
  zi <- hurdle_negbin()

  expect_s3_class(zi, "ratiod_zi")
  expect_equal(zi$type, "hurdle_negbin")
  expect_null(zi$formula)
  expect_equal(zi$distribution, "negbin")
})


test_that("print.ratiod_zi works", {
  zi <- zi_poisson(~ habitat)
  output <- capture.output(print(zi))
  expect_true(any(grepl("Zero-Inflation", output)))
  expect_true(any(grepl("Poisson", output)))
})


test_that("validate_zi returns NULL for NULL input", {
  result <- validate_zi(NULL)
  expect_null(result)
})


test_that("validate_zi passes valid ratiod_zi", {
  zi <- zi_poisson()
  result <- validate_zi(zi)
  expect_s3_class(result, "ratiod_zi")
})


test_that("validate_zi rejects non-ratiod_zi objects", {
  expect_error(
    validate_zi(list(type = "zi_poisson")),
    "ratiod_zi object"
  )
})


test_that("validate_zi checks formula variables exist", {
  zi <- zi_poisson(~ nonexistent_var)
  df <- data.frame(x = 1:10, y = 10:1)

  expect_error(
    validate_zi(zi, df),
    "not in data"
  )
})


test_that("prepare_zi_for_hmc returns none for NULL zi", {
  zi_info <- tulpaRatio:::prepare_zi_for_hmc(NULL, data.frame(), 10)

  expect_equal(zi_info$type, "none")
  expect_equal(nrow(zi_info$X_zi), 10)
  expect_equal(ncol(zi_info$X_zi), 1)
})


test_that("prepare_zi_for_hmc creates design matrix for intercept-only", {
  zi <- zi_poisson()
  df <- data.frame(x = 1:10, y = 10:1)

  zi_info <- tulpaRatio:::prepare_zi_for_hmc(zi, df, 10)

  expect_equal(zi_info$type, "zi_poisson")
  expect_equal(nrow(zi_info$X_zi), 10)
  expect_equal(ncol(zi_info$X_zi), 1)
  expect_true(all(zi_info$X_zi[, 1] == 1))  # Intercept
})


test_that("prepare_zi_for_hmc creates design matrix with predictors", {
  zi <- zi_poisson(~ x)
  df <- data.frame(x = 1:10, y = 10:1)

  zi_info <- tulpaRatio:::prepare_zi_for_hmc(zi, df, 10)

  expect_equal(zi_info$type, "zi_poisson")
  expect_equal(nrow(zi_info$X_zi), 10)
  expect_equal(ncol(zi_info$X_zi), 2)  # Intercept + x
  expect_equal(zi_info$p_zi, 2)
})


test_that("ratiod_zinegbin creates ZI family", {
  fam <- ratiod_zinegbin()

  expect_s3_class(fam, "ratiod_family_zi")
  expect_true(fam$zero_inflated)
  expect_equal(fam$zi_type, "mixture")
})


test_that("ratiod_hurdle_negbin creates hurdle family", {
  fam <- ratiod_hurdle_negbin()

  expect_s3_class(fam, "ratiod_family_zi")
  expect_true(fam$zero_inflated)
  expect_equal(fam$zi_type, "hurdle")
})


test_that("is_zi_family correctly identifies ZI families", {
  expect_true(tulpaRatio:::is_zi_family(ratiod_zinegbin()))
  expect_true(tulpaRatio:::is_zi_family(ratiod_zipois()))
  expect_false(tulpaRatio:::is_zi_family(ratiod_negbin_negbin()))
})


test_that("is_hurdle_family correctly identifies hurdle families", {
  expect_true(tulpaRatio:::is_hurdle_family(ratiod_hurdle_negbin()))
  expect_true(tulpaRatio:::is_hurdle_family(ratiod_hurdle_pois()))
  expect_false(tulpaRatio:::is_hurdle_family(ratiod_zinegbin()))
})


# ============================================================================
# Tests for new ZI variants (v1.2.0)
# ============================================================================

test_that("ratiod_zibinomial creates correct structure", {
  fam <- ratiod_zibinomial()

  expect_s3_class(fam, "ratiod_family_zi")
  expect_equal(fam$name, "zibinomial")
  expect_true(fam$zero_inflated)
  expect_equal(fam$zi_type, "mixture")
  expect_equal(fam$numerator$base_distribution, "binomial")
  expect_equal(fam$numerator$link, "logit")
})


test_that("ratiod_oibinomial creates one-inflated structure", {
  fam <- ratiod_oibinomial()

  expect_s3_class(fam, "ratiod_family_zi")
  expect_equal(fam$name, "oibinomial")
  expect_false(fam$zero_inflated)
  expect_true(fam$one_inflated)
  expect_equal(fam$zi_type, "one_inflated")
})


test_that("ratiod_zoibinomial creates ZOIB structure", {
  fam <- ratiod_zoibinomial()

  expect_s3_class(fam, "ratiod_family_zi")
  expect_equal(fam$name, "zoibinomial")
  expect_true(fam$zero_inflated)
  expect_true(fam$one_inflated)
  expect_equal(fam$zi_type, "zoib")
  expect_true(!is.null(fam$numerator$link_zi))
  expect_true(!is.null(fam$numerator$link_oi))
})


test_that("ratiod_hurdle_binomial creates correct structure", {
  fam <- ratiod_hurdle_binomial()

  expect_s3_class(fam, "ratiod_family_zi")
  expect_equal(fam$name, "hurdle_binomial")
  expect_true(fam$zero_inflated)
  expect_equal(fam$zi_type, "hurdle")
  expect_equal(fam$numerator$base_distribution, "binomial")
})


test_that("is_oi_family identifies one-inflated families", {
  expect_true(tulpaRatio:::is_oi_family(ratiod_oibinomial()))
  expect_true(tulpaRatio:::is_oi_family(ratiod_zoibinomial()))
  expect_false(tulpaRatio:::is_oi_family(ratiod_zinegbin()))
  expect_false(tulpaRatio:::is_oi_family(ratiod_zibinomial()))
})


test_that("is_zoib_family identifies ZOIB families", {
  expect_true(tulpaRatio:::is_zoib_family(ratiod_zoibinomial()))
  expect_false(tulpaRatio:::is_zoib_family(ratiod_oibinomial()))
  expect_false(tulpaRatio:::is_zoib_family(ratiod_zibinomial()))
})


test_that("print works for new ZI families", {
  fam <- ratiod_zoibinomial()
  output <- capture.output(print(fam))
  expect_true(any(grepl("zoibinomial", output)))
})


test_that("all ZI families validate link functions", {
  expect_error(ratiod_zibinomial(link_num = "invalid"))
  expect_error(ratiod_oibinomial(link_oi = "invalid"))
  expect_error(ratiod_zoibinomial(link_zi = "identity"))
  expect_error(ratiod_hurdle_binomial(link_hurdle = "log"))
})


# ============================================================================
# Explicit zi = binomial-family ZI/hurdle/OI/ZOIB constructors (#33)
# ============================================================================

test_that("zi_binomial creates correct structure", {
  zi <- zi_binomial()

  expect_s3_class(zi, "ratiod_zi")
  expect_equal(zi$type, "zi_binomial")
  expect_null(zi$formula)
  expect_equal(zi$distribution, "binomial")
})


test_that("hurdle_binomial creates correct structure", {
  zi <- hurdle_binomial()

  expect_s3_class(zi, "ratiod_zi")
  expect_equal(zi$type, "hurdle_binomial")
  expect_null(zi$formula)
  expect_equal(zi$distribution, "binomial")
})


test_that("oi_binomial creates correct structure", {
  zi <- oi_binomial()

  expect_s3_class(zi, "ratiod_zi")
  expect_equal(zi$type, "oi_binomial")
  expect_null(zi$formula)
  expect_equal(zi$distribution, "binomial")
})


test_that("zoib creates correct structure with separate zi/oi formulas", {
  zi <- zoib(~ habitat, ~ effort)

  expect_s3_class(zi, "ratiod_zi")
  expect_equal(zi$type, "zoib")
  expect_true(inherits(zi$formula, "formula"))
  expect_true(inherits(zi$oi_formula, "formula"))
  expect_equal(zi$distribution, "binomial")
})


test_that("validate_zi accepts the new binomial-family ZI types", {
  expect_s3_class(validate_zi(zi_binomial()), "ratiod_zi")
  expect_s3_class(validate_zi(hurdle_binomial()), "ratiod_zi")
  expect_s3_class(validate_zi(oi_binomial()), "ratiod_zi")
  expect_s3_class(validate_zi(zoib()), "ratiod_zi")
})


test_that("validate_zi checks zoib's oi_formula variables exist", {
  zi <- zoib(oi_formula = ~ nonexistent_var)
  df <- data.frame(x = 1:10, y = 10:1)

  expect_error(
    validate_zi(zi, df),
    "not in data"
  )
})


test_that("print works for the new binomial-family ZI specs", {
  output <- capture.output(print(zi_binomial()))
  expect_true(any(grepl("Zero-Inflated Binomial", output)))

  output <- capture.output(print(zoib(~ habitat, ~ effort)))
  expect_true(any(grepl("Zero-and-One-Inflated Binomial", output)))
  expect_true(any(grepl("Zero-inflation formula", output)))
  expect_true(any(grepl("One-inflation formula", output)))
})


test_that("prepare_zi_for_hmc builds OI-only design (no ZI coefficient)", {
  zi <- oi_binomial(~ x)
  df <- data.frame(x = 1:10, y = 10:1)

  zi_info <- tulpaRatio:::prepare_zi_for_hmc(zi, df, 10)

  expect_equal(zi_info$type, "oi_binomial")
  expect_equal(zi_info$p_zi, 0L)
  expect_equal(ncol(zi_info$X_oi), 2)  # Intercept + x
  expect_equal(zi_info$p_oi, 2)
})


test_that("prepare_zi_for_hmc builds both ZI and OI designs for zoib", {
  zi <- zoib(~ x, ~ 1)
  df <- data.frame(x = 1:10, y = 10:1)

  zi_info <- tulpaRatio:::prepare_zi_for_hmc(zi, df, 10)

  expect_equal(zi_info$type, "zoib")
  expect_equal(ncol(zi_info$X_zi), 2)  # Intercept + x
  expect_equal(zi_info$p_zi, 2)
  expect_equal(ncol(zi_info$X_oi), 1)  # Intercept only
  expect_equal(zi_info$p_oi, 1)
})


test_that("prepare_zi_for_hmc auto-detects the specific ZI string from a family", {
  df <- data.frame(x = 1:10, y = 10:1)

  # zi_negbin family: must produce "zi_negbin", never a bare "zi" (#33)
  zi_info <- tulpaRatio:::prepare_zi_for_hmc(NULL, df, 10, ratiod_zinegbin())
  expect_equal(zi_info$type, "zi_negbin")

  # hurdle_negbin family: must produce "hurdle_negbin", never a bare "hurdle"
  zi_info <- tulpaRatio:::prepare_zi_for_hmc(NULL, df, 10, ratiod_hurdle_negbin())
  expect_equal(zi_info$type, "hurdle_negbin")

  # Binomial-family ZI variants: previously unreachable through auto-detection
  zi_info <- tulpaRatio:::prepare_zi_for_hmc(NULL, df, 10, ratiod_zibinomial())
  expect_equal(zi_info$type, "zi_binomial")
  expect_equal(zi_info$p_zi, 1L)
  expect_equal(zi_info$p_oi, 0L)

  zi_info <- tulpaRatio:::prepare_zi_for_hmc(NULL, df, 10, ratiod_hurdle_binomial())
  expect_equal(zi_info$type, "hurdle_binomial")

  zi_info <- tulpaRatio:::prepare_zi_for_hmc(NULL, df, 10, ratiod_oibinomial())
  expect_equal(zi_info$type, "oi_binomial")
  expect_equal(zi_info$p_zi, 0L)
  expect_equal(zi_info$p_oi, 1L)

  zi_info <- tulpaRatio:::prepare_zi_for_hmc(NULL, df, 10, ratiod_zoibinomial())
  expect_equal(zi_info$type, "zoib")
  expect_equal(zi_info$p_zi, 1L)
  expect_equal(zi_info$p_oi, 1L)
})
