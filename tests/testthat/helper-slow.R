# Tier 3 (samplers, multi-seed coverage, benchmarks) runs only when
# TULPA_SLOW_TESTS is "true", the variable the tulpa engine's test tiers use.
skip_unless_slow <- function() {
  if (!identical(Sys.getenv("TULPA_SLOW_TESTS"), "true"))
    testthat::skip("slow tier: set TULPA_SLOW_TESTS=true")
}
