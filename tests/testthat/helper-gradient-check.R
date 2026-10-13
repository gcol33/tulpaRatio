# Relative deviation per parameter between the analytic gradient and the
# central difference `cpp_gradient_check()` returns, with an absolute allowance
# first.
#
# A central difference of a log posterior of magnitude L with step h carries a
# roundoff error of about eps*L/h. A purely relative criterion demands more than
# that from any entry whose true value is small: the HSGP basis coefficients run
# to 6e-4, where 1e-4 relative is 6e-8, below what the difference can resolve.
# Sweeping h on that field traces the expected V (1.2e-4 at h = 1e-3, 1.5e-7 at
# h = 1e-5, 6.3e-6 at h = 1e-7) rather than the flat line a missing term would
# give. The allowance is 1e-6 or four times eps*L/h, whichever is larger: the
# space-time fixtures reach L ~ 2e5, where eps*L/h is 3.5e-6 and every entry
# differs from its difference by about 2e-6 whatever its size. A term actually
# absent from one side is wrong by order 1, so the allowance cannot hide one.
grad_rel_dev <- function(r, h = 1e-5) {
  atol <- max(1e-6, 4 * .Machine$double.eps * abs(r$log_post) / h)
  pmax(0, abs(r$analytic - r$finite_diff) - atol) /
    pmax(1e-6, pmax(abs(r$analytic), abs(r$finite_diff)))
}
