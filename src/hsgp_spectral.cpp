#include <Rcpp.h>
#include <tulpa/hsgp_spectral.h>

// [[Rcpp::export]]
Rcpp::NumericMatrix cpp_hsgp_spectrum(Rcpp::NumericVector eigenvalues,
                                     Rcpp::NumericVector sigma2,
                                     Rcpp::NumericVector lengthscale,
                                     int dimension, int kernel) {
    if (dimension < 1 || dimension > 3 || kernel < 0 || kernel > 2)
        Rcpp::stop("HSGP dimension must be 1, 2 or 3 and kernel must be 0, 1 or 2.");
    if (sigma2.size() != lengthscale.size())
        Rcpp::stop("HSGP scale and lengthscale draws must have equal lengths.");
    Rcpp::NumericMatrix out(sigma2.size(), eigenvalues.size());
    for (int i = 0; i < sigma2.size(); ++i) {
        if (!R_finite(sigma2[i]) || sigma2[i] <= 0 ||
            !R_finite(lengthscale[i]) || lengthscale[i] <= 0)
            Rcpp::stop("HSGP scales and lengthscales must be finite and positive.");
        for (int j = 0; j < eigenvalues.size(); ++j) {
            if (!R_finite(eigenvalues[j]) || eigenvalues[j] < 0)
                Rcpp::stop("HSGP eigenvalues must be finite and nonnegative.");
            out(i, j) = tulpa_hsgp_spectral::spectral_density(
                eigenvalues[j], sigma2[i], lengthscale[i], dimension, kernel);
        }
    }
    return out;
}
