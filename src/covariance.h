#ifndef TULPARATIO_COVARIANCE_H
#define TULPARATIO_COVARIANCE_H
#include <Rcpp.h>
#include <cmath>
#include <tulpa/cov_kernel.h>
#include <tulpa/gp_conditioning.h>
#include "enum_names.h"
namespace ratiod_cov {
using tulpa::CovType;
inline constexpr ratiod_enum::EnumName<CovType> COV_TYPE_NAMES[]={
  {"exponential",CovType::EXPONENTIAL},{"matern",CovType::MATERN32},
  {"gaussian",CovType::GAUSSIAN},{"spherical",CovType::SPHERICAL},{"matern52",CovType::MATERN52}};
inline CovType parse_cov_type(const std::string& name,double nu=1.5) {
  if(name=="matern") {
    if(nu==.5) return CovType::EXPONENTIAL;
    if(nu==1.5) return CovType::MATERN32;
    if(nu==2.5) return CovType::MATERN52;
    Rcpp::stop("Matern smoothness must be 0.5, 1.5 or 2.5");
  }
  return ratiod_enum::parse_enum(COV_TYPE_NAMES,name,"covariance");
}
template<class Cov>
inline double range_scale(Cov type) {
  return type==Cov::GAUSSIAN ? std::sqrt(2.0):1.0;
}
inline double compute_cov(double distance,double sigma2,double lengthscale,CovType type) {
  const double range=range_scale(type)*lengthscale;
  return tulpa::cov_value(distance,sigma2,range,type);
}
using tulpa_gp_conditioning::nngp_chol;
using tulpa_gp_conditioning::nngp_forward_solve;
using tulpa_gp_conditioning::nngp_back_solve;
using tulpa_gp_conditioning::nngp_floor_cond_var;
}
#endif
