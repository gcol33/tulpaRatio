#include "engine_fixture.h"
#include <array>
#include "lik_specs/family_weights.h"
using namespace tulpaRatio::margins;

template<class Num,class Den=NoMargin>
std::array<double,3> ratio_accumulation() {
  const int counts[1]={3},denom_counts[1]={8};
  const double num_y[1]={3},denom_y[1]={8},eta[2]={-.35,.2};
  Response nr,dr;
  if constexpr(std::is_same_v<Num,Gamma> || std::is_same_v<Num,Lognormal>) nr.continuous=num_y;
  else nr.counts=counts;
  nr.trials=denom_counts;
  if constexpr(std::is_same_v<Den,Gamma> || std::is_same_v<Den,Lognormal>) dr.continuous=denom_y;
  else if constexpr(!std::is_same_v<Den,NoMargin>) dr.counts=denom_counts;
  Ratio<Num,Den> ratio{Num(2.4),Den(1.7),nr,dr,1};
  double extra[2]={Num::n_extra ? 5.0:-7.0,-7.0},residual[2]={};
  const double ll=ratio(0,eta,residual,extra);
  return {Num::n_extra ? extra[0]:5.0,Den::n_extra ? extra[Num::n_extra]:-7.0,13.0+ll};
}

// [[Rcpp::export]]
Rcpp::NumericMatrix cpp_test_residual_accumulation() {
  const std::array<std::array<double,3>,7> rows={ratio_accumulation<Binomial>(),
    ratio_accumulation<Poisson,Gamma>(),ratio_accumulation<NegBin,Gamma>(),
    ratio_accumulation<NegBin,NegBin>(),ratio_accumulation<Gamma,Gamma>(),
    ratio_accumulation<Lognormal,Lognormal>(),ratio_accumulation<BetaBinomial>()};
  Rcpp::NumericMatrix out(rows.size(),3);
  for(size_t i=0;i<rows.size();++i) for(int j=0;j<3;++j) out(i,j)=rows[i][j];
  return out;
}

// [[Rcpp::export]]
Rcpp::NumericMatrix cpp_test_collapsed_margin(int family,int y,double yd,double en,
    double ed,double pn,double pd,bool shared) {
  const std::vector<std::string> families={"binomial","negbin_negbin","poisson_gamma","negbin_gamma"};
  if(family<0 || family>=static_cast<int>(families.size())) Rcpp::stop("Unknown family");
  auto data=engine_fixture::make_model("none",1,1,1,1,families[family]);
  data.response.y_num={y};data.response.y_denom={static_cast<int>(yd)};
  data.response.y_denom_cont={yd};
  const auto layout=engine_fixture::fixture_layout(data);
  std::vector<double> q(layout.total_params,0),score(data.n_processes),weights(data.n_processes*data.n_processes);
  const auto& entry=tulpaRatio::family_entry(families[family]);
  if(entry.n_extra) q[layout.extra_offset]=std::log(family==1 || family==3 ? pn:pd);
  if(entry.n_extra==2) q[layout.extra_offset+1]=std::log(pd);
  double eta[2]={en,ed};
  data.spec.eta_weights_fn(0,eta,0,0,q,data,layout,&data.response,score.data(),weights.data());
  const auto density=[&](bool both) {
    const double ll=data.spec.ll_double(0,eta,0,0,q,data,layout,&data.response);
    double extra_prior=0;
    if(entry.n_extra) extra_prior+=ratiod::math::log_prior_gamma(q[layout.extra_offset],data.phi_prior_shape,data.phi_prior_rate);
    if(entry.n_extra==2) extra_prior+=ratiod::math::log_prior_gamma(q[layout.extra_offset+1],data.phi_prior_shape,data.phi_prior_rate);
    if(both || data.n_processes==1) return ll-extra_prior;
    if(family==2) return Poisson().evaluate<false>({double(y),en}).ll;
    return NegBin(pn).evaluate<false>({double(y),en}).ll;
  };
  Rcpp::NumericMatrix out(2,3);
  for(int i=0;i<2;++i) {
    const bool both=i==0 || shared;
    out(i,0)=density(both);out(i,1)=score[0];out(i,2)=weights[0];
    if(both && data.n_processes==2) {out(i,1)+=score[1];out(i,2)+=weights[3];}
  }
  return out;
}

// [[Rcpp::export]]
Rcpp::NumericMatrix cpp_test_negbin_h_kernels(int y,double mu,double phi) {
  const NegBin margin(phi);
  Observation observation{double(y),std::log(mu)};
  observation.log_factorial=std::lgamma(y+1.);
  observation.lgamma_difference=tulpa::math::lgamma_ratio_count(y,phi);
  observation.digamma_difference=tulpa::math::polygamma_rising_difference(0,y,phi);
  const std::array<Result,2> rows={margin.evaluate<false>(observation),margin.evaluate<true>(observation)};
  Rcpp::NumericMatrix out(2,3);
  for(int i=0;i<2;++i) {out(i,0)=rows[i].ll;out(i,1)=rows[i].residual;out(i,2)=phi*rows[i].extra;}
  return out;
}

#include "covariance.h"
// [[Rcpp::export]]
Rcpp::NumericMatrix cpp_test_spatial_covariance(Rcpp::NumericVector distance,
    double variance,double range,std::string covariance,double nu=1.5) {
  const auto type=ratiod_cov::parse_cov_type(covariance,nu);
  const double scale=ratiod_cov::range_scale(type);
  Rcpp::NumericMatrix out(distance.size(),2);
  for(R_xlen_t i=0;i<distance.size();++i) {
    out(i,0)=tulpa::cov_value(distance[i],variance,range*scale,type);
    out(i,1)=scale*tulpa::cov_dphi(distance[i],range*scale,out(i,0),variance,type);
  }
  return out;
}
