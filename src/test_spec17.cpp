#include "engine_fixture.h"
#include <array>

namespace {
using namespace tulpaRatio::margins;

template<class Margin>
Observation fixture_observation(const engine_fixture::Fixture& data,int i,double eta,
    double phi,bool numerator,const double* mean) {
  Observation out{0,eta};
  if constexpr(std::is_same_v<Margin,NoMargin>) return out;
  if constexpr(std::is_same_v<Margin,Gamma> || std::is_same_v<Margin,Lognormal>) {
    out.y=numerator ? data.response.y_num_cont[i]:data.response.y_denom_cont[i];
    out.log_y=out.y>0 ? std::log(out.y):0;
  } else {
    out.y=numerator ? data.response.y_num[i]:data.response.y_denom[i];
    out.log_factorial=std::lgamma(out.y+1);
    if constexpr(std::is_same_v<Margin,NegBin>) {
      out.digamma_difference=tulpa::math::polygamma_rising_difference(0,int(out.y),phi);
      out.lgamma_difference=tulpa::math::lgamma_ratio_count(int(out.y),phi);
    }
    if constexpr(std::is_same_v<Margin,Binomial> || std::is_same_v<Margin,BetaBinomial>) {
      out.trials=data.response.y_denom[i];
      out.log_factorial=std::lgamma(out.trials+1.)-std::lgamma(out.y+1.)-
        std::lgamma(out.trials-out.y+1.);
    }
  }
  out.mean=mean;
  return out;
}

template<class Margin>
double raw_extra(const Margin& margin,double value) {
  if constexpr(Margin::log_extra) return value/margin.phi;
  return value;
}

template<class Num,class Den=NoMargin>
Rcpp::List margin_fixture(const std::string& family,const std::string& field) {
  if(field!="none" && field!="re" && field!="rw1") Rcpp::stop("Unknown margin fixture field");
  auto data=engine_fixture::make_model(field,80,5,5,17,family);
  constexpr bool single=std::is_same_v<Den,NoMargin>;
  const int p_num=2,p_denom=single ? 0:2;
  const int re_sigma=field=="re" ? p_num+p_denom:-1;
  const int re_start=re_sigma<0 ? -1:re_sigma+1;
  const int extra_start=p_num+p_denom+(field=="re" ? 6:0);
  const int phi_num=Num::n_extra ? extra_start:-1;
  const int phi_den=Den::n_extra ? extra_start+Num::n_extra:-1;
  const int tau_index=field=="rw1" ? extra_start+Num::n_extra+Den::n_extra:-1;
  const int temporal_start=tau_index<0 ? -1:tau_index+1;
  const int p=extra_start+Num::n_extra+Den::n_extra+(field=="rw1" ? 6:0);
  std::mt19937 rng(1017);
  std::normal_distribution<double> normal(0,.35);
  std::vector<double> q(p),g(p,0);
  for(double& x:q) x=normal(rng);
  const double pn=phi_num<0 ? 1:std::exp(q[phi_num]);
  const double pd=phi_den<0 ? 1:std::exp(q[phi_den]);
  const Num num(pn);const Den den(pd);
  double prior=0;
  if(re_sigma>=0) {
    const double sigma=std::exp(q[re_sigma]);
    const double tau=1/(sigma*sigma+1e-10);
    prior=prior+tulpa::math::log_prior_half_cauchy(q[re_sigma],1.);
    const double ratio_sq=sigma*sigma;
    g[re_sigma]=-2*ratio_sq/(1+ratio_sq)+1;
    double scale_score=0;
    for(int i=0;i<5;++i) {
      const double effect=q[re_start+i];
      prior=prior-.5*tau*effect*effect+.5*std::log(tau);
      g[re_start+i]=-tau*effect;
      scale_score+=tau*effect*effect-1;
    }
    prior=prior-.5*5*std::log(2*M_PI);
    g[re_sigma]+=scale_score;
  }
  for(int j=0;j<p_num+p_denom;++j) {
    prior=prior+ratiod::math::log_prior_normal(q[j],.01);
    g[j]=-.01*q[j];
  }
  for(int j:{phi_num,phi_den}) if(j>=0) {
    prior=prior+ratiod::math::log_prior_gamma(q[j],1,.01);
    g[j]=1-.01*std::exp(q[j]);
  }
  std::vector<double> temporal(5),temporal_score(5,0);
  double sum=0,tau=1;
  if(tau_index>=0) {
    tau=std::exp(q[tau_index]);
    sum=tulpa::s2z_component_sum(q.data(),temporal_start,5);
    std::copy(q.begin()+temporal_start,q.begin()+temporal_start+5,temporal.begin());
    tulpa::s2z_centre_component(temporal.data(),0,5);
    prior=prior+ratiod::math::pc_log_density(q[tau_index],-.5,3,.01);
    double quad=0;
    for(int t=1;t<5;++t) {
      const double difference=q[temporal_start+t]-q[temporal_start+t-1];
      quad=quad+difference*difference;
    }
    double augmented=0;
    augmented=augmented+tulpa::s2z_aug_coef(1.,5)*sum*sum;
    quad=quad+augmented;
    prior=prior+.5*5*q[tau_index];
    prior=prior-.5*tau*quad;
    g[tau_index]=ratiod::math::pc_scale_gradient(1/std::sqrt(tau),-.5,3,.01);
  }
  const int N=data.N;
  std::vector<double> en(N),ed(N),mn(N),md(N);
  for(int i=0;i<N;++i) {
    for(int j=0;j<p_num;++j) en[i]+=data.processes[0].X_flat[i*p_num+j]*q[j];
    if constexpr(!single) for(int j=0;j<p_denom;++j)
      ed[i]+=data.processes[1].X_flat[i*p_denom+j]*q[p_num+j];
    if(re_start>=0) {
      en[i]+=q[re_start+data.re_group[i]-1];
      if constexpr(!single) ed[i]+=q[re_start+data.re_group[i]-1];
    }
    if(temporal_start>=0) {
      const double value=temporal[data.temporal_time_idx[i]-1];
      en[i]+=value;
      if constexpr(!single) ed[i]+=value;
    }
  }
  constexpr bool multipass=!std::is_same_v<Num,Binomial> &&
    !std::is_same_v<Num,BetaBinomial> && !std::is_same_v<Num,Lognormal>;
  if constexpr(multipass) {
    using Array=Eigen::Array<double,Eigen::Dynamic,1>;
    Eigen::Map<Array>(mn.data(),N)=Eigen::Map<const Array>(en.data(),N).exp();
    Eigen::Map<Array>(md.data(),N)=Eigen::Map<const Array>(ed.data(),N).exp();
  }
  double gn[2]={},gd[2]={},extra_n=0,extra_d=0;
  double likelihood=0;
  for(int i=0;i<N;++i) {
    const auto xn=fixture_observation<Num>(data,i,en[i],pn,true,multipass ? &mn[i]:nullptr);
    const auto xd=fixture_observation<Den>(data,i,ed[i],pd,false,multipass ? &md[i]:nullptr);
    const auto a=num.template evaluate<true>(xn);
    const auto b=den.template evaluate<true>(xd);
    extra_n+=raw_extra(num,a.extra);extra_d+=raw_extra(den,b.extra);
    for(int j=0;j<p_num;++j) gn[j]+=data.processes[0].X_flat[i*p_num+j]*a.residual;
    if constexpr(!single) for(int j=0;j<p_denom;++j)
      gd[j]+=data.processes[1].X_flat[i*p_denom+j]*b.residual;
    double residual=a.residual;
    if constexpr(!single) residual+=b.residual;
    if(re_start>=0) g[re_start+data.re_group[i]-1]+=residual;
    if(temporal_start>=0) temporal_score[data.temporal_time_idx[i]-1]+=residual;
    using namespace ratiod::math;
    double value;
    if constexpr(std::is_same_v<Num,Binomial>) value=log_lik_binomial(data.response.y_num[i],data.response.y_denom[i],en[i]);
    else if constexpr(std::is_same_v<Num,BetaBinomial>) value=log_lik_beta_binomial(data.response.y_num[i],data.response.y_denom[i],en[i],pn);
    else if constexpr(std::is_same_v<Num,Lognormal>) value=log_lik_lognormal_lognormal(data.response.y_num_cont[i],data.response.y_denom_cont[i],en[i],ed[i],pn,pd);
    else if constexpr(std::is_same_v<Num,Gamma>) value=log_lik_gamma_gamma(data.response.y_num_cont[i],data.response.y_denom_cont[i],std::exp(en[i]),std::exp(ed[i]),pn,pd);
    else {
      if constexpr(std::is_same_v<Num,Poisson>) value=log_lik_poisson(data.response.y_num[i],std::exp(en[i]));
      else value=log_lik_negbin(data.response.y_num[i],std::exp(en[i]),pn);
      if constexpr(std::is_same_v<Den,Gamma>) value=value+log_lik_gamma(data.response.y_denom_cont[i],pd,std::exp(ed[i]));
      else value=value+log_lik_negbin(data.response.y_denom[i],std::exp(ed[i]),pd);
    }
    likelihood+=value;
  }
  for(int j=0;j<p_num;++j) g[j]+=gn[j];
  for(int j=0;j<p_denom;++j) g[p_num+j]+=gd[j];
  if(phi_num>=0) g[phi_num]+=pn*extra_n;
  if(phi_den>=0) g[phi_den]+=pd*extra_d;
  if(temporal_start>=0) {
    std::copy(temporal_score.begin(),temporal_score.end(),g.begin()+temporal_start);
    tulpa::s2z_centre_blocks(g.data()+temporal_start,1,5);
    const double augmented_gradient=-tulpa::s2z_aug_coef(tau,5)*sum;
    for(int t=0;t<5;++t) g[temporal_start+t]+=augmented_gradient;
    double quad=0;
    for(int t=0;t<5;++t) {
      double gradient=0;
      if(t>0) {
        gradient+=tau*(temporal[t-1]-temporal[t]);
        quad+=(temporal[t]-temporal[t-1])*(temporal[t]-temporal[t-1]);
      }
      if(t<4) gradient+=tau*(temporal[t+1]-temporal[t]);
      g[temporal_start+t]+=gradient;
    }
    double augmented=0;
    augmented+=tulpa::s2z_aug_coef(1.,5)*sum*sum;
    g[tau_index]+=.5*5-.5*tau*(quad+augmented);
  }
  return Rcpp::List::create(Rcpp::Named("analytic")=g,Rcpp::Named("log_post")=prior+likelihood);
}
}

// [[Rcpp::export]]
Rcpp::List cpp_test_spec17(std::string family,std::string field) {
  using namespace tulpaRatio::margins;
  if(family=="binomial") return margin_fixture<Binomial>(family,field);
  if(family=="poisson_gamma") return margin_fixture<Poisson,Gamma>(family,field);
  if(family=="negbin_gamma") return margin_fixture<NegBin,Gamma>(family,field);
  if(family=="negbin_negbin") return margin_fixture<NegBin,NegBin>(family,field);
  if(family=="gamma_gamma") return margin_fixture<Gamma,Gamma>(family,field);
  if(family=="lognormal") return margin_fixture<Lognormal,Lognormal>(family,field);
  if(family=="beta_binomial") return margin_fixture<BetaBinomial>(family,field);
  Rcpp::stop("Unknown margin fixture family");
}
