#include <RcppEigen.h>
#include "hmc_grad_helpers.h"
#include "hmc_gp_collapsed.h"

using namespace Rcpp;

namespace ratiod_hmc {

Rcpp::NumericMatrix residual_accumulation_probe() {
  ModelData data;
  data.N = 1; data.y_num = {3}; data.y_denom = {8};
  data.y_num_cont = {3.0}; data.y_denom_cont = {8.0};
  vectorized::VecGradWorkspace ws;
  ws.init(1); ws.precompute(data); ws.build_digamma_tables(2.4, 1.7, true);
  Rcpp::NumericMatrix out(7,3);
  double en = -0.35, ed = 0.2, rn, rd;
  int row = 0;
  const auto run = [&](auto tag) {
    double gn = 5.0, gd = -7.0, ll = 13.0;
    vectorized::compute_residuals_impl(tag,1,&en,&ed,data,2.4,1.7,&rn,&rd,gn,gd,ll,true,ws);
    out(row,0) = gn; out(row,1) = gd; out(row++,2) = ll;
  };
  run(vectorized::ModelTag<ModelType::BINOMIAL>{});
  run(vectorized::ModelTag<ModelType::POISSON_GAMMA>{});
  run(vectorized::ModelTag<ModelType::NEGBIN_GAMMA>{});
  run(vectorized::ModelTag<ModelType::NEGBIN_NEGBIN>{});
  run(vectorized::ModelTag<ModelType::GAMMA_GAMMA>{});
  run(vectorized::ModelTag<ModelType::LOGNORMAL>{});
  run(vectorized::ModelTag<ModelType::BETA_BINOMIAL>{});
  return out;
}

void compute_gradient_analytical(const std::vector<double>&, const ModelData&,
    const ParamLayout&, std::vector<double>&, double*);

Rcpp::NumericMatrix precision_h_probe(double y, double mu, double phi, int family) {
  ModelData data;
  data.N = 1; data.p_num = 1; data.p_denom = 1;
  data.y_num = {0}; data.y_denom = {40};
  data.y_num_cont = {y}; data.y_denom_cont = {y};
  data.X_num_flat = {1}; data.X_denom_flat = {1};
  data.spatial_group = {1}; data.gp_data.obs_to_loc = {0}; data.gp_data.shared = true;
  data.model_type = family == 0 ? ModelType::POISSON_GAMMA :
      family == 1 ? ModelType::NEGBIN_GAMMA : family == 2 ? ModelType::GAMMA_GAMMA : ModelType::BETA_BINOMIAL;
  if (family == 3) { data.y_num={static_cast<int>(y)}; data.p_denom=0; data.link_num=LinkType::LOGIT; }
  data.phi_prior_shape=0; data.phi_prior_rate=0;
  data.unique_id = 17215 + family;
  ParamLayout layout{};
  layout.beta_num_start = 0; layout.beta_num_end = 1;
  layout.beta_denom_start = 1; layout.beta_denom_end = 2;
  layout.has_phi_num = true; layout.log_phi_num_idx = 2;
  layout.has_phi_denom = true; layout.log_phi_denom_idx = 3;
  layout.total_params = 4;
  const double eta = std::log(mu), lp = std::log(phi);
  const std::vector<double> params = {family == 2 ? eta : 0, eta, family < 2 ? std::log(2.) : lp, lp};
  Rcpp::NumericMatrix out(5, 2);
  rownames(out) = Rcpp::CharacterVector::create("vectorized", "fused", "collapsed_ICAR", "collapsed_GP", "scalar_H");
  colnames(out) = Rcpp::CharacterVector::create("value", "dphi");
  const double offset = family == 0 ? 1 : family == 1 ? 2*std::log1p(.5) : 0;
  for (int k = 0; k < 2; ++k) {
    vectorized::VecGradWorkspace ws;
    std::vector<double> g(4, 0), gt, gs; double ll = 0;
    auto run = [&](auto tag) {
      constexpr ModelType mt = decltype(tag)::value;
      if (k == 0) vectorized::compute_obs_gradients_vectorized<mt>(params,data,layout,g,ll,true,gt,gs,ws);
      else vectorized::compute_gradient_fused<mt>(params,data,layout,g,ll,true,gt,gs,ws);
    };
    if (family == 0) run(std::integral_constant<ModelType,ModelType::POISSON_GAMMA>{});
    if (family == 1) run(std::integral_constant<ModelType,ModelType::NEGBIN_GAMMA>{});
    if (family == 2) run(std::integral_constant<ModelType,ModelType::GAMMA_GAMMA>{});
    if (family == 3) run(std::integral_constant<ModelType,ModelType::BETA_BINOMIAL>{});
    out(k,0) = ll + offset; out(k,1) = g[family==3?2:3]/std::exp(lp);
  }
  if (family < 2) {
    double bn=0, bd=eta, w=0;
    out(2,0) = compute_unit_lik(0,0,&bn,&bd,2,phi,nullptr,data,false).ll + offset;
    if (family == 0) out(3,0) = compute_loc_lik(0,&w,&bn,&bd,2,phi,data,false).ll + offset;
    else out(3,0) = NA_REAL;
  } else { out(2,0) = out(3,0) = NA_REAL; }
  out(2,1) = out(3,1) = NA_REAL;
  layout.has_zi=true; data.p_zi=0; layout.beta_zi_start=layout.beta_zi_end=4;
  std::vector<double> g(4,0);
  compute_gradient_analytical(params,data,layout,g,nullptr);
  out(4,0)=NA_REAL; out(4,1)=g[family==3?2:3]/std::exp(lp);
  return out;
}

Rcpp::NumericMatrix collapsed_margin_probe(int family, int y, double yd, double en,
    double ed, double pn, double pd, bool shared) {
  ModelData data;
  data.N=1; data.p_num=1; data.p_denom=1;
  data.y_num={y}; data.y_denom={static_cast<int>(yd)}; data.y_denom_cont={yd};
  data.X_num_flat={1}; data.X_denom_flat={1}; data.spatial_group={1};
  data.gp_data.obs_to_loc={0}; data.gp_data.shared=shared;
  data.model_type=static_cast<ModelType>(family);
  const bool binomial=data.model_type==ModelType::BINOMIAL;
  const double w=0;
  const auto a=compute_unit_lik(0,0,&en,&ed,pn,pd,nullptr,data,binomial);
  const auto b=compute_loc_lik(0,&w,&en,&ed,pn,pd,data,binomial);
  Rcpp::NumericMatrix out(2,3);
  out(0,0)=a.ll; out(0,1)=a.grad; out(0,2)=a.neg_hess;
  out(1,0)=b.ll; out(1,1)=b.grad; out(1,2)=b.neg_hess;
  return out;
}

Rcpp::NumericVector collapsed_gp_reference_probe(Rcpp::NumericVector beta, int y, int n,
                                                double sigma) {
  ModelData data;
  data.N=1; data.p_num=1; data.p_denom=0; data.model_type=ModelType::BINOMIAL;
  data.y_num={y}; data.y_denom={n}; data.X_num_flat={1};
  data.gp_data.n_obs=1; data.gp_data.nn=1; data.gp_data.nn_order={0};
  data.gp_data.nn_idx={0}; data.gp_data.nn_dist={0}; data.gp_data.obs_to_loc={0};
  data.gp_data.shared=false; data.unique_id=224;
  Rcpp::NumericVector out(beta.size());
  for (int k=0;k<beta.size();++k) {
    CollapsedGPWorkspace ws;
    ws.bind(data.unique_id);
    build_nngp_B_D(sigma*sigma,1.,data.gp_data,ws);
    ws.cached_sigma2=sigma*sigma; ws.cached_phi=1.;
    ws.w_star[0]=std::log((y+.5)/(n-y+.5))-beta[k];
    ws.mode_found=true;
    const double b=beta[k],unused=0.;
    out[k]=collapsed_gp_find_mode(&b,&unused,sigma*sigma,1.,1.,1.,data,ws,100,1e-10)
      + ws.laplace_log_det;
  }
  return out;
}

Rcpp::NumericVector negbin_table_probe(int y, double mu, double phi) {
  ModelData data;
  data.N = 1;
  data.y_num = {y};
  data.y_denom = {0};
  vectorized::VecGradWorkspace ws;
  ws.init(1);
  ws.max_y_num_val = y;
  ws.max_y_denom_val = 0;
  ws.lgamma_y_num_p1[0] = std::lgamma(y + 1.0);
  ws.lgamma_y_denom_p1[0] = 0.0;
  ws.build_digamma_tables(phi, 1.0, true);
  double eta_n = std::log(mu), eta_d = 0.0;
  double rn = 0.0, rd = 0.0, gn = 0.0, gd = 0.0, ll = 0.0;
  vectorized::compute_residuals_impl(
      vectorized::ModelTag<ModelType::NEGBIN_NEGBIN>{}, 1, &eta_n, &eta_d,
      data, phi, 1.0, &rn, &rd, gn, gd, ll, true, ws);
  return Rcpp::NumericVector::create(ll + std::log(2.0), rn, phi * gn);
}

Rcpp::NumericMatrix negbin_h_kernels(int y, double mu, double phi) {
  ModelData data;
  data.N = 1; data.p_num = 1; data.p_denom = 1;
  data.y_num = {y}; data.y_denom = {0};
  data.y_num_cont = {static_cast<double>(y)}; data.y_denom_cont = {1.0};
  data.X_num_flat = {1.0}; data.X_denom_flat = {1.0};
  data.unique_id = 1;
  ParamLayout layout;
  layout.beta_num_start = 0; layout.beta_denom_start = 1;
  layout.has_phi_num = true; layout.log_phi_num_idx = 2;
  layout.has_phi_denom = true; layout.log_phi_denom_idx = 3;
  const std::vector<double> params = {std::log(mu), 0.0, std::log(phi), 0.0};
  Rcpp::NumericMatrix out(4, 3);
  rownames(out) = Rcpp::CharacterVector::create("vectorized_nb_nb", "fused_nb_nb",
                                              "vectorized_nb_gamma", "fused_nb_gamma");
  for (int k = 0; k < 4; ++k) {
    vectorized::VecGradWorkspace ws;
    std::vector<double> grad(4, 0.0), gt, gs;
    double ll = 0.0;
    if (k == 0) vectorized::compute_obs_gradients_vectorized<ModelType::NEGBIN_NEGBIN>(
        params, data, layout, grad, ll, true, gt, gs, ws);
    if (k == 1) vectorized::compute_gradient_fused<ModelType::NEGBIN_NEGBIN>(
        params, data, layout, grad, ll, true, gt, gs, ws);
    if (k == 2) vectorized::compute_obs_gradients_vectorized<ModelType::NEGBIN_GAMMA>(
        params, data, layout, grad, ll, true, gt, gs, ws);
    if (k == 3) vectorized::compute_gradient_fused<ModelType::NEGBIN_GAMMA>(
        params, data, layout, grad, ll, true, gt, gs, ws);
    out(k, 0) = ll + (k < 2 ? std::log(2.0) : 1.0);
    out(k, 1) = grad[0]; out(k, 2) = grad[2];
  }
  return out;
}

}  // namespace ratiod_hmc
