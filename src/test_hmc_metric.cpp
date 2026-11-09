// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
#include "hmc_sampler.h"
#include "hmc_integrator.h"

template<class Run>
auto with_test_metric(std::string type, std::string block, int n, Run run) {
    using namespace ratiod_hmc;
    DenseMassMatrix mass;
    const auto kind = type == "dense" ? MassMatrixType::DENSE
                    : type == "block" ? MassMatrixType::BLOCK_DIAG : MassMatrixType::DIAG;
    mass.init(n,kind); mass.adapted = true;
    Eigen::MatrixXd C = Eigen::MatrixXd::Zero(n,n);
    for (int i = 0; i < n; ++i) {
        C(i,i) = type == "identity" ? 1.0 : 1.2+0.1*i;
        mass.inv_mass_diag[i] = C(i,i);
        mass.sqrt_mass_diag[i] = 1/std::sqrt(C(i,i));
    }
    if (type == "dense") {
        C.array() += 0.08;
        Eigen::MatrixXd L = C.llt().matrixL();
        mass.inv_mass_dense.assign(C.data(),C.data()+n*n);
        mass.L_inv_mass.assign(L.data(),L.data()+n*n);
    } else if (type == "block") {
        mass.in_block.assign(n,false);
        for (int start : {0,2}) {
            MassBlock b; b.init(start,2); b.adapted = true;
            C(start,start+1) = C(start+1,start) = 0.12;
            Eigen::Matrix2d B = C.block<2,2>(start,start);
            Eigen::Matrix2d L = B.llt().matrixL();
            for(int j=0;j<2;++j) for(int i=0;i<2;++i) {
                b.inv_mass[i*4+j] = B(i,j); b.L_inv_mass[i*4+j] = L(i,j);
            }
            mass.blocks.push_back(b);
            mass.in_block[start] = mass.in_block[start+1] = true;
        }
    }
    if (block != "none") {
        Eigen::Matrix2d Q; Q << 3.0,0.4,0.4,2.0;
        Eigen::Matrix2d inverse = Q.inverse();
        Eigen::Matrix2d L = Q.llt().matrixL();
        if (block == "precision") mass.precision_block.init(1,2,inverse.data(),L.data());
        else if (block == "kronecker") {
            Eigen::Matrix2d Qt; Qt << 2.0,0.2,0.2,1.5;
            Eigen::Matrix2d Ti = Qt.inverse(), Lt = Qt.llt().matrixL();
            mass.kronecker_block.init(1,2,2,inverse.data(),L.data(),Ti.data(),Lt.data());
            for(int i=1;i<5;++i) { C.row(i).setZero(); C.col(i).setZero(); }
            for(int t=0;t<2;++t) for(int u=0;u<2;++u)
                for(int a=0;a<2;++a) for(int b=0;b<2;++b)
                    C(1+a+2*t,1+b+2*u) = inverse(a,b)*Ti(t,u);
            return run(mass,C);
        } else {
            mass.sparse_gmrf.init(1,1,2);
            mass.sparse_gmrf.Q_sparse = Q.sparseView();
            mass.sparse_gmrf.llt.compute(mass.sparse_gmrf.Q_sparse);
            mass.sparse_gmrf.factorized = true;
        }
        C.row(1).setZero(); C.row(2).setZero();
        C.col(1).setZero(); C.col(2).setZero();
        C.block<2,2>(1,1) = inverse;
    }
    return run(mass,C);
}

void metric_gaussian_gradient(const std::vector<double>& q,
    const ratiod_hmc::ModelData&, const ratiod_hmc::ParamLayout&,
    std::vector<double>& gradient, double* lp) {
    gradient.resize(q.size());
    double sum = 0.0;
    for (size_t i=0;i<q.size();++i) { gradient[i] = -q[i]; sum += q[i]*q[i]; }
    if(lp) *lp = -0.5*sum;
}

// [[Rcpp::export]]
Rcpp::List cpp_test_metric_step(std::string type, std::string block, int n = 5,
                                double epsilon = 0.15, bool extreme = false) {
  using namespace ratiod_hmc;
  return with_test_metric(type,block,n,[&](const auto& mass,const auto& C) {
    NUTSWorkspace ws; ws.init(n,1); ws.gradient_fn = metric_gaussian_gradient;
    Rcpp::NumericVector q0(n), p0(n);
    for(int i=0;i<n;++i) {
      q0[i] = extreme ? 1e11 : 0.1+0.15*i;
      p0[i] = 0.3+0.2*i;
      ws.q_at(0)[i] = q0[i]; ws.p_at(0)[i] = p0[i]; ws.grad_at(0)[i] = -q0[i];
    }
    ModelData data; ParamLayout layout;
    const auto step = leapfrog_step_inplace(ws,0,epsilon,mass,data,layout);
    Rcpp::NumericVector q(ws.q_at(0),ws.q_at(0)+n), p(ws.p_at(0),ws.p_at(0)+n);
    return Rcpp::List::create(Rcpp::Named("q0")=q0,Rcpp::Named("p0")=p0,
        Rcpp::Named("q")=q,Rcpp::Named("p")=p,Rcpp::Named("inverse_mass")=C,
        Rcpp::Named("log_prob")=step.log_prob,Rcpp::Named("divergent")=step.divergent);
  });
}

// [[Rcpp::export]]
Rcpp::List cpp_test_metric_epsilon(std::string type, int n = 5) {
  using namespace ratiod_hmc;
  return with_test_metric(type,"none",n,[&](const auto& mass,const auto& C) {
   const auto run = [&](const auto& metric) {
    std::vector<double> q(n,0.3), p(n);
    std::mt19937 rng(170183), clone = rng;
    metric.sample_momentum(p.data(),clone);
    int calls = 0;
    const auto evaluate = [&](const double* x,double* g,double& lp) {
      ++calls; double sum = 0;
      for(int i=0;i<n;++i) { g[i] = -x[i]; sum += x[i]*x[i]; }
      lp = -0.5*sum;
    };
    const double epsilon = metric_find_epsilon(q,metric,rng,evaluate);
    return Rcpp::List::create(Rcpp::Named("epsilon")=epsilon,Rcpp::Named("calls")=calls,
      Rcpp::Named("q")=q,Rcpp::Named("p")=p,Rcpp::Named("inverse_mass")=C);
   };
   if(type == "identity") return run(IdentityMetric{n});
   if(type == "diag") return run(DiagonalMetric{mass.inv_mass_diag});
   return run(mass);
  });
}

// [[Rcpp::export]]
Rcpp::List cpp_test_metric_geometry(std::string type, std::string block,
                                    int n = 5, int n_draws = 20000) {
  return with_test_metric(type,block,n,[&](const auto& mass,const auto& C) {
    Eigen::VectorXd p = Eigen::VectorXd::LinSpaced(n,0.3,1.2), velocity(n);
    mass.inv_mass_times_p(p.data(),velocity.data());
    Eigen::VectorXd qminus = Eigen::VectorXd::Zero(n), qplus = qminus, scratch(n);
    qplus[0] = 1.0; qplus[1] = -1.0;
    const bool uturn = ratiod_hmc::nuts_check_uturn_fast(qminus.data(),qplus.data(),
        p.data(),p.data(),mass,scratch.data(),n);
    Eigen::MatrixXd draws(n,n_draws);
    std::mt19937 rng(170183);
    for (int j=0;j<n_draws;++j) mass.sample_momentum(draws.col(j).data(),rng);
    Eigen::VectorXd mean = draws.rowwise().mean();
    Eigen::MatrixXd centered = draws.colwise()-mean;
    Eigen::MatrixXd covariance = centered*centered.transpose()/(n_draws-1.0);
    return Rcpp::List::create(Rcpp::Named("inverse_mass")=C,
        Rcpp::Named("momentum")=p,Rcpp::Named("velocity")=velocity,
        Rcpp::Named("kinetic")=mass.kinetic_energy(p.data()),
        Rcpp::Named("covariance")=covariance,Rcpp::Named("mean")=mean,
        Rcpp::Named("uturn")=uturn);
  });
}
