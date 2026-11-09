#ifndef TULPARATIO_HMC_GRAD_INTERNAL_H
#define TULPARATIO_HMC_GRAD_INTERNAL_H
#include "hmc_sampler.h"
namespace ratiod_hmc {
extern GradientMode g_gradient_mode;
void verify_gradient_or_fallback(const std::vector<double>&, const ModelData&, const ParamLayout&);
LeapfrogResult leapfrog_step_mass(const std::vector<double>&, const std::vector<double>&,
    double, const std::vector<double>&, const ModelData&, const ParamLayout&);
bool can_use_analytical_gradient(const ModelData&, const ParamLayout&);
void compute_gradient_analytical(const std::vector<double>&, const ModelData&, const ParamLayout&, std::vector<double>&, double*);
void compute_gradient_numerical(const std::vector<double>&, const ModelData&, const ParamLayout&, std::vector<double>&, double*);
void compute_gradient_arena(const std::vector<double>&, const ModelData&, const ParamLayout&, std::vector<double>&, double*);
void compute_gradient_forward(const std::vector<double>&, const ModelData&, const ParamLayout&, std::vector<double>&, double*);
void compute_gradient_gp_handcoded(const std::vector<double>&, const ModelData&, const ParamLayout&, std::vector<double>&, double*);
void compute_gradient_gp_collapsed(const std::vector<double>&, const ModelData&, const ParamLayout&, std::vector<double>&, double*);
void compute_gradient_icar_collapsed(const std::vector<double>&, const ModelData&, const ParamLayout&, std::vector<double>&, double*);
void compute_gradient_gp_plus_temporal_gmrf(const std::vector<double>&, const ModelData&, const ParamLayout&, std::vector<double>&, double*);
void compute_gradient_temporal_gp(const std::vector<double>&, const ModelData&, const ParamLayout&, std::vector<double>&, double*);
void compute_gradient_svc_handcoded(const std::vector<double>&, const ModelData&, const ParamLayout&, std::vector<double>&, double*);
void compute_gradient_svc_hsgp_handcoded(const std::vector<double>&, const ModelData&, const ParamLayout&, std::vector<double>&, double*);
void compute_gradient_tvc_handcoded(const std::vector<double>&, const ModelData&, const ParamLayout&, std::vector<double>&, double*);
void compute_gradient_latent_handcoded(const std::vector<double>&, const ModelData&, const ParamLayout&, std::vector<double>&, double*);
void compute_gradient_hsgp(const std::vector<double>&, const ModelData&, const ParamLayout&, std::vector<double>&, double*);
void compute_gradient_msgp_hsgp(const std::vector<double>&, const ModelData&, const ParamLayout&, std::vector<double>&, double*);
void compute_gradient_ms_temporal_handcoded(const std::vector<double>&, const ModelData&, const ParamLayout&, std::vector<double>&, double*);
void compute_gradient_spatiotemporal_handcoded(const std::vector<double>&, const ModelData&, const ParamLayout&, std::vector<double>&, double*);
void compute_gradient_composite(const std::vector<double>&, const ModelData&, const ParamLayout&, std::vector<double>&, double*);
void compute_gradient_field_coordinates(const std::vector<double>&, const ModelData&, const ParamLayout&, std::vector<double>&, double*);
}
#endif
