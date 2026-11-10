#ifndef TULPARATIO_ST_RANGE_H
#define TULPARATIO_ST_RANGE_H
#include <tulpa/field_range.h>
#include "hmc_sampler.h"

namespace ratiod_hmc {
template<class T>
T st_gp_range_value(const T& coordinate, const ModelData& data, bool space) {
    return tulpa::field_range_from_coordinate(coordinate,
        space ? data.st_phi_space_prior_lower : data.st_phi_time_prior_lower,
        space ? data.st_phi_space_prior_upper : data.st_phi_time_prior_upper,
        data.st_parameterization != 2);
}
inline double st_gp_range_slope(double coordinate, const ModelData& data, bool space) {
    return tulpa::field_range_log_slope(coordinate,
        space ? data.st_phi_space_prior_lower : data.st_phi_time_prior_lower,
        space ? data.st_phi_space_prior_upper : data.st_phi_time_prior_upper,
        data.st_parameterization != 2);
}
}
#endif
