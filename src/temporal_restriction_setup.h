#ifndef TULPARATIO_TEMPORAL_RESTRICTION_SETUP_H
#define TULPARATIO_TEMPORAL_RESTRICTION_SETUP_H
#include <tulpa/field_setup_api.h>
#include "hmc_sampler.h"

namespace ratiod_hmc {
inline void prepare_temporal_restriction(ModelData& data) {
    data.temporal_restricted_slopes = tulpa::RestrictedSlopes{};
    data.ms_trend_restricted_slopes = tulpa::RestrictedSlopes{};
    if (!data.rtr.active) return;
    tulpa::ModelData native;
    native.N = data.N;
    native.set_temporal_restriction(data.rtr.basis, data.rtr.rank);
    native.temporal_type = tulpa::TemporalType::RW2;
    auto prepare = [&](int n, int groups, const std::vector<int>& time,
                       const std::vector<int>& group) {
        native.n_times = n;
        native.n_temporal_groups = groups;
        native.temporal_time_idx = time;
        native.temporal_group_idx = group;
        tulpa::prepare_field(native, tulpa::FieldSetup::TEMPORAL_RESTRICTION);
        return native.temporal_restricted_slopes;
    };
    if (data.temporal_type == TemporalType::RW2 && !data.temporal_cyclic)
        data.temporal_restricted_slopes = prepare(data.n_times, data.n_temporal_groups,
            data.temporal_time_idx, data.temporal_group_idx);
    if (data.has_multiscale_temporal && data.multiscale_temporal_data.trend_type == TemporalType::RW2) {
        const auto& ms = data.multiscale_temporal_data;
        data.ms_trend_restricted_slopes = prepare(ms.n_times, 1, ms.time_index, {});
    }
}
}
#endif
