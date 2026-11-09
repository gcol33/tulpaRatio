#ifndef RATIOD_HMC_FIELD_COORDINATES_H
#define RATIOD_HMC_FIELD_COORDINATES_H

#include <tulpa/field_coordinates.h>
#include <cmath>
#include <vector>
#include <algorithm>

namespace ratiod_hmc {

using tulpa::FieldCoordinate;
using FieldCoordinates = tulpa::FieldCoordinates<GradientFn>;

FieldCoordinates make_field_coordinates(const std::vector<double>& initial,
                                         const ModelData& data,
                                         const ParamLayout& layout);

inline thread_local const FieldCoordinates* field_coordinates = nullptr;

class FieldCoordinateScope {
    const FieldCoordinates* previous_;
public:
    explicit FieldCoordinateScope(const FieldCoordinates* current)
        : previous_(field_coordinates) { field_coordinates = current; }
    ~FieldCoordinateScope() { field_coordinates = previous_; }
};

} // namespace ratiod_hmc
#endif
