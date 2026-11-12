#ifndef TULPARATIO_ENGINE_OPTIONS_H
#define TULPARATIO_ENGINE_OPTIONS_H
#include <tulpa/types.h>
#include "enum_names.h"
namespace tulpaRatio {
inline constexpr ratiod_enum::EnumName<tulpa::TemporalType> TEMPORAL_NAMES[]={
 {"none",tulpa::TemporalType::NONE},{"rw1",tulpa::TemporalType::RW1},
 {"rw2",tulpa::TemporalType::RW2},{"ar1",tulpa::TemporalType::AR1},
 {"iid",tulpa::TemporalType::IID},{"gp",tulpa::TemporalType::GP},
 {"multiscale",tulpa::TemporalType::MULTISCALE}};
inline constexpr ratiod_enum::EnumName<tulpa::TemporalType> TVC_NAMES[]={
 {"rw1",tulpa::TemporalType::RW1},{"rw2",tulpa::TemporalType::RW2},
 {"ar1",tulpa::TemporalType::AR1},{"iid",tulpa::TemporalType::IID}};
inline constexpr ratiod_enum::EnumName<tulpa::MassMatrixType> METRIC_NAMES[]={
 {"auto",tulpa::MassMatrixType::AUTO},{"diag",tulpa::MassMatrixType::DIAG},
 {"dense",tulpa::MassMatrixType::DENSE},{"block_diag",tulpa::MassMatrixType::BLOCK_DIAG}};
inline constexpr ratiod_enum::EnumName<tulpa::GradientMode> GRADIENT_NAMES[]={
 {"auto",tulpa::GradientMode::AUTO},{"N",tulpa::GradientMode::NUMERICAL},
 {"A",tulpa::GradientMode::AUTODIFF_FWD},{"A_r",tulpa::GradientMode::AUTODIFF_ARENA},
 {"H",tulpa::GradientMode::HANDCODED},{"numerical",tulpa::GradientMode::NUMERICAL},
 {"forward",tulpa::GradientMode::AUTODIFF_FWD},{"arena",tulpa::GradientMode::AUTODIFF_ARENA},
 {"handcoded",tulpa::GradientMode::HANDCODED}};
inline constexpr ratiod_enum::EnumName<tulpa::MSGPSampler> MSGP_NAMES[]={
 {"auto",tulpa::MSGPSampler::AUTO},{"lbfgs",tulpa::MSGPSampler::LBFGS}};
}
#endif
