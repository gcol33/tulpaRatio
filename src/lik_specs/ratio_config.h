#ifndef TULPARATIO_RATIO_CONFIG_H
#define TULPARATIO_RATIO_CONFIG_H
#include <string>

namespace tulpaRatio {
struct RatioConfig {
    std::string family;
    std::string zi = "none";
    std::string num_link = "logit";
    std::string denom_link = "log";
};
}
#endif
