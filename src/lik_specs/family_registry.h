#ifndef TULPARATIO_FAMILY_REGISTRY_H
#define TULPARATIO_FAMILY_REGISTRY_H

#include <string>
#include <vector>
#include <tulpa/likelihood.h>
#include "ratio_config.h"

namespace tulpaRatio {

enum class ResponseShape { Trials, Count, Positive, Exposure };
enum class LinkDomain { Probability, Positive, LogLocation };

struct FamilyEntry {
    const char* name;
    const char* spec_name;
    int processes;
    int n_extra;
    const char* num_link;
    const char* denom_link;
    std::vector<std::string> zi;
    ResponseShape num_response;
    ResponseShape denom_response;
    tulpa::LikelihoodFn<double> likelihood;
    tulpa::LikelihoodFn<tulpa::arena::Var> likelihood_arena;
    tulpa::FullGradFn gradient;
    LinkDomain num_domain, denom_domain;
    tulpa::EtaWeightsFn weights;
    tulpa::EtaWeightsArenaFn weights_arena;
};

const std::vector<FamilyEntry>& family_entries();
const FamilyEntry& family_entry(const std::string& name);
tulpa::LikelihoodSpec build_ratio_likelihood_spec(const RatioConfig& cfg);

}
#endif
