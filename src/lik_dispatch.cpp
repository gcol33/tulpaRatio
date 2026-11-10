#include <Rcpp.h>
#include <algorithm>
#include "lik_specs/family_registry.h"
#include "lik_specs/family_kernel.h"
#include "lik_specs/family_weights.h"
#include "lik_specs/lik_grad_h_kernel.h"

namespace tulpaRatio {

template<class Margin>
constexpr LinkDomain margin_link_domain() {
    if constexpr (std::is_same<Margin,margins::Binomial>::value ||
                  std::is_same<Margin,margins::BetaBinomial>::value) return LinkDomain::Probability;
    if constexpr (std::is_same<Margin,margins::Lognormal>::value) return LinkDomain::LogLocation;
    return LinkDomain::Positive;
}

template<class Num, class Den = margins::NoMargin>
FamilyEntry make_family_entry(const char* name, const char* spec_name,
    const char* link, std::vector<std::string> zi, tulpa::FullGradFn gradient,
    ResponseShape fixed = ResponseShape::Trials, bool legacy_likelihood = true) {
    constexpr bool single = std::is_same<Den,margins::NoMargin>::value;
    return {name,spec_name,single ? 1 : 2,Num::n_extra+Den::n_extra,link,"log",std::move(zi),
        lik::continuous_response<Num> ? ResponseShape::Positive : ResponseShape::Count,
        single ? fixed : lik::continuous_response<Den> ? ResponseShape::Positive : ResponseShape::Count,
        &lik::family_likelihood<Num,Den,double>,
        &lik::family_likelihood<Num,Den,tulpa::arena::Var>,gradient,legacy_likelihood,
        margin_link_domain<Num>(),margin_link_domain<Den>(),

        &lik::family_weights<Num,Den,double>, &lik::family_weights<Num,Den,tulpa::arena::Var>};
}

const std::vector<FamilyEntry>& family_entries() {
    using namespace margins;
    static const std::vector<FamilyEntry> entries = {
        make_family_entry<Binomial>("binomial","tulpaRatio_binomial","logit",
            {"none","zi_binomial","hurdle_binomial","oi_binomial","zoib"},&grad_h_binomial),
        make_family_entry<Poisson,Gamma>("poisson_gamma","tulpaRatio_poisson_gamma","log",
            {"none","zi_poisson","hurdle_poisson"},&grad_h_poisson_gamma),
        make_family_entry<NegBin,Gamma>("negbin_gamma","tulpaRatio_negbin_gamma","log",
            {"none","zi_negbin","hurdle_negbin"},&grad_h_negbin_gamma),
        make_family_entry<NegBin,NegBin>("negbin_negbin","tulpaRatio_negbin_negbin","log",
            {"none","zi_negbin","hurdle_negbin"},&grad_h_negbin_negbin),
        make_family_entry<Gamma,Gamma>("gamma_gamma","tulpaRatio_gamma_gamma","log",
            {"none"},&grad_h_gamma_gamma),
        make_family_entry<Lognormal,Lognormal>("lognormal","tulpaRatio_lognormal","log",
            {"none"},&grad_h_lognormal),
        make_family_entry<BetaBinomial>("beta_binomial","tulpaRatio_beta_binomial","logit",
            {"none"},&grad_h_beta_binomial),
        make_family_entry<Poisson>("poisson_fixed","tulpaRatio_poisson_fixed","log",
            {"none","zi_poisson","hurdle_poisson"},nullptr,ResponseShape::Exposure,false),
        make_family_entry<NegBin>("negbin_fixed","tulpaRatio_negbin_fixed","log",
            {"none","zi_negbin","hurdle_negbin"},nullptr,ResponseShape::Exposure,false),
        make_family_entry<Lognormal>("lognormal_fixed","tulpaRatio_lognormal_fixed","log",
            {"none"},nullptr,ResponseShape::Exposure,false),
        make_family_entry<Poisson,NegBin>("poisson_negbin","tulpaRatio_poisson_negbin","log",
            {"none","zi_poisson","hurdle_poisson"},nullptr,ResponseShape::Trials,false)
    };
    return entries;
}

const FamilyEntry& family_entry(const std::string& name) {
    for (const auto& entry : family_entries()) if (name == entry.name) return entry;
    Rcpp::stop("Unknown likelihood family '%s'",name.c_str());
}

tulpa::LikelihoodSpec build_ratio_likelihood_spec(const RatioConfig& cfg) {
    const auto& entry = family_entry(cfg.family);
    if (std::find(entry.zi.begin(),entry.zi.end(),cfg.zi) == entry.zi.end())
        Rcpp::stop("Family '%s' does not support inflation '%s'",cfg.family.c_str(),cfg.zi.c_str());
    const auto valid = [](const std::string& name, LinkDomain domain) {
        ratiod_link::LinkType link;
        if (!ratiod_link::parse_link(name, link)) return false;
        if (domain == LinkDomain::Probability) return ratiod_link::is_probability_link(link);
        if (domain == LinkDomain::LogLocation) return link == ratiod_link::LinkType::LOG;
        return link == ratiod_link::LinkType::LOG || link == ratiod_link::LinkType::IDENTITY ||
               link == ratiod_link::LinkType::INVERSE;
    };
    if (!valid(cfg.num_link, entry.num_domain) ||
        (entry.processes > 1 && !valid(cfg.denom_link, entry.denom_domain)) ||
        !valid(cfg.zi_link, LinkDomain::Probability) || !valid(cfg.oi_link, LinkDomain::Probability))
        Rcpp::stop("Unsupported likelihood links for family '%s'",cfg.family.c_str());
    tulpa::LikelihoodSpec spec;
    spec.name=entry.spec_name;
    spec.n_processes=entry.processes;
    spec.n_extra_params=entry.n_extra;
    spec.ll_double=entry.likelihood;
    spec.ll_arena=entry.likelihood_arena;
    if (cfg.zi == "none" && cfg.num_link == entry.num_link &&
        cfg.denom_link == entry.denom_link) spec.gradient_fn=entry.gradient;
    if (cfg.zi == "none") {
        spec.eta_weights_fn=entry.weights;
        spec.eta_weights_arena_fn=entry.weights_arena;
    }
    return spec;
}

}

// [[Rcpp::export]]
Rcpp::List cpp_ratio_family_entries() {
    const auto& entries = tulpaRatio::family_entries();
    Rcpp::List out(entries.size());
    Rcpp::CharacterVector names(entries.size());
    const auto domain = [](tulpaRatio::ResponseShape shape) {
        return shape == tulpaRatio::ResponseShape::Positive ? "positive" :
               shape == tulpaRatio::ResponseShape::Exposure ? "exposure" :
               shape == tulpaRatio::ResponseShape::Trials ? "trials" : "count";
    };
    for (size_t k=0;k<entries.size();++k) {
        const auto& e=entries[k]; names[k]=e.name;
        out[k]=Rcpp::List::create(Rcpp::Named("n_processes")=e.processes,
            Rcpp::Named("n_extra")=e.n_extra,Rcpp::Named("num_link")=e.num_link,
            Rcpp::Named("denom_link")=e.denom_link,Rcpp::Named("spec_inflation")=e.zi,
            Rcpp::Named("handcoded")=e.legacy_likelihood,
            Rcpp::Named("responses")=Rcpp::CharacterVector::create(
                Rcpp::Named("num")=domain(e.num_response),Rcpp::Named("denom")=domain(e.denom_response)));
    }
    out.attr("names")=names;
    return out;
}

// [[Rcpp::export]]
Rcpp::NumericVector cpp_test_family_likelihood(std::string family, double shape,
                                              double rate, double scale) {
    using namespace tulpaRatio;
    const auto& entry=family_entry(family);
    RatioConfig cfg;
    cfg.family=family; cfg.num_link=entry.num_link;
    const auto spec=build_ratio_likelihood_spec(cfg);
    lik::ResponseData response{{3},{8},{3.0},{4.0}};
    ratiod_link::parse_link(cfg.num_link, response.num_link);
    tulpa::ModelData data;
    data.phi_prior_shape=shape; data.phi_prior_rate=rate; data.sigma_re_scale=scale;
    tulpa::ParamLayout layout;
    layout.extra_offset=0;
    const double eta[2]={-.35,.2};
    std::vector<double> params;
    for (int k=0;k<entry.n_extra;++k) params.push_back(std::log(k==0 ? 2.4 : 1.7));
    const double ll=spec.ll_double(0,eta,0.,0.,params,data,layout,&response);
    tulpa::arena::ArenaScope scope;
    tulpa::arena::Var en(-.35),ed(.2),zero(0.);
    const tulpa::arena::Var ae[2]={en,ed};
    std::vector<tulpa::arena::Var> ap;
    for (double p : params) ap.emplace_back(p);
    auto al=spec.ll_arena(0,ae,zero,zero,ap,data,layout,&response);
    al.backward();
    Rcpp::NumericVector out(4+entry.n_extra);
    out[0]=ll; out[1]=al.val(); out[2]=en.adj(); out[3]=ed.adj();
    for (int k=0;k<entry.n_extra;++k) out[4+k]=ap[k].adj();
    return out;
}

// [[Rcpp::export]]
Rcpp::List cpp_test_family_weights(std::string family, Rcpp::NumericVector point) {
    using namespace tulpaRatio;
    const auto& entry = family_entry(family);
    RatioConfig cfg;
    cfg.family = family; cfg.num_link = entry.num_link;
    const auto spec = build_ratio_likelihood_spec(cfg);
    const int np = entry.processes;
    if (point.size() != np + entry.n_extra) Rcpp::stop("point must contain process predictors and log extra parameters");
    lik::ResponseData response{{3}, {8}, {3.0}, {4.0}};
    tulpa::ModelData data;
    data.N = 1; data.n_processes = np;
    tulpa::ParamLayout layout;
    layout.extra_offset = 0;
    std::vector<double> q(point.begin() + np, point.end()), score(np), weights(np * np);
    spec.eta_weights_fn(0, point.begin(), 0., 0., q, data, layout, &response, score.data(), weights.data());
    const double ll = spec.ll_double(0, point.begin(), 0., 0., q, data, layout, &response);
    tulpa::arena::ArenaScope scope;
    const auto input = tulpa::arena::make_vars(scope.arena(), std::vector<double>(point.begin(), point.end()));
    std::vector<tulpa::arena::Var> parameters(input.begin() + np, input.end()), a_score(np), a_weights(np * np);
    spec.eta_weights_arena_fn(0, input.data(), tulpa::arena::Var(0), tulpa::arena::Var(0), parameters,
        data, layout, &response, a_score.data(), a_weights.data());
    std::vector<tulpa::arena::Var> outputs = a_score;
    outputs.insert(outputs.end(), a_weights.begin(), a_weights.end());
    Rcpp::NumericMatrix derivatives(outputs.size(), input.size());
    Rcpp::NumericVector values(outputs.size());
    for (size_t i = 0; i < outputs.size(); ++i) {
        values[i] = outputs[i].val();
        outputs[i].backward();
        const auto gradient = tulpa::arena::get_adjoints(input);
        for (size_t j = 0; j < input.size(); ++j) derivatives(i, j) = gradient[j];
    }
    return Rcpp::List::create(Rcpp::Named("ll") = ll, Rcpp::Named("score") = score,
        Rcpp::Named("weights") = weights, Rcpp::Named("values") = values,
        Rcpp::Named("derivatives") = derivatives);
}
