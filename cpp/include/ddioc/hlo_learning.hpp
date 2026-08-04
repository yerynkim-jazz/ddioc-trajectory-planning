#pragma once

#include "ddioc/dynamics_learning.hpp"

#include <string>
#include <vector>

namespace ddioc::ddioc {

struct HLOLearningResult {
    std::vector<double> omega;
    std::vector<double> feature_scales;
    std::vector<std::vector<double>> omega_history;
};

std::vector<double> compute_feature_scales(
    const std::vector<Matrix>& segments_X,
    const std::vector<Matrix>& segments_U,
    const std::vector<std::vector<double>>& segments_V,
    const std::vector<double>& segments_target_lat_m,
    const std::vector<std::vector<double>>& segments_ref_orientation,
    const std::vector<std::vector<double>>& segments_ref_curvature,
    double dt_s,
    const std::vector<double>& X_mean,
    const std::vector<double>& X_std,
    const std::vector<double>& U_mean,
    const std::vector<double>& U_std);

HLOLearningResult learn_hlo_omega(
    const std::vector<Matrix>& segments_X,
    const std::vector<Matrix>& segments_U,
    const std::vector<std::vector<double>>& segments_V,
    const std::vector<double>& segments_target_lat_m,
    const std::vector<std::vector<double>>& segments_ref_orientation,
    const std::vector<std::vector<double>>& segments_ref_curvature,
    const KoopmanModel& model,
    double dt_s,
    const std::vector<double>& X_mean,
    const std::vector<double>& X_std,
    const std::vector<double>& U_mean,
    const std::vector<double>& U_std,
    const std::vector<double>& feature_scales,
    int ioc_every,
    int window);

double eval_ioc_residual(
    const std::vector<Matrix>& segments_X,
    const std::vector<Matrix>& segments_U,
    const std::vector<std::vector<double>>& segments_V,
    const std::vector<double>& segments_target_lat_m,
    const std::vector<std::vector<double>>& segments_ref_orientation,
    const std::vector<std::vector<double>>& segments_ref_curvature,
    const std::vector<double>& omega,
    const std::vector<double>& feature_scales,
    double dt_s,
    const std::vector<double>& X_mean,
    const std::vector<double>& X_std,
    const std::vector<double>& U_mean,
    const std::vector<double>& U_std);

std::vector<double> load_hlo_omega_from_learned_objective_json(const std::string& path);
void save_learned_objective_json(const std::string& path, const std::vector<double>& omega, const std::vector<double>& feature_scales);

}  // namespace ddioc::ddioc