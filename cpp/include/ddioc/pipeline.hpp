#pragma once

#include "ddioc/dynamics_learning.hpp"
#include "ddioc/hlo_learning.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace ddioc::ddioc {

struct PipelineConfig {
    std::string merged_csv_path;
    std::string output_root;
    int n_traj = 200;
    int min_traj_len = 30;
    double train_ratio = 0.8;
    std::uint32_t split_seed = 0;
    std::string lift = "poly";
    int degree = 2;
    int tanh_features = 28;
    int seg_len = 31;
    int segment_stride = 10;
    double reg = 1e-6;
    int ioc_every = 20;
    int window = 200;
    bool verbose = false;
};

struct PipelineMetrics {
    MetricSummary train_one_step;
    MetricSummary test_one_step;
    MetricSummary train_rollout10;
    MetricSummary test_rollout10;
    double ioc_train_residual = 0.0;
    double ioc_test_residual = 0.0;
};

struct PipelineResult {
    std::string merged_csv_path;
    std::string output_root;
    std::string lift;
    int n_traj_loaded = 0;
    int train_segments = 0;
    int test_segments = 0;
    double dt_s = 0.0;
    std::vector<std::string> train_ids;
    std::vector<double> omega;
    std::vector<double> feature_scales;
    std::vector<std::vector<double>> omega_history;
    PipelineMetrics metrics;
    std::string learned_objective_json_path;
    std::string omega_history_json_path;
};

PipelineResult learn_dynamics_and_weights(const PipelineConfig& config);

}  // namespace ddioc::ddioc