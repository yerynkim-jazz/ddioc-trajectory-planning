#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ddioc {

constexpr std::size_t kStateDim = 4;
constexpr std::size_t kPlannerWeightDim = 5;
constexpr std::size_t kFeatureDim = 10;

using State = std::array<double, kStateDim>;
using PlannerWeightArray = std::array<double, kPlannerWeightDim>;
using FeatureVector = std::array<double, kFeatureDim>;

struct DemoGenerationConfig {
    double dt_s = 0.1;
    int horizon = 40;
    double constant_speed_mps = 8.0;
    double control_limit = 0.25;
    int n_demos = 8;
    std::uint32_t seed = 7;
    int restarts = 4;
    double target_lat_offset_m = 3.5;
    std::string output_json = "outputs/examples/synthetic_ioc_demo_cpp/synthetic_demos.json";
};

struct SyntheticDemoTrajectory {
    State x0{};
    std::vector<State> states;
    std::vector<double> controls;
    std::vector<double> velocity_mps;
    double true_hlo_cost = 0.0;
    FeatureVector feature_sums{};
};

struct LQRPlannerWeights {
    double w_d = 1.0;
    double w_a1 = 0.2;
    double w_a2 = 0.1;
    double w_a3 = 0.1;
    double w_a4 = 0.05;
};

struct LQRPlannerResult {
    std::vector<State> states;
    std::vector<double> controls;
    std::vector<double> velocity_mps;
    std::vector<std::array<double, kStateDim>> gains;
};

struct OurMethodConfig {
    int pref_samples_per_demo = 12;
    double preference_noise_std = 0.03;
    double margin = 0.02;
    double omega_reg = 2e-2;
    double omega_entropy_reg = 5e-3;
    double omega_min = 0.02;
    double omega_max = 0.55;
    int hlo_restarts = 4;
    int planner_restarts = 6;
    int planner_iterations = 120;
};

struct LearnedHLOResult {
    FeatureVector omega{};
    FeatureVector feature_scales{};
};

struct OurMethodResult {
    LearnedHLOResult learned_hlo{};
    LQRPlannerWeights tuned_planner_weights{};
    std::vector<LQRPlannerResult> planned_trajectories;
    double mean_learned_hlo_cost = 0.0;
};

struct ClassicalIOCResult {
    LQRPlannerWeights tuned_planner_weights{};
    std::vector<LQRPlannerResult> planned_trajectories;
    double mean_tracking_sse = 0.0;
};

struct EvaluationSummary {
    double our_method_mean_gt_hlo_cost = 0.0;
    double classical_ioc_mean_gt_hlo_cost = 0.0;
    double expert_mean_gt_hlo_cost = 0.0;
    double our_method_std_gt_hlo_cost = 0.0;
    double classical_ioc_std_gt_hlo_cost = 0.0;
    double expert_std_gt_hlo_cost = 0.0;
    int n_test = 0;
};

struct GroundTruthHLO {
    std::array<std::string, kFeatureDim> feature_names{};
    FeatureVector omega{};
};

GroundTruthHLO get_ground_truth_hlo();
LQRPlannerWeights get_default_lqr_planner_weights();

std::vector<SyntheticDemoTrajectory> generate_synthetic_demonstrations(
    const GroundTruthHLO& hlo,
    const DemoGenerationConfig& config);

OurMethodResult run_our_method(
    const std::vector<SyntheticDemoTrajectory>& demos,
    const GroundTruthHLO& basis_hlo,
    const DemoGenerationConfig& demo_config,
    const OurMethodConfig& method_config,
    const LQRPlannerWeights& initial_planner_weights);

ClassicalIOCResult run_classical_ioc_benchmark(
    const std::vector<SyntheticDemoTrajectory>& demos,
    const DemoGenerationConfig& demo_config,
    const OurMethodConfig& method_config,
    const LQRPlannerWeights& initial_planner_weights);

EvaluationSummary evaluate_on_unseen_initial_states(
    const GroundTruthHLO& hlo,
    const DemoGenerationConfig& demo_config,
    const OurMethodResult& our_method,
    const ClassicalIOCResult& classical_ioc,
    int n_test,
    int seed_offset);

LQRPlannerResult plan_with_lqr(
    const State& x0,
    const LQRPlannerWeights& weights,
    int horizon,
    double dt_s,
    double constant_speed_mps,
    double target_lat_offset_m,
    double control_limit);

bool save_synthetic_demonstrations(
    const std::vector<SyntheticDemoTrajectory>& demos,
    const GroundTruthHLO& hlo,
    const DemoGenerationConfig& config,
    const std::string& path,
    std::string* error_message);

}  // namespace ddioc