#pragma once

#include "ddioc/synthetic_ioc.hpp"

#include <vector>

namespace ddioc::lioc {

using CostParameters = LQRPlannerWeights;

struct BilevelIOCResult {
    CostParameters theta{};
    double final_cost = 0.0;
    int n_evaluations = 0;
    std::vector<LQRPlannerResult> planned_trajectories;
};

double trajectory_tracking_error(
    const LQRPlannerResult& generated,
    const SyntheticDemoTrajectory& expert,
    double control_weight = 1.0);

BilevelIOCResult solve_synthetic_bilevel_ioc(
    const std::vector<SyntheticDemoTrajectory>& dataset,
    const DemoGenerationConfig& config,
    const OurMethodConfig& method_config,
    const CostParameters& theta_init);

}  // namespace ddioc::lioc