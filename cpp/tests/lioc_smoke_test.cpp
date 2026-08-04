#include "ddioc/linear_ioc_bilevel.hpp"

#include <iostream>

int main() {
    const ddioc::GroundTruthHLO hlo = ddioc::get_ground_truth_hlo();
    const ddioc::DemoGenerationConfig config{};
    const ddioc::OurMethodConfig method_config{};
    const ddioc::LQRPlannerWeights initial = ddioc::get_default_lqr_planner_weights();

    const auto demos = ddioc::generate_synthetic_demonstrations(hlo, config);
    const auto result = ddioc::lioc::solve_synthetic_bilevel_ioc(demos, config, method_config, initial);

    if (result.planned_trajectories.size() != demos.size()) {
        std::cerr << "planned trajectory count mismatch\n";
        return 1;
    }
    if (!(result.final_cost >= 0.0)) {
        std::cerr << "unexpected negative objective\n";
        return 1;
    }
    return 0;
}