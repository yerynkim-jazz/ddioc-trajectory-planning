#include "ddioc/linear_ioc_bilevel.hpp"

#include <iomanip>
#include <iostream>

int main() {
    std::cout << std::fixed << std::setprecision(6);

    const ddioc::GroundTruthHLO hlo = ddioc::get_ground_truth_hlo();
    const ddioc::DemoGenerationConfig config{};
    const ddioc::OurMethodConfig method_config{};
    const ddioc::LQRPlannerWeights initial = ddioc::get_default_lqr_planner_weights();

    const auto demos = ddioc::generate_synthetic_demonstrations(hlo, config);
    const auto result = ddioc::lioc::solve_synthetic_bilevel_ioc(demos, config, method_config, initial);

    std::cout << "Synthetic bilevel IOC benchmark\n";
    std::cout << "- demonstrations: " << demos.size() << '\n';
    std::cout << "- tuned w_d: " << result.theta.w_d << '\n';
    std::cout << "- tuned w_a1: " << result.theta.w_a1 << '\n';
    std::cout << "- tuned w_a2: " << result.theta.w_a2 << '\n';
    std::cout << "- tuned w_a3: " << result.theta.w_a3 << '\n';
    std::cout << "- tuned w_a4: " << result.theta.w_a4 << '\n';
    std::cout << "- objective: " << result.final_cost << '\n';
    std::cout << "- planned trajectories: " << result.planned_trajectories.size() << '\n';

    return 0;
}