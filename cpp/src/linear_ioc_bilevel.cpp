#include "ddioc/linear_ioc_bilevel.hpp"

namespace ddioc::lioc {

double trajectory_tracking_error(
    const LQRPlannerResult& generated,
    const SyntheticDemoTrajectory& expert,
    double control_weight) {
    double x_error = 0.0;
    for (std::size_t step = 0; step < generated.states.size() && step < expert.states.size(); ++step) {
        for (std::size_t index = 0; index < kStateDim; ++index) {
            const double diff = generated.states[step][index] - expert.states[step][index];
            x_error += diff * diff;
        }
    }
    double u_error = 0.0;
    for (std::size_t step = 0; step < generated.controls.size() && step < expert.controls.size(); ++step) {
        const double diff = generated.controls[step] - expert.controls[step];
        u_error += diff * diff;
    }
    return x_error + control_weight * u_error;
}

BilevelIOCResult solve_synthetic_bilevel_ioc(
    const std::vector<SyntheticDemoTrajectory>& dataset,
    const DemoGenerationConfig& config,
    const OurMethodConfig& method_config,
    const CostParameters& theta_init) {
    const ClassicalIOCResult benchmark = run_classical_ioc_benchmark(dataset, config, method_config, theta_init);
    BilevelIOCResult result{};
    result.theta = benchmark.tuned_planner_weights;
    result.final_cost = benchmark.mean_tracking_sse;
    result.n_evaluations = method_config.planner_restarts * method_config.planner_iterations;
    result.planned_trajectories = benchmark.planned_trajectories;
    return result;
}

}  // namespace ddioc::lioc