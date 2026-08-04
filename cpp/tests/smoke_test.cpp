#include "ddioc/synthetic_ioc.hpp"

#include <cmath>
#include <iostream>
#include <string>

namespace {

bool check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << message << '\n';
        return false;
    }
    return true;
}

}  // namespace

int main() {
    const ddioc::GroundTruthHLO hlo = ddioc::get_ground_truth_hlo();
    const ddioc::DemoGenerationConfig config{};
    const ddioc::LQRPlannerWeights planner_weights = ddioc::get_default_lqr_planner_weights();

    const auto demos = ddioc::generate_synthetic_demonstrations(hlo, config);
    if (!check(static_cast<int>(demos.size()) == config.n_demos, "unexpected synthetic demo count")) {
        return 1;
    }

    const auto plan = ddioc::plan_with_lqr(
        demos.front().x0,
        planner_weights,
        config.horizon,
        config.dt_s,
        config.constant_speed_mps,
        config.target_lat_offset_m,
        config.control_limit);

    if (!check(static_cast<int>(plan.states.size()) == config.horizon + 1, "unexpected LQR state horizon")) {
        return 1;
    }
    if (!check(static_cast<int>(plan.controls.size()) == config.horizon, "unexpected LQR control horizon")) {
        return 1;
    }

    double control_norm = 0.0;
    for (double control : plan.controls) {
        control_norm += std::abs(control);
    }
    if (!check(control_norm > 0.0, "planner produced zero control sequence")) {
        return 1;
    }

    return 0;
}