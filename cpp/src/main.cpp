#include "ddioc/synthetic_ioc.hpp"

#include <iomanip>
#include <iostream>

namespace {

void print_planner_weights(const ddioc::LQRPlannerWeights& weights) {
    std::cout << "- w_d: " << weights.w_d << '\n';
    std::cout << "- w_a1: " << weights.w_a1 << '\n';
    std::cout << "- w_a2: " << weights.w_a2 << '\n';
    std::cout << "- w_a3: " << weights.w_a3 << '\n';
    std::cout << "- w_a4: " << weights.w_a4 << '\n';
}

}  // namespace

int main() {
    std::cout << std::fixed << std::setprecision(6);

    const ddioc::GroundTruthHLO hlo = ddioc::get_ground_truth_hlo();
    const ddioc::LQRPlannerWeights planner_weights = ddioc::get_default_lqr_planner_weights();
    const ddioc::DemoGenerationConfig config{};
    const ddioc::OurMethodConfig method_config{};

    std::cout << "GT HLO\n";
    for (std::size_t index = 0; index < hlo.feature_names.size(); ++index) {
        std::cout << "- " << hlo.feature_names[index] << ": " << hlo.omega[index] << '\n';
    }

    std::cout << "\nLQR planner weights\n";
    print_planner_weights(planner_weights);

    const auto demos = ddioc::generate_synthetic_demonstrations(hlo, config);
    const auto our_method = ddioc::run_our_method(demos, hlo, config, method_config, planner_weights);
    const auto classical_ioc = ddioc::run_classical_ioc_benchmark(demos, config, method_config, planner_weights);
    const auto evaluation = ddioc::evaluate_on_unseen_initial_states(hlo, config, our_method, classical_ioc, 24, 1000);

    std::cout << "\nSynthetic demonstrations\n";
    std::cout << "- count: " << demos.size() << '\n';
    if (!demos.empty()) {
        const auto& first_demo = demos.front();
        std::cout << "- first demo x0: [" << first_demo.x0[0] << ", " << first_demo.x0[1] << ", " << first_demo.x0[2] << ", " << first_demo.x0[3] << "]\n";
        std::cout << "- first demo cost: " << first_demo.true_hlo_cost << '\n';
        const auto& final_state = first_demo.states.back();
        std::cout << "- first final state: [" << final_state[0] << ", " << final_state[1] << ", " << final_state[2] << ", " << final_state[3] << "]\n";
    }

    std::cout << "\nOur method\n";
    std::cout << "- learned omega: [";
    for (std::size_t index = 0; index < our_method.learned_hlo.omega.size(); ++index) {
        if (index > 0) {
            std::cout << ", ";
        }
        std::cout << our_method.learned_hlo.omega[index];
    }
    std::cout << "]\n";
    std::cout << "- tuned planner weights\n";
    print_planner_weights(our_method.tuned_planner_weights);
    std::cout << "- mean learned HLO cost: " << our_method.mean_learned_hlo_cost << '\n';

    std::cout << "\nClassical IOC benchmark\n";
    print_planner_weights(classical_ioc.tuned_planner_weights);
    std::cout << "- mean tracking SSE: " << classical_ioc.mean_tracking_sse << '\n';

    std::cout << "\nEvaluation on unseen initial states\n";
    std::cout << "- n_test: " << evaluation.n_test << '\n';
    std::cout << "- our method mean GT HLO cost: " << evaluation.our_method_mean_gt_hlo_cost << '\n';
    std::cout << "- our method std GT HLO cost: " << evaluation.our_method_std_gt_hlo_cost << '\n';
    std::cout << "- classical IOC mean GT HLO cost: " << evaluation.classical_ioc_mean_gt_hlo_cost << '\n';
    std::cout << "- classical IOC std GT HLO cost: " << evaluation.classical_ioc_std_gt_hlo_cost << '\n';
    std::cout << "- expert mean GT HLO cost: " << evaluation.expert_mean_gt_hlo_cost << '\n';
    std::cout << "- expert std GT HLO cost: " << evaluation.expert_std_gt_hlo_cost << '\n';

    std::string error_message;
    if (!ddioc::save_synthetic_demonstrations(demos, hlo, config, config.output_json, &error_message)) {
        std::cerr << "\nFailed to save JSON output: " << error_message << '\n';
        return 1;
    }
    std::cout << "- saved demos: " << config.output_json << '\n';

    return 0;
}