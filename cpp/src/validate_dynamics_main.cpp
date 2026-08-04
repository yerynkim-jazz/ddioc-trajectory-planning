#include "ddioc/dynamics_learning.hpp"

#include <iomanip>
#include <iostream>

int main() {
    using namespace ddioc::ddioc;

    std::cout << std::fixed << std::setprecision(6);

    const DynamicsValidationConfig config{};
    const TrajectoryDataset dataset = build_synthetic_validation_dataset(config);
    const SegmentDataset segments = build_validation_segments(dataset, config);

    const KoopmanModel poly = learn_koopman_dynamics_poly(segments.segments_X_train, segments.segments_U_train, 4, config.poly_degree, 1e-6);
    const KoopmanModel tanh = learn_koopman_dynamics_tanh(segments.segments_X_train, segments.segments_U_train, 4, config.tanh_features, config.seed, 1e-6);

    const MetricSummary poly_one = eval_koopman_one_step_mse(segments.segments_X_test, segments.segments_U_test, poly);
    const MetricSummary tanh_one = eval_koopman_one_step_mse(segments.segments_X_test, segments.segments_U_test, tanh);
    const MetricSummary poly_roll = eval_koopman_rollout_rmse(segments.segments_X_test, segments.segments_U_test, poly, 30, 300);
    const MetricSummary tanh_roll = eval_koopman_rollout_rmse(segments.segments_X_test, segments.segments_U_test, tanh, 30, 300);
    const double poly_roll_raw = eval_rollout_rmse_raw(segments.segments_X_test, segments.segments_U_test, poly, dataset.X_mean, dataset.X_std, 30);
    const double tanh_roll_raw = eval_rollout_rmse_raw(segments.segments_X_test, segments.segments_U_test, tanh, dataset.X_mean, dataset.X_std, 30);

    std::cout << "Synthetic DDIOC dynamics validation\n";
    std::cout << "- train trajectories: " << dataset.X_train.size() << '\n';
    std::cout << "- test trajectories: " << dataset.X_test.size() << '\n';
    std::cout << "- train segments: " << segments.segments_X_train.size() << '\n';
    std::cout << "- test segments: " << segments.segments_X_test.size() << '\n';
    std::cout << "\nOne-step MSE (normalized)\n";
    std::cout << "- poly: " << poly_one.mse_mean << '\n';
    std::cout << "- tanh: " << tanh_one.mse_mean << '\n';
    std::cout << "\nRollout RMSE (normalized, horizon=30)\n";
    std::cout << "- poly: " << poly_roll.rmse_mean << '\n';
    std::cout << "- tanh: " << tanh_roll.rmse_mean << '\n';
    std::cout << "\nRollout RMSE (raw, horizon=30)\n";
    std::cout << "- poly: " << poly_roll_raw << '\n';
    std::cout << "- tanh: " << tanh_roll_raw << '\n';

    return 0;
}