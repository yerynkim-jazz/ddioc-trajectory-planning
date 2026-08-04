#include "ddioc/dynamics_learning.hpp"

#include <iostream>

int main() {
    using namespace ddioc::ddioc;

    DynamicsValidationConfig config{};
    config.n_traj = 24;
    config.horizon = 80;

    const TrajectoryDataset dataset = build_synthetic_validation_dataset(config);
    const SegmentDataset segments = build_validation_segments(dataset, config);
    const KoopmanModel poly = learn_koopman_dynamics_poly(segments.segments_X_train, segments.segments_U_train, 4, config.poly_degree, 1e-6);
    const MetricSummary summary = eval_koopman_one_step_mse(segments.segments_X_test, segments.segments_U_test, poly, 32);

    if (segments.segments_X_train.empty() || segments.segments_X_test.empty()) {
        std::cerr << "no validation segments built\n";
        return 1;
    }
    if (!(summary.mse_mean >= 0.0)) {
        std::cerr << "invalid one-step mse\n";
        return 1;
    }
    return 0;
}