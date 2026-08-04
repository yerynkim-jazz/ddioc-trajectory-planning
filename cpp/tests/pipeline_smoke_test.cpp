#include "ddioc/pipeline.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace {

void write_test_csv(const std::filesystem::path& path) {
    std::ofstream out(path);
    out << "lc_id_label,time_s,lateral_offset_m_sg,target_orientation_rad_sg,target_curvature_1pm_sg,target_curvature_1pm_dot,target_curvature_1pm_ddot,reference_orientation_sg,reference_curvature_sg,target_speed_mps\n";
    for (int traj = 0; traj < 3; ++traj) {
        for (int step = 0; step < 8; ++step) {
            const double t = 0.1 * static_cast<double>(step);
            const double lat = 0.2 * traj + 0.03 * step;
            const double psi = 0.01 * step;
            const double kappa = 0.002 * step;
            const double kappa_dot = 0.001 * (step + traj);
            const double u = 0.0005 * (step + 1);
            out << "traj_" << traj << ','
                << t << ','
                << lat << ','
                << psi << ','
                << kappa << ','
                << kappa_dot << ','
                << u << ','
                << 0.0 << ','
                << 0.0 << ','
                << 8.0 + traj << '\n';
        }
    }
}

}  // namespace

int main() {
    using namespace ddioc::ddioc;

    const std::filesystem::path temp_root = std::filesystem::temp_directory_path() / "ddioc_pipeline_smoke";
    std::filesystem::create_directories(temp_root);
    const std::filesystem::path csv_path = temp_root / "merged.csv";
    const std::filesystem::path output_root = temp_root / "outputs";
    write_test_csv(csv_path);

    PipelineConfig config{};
    config.merged_csv_path = csv_path.string();
    config.output_root = output_root.string();
    config.n_traj = 3;
    config.min_traj_len = 6;
    config.train_ratio = 2.0 / 3.0;
    config.split_seed = 1;
    config.lift = "poly";
    config.degree = 2;
    config.seg_len = 5;
    config.segment_stride = 2;
    config.window = 2;
    config.ioc_every = 1;

    const PipelineResult result = learn_dynamics_and_weights(config);
    if (result.omega.size() != 9) {
        std::cerr << "unexpected omega size\n";
        return 1;
    }
    double omega_sum = 0.0;
    for (double value : result.omega) {
        if (!(value >= 0.0)) {
            std::cerr << "negative omega\n";
            return 1;
        }
        omega_sum += value;
    }
    if (std::abs(omega_sum - 1.0) > 1e-6) {
        std::cerr << "omega does not sum to one\n";
        return 1;
    }
    if (!std::filesystem::exists(result.learned_objective_json_path) || !std::filesystem::exists(result.omega_history_json_path)) {
        std::cerr << "expected artifacts missing\n";
        return 1;
    }
    if (result.train_segments <= 0 || result.test_segments <= 0) {
        std::cerr << "segments missing\n";
        return 1;
    }
    return 0;
}