#include "ddioc/pipeline.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    using namespace ddioc::ddioc;

    PipelineConfig config{};
    config.output_root = "outputs/cpp_pipeline";

    for (int idx = 1; idx < argc; ++idx) {
        const std::string arg = argv[idx];
        const auto require_value = [&](const std::string& name) -> std::string {
            if (idx + 1 >= argc) {
                throw std::runtime_error("missing value for " + name);
            }
            return argv[++idx];
        };
        if (arg == "--merged_csv" || arg == "--csv") {
            config.merged_csv_path = require_value(arg);
        } else if (arg == "--output_root") {
            config.output_root = require_value(arg);
        } else if (arg == "--n_traj") {
            config.n_traj = std::stoi(require_value(arg));
        } else if (arg == "--min_traj_len") {
            config.min_traj_len = std::stoi(require_value(arg));
        } else if (arg == "--train_ratio") {
            config.train_ratio = std::stod(require_value(arg));
        } else if (arg == "--split_seed") {
            config.split_seed = static_cast<std::uint32_t>(std::stoul(require_value(arg)));
        } else if (arg == "--lift") {
            config.lift = require_value(arg);
        } else if (arg == "--degree") {
            config.degree = std::stoi(require_value(arg));
        } else if (arg == "--tanh_features") {
            config.tanh_features = std::stoi(require_value(arg));
        } else if (arg == "--seg_len") {
            config.seg_len = std::stoi(require_value(arg));
        } else if (arg == "--segment_stride") {
            config.segment_stride = std::stoi(require_value(arg));
        } else if (arg == "--reg") {
            config.reg = std::stod(require_value(arg));
        } else if (arg == "--ioc_every") {
            config.ioc_every = std::stoi(require_value(arg));
        } else if (arg == "--window") {
            config.window = std::stoi(require_value(arg));
        } else if (arg == "--verbose") {
            config.verbose = true;
        } else if (arg == "--help") {
            std::cout << "Usage: ddioc_pipeline --merged_csv path [--output_root dir] [--lift poly|tanh] [--degree N] [--tanh_features N] [--seg_len N] [--segment_stride N] [--reg x] [--ioc_every N] [--window N] [--verbose]\n";
            return 0;
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    if (config.merged_csv_path.empty()) {
        std::cerr << "Provide --merged_csv\n";
        return 1;
    }

    try {
        const PipelineResult result = learn_dynamics_and_weights(config);
        std::cout << "loaded trajectories: " << result.n_traj_loaded << "\n";
        std::cout << "train segments: " << result.train_segments << "\n";
        std::cout << "test segments: " << result.test_segments << "\n";
        std::cout << "omega:";
        for (double value : result.omega) {
            std::cout << ' ' << value;
        }
        std::cout << "\nlearned objective: " << result.learned_objective_json_path << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}