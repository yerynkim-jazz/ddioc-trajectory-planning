#include "ddioc/pipeline.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace ddioc::ddioc {
namespace {

std::string json_escape(const std::string& value) {
    std::string out;
    out.reserve(value.size());
    for (char ch : value) {
        switch (ch) {
            case '\\':
                out += "\\\\";
                break;
            case '"':
                out += "\\\"";
                break;
            case '\n':
                out += "\\n";
                break;
            default:
                out.push_back(ch);
                break;
        }
    }
    return out;
}

std::string json_array(const std::vector<double>& values) {
    std::ostringstream out;
    out << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            out << ", ";
        }
        out << values[i];
    }
    out << ']';
    return out.str();
}

std::string json_string_array(const std::vector<std::string>& values) {
    std::ostringstream out;
    out << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            out << ", ";
        }
        out << '"' << json_escape(values[i]) << '"';
    }
    out << ']';
    return out.str();
}

std::string json_matrix(const std::vector<std::vector<double>>& values) {
    std::ostringstream out;
    out << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            out << ", ";
        }
        out << json_array(values[i]);
    }
    out << ']';
    return out.str();
}

std::string metric_summary_json(const MetricSummary& metric) {
    std::ostringstream out;
    out << "{\"mse_mean\": " << metric.mse_mean
        << ", \"rmse_mean\": " << metric.rmse_mean
        << ", \"n_segments\": " << metric.n_segments << '}';
    return out.str();
}

void write_pipeline_json(const std::filesystem::path& path, const PipelineResult& result) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to write pipeline json");
    }
    out << "{\n";
    out << "  \"merged_csv\": \"" << json_escape(result.merged_csv_path) << "\",\n";
    out << "  \"output_root\": \"" << json_escape(result.output_root) << "\",\n";
    out << "  \"lift\": \"" << json_escape(result.lift) << "\",\n";
    out << "  \"n_traj_loaded\": " << result.n_traj_loaded << ",\n";
    out << "  \"train_segments\": " << result.train_segments << ",\n";
    out << "  \"test_segments\": " << result.test_segments << ",\n";
    out << "  \"dt_s\": " << result.dt_s << ",\n";
    out << "  \"train_ids\": " << json_string_array(result.train_ids) << ",\n";
    out << "  \"omega\": " << json_array(result.omega) << ",\n";
    out << "  \"feature_scales\": " << json_array(result.feature_scales) << ",\n";
    out << "  \"omega_history\": " << json_matrix(result.omega_history) << ",\n";
    out << "  \"metrics\": {\n";
    out << "    \"koopman_one_step\": {\"train\": " << metric_summary_json(result.metrics.train_one_step)
        << ", \"test\": " << metric_summary_json(result.metrics.test_one_step) << "},\n";
    out << "    \"koopman_rollout10\": {\"train\": " << metric_summary_json(result.metrics.train_rollout10)
        << ", \"test\": " << metric_summary_json(result.metrics.test_rollout10) << "},\n";
    out << "    \"ioc_residual\": {\"train\": " << result.metrics.ioc_train_residual
        << ", \"test\": " << result.metrics.ioc_test_residual << "}\n";
    out << "  }\n";
    out << "}\n";
}

}  // namespace

PipelineResult learn_dynamics_and_weights(const PipelineConfig& config) {
    PreparedSegments data = prepare_train_test_segments_from_merged_csv(
        config.merged_csv_path,
        config.n_traj,
        config.min_traj_len,
        config.train_ratio,
        config.split_seed,
        config.seg_len,
        config.segment_stride);

    if (config.verbose) {
        std::cerr << "[PIPELINE] loaded train=" << data.segments_X.size() << " test=" << data.segments_X_test.size() << " dt_s=" << data.dt_s << "\n";
    }

    KoopmanModel model{};
    if (config.lift == "poly") {
        model = learn_koopman_dynamics_poly(data.segments_X, data.segments_U, 4, config.degree, config.reg);
    } else if (config.lift == "tanh") {
        model = learn_koopman_dynamics_tanh(data.segments_X, data.segments_U, 4, config.tanh_features, config.split_seed, config.reg);
    } else {
        throw std::runtime_error("unsupported lift: " + config.lift);
    }

    const std::vector<double> feature_scales = compute_feature_scales(
        data.segments_X,
        data.segments_U,
        data.segments_V,
        data.segments_target_lat_m,
        data.segments_ref_orientation,
        data.segments_ref_curvature,
        data.dt_s,
        data.X_mean,
        data.X_std,
        data.U_mean,
        data.U_std);

    const HLOLearningResult hlo = learn_hlo_omega(
        data.segments_X,
        data.segments_U,
        data.segments_V,
        data.segments_target_lat_m,
        data.segments_ref_orientation,
        data.segments_ref_curvature,
        model,
        data.dt_s,
        data.X_mean,
        data.X_std,
        data.U_mean,
        data.U_std,
        feature_scales,
        config.ioc_every,
        config.window);

    PipelineResult result{};
    result.merged_csv_path = config.merged_csv_path;
    result.output_root = config.output_root;
    result.lift = config.lift;
    result.n_traj_loaded = static_cast<int>(data.traj_X_raw.size());
    result.train_segments = static_cast<int>(data.segments_X.size());
    result.test_segments = static_cast<int>(data.segments_X_test.size());
    result.dt_s = data.dt_s;
    result.train_ids = data.ids_train;
    result.omega = hlo.omega;
    result.feature_scales = hlo.feature_scales;
    result.omega_history = hlo.omega_history;
    result.metrics.train_one_step = eval_koopman_one_step_mse(data.segments_X, data.segments_U, model, 0);
    result.metrics.test_one_step = eval_koopman_one_step_mse(data.segments_X_test, data.segments_U_test, model, 0);
    result.metrics.train_rollout10 = eval_koopman_rollout_rmse(data.segments_X, data.segments_U, model, 10, 200);
    result.metrics.test_rollout10 = eval_koopman_rollout_rmse(data.segments_X_test, data.segments_U_test, model, 10, 200);

    const int train_window = std::max(1, std::min(config.window, static_cast<int>(data.segments_X.size())));
    result.metrics.ioc_train_residual = eval_ioc_residual(
        std::vector<Matrix>(data.segments_X.end() - train_window, data.segments_X.end()),
        std::vector<Matrix>(data.segments_U.end() - train_window, data.segments_U.end()),
        std::vector<std::vector<double>>(data.segments_V.end() - train_window, data.segments_V.end()),
        std::vector<double>(data.segments_target_lat_m.end() - train_window, data.segments_target_lat_m.end()),
        std::vector<std::vector<double>>(data.segments_ref_orientation.end() - train_window, data.segments_ref_orientation.end()),
        std::vector<std::vector<double>>(data.segments_ref_curvature.end() - train_window, data.segments_ref_curvature.end()),
        result.omega,
        result.feature_scales,
        data.dt_s,
        data.X_mean,
        data.X_std,
        data.U_mean,
        data.U_std);

    if (!data.segments_X_test.empty()) {
        const int test_window = std::max(1, std::min(config.window, static_cast<int>(data.segments_X_test.size())));
        result.metrics.ioc_test_residual = eval_ioc_residual(
            std::vector<Matrix>(data.segments_X_test.end() - test_window, data.segments_X_test.end()),
            std::vector<Matrix>(data.segments_U_test.end() - test_window, data.segments_U_test.end()),
            std::vector<std::vector<double>>(data.segments_V_test.end() - test_window, data.segments_V_test.end()),
            std::vector<double>(data.segments_target_lat_m_test.end() - test_window, data.segments_target_lat_m_test.end()),
            std::vector<std::vector<double>>(data.segments_ref_orientation_test.end() - test_window, data.segments_ref_orientation_test.end()),
            std::vector<std::vector<double>>(data.segments_ref_curvature_test.end() - test_window, data.segments_ref_curvature_test.end()),
            result.omega,
            result.feature_scales,
            data.dt_s,
            data.X_mean,
            data.X_std,
            data.U_mean,
            data.U_std);
    }

    const std::filesystem::path output_root(config.output_root);
    std::filesystem::create_directories(output_root);
    result.learned_objective_json_path = (output_root / "learned_objective.json").string();
    result.omega_history_json_path = (output_root / "omega_history.json").string();
    save_learned_objective_json(result.learned_objective_json_path, result.omega, result.feature_scales);
    write_pipeline_json(result.omega_history_json_path, result);

    if (config.verbose) {
        std::cerr << "[PIPELINE] wrote outputs under " << output_root << "\n";
    }
    return result;
}

}  // namespace ddioc::ddioc