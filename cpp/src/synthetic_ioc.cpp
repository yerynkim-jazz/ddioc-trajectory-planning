#include "ddioc/synthetic_ioc.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>

namespace ddioc {
namespace {

using Matrix4 = std::array<std::array<double, kStateDim>, kStateDim>;
using Vector4 = std::array<double, kStateDim>;

double wrap_to_pi(double angle) {
    return std::atan2(std::sin(angle), std::cos(angle));
}

double clamp_value(double value, double lower, double upper) {
    return std::max(lower, std::min(value, upper));
}

double mean(const std::vector<double>& values) {
    if (values.empty()) {
        return 0.0;
    }
    return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
}

double standard_deviation(const std::vector<double>& values) {
    if (values.empty()) {
        return 0.0;
    }
    const double mu = mean(values);
    double accum = 0.0;
    for (double value : values) {
        const double diff = value - mu;
        accum += diff * diff;
    }
    return std::sqrt(accum / static_cast<double>(values.size()));
}

FeatureVector project_to_simplex(const FeatureVector& weights, double minimum) {
    FeatureVector projected{};
    double total = 0.0;
    for (std::size_t index = 0; index < kFeatureDim; ++index) {
        projected[index] = std::max(weights[index], minimum);
        total += projected[index];
    }
    if (total <= 0.0) {
        const double uniform = 1.0 / static_cast<double>(kFeatureDim);
        projected.fill(uniform);
        return projected;
    }
    for (double& value : projected) {
        value /= total;
    }
    return projected;
}

FeatureVector project_to_bounded_simplex(const FeatureVector& weights, double minimum, double maximum, int iterations = 12) {
    FeatureVector projected = project_to_simplex(weights, minimum);
    for (int iter = 0; iter < std::max(1, iterations); ++iter) {
        double sum = 0.0;
        int free_count = 0;
        std::array<bool, kFeatureDim> is_free{};
        for (std::size_t index = 0; index < kFeatureDim; ++index) {
            projected[index] = clamp_value(projected[index], minimum, maximum);
            sum += projected[index];
        }
        if (std::abs(sum - 1.0) <= 1e-10) {
            break;
        }
        for (std::size_t index = 0; index < kFeatureDim; ++index) {
            is_free[index] = projected[index] > minimum + 1e-12 && projected[index] < maximum - 1e-12;
            if (is_free[index]) {
                ++free_count;
            }
        }
        if (free_count == 0) {
            const double scale = sum <= 1e-12 ? 1.0 : (1.0 / sum);
            for (double& value : projected) {
                value = clamp_value(value * scale, minimum, maximum);
            }
            break;
        }
        const double offset = (1.0 - sum) / static_cast<double>(free_count);
        for (std::size_t index = 0; index < kFeatureDim; ++index) {
            if (is_free[index]) {
                projected[index] += offset;
            }
        }
    }
    return project_to_simplex(projected, minimum);
}

PlannerWeightArray planner_weights_to_array(const LQRPlannerWeights& weights) {
    return {weights.w_d, weights.w_a1, weights.w_a2, weights.w_a3, weights.w_a4};
}

LQRPlannerWeights array_to_planner_weights(const PlannerWeightArray& values) {
    return {values[0], values[1], values[2], values[3], values[4]};
}

Vector4 subtract_state(const State& lhs, const State& rhs) {
    Vector4 out{};
    for (std::size_t index = 0; index < kStateDim; ++index) {
        out[index] = lhs[index] - rhs[index];
    }
    return out;
}

Vector4 matrix_vector_mul(const Matrix4& matrix, const Vector4& vector) {
    Vector4 out{};
    for (std::size_t row = 0; row < kStateDim; ++row) {
        for (std::size_t col = 0; col < kStateDim; ++col) {
            out[row] += matrix[row][col] * vector[col];
        }
    }
    return out;
}

Matrix4 matrix_add(const Matrix4& lhs, const Matrix4& rhs) {
    Matrix4 out{};
    for (std::size_t row = 0; row < kStateDim; ++row) {
        for (std::size_t col = 0; col < kStateDim; ++col) {
            out[row][col] = lhs[row][col] + rhs[row][col];
        }
    }
    return out;
}

Matrix4 matrix_subtract(const Matrix4& lhs, const Matrix4& rhs) {
    Matrix4 out{};
    for (std::size_t row = 0; row < kStateDim; ++row) {
        for (std::size_t col = 0; col < kStateDim; ++col) {
            out[row][col] = lhs[row][col] - rhs[row][col];
        }
    }
    return out;
}

Matrix4 matrix_mul(const Matrix4& lhs, const Matrix4& rhs) {
    Matrix4 out{};
    for (std::size_t row = 0; row < kStateDim; ++row) {
        for (std::size_t col = 0; col < kStateDim; ++col) {
            for (std::size_t inner = 0; inner < kStateDim; ++inner) {
                out[row][col] += lhs[row][inner] * rhs[inner][col];
            }
        }
    }
    return out;
}

Matrix4 transpose(const Matrix4& matrix) {
    Matrix4 out{};
    for (std::size_t row = 0; row < kStateDim; ++row) {
        for (std::size_t col = 0; col < kStateDim; ++col) {
            out[row][col] = matrix[col][row];
        }
    }
    return out;
}

Matrix4 outer_product(const std::array<double, kStateDim>& lhs, const std::array<double, kStateDim>& rhs) {
    Matrix4 out{};
    for (std::size_t row = 0; row < kStateDim; ++row) {
        for (std::size_t col = 0; col < kStateDim; ++col) {
            out[row][col] = lhs[row] * rhs[col];
        }
    }
    return out;
}

struct DynamicsMatrices {
    Matrix4 A{};
    std::array<double, kStateDim> B{};
    std::array<std::array<double, 2>, kStateDim> D{};
};

DynamicsMatrices build_discrete_dynamics_matrices(double dt_s, double speed_mps) {
    const double dt = dt_s;
    const double v = speed_mps;
    DynamicsMatrices matrices{};
    matrices.A = {{{1.0, v * dt, 0.5 * v * v * dt * dt, (1.0 / 6.0) * v * v * dt * dt * dt},
                   {0.0, 1.0, v * dt, 0.5 * v * dt * dt},
                   {0.0, 0.0, 1.0, dt},
                   {0.0, 0.0, 0.0, 1.0}}};
    matrices.B = {{(1.0 / 24.0) * v * v * dt * dt * dt * dt,
                   (1.0 / 6.0) * v * dt * dt * dt,
                   0.5 * dt * dt,
                   dt}};
    matrices.D = {{{dt, -v * dt},
                   {0.0, 0.0},
                   {0.0, 0.0},
                   {0.0, 0.0}}};
    return matrices;
}

struct CostMatrices {
    Matrix4 Q{};
    Matrix4 Q_terminal{};
    double R = 0.0;
};

CostMatrices build_lqr_cost_matrices(const LQRPlannerWeights& weights, double speed_mps, double v_bar = 0.1, double terminal_scale = 10.0) {
    const double v_eff = std::max(speed_mps, v_bar);
    const double v2 = v_eff * v_eff;
    const double v4 = v2 * v2;
    CostMatrices matrices{};
    matrices.Q[0][0] = weights.w_d;
    matrices.Q[1][1] = weights.w_a1 * v2;
    matrices.Q[2][2] = weights.w_a2 * v4;
    matrices.Q[3][3] = weights.w_a3 * v4;
    matrices.Q_terminal = matrices.Q;
    for (std::size_t row = 0; row < kStateDim; ++row) {
        for (std::size_t col = 0; col < kStateDim; ++col) {
            matrices.Q_terminal[row][col] *= terminal_scale;
        }
    }
    matrices.R = weights.w_a4 * v4;
    return matrices;
}

FeatureVector evaluate_features(
    const GroundTruthHLO& hlo,
    const std::vector<State>& states,
    const std::vector<double>& controls,
    const std::vector<double>& velocity_mps,
    double dt_s,
    double target_lat_offset_m) {
    (void)hlo;
    FeatureVector features{};
    for (std::size_t index = 0; index < controls.size(); ++index) {
        const double d_next = states[index + 1][0];
        const double d_prev = states[index][0];
        const double psi_next = states[index + 1][1];
        const double kappa_next = states[index + 1][2];
        const double kappa_prev = states[index][2];
        const double kappa_dot_next = states[index + 1][3];
        const double v_eff = std::max(velocity_mps[index], 1e-3);
        const double lat_err = d_next - target_lat_offset_m;
        const double lat_rate = (d_next - d_prev) / dt_s;
        const double kappa_err = kappa_next;
        const double delta_kappa_err = kappa_err - kappa_prev;
        const double psi_err = wrap_to_pi(psi_next);
        features[0] += lat_err * lat_err;
        features[1] += (v_eff * v_eff) * (lat_rate * lat_rate);
        features[2] += lat_err * lat_err * lat_err * lat_err;
        features[3] += (v_eff * v_eff) * std::abs(lat_err * psi_err);
        features[4] += std::pow(v_eff, 4.0) * (kappa_dot_next * kappa_dot_next);
        features[5] += std::pow(v_eff, 4.0) * (kappa_err * kappa_err);
        features[6] += std::pow(v_eff, 4.0) * std::pow(kappa_err, 4.0);
        features[7] += std::pow(v_eff, 4.0) * (delta_kappa_err * delta_kappa_err);
        features[8] += std::pow(v_eff, 4.0) * (controls[index] * controls[index]);
        features[9] += (v_eff * v_eff) * (psi_err * psi_err);
    }
    return features;
}

double evaluate_cost(
    const GroundTruthHLO& hlo,
    const std::vector<State>& states,
    const std::vector<double>& controls,
    const std::vector<double>& velocity_mps,
    double dt_s,
    double target_lat_offset_m) {
    const FeatureVector features = evaluate_features(hlo, states, controls, velocity_mps, dt_s, target_lat_offset_m);
    double cost = 0.0;
    for (std::size_t index = 0; index < kFeatureDim; ++index) {
        cost += hlo.omega[index] * features[index];
    }
    return cost;
}

LQRPlannerResult rollout_linearized_kinematics(const State& x0, const std::vector<double>& controls, double dt_s, double constant_speed_mps) {
    LQRPlannerResult result{};
    result.states.resize(controls.size() + 1);
    result.controls = controls;
    result.velocity_mps.assign(controls.size(), constant_speed_mps);
    result.states[0] = x0;
    for (std::size_t index = 0; index < controls.size(); ++index) {
        const State& state = result.states[index];
        const double d = state[0];
        const double psi = state[1];
        const double kappa = state[2];
        const double kappa_dot = state[3];
        result.states[index + 1] = State{
            d + dt_s * constant_speed_mps * psi,
            psi + dt_s * constant_speed_mps * kappa,
            kappa + dt_s * kappa_dot,
            kappa_dot + dt_s * controls[index]};
    }
    return result;
}

template <typename Candidate, typename Objective, typename Projector>
Candidate random_search_optimize(
    const Candidate& initial,
    std::mt19937& rng,
    int restarts,
    int iterations,
    double noise_scale,
    const Objective& objective,
    const Projector& projector) {
    std::normal_distribution<double> noise(0.0, 1.0);
    Candidate best = initial;
    projector(best);
    double best_value = objective(best);
    for (int restart = 0; restart < std::max(1, restarts); ++restart) {
        Candidate current = best;
        if (restart > 0) {
            for (double& value : current) {
                value += noise_scale * noise(rng);
            }
            projector(current);
        }
        double current_value = objective(current);
        double local_scale = noise_scale;
        for (int iter = 0; iter < std::max(1, iterations); ++iter) {
            Candidate candidate = current;
            for (double& value : candidate) {
                value += local_scale * noise(rng);
            }
            projector(candidate);
            const double candidate_value = objective(candidate);
            if (candidate_value < current_value) {
                current = candidate;
                current_value = candidate_value;
                if (candidate_value < best_value) {
                    best = candidate;
                    best_value = candidate_value;
                }
            } else {
                local_scale *= 0.995;
            }
        }
    }
    return best;
}

std::vector<State> sample_initial_states(std::mt19937& rng, int count) {
    std::uniform_real_distribution<double> lat(-0.75, 0.75);
    std::uniform_real_distribution<double> psi(-0.15, 0.15);
    std::uniform_real_distribution<double> kappa(-0.03, 0.03);
    std::uniform_real_distribution<double> kappa_dot(-0.03, 0.03);
    std::vector<State> states;
    states.reserve(std::max(0, count));
    for (int index = 0; index < count; ++index) {
        states.push_back(State{lat(rng), psi(rng), kappa(rng), kappa_dot(rng)});
    }
    return states;
}

SyntheticDemoTrajectory generate_demo_for_state(
    const State& x0,
    const GroundTruthHLO& hlo,
    const DemoGenerationConfig& config,
    std::mt19937& rng) {
    std::vector<double> initial_controls(static_cast<std::size_t>(config.horizon), 0.0);
    const auto objective = [&](const std::vector<double>& controls) {
        const LQRPlannerResult rollout = rollout_linearized_kinematics(x0, controls, config.dt_s, config.constant_speed_mps);
        return evaluate_cost(hlo, rollout.states, controls, rollout.velocity_mps, config.dt_s, config.target_lat_offset_m);
    };
    const auto projector = [&](std::vector<double>& controls) {
        for (double& control : controls) {
            control = clamp_value(control, -config.control_limit, config.control_limit);
        }
    };
    std::vector<double> best_controls = random_search_optimize(
        initial_controls,
        rng,
        config.restarts,
        220,
        0.05,
        objective,
        projector);
    const LQRPlannerResult rollout = rollout_linearized_kinematics(x0, best_controls, config.dt_s, config.constant_speed_mps);
    SyntheticDemoTrajectory demo{};
    demo.x0 = x0;
    demo.states = rollout.states;
    demo.controls = best_controls;
    demo.velocity_mps = rollout.velocity_mps;
    demo.true_hlo_cost = evaluate_cost(hlo, demo.states, demo.controls, demo.velocity_mps, config.dt_s, config.target_lat_offset_m);
    demo.feature_sums = evaluate_features(hlo, demo.states, demo.controls, demo.velocity_mps, config.dt_s, config.target_lat_offset_m);
    return demo;
}

FeatureVector learn_feature_scales_from_demos(const std::vector<SyntheticDemoTrajectory>& demos) {
    std::array<std::vector<double>, kFeatureDim> columns;
    for (const SyntheticDemoTrajectory& demo : demos) {
        for (std::size_t index = 0; index < kFeatureDim; ++index) {
            columns[index].push_back(demo.feature_sums[index]);
        }
    }
    FeatureVector scales{};
    for (std::size_t index = 0; index < kFeatureDim; ++index) {
        auto& values = columns[index];
        std::sort(values.begin(), values.end());
        if (values.empty()) {
            scales[index] = 1.0;
        } else {
            const std::size_t mid = values.size() / 2;
            const double median = values.size() % 2 == 0 ? 0.5 * (values[mid - 1] + values[mid]) : values[mid];
            scales[index] = std::max(median, 1e-6);
        }
    }
    return scales;
}

double learned_hlo_cost_for_planner_result(
    const LQRPlannerResult& planner_result,
    const LearnedHLOResult& learned_hlo,
    const GroundTruthHLO& basis_hlo,
    const DemoGenerationConfig& demo_config) {
    const FeatureVector features = evaluate_features(
        basis_hlo,
        planner_result.states,
        planner_result.controls,
        planner_result.velocity_mps,
        demo_config.dt_s,
        demo_config.target_lat_offset_m);
    double cost = 0.0;
    for (std::size_t index = 0; index < kFeatureDim; ++index) {
        cost += learned_hlo.omega[index] * (features[index] / learned_hlo.feature_scales[index]);
    }
    return cost;
}

LearnedHLOResult learn_hlo_from_demos(
    const std::vector<SyntheticDemoTrajectory>& demos,
    const GroundTruthHLO& basis_hlo,
    const DemoGenerationConfig& demo_config,
    const OurMethodConfig& method_config,
    std::mt19937& rng) {
    const FeatureVector feature_scales = learn_feature_scales_from_demos(demos);
    std::vector<FeatureVector> deltas;
    std::normal_distribution<double> noise(0.0, method_config.preference_noise_std);
    FeatureVector uniform{};
    uniform.fill(1.0 / static_cast<double>(kFeatureDim));

    for (const SyntheticDemoTrajectory& demo : demos) {
        FeatureVector expert{};
        for (std::size_t index = 0; index < kFeatureDim; ++index) {
            expert[index] = demo.feature_sums[index] / feature_scales[index];
        }
        for (int sample = 0; sample < method_config.pref_samples_per_demo; ++sample) {
            std::vector<double> candidate_controls = demo.controls;
            for (double& control : candidate_controls) {
                control = clamp_value(control + noise(rng), -demo_config.control_limit, demo_config.control_limit);
            }
            const LQRPlannerResult rollout = rollout_linearized_kinematics(demo.x0, candidate_controls, demo_config.dt_s, demo_config.constant_speed_mps);
            const FeatureVector candidate_features = evaluate_features(
                basis_hlo,
                rollout.states,
                candidate_controls,
                rollout.velocity_mps,
                demo_config.dt_s,
                demo_config.target_lat_offset_m);
            FeatureVector delta{};
            double norm_sq = 0.0;
            for (std::size_t index = 0; index < kFeatureDim; ++index) {
                delta[index] = (candidate_features[index] / feature_scales[index]) - expert[index];
                norm_sq += delta[index] * delta[index];
            }
            const double norm = std::sqrt(norm_sq);
            if (norm > 1e-10) {
                for (double& value : delta) {
                    value /= norm;
                }
                deltas.push_back(delta);
            }
        }
    }

    if (deltas.empty()) {
        return {uniform, feature_scales};
    }

    const auto objective = [&](const FeatureVector& raw_weights) {
        const FeatureVector weights = project_to_bounded_simplex(raw_weights, method_config.omega_min, method_config.omega_max);
        double loss = 0.0;
        for (const FeatureVector& delta : deltas) {
            double score = 0.0;
            for (std::size_t index = 0; index < kFeatureDim; ++index) {
                score += delta[index] * weights[index];
            }
            loss += std::log1p(std::exp(method_config.margin - score));
        }
        loss /= static_cast<double>(deltas.size());
        double prior_reg = 0.0;
        double entropy = 0.0;
        for (std::size_t index = 0; index < kFeatureDim; ++index) {
            const double diff = weights[index] - uniform[index];
            prior_reg += diff * diff;
            entropy += weights[index] * std::log(std::max(weights[index], 1e-12));
        }
        return loss + method_config.omega_reg * prior_reg - method_config.omega_entropy_reg * (-entropy);
    };
    const auto projector = [&](FeatureVector& weights) {
        weights = project_to_bounded_simplex(weights, method_config.omega_min, method_config.omega_max);
    };

    FeatureVector initial = uniform;
    FeatureVector best = random_search_optimize(initial, rng, method_config.hlo_restarts, 320, 0.2, objective, projector);
    return {project_to_bounded_simplex(best, method_config.omega_min, method_config.omega_max), feature_scales};
}

double tracking_sse_for_planner_result(const LQRPlannerResult& planner_result, const SyntheticDemoTrajectory& demo, double control_weight = 0.1) {
    double state_error = 0.0;
    for (std::size_t step = 0; step < planner_result.states.size(); ++step) {
        for (std::size_t index = 0; index < kStateDim; ++index) {
            const double diff = planner_result.states[step][index] - demo.states[step][index];
            state_error += diff * diff;
        }
    }
    state_error /= static_cast<double>(planner_result.states.size() * kStateDim);
    double control_error = 0.0;
    for (std::size_t step = 0; step < planner_result.controls.size(); ++step) {
        const double diff = planner_result.controls[step] - demo.controls[step];
        control_error += diff * diff;
    }
    control_error /= static_cast<double>(std::max<std::size_t>(1, planner_result.controls.size()));
    return state_error + control_weight * control_error;
}

LQRPlannerWeights tune_lqr_weights(
    const std::vector<SyntheticDemoTrajectory>& demos,
    const DemoGenerationConfig& demo_config,
    std::mt19937& rng,
    int restarts,
    int iterations,
    const LQRPlannerWeights& initial_weights,
    const std::function<double(const LQRPlannerResult&, const SyntheticDemoTrajectory&)>& trajectory_cost) {
    PlannerWeightArray initial = planner_weights_to_array(initial_weights);
    for (double& value : initial) {
        value = std::log(std::max(value, 1e-8));
    }
    const auto projector = [](PlannerWeightArray& values) {
        for (double& value : values) {
            value = clamp_value(value, -8.0, 4.0);
        }
    };
    const auto objective = [&](const PlannerWeightArray& log_theta) {
        PlannerWeightArray positive{};
        for (std::size_t index = 0; index < kPlannerWeightDim; ++index) {
            positive[index] = std::exp(log_theta[index]);
        }
        const LQRPlannerWeights weights = array_to_planner_weights(positive);
        std::vector<double> costs;
        costs.reserve(demos.size());
        for (const SyntheticDemoTrajectory& demo : demos) {
            const LQRPlannerResult planner_result = plan_with_lqr(
                demo.x0,
                weights,
                demo_config.horizon,
                demo_config.dt_s,
                demo_config.constant_speed_mps,
                demo_config.target_lat_offset_m,
                demo_config.control_limit);
            costs.push_back(trajectory_cost(planner_result, demo));
        }
        return mean(costs);
    };
    const PlannerWeightArray best_log = random_search_optimize(initial, rng, restarts, iterations, 0.35, objective, projector);
    PlannerWeightArray positive{};
    for (std::size_t index = 0; index < kPlannerWeightDim; ++index) {
        positive[index] = std::exp(best_log[index]);
    }
    return array_to_planner_weights(positive);
}

std::string escape_json(const std::string& value) {
    std::ostringstream out;
    for (char c : value) {
        switch (c) {
            case '\\':
                out << "\\\\";
                break;
            case '"':
                out << "\\\"";
                break;
            case '\n':
                out << "\\n";
                break;
            default:
                out << c;
                break;
        }
    }
    return out.str();
}

}  // namespace

GroundTruthHLO get_ground_truth_hlo() {
    GroundTruthHLO hlo{};
    hlo.feature_names = {{"lat_err_sq",
                          "lat_rate_sq_v2",
                          "lat_err_quartic",
                          "lat_psi_cross_abs_v2",
                          "kappa_dot_abs_sq_v4",
                          "kappa_err_sq_v4",
                          "kappa_err_quartic_v4",
                          "delta_kappa_err_sq_v4",
                          "u_sq_v4",
                          "psi_err_sq_v2"}};
    const FeatureVector raw = {{0.14, 0.08, 0.15, 0.10, 0.11, 0.08, 0.10, 0.09, 0.07, 0.08}};
    hlo.omega = project_to_simplex(raw, 1e-8);
    return hlo;
}

LQRPlannerWeights get_default_lqr_planner_weights() {
    return {};
}

std::vector<SyntheticDemoTrajectory> generate_synthetic_demonstrations(
    const GroundTruthHLO& hlo,
    const DemoGenerationConfig& config) {
    std::mt19937 rng(config.seed);
    const std::vector<State> initial_states = sample_initial_states(rng, config.n_demos);
    std::vector<SyntheticDemoTrajectory> demos;
    demos.reserve(initial_states.size());
    for (const State& x0 : initial_states) {
        demos.push_back(generate_demo_for_state(x0, hlo, config, rng));
    }
    return demos;
}

LQRPlannerResult plan_with_lqr(
    const State& x0,
    const LQRPlannerWeights& weights,
    int horizon,
    double dt_s,
    double constant_speed_mps,
    double target_lat_offset_m,
    double control_limit) {
    const DynamicsMatrices dynamics = build_discrete_dynamics_matrices(dt_s, constant_speed_mps);
    const CostMatrices cost = build_lqr_cost_matrices(weights, constant_speed_mps);

    std::vector<State> ref_states(static_cast<std::size_t>(horizon) + 1, State{target_lat_offset_m, 0.0, 0.0, 0.0});
    Vector4 error0 = subtract_state(x0, ref_states[0]);
    Matrix4 P = cost.Q_terminal;
    std::vector<std::array<double, kStateDim>> gains_rev;
    gains_rev.reserve(static_cast<std::size_t>(horizon));

    const Matrix4 A_t = transpose(dynamics.A);
    for (int step = 0; step < horizon; ++step) {
        const auto P_B = matrix_vector_mul(P, dynamics.B);
        double middle = cost.R;
        for (std::size_t index = 0; index < kStateDim; ++index) {
            middle += dynamics.B[index] * P_B[index];
        }
        std::array<double, kStateDim> B_t_P_A{};
        for (std::size_t col = 0; col < kStateDim; ++col) {
            for (std::size_t row = 0; row < kStateDim; ++row) {
                for (std::size_t inner = 0; inner < kStateDim; ++inner) {
                    B_t_P_A[col] += dynamics.B[row] * P[row][inner] * dynamics.A[inner][col];
                }
            }
            B_t_P_A[col] /= middle;
        }
        gains_rev.push_back(B_t_P_A);
        const auto BK = outer_product(dynamics.B, B_t_P_A);
        const auto A_minus_BK = matrix_subtract(dynamics.A, BK);
        P = matrix_add(cost.Q, matrix_mul(A_t, matrix_mul(P, A_minus_BK)));
    }

    std::vector<std::array<double, kStateDim>> gains(gains_rev.rbegin(), gains_rev.rend());
    std::vector<Vector4> errors(static_cast<std::size_t>(horizon) + 1);
    std::vector<State> states(static_cast<std::size_t>(horizon) + 1);
    std::vector<double> controls(static_cast<std::size_t>(horizon), 0.0);
    errors[0] = error0;
    states[0] = x0;

    for (int step = 0; step < horizon; ++step) {
        double control = 0.0;
        for (std::size_t index = 0; index < kStateDim; ++index) {
            control -= gains[static_cast<std::size_t>(step)][index] * errors[static_cast<std::size_t>(step)][index];
        }
        control = clamp_value(control, -control_limit, control_limit);
        controls[static_cast<std::size_t>(step)] = control;
        Vector4 next_error = matrix_vector_mul(dynamics.A, errors[static_cast<std::size_t>(step)]);
        for (std::size_t index = 0; index < kStateDim; ++index) {
            next_error[index] += dynamics.B[index] * control;
        }
        errors[static_cast<std::size_t>(step) + 1] = next_error;
        for (std::size_t index = 0; index < kStateDim; ++index) {
            states[static_cast<std::size_t>(step) + 1][index] = ref_states[static_cast<std::size_t>(step) + 1][index] + next_error[index];
        }
    }

    LQRPlannerResult result{};
    result.states = std::move(states);
    result.controls = std::move(controls);
    result.velocity_mps.assign(static_cast<std::size_t>(horizon), constant_speed_mps);
    result.gains = std::move(gains);
    return result;
}

OurMethodResult run_our_method(
    const std::vector<SyntheticDemoTrajectory>& demos,
    const GroundTruthHLO& basis_hlo,
    const DemoGenerationConfig& demo_config,
    const OurMethodConfig& method_config,
    const LQRPlannerWeights& initial_planner_weights) {
    std::mt19937 rng(demo_config.seed + 101U);
    const LearnedHLOResult learned_hlo = learn_hlo_from_demos(demos, basis_hlo, demo_config, method_config, rng);
    const auto trajectory_cost = [&](const LQRPlannerResult& planner_result, const SyntheticDemoTrajectory&) {
        return learned_hlo_cost_for_planner_result(planner_result, learned_hlo, basis_hlo, demo_config);
    };
    const LQRPlannerWeights tuned_weights = tune_lqr_weights(
        demos,
        demo_config,
        rng,
        method_config.planner_restarts,
        method_config.planner_iterations,
        initial_planner_weights,
        trajectory_cost);

    std::vector<LQRPlannerResult> planned_trajectories;
    planned_trajectories.reserve(demos.size());
    std::vector<double> learned_costs;
    learned_costs.reserve(demos.size());
    for (const SyntheticDemoTrajectory& demo : demos) {
        LQRPlannerResult plan = plan_with_lqr(
            demo.x0,
            tuned_weights,
            demo_config.horizon,
            demo_config.dt_s,
            demo_config.constant_speed_mps,
            demo_config.target_lat_offset_m,
            demo_config.control_limit);
        learned_costs.push_back(learned_hlo_cost_for_planner_result(plan, learned_hlo, basis_hlo, demo_config));
        planned_trajectories.push_back(std::move(plan));
    }

    OurMethodResult result{};
    result.learned_hlo = learned_hlo;
    result.tuned_planner_weights = tuned_weights;
    result.planned_trajectories = std::move(planned_trajectories);
    result.mean_learned_hlo_cost = mean(learned_costs);
    return result;
}

ClassicalIOCResult run_classical_ioc_benchmark(
    const std::vector<SyntheticDemoTrajectory>& demos,
    const DemoGenerationConfig& demo_config,
    const OurMethodConfig& method_config,
    const LQRPlannerWeights& initial_planner_weights) {
    std::mt19937 rng(demo_config.seed + 202U);
    const auto trajectory_cost = [](const LQRPlannerResult& planner_result, const SyntheticDemoTrajectory& demo) {
        return tracking_sse_for_planner_result(planner_result, demo);
    };
    const LQRPlannerWeights tuned_weights = tune_lqr_weights(
        demos,
        demo_config,
        rng,
        method_config.planner_restarts,
        method_config.planner_iterations,
        initial_planner_weights,
        trajectory_cost);

    ClassicalIOCResult result{};
    result.tuned_planner_weights = tuned_weights;
    result.planned_trajectories.reserve(demos.size());
    std::vector<double> tracking_errors;
    tracking_errors.reserve(demos.size());
    for (const SyntheticDemoTrajectory& demo : demos) {
        LQRPlannerResult plan = plan_with_lqr(
            demo.x0,
            tuned_weights,
            demo_config.horizon,
            demo_config.dt_s,
            demo_config.constant_speed_mps,
            demo_config.target_lat_offset_m,
            demo_config.control_limit);
        tracking_errors.push_back(tracking_sse_for_planner_result(plan, demo));
        result.planned_trajectories.push_back(std::move(plan));
    }
    result.mean_tracking_sse = mean(tracking_errors);
    return result;
}

EvaluationSummary evaluate_on_unseen_initial_states(
    const GroundTruthHLO& hlo,
    const DemoGenerationConfig& demo_config,
    const OurMethodResult& our_method,
    const ClassicalIOCResult& classical_ioc,
    int n_test,
    int seed_offset) {
    std::mt19937 rng(demo_config.seed + static_cast<std::uint32_t>(seed_offset));
    const std::vector<State> test_x0 = sample_initial_states(rng, n_test);
    std::vector<double> out_our;
    std::vector<double> out_ioc;
    std::vector<double> out_expert;
    out_our.reserve(test_x0.size());
    out_ioc.reserve(test_x0.size());
    out_expert.reserve(test_x0.size());

    for (const State& x0 : test_x0) {
        const LQRPlannerResult our_plan = plan_with_lqr(
            x0,
            our_method.tuned_planner_weights,
            demo_config.horizon,
            demo_config.dt_s,
            demo_config.constant_speed_mps,
            demo_config.target_lat_offset_m,
            demo_config.control_limit);
        const LQRPlannerResult ioc_plan = plan_with_lqr(
            x0,
            classical_ioc.tuned_planner_weights,
            demo_config.horizon,
            demo_config.dt_s,
            demo_config.constant_speed_mps,
            demo_config.target_lat_offset_m,
            demo_config.control_limit);
        const SyntheticDemoTrajectory expert_demo = generate_demo_for_state(x0, hlo, demo_config, rng);
        out_our.push_back(evaluate_cost(hlo, our_plan.states, our_plan.controls, our_plan.velocity_mps, demo_config.dt_s, demo_config.target_lat_offset_m));
        out_ioc.push_back(evaluate_cost(hlo, ioc_plan.states, ioc_plan.controls, ioc_plan.velocity_mps, demo_config.dt_s, demo_config.target_lat_offset_m));
        out_expert.push_back(expert_demo.true_hlo_cost);
    }

    EvaluationSummary summary{};
    summary.our_method_mean_gt_hlo_cost = mean(out_our);
    summary.classical_ioc_mean_gt_hlo_cost = mean(out_ioc);
    summary.expert_mean_gt_hlo_cost = mean(out_expert);
    summary.our_method_std_gt_hlo_cost = standard_deviation(out_our);
    summary.classical_ioc_std_gt_hlo_cost = standard_deviation(out_ioc);
    summary.expert_std_gt_hlo_cost = standard_deviation(out_expert);
    summary.n_test = n_test;
    return summary;
}

bool save_synthetic_demonstrations(
    const std::vector<SyntheticDemoTrajectory>& demos,
    const GroundTruthHLO& hlo,
    const DemoGenerationConfig& config,
    const std::string& path,
    std::string* error_message) {
    try {
        const std::filesystem::path output_path(path);
        std::filesystem::create_directories(output_path.parent_path());
        std::ofstream out(output_path);
        if (!out) {
            throw std::runtime_error("failed to open output JSON");
        }
        out << std::fixed << std::setprecision(10);
        out << "{\n";
        out << "  \"feature_names\": [";
        for (std::size_t index = 0; index < hlo.feature_names.size(); ++index) {
            if (index > 0) {
                out << ", ";
            }
            out << '"' << escape_json(hlo.feature_names[index]) << '"';
        }
        out << "],\n";
        out << "  \"omega\": [";
        for (std::size_t index = 0; index < hlo.omega.size(); ++index) {
            if (index > 0) {
                out << ", ";
            }
            out << hlo.omega[index];
        }
        out << "],\n";
        out << "  \"config\": {\n";
        out << "    \"dt_s\": " << config.dt_s << ",\n";
        out << "    \"horizon\": " << config.horizon << ",\n";
        out << "    \"constant_speed_mps\": " << config.constant_speed_mps << ",\n";
        out << "    \"control_limit\": " << config.control_limit << ",\n";
        out << "    \"n_demos\": " << config.n_demos << ",\n";
        out << "    \"seed\": " << config.seed << ",\n";
        out << "    \"restarts\": " << config.restarts << ",\n";
        out << "    \"target_lat_offset_m\": " << config.target_lat_offset_m << "\n";
        out << "  },\n";
        out << "  \"demos\": [\n";
        for (std::size_t demo_index = 0; demo_index < demos.size(); ++demo_index) {
            const SyntheticDemoTrajectory& demo = demos[demo_index];
            out << "    {\n";
            out << "      \"x0\": [" << demo.x0[0] << ", " << demo.x0[1] << ", " << demo.x0[2] << ", " << demo.x0[3] << "],\n";
            out << "      \"states\": [\n";
            for (std::size_t state_index = 0; state_index < demo.states.size(); ++state_index) {
                const State& state = demo.states[state_index];
                out << "        [" << state[0] << ", " << state[1] << ", " << state[2] << ", " << state[3] << "]";
                out << (state_index + 1 < demo.states.size() ? ",\n" : "\n");
            }
            out << "      ],\n";
            out << "      \"controls\": [";
            for (std::size_t control_index = 0; control_index < demo.controls.size(); ++control_index) {
                if (control_index > 0) {
                    out << ", ";
                }
                out << demo.controls[control_index];
            }
            out << "],\n";
            out << "      \"velocity_mps\": [";
            for (std::size_t velocity_index = 0; velocity_index < demo.velocity_mps.size(); ++velocity_index) {
                if (velocity_index > 0) {
                    out << ", ";
                }
                out << demo.velocity_mps[velocity_index];
            }
            out << "],\n";
            out << "      \"true_hlo_cost\": " << demo.true_hlo_cost << ",\n";
            out << "      \"feature_sums\": [";
            for (std::size_t feature_index = 0; feature_index < demo.feature_sums.size(); ++feature_index) {
                if (feature_index > 0) {
                    out << ", ";
                }
                out << demo.feature_sums[feature_index];
            }
            out << "]\n";
            out << "    }" << (demo_index + 1 < demos.size() ? ",\n" : "\n");
        }
        out << "  ]\n";
        out << "}\n";
        return true;
    } catch (const std::exception& ex) {
        if (error_message != nullptr) {
            *error_message = ex.what();
        }
        return false;
    }
}

}  // namespace ddioc