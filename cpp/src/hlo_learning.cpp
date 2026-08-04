#include "ddioc/hlo_learning.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace ddioc::ddioc {
namespace {

using Vector = std::vector<double>;

double wrap_to_pi(double angle) {
    return std::atan2(std::sin(angle), std::cos(angle));
}

double mean_value(const std::vector<double>& values) {
    if (values.empty()) {
        return 0.0;
    }
    return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
}

Vector project_simplex_with_lower_bounds(const Vector& omega_in) {
    Vector lower(9, 1e-5);
    lower[3] = 0.02;
    lower[6] = 0.02;
    lower[7] = 0.01;
    Vector omega = omega_in;
    const double remainder = 1.0 - std::accumulate(lower.begin(), lower.end(), 0.0);
    for (std::size_t i = 0; i < omega.size(); ++i) {
        omega[i] = std::max(0.0, omega[i] - lower[i]);
    }
    const double sum = std::accumulate(omega.begin(), omega.end(), 0.0);
    if (sum > 1e-12) {
        for (double& value : omega) {
            value *= remainder / sum;
        }
    } else {
        for (double& value : omega) {
            value = remainder / static_cast<double>(omega.size());
        }
    }
    for (std::size_t i = 0; i < omega.size(); ++i) {
        omega[i] += lower[i];
    }
    return omega;
}

Vector denormalize_state(const Vector& x_n, const Vector& mean, const Vector& stddev) {
    Vector out(x_n.size(), 0.0);
    for (std::size_t i = 0; i < x_n.size(); ++i) {
        out[i] = x_n[i] * stddev[i] + mean[i];
    }
    return out;
}

double denormalize_control(double u_n, const Vector& mean, const Vector& stddev) {
    return u_n * stddev[0] + mean[0];
}

Vector feature_vector_for_transition(
    const Vector& xk_n,
    const Vector& xkp1_n,
    double u_n,
    double v,
    double target_lat,
    double dt_s,
    const Vector& X_mean,
    const Vector& X_std,
    const Vector& U_mean,
    const Vector& U_std,
    double ref_psi_k,
    double ref_psi_kp1,
    double ref_kappa_k,
    double ref_kappa_kp1) {
    (void)ref_psi_k;
    const Vector xk = denormalize_state(xk_n, X_mean, X_std);
    const Vector xkp1 = denormalize_state(xkp1_n, X_mean, X_std);
    const double u = denormalize_control(u_n, U_mean, U_std);
    const double v_eff = std::max(v, 1e-3);
    const double v2 = v_eff * v_eff;
    const double v4 = v2 * v2;
    Vector phi(9, 0.0);
    const double lat_err = xkp1[0] - target_lat;
    const double lat_rate = (xkp1[0] - xk[0]) / std::max(dt_s, 1e-9);
    const double kappa_err = xkp1[2] - ref_kappa_kp1;
    const double kappa_err_prev = xk[2] - ref_kappa_k;
    const double delta_kappa_err = kappa_err - kappa_err_prev;
    const double delta_kappa_dot = xkp1[3] - xk[3];
    const double psi_err = wrap_to_pi(xkp1[1] - ref_psi_kp1);

    phi[0] = lat_err * lat_err;
    phi[1] = v2 * lat_rate * lat_rate;
    phi[2] = v4 * xkp1[2] * xkp1[2];
    phi[3] = v4 * xkp1[3] * xkp1[3];
    phi[4] = v4 * kappa_err * kappa_err;
    phi[5] = v4 * delta_kappa_err * delta_kappa_err;
    phi[6] = v4 * delta_kappa_dot * delta_kappa_dot;
    phi[7] = v4 * u * u;
    phi[8] = v2 * psi_err * psi_err;
    return phi;
}

Vector segment_feature_sums(
    const Matrix& Xseg,
    const Matrix& Useg,
    const std::vector<double>& Vseg,
    double target_lat,
    const std::vector<double>& ref_orientation,
    const std::vector<double>& ref_curvature,
    double dt_s,
    const Vector& X_mean,
    const Vector& X_std,
    const Vector& U_mean,
    const Vector& U_std) {
    Vector sums(9, 0.0);
    for (int k = 0; k < Useg.rows; ++k) {
        Vector xk = {Xseg(k, 0), Xseg(k, 1), Xseg(k, 2), Xseg(k, 3)};
        Vector xkp1 = {Xseg(k + 1, 0), Xseg(k + 1, 1), Xseg(k + 1, 2), Xseg(k + 1, 3)};
        const Vector phi = feature_vector_for_transition(
            xk,
            xkp1,
            Useg(k, 0),
            k + 1 < static_cast<int>(Vseg.size()) ? Vseg[static_cast<std::size_t>(k + 1)] : Vseg.back(),
            target_lat,
            dt_s,
            X_mean,
            X_std,
            U_mean,
            U_std,
            ref_orientation[static_cast<std::size_t>(k)],
            ref_orientation[static_cast<std::size_t>(k + 1)],
            ref_curvature[static_cast<std::size_t>(k)],
            ref_curvature[static_cast<std::size_t>(k + 1)]);
        for (std::size_t i = 0; i < sums.size(); ++i) {
            sums[i] += phi[i];
        }
    }
    return sums;
}

std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> parts;
    std::string current;
    bool in_quotes = false;
    for (char ch : line) {
        if (ch == '"') {
            in_quotes = !in_quotes;
        } else if (ch == ',' && !in_quotes) {
            parts.push_back(current);
            current.clear();
        } else {
            current.push_back(ch);
        }
    }
    parts.push_back(current);
    return parts;
}

}  // namespace

std::vector<double> compute_feature_scales(
    const std::vector<Matrix>& segments_X,
    const std::vector<Matrix>& segments_U,
    const std::vector<std::vector<double>>& segments_V,
    const std::vector<double>& segments_target_lat_m,
    const std::vector<std::vector<double>>& segments_ref_orientation,
    const std::vector<std::vector<double>>& segments_ref_curvature,
    double dt_s,
    const std::vector<double>& X_mean,
    const std::vector<double>& X_std,
    const std::vector<double>& U_mean,
    const std::vector<double>& U_std) {
    std::vector<std::vector<double>> per_feature(9);
    for (std::size_t seg = 0; seg < segments_X.size(); ++seg) {
        const Vector sums = segment_feature_sums(
            segments_X[seg],
            segments_U[seg],
            segments_V[seg],
            segments_target_lat_m[seg],
            segments_ref_orientation[seg],
            segments_ref_curvature[seg],
            dt_s,
            X_mean,
            X_std,
            U_mean,
            U_std);
        for (std::size_t i = 0; i < sums.size(); ++i) {
            per_feature[i].push_back(sums[i]);
        }
    }
    Vector scales(9, 1e-6);
    for (std::size_t i = 0; i < per_feature.size(); ++i) {
        auto values = per_feature[i];
        std::sort(values.begin(), values.end());
        if (!values.empty()) {
            const std::size_t mid = values.size() / 2;
            const double median = values.size() % 2 == 0 ? 0.5 * (values[mid - 1] + values[mid]) : values[mid];
            scales[i] = std::max(median, 1e-6);
        }
    }
    return scales;
}

HLOLearningResult learn_hlo_omega(
    const std::vector<Matrix>& segments_X,
    const std::vector<Matrix>& segments_U,
    const std::vector<std::vector<double>>& segments_V,
    const std::vector<double>& segments_target_lat_m,
    const std::vector<std::vector<double>>& segments_ref_orientation,
    const std::vector<std::vector<double>>& segments_ref_curvature,
    const KoopmanModel& model,
    double dt_s,
    const std::vector<double>& X_mean,
    const std::vector<double>& X_std,
    const std::vector<double>& U_mean,
    const std::vector<double>& U_std,
    const std::vector<double>& feature_scales,
    int ioc_every,
    int window) {
    (void)model;
    Vector omega(9, 1.0 / 9.0);
    omega = project_simplex_with_lower_bounds(omega);
    HLOLearningResult result{};
    result.feature_scales = feature_scales;
    result.omega_history.push_back(omega);

    std::vector<Vector> feature_buffer;
    const int stride = std::max(1, ioc_every);
    const int max_window = std::max(1, window);
    for (std::size_t seg = 0; seg < segments_X.size(); ++seg) {
        feature_buffer.push_back(segment_feature_sums(
            segments_X[seg],
            segments_U[seg],
            segments_V[seg],
            segments_target_lat_m[seg],
            segments_ref_orientation[seg],
            segments_ref_curvature[seg],
            dt_s,
            X_mean,
            X_std,
            U_mean,
            U_std));
        if (static_cast<int>(feature_buffer.size()) > max_window) {
            feature_buffer.erase(feature_buffer.begin());
        }
        if ((seg % static_cast<std::size_t>(stride) == 0U) || seg + 1 == segments_X.size()) {
            Vector aggregated(9, 0.0);
            for (const Vector& features : feature_buffer) {
                for (std::size_t i = 0; i < aggregated.size(); ++i) {
                    aggregated[i] += features[i] / feature_scales[i];
                }
            }
            double sum = std::accumulate(aggregated.begin(), aggregated.end(), 0.0);
            if (sum <= 1e-12) {
                sum = 1.0;
            }
            for (double& value : aggregated) {
                value /= sum;
            }
            omega = project_simplex_with_lower_bounds(aggregated);
            result.omega_history.push_back(omega);
        }
    }
    result.omega = omega;
    return result;
}

double eval_ioc_residual(
    const std::vector<Matrix>& segments_X,
    const std::vector<Matrix>& segments_U,
    const std::vector<std::vector<double>>& segments_V,
    const std::vector<double>& segments_target_lat_m,
    const std::vector<std::vector<double>>& segments_ref_orientation,
    const std::vector<std::vector<double>>& segments_ref_curvature,
    const std::vector<double>& omega,
    const std::vector<double>& feature_scales,
    double dt_s,
    const std::vector<double>& X_mean,
    const std::vector<double>& X_std,
    const std::vector<double>& U_mean,
    const std::vector<double>& U_std) {
    std::vector<double> costs;
    costs.reserve(segments_X.size());
    for (std::size_t seg = 0; seg < segments_X.size(); ++seg) {
        const Vector sums = segment_feature_sums(
            segments_X[seg],
            segments_U[seg],
            segments_V[seg],
            segments_target_lat_m[seg],
            segments_ref_orientation[seg],
            segments_ref_curvature[seg],
            dt_s,
            X_mean,
            X_std,
            U_mean,
            U_std);
        double cost = 0.0;
        for (std::size_t i = 0; i < omega.size(); ++i) {
            cost += omega[i] * (sums[i] / feature_scales[i]);
        }
        costs.push_back(cost);
    }
    return mean_value(costs);
}

std::vector<double> load_hlo_omega_from_learned_objective_json(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open learned objective json");
    }
    std::string line;
    while (std::getline(in, line)) {
        if (line.find("\"omega\"") == std::string::npos) {
            continue;
        }
        const auto left = line.find('[');
        const auto right = line.find(']');
        if (left == std::string::npos || right == std::string::npos || right <= left) {
            break;
        }
        const auto parts = split_csv_line(line.substr(left + 1, right - left - 1));
        Vector omega;
        for (const std::string& part : parts) {
            if (!part.empty()) {
                omega.push_back(std::stod(part));
            }
        }
        return omega;
    }
    throw std::runtime_error("omega field not found in learned objective json");
}

void save_learned_objective_json(const std::string& path, const std::vector<double>& omega, const std::vector<double>& feature_scales) {
    std::filesystem::path output(path);
    std::filesystem::create_directories(output.parent_path());
    std::ofstream out(output);
    if (!out) {
        throw std::runtime_error("failed to write learned objective json");
    }
    out << "{\n  \"omega\": [";
    for (std::size_t i = 0; i < omega.size(); ++i) {
        if (i > 0) {
            out << ", ";
        }
        out << omega[i];
    }
    out << "],\n  \"feature_scales\": [";
    for (std::size_t i = 0; i < feature_scales.size(); ++i) {
        if (i > 0) {
            out << ", ";
        }
        out << feature_scales[i];
    }
    out << "]\n}\n";
}

}  // namespace ddioc::ddioc