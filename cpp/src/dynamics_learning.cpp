#include "ddioc/dynamics_learning.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>

namespace ddioc::ddioc {
namespace {

using Vector = std::vector<double>;

double clip(double value, double lower, double upper) {
    return std::max(lower, std::min(value, upper));
}

Matrix identity(int size) {
    Matrix out(size, size, 0.0);
    for (int index = 0; index < size; ++index) {
        out(index, index) = 1.0;
    }
    return out;
}

Matrix transpose(const Matrix& matrix) {
    Matrix out(matrix.cols, matrix.rows, 0.0);
    for (int row = 0; row < matrix.rows; ++row) {
        for (int col = 0; col < matrix.cols; ++col) {
            out(col, row) = matrix(row, col);
        }
    }
    return out;
}

Matrix multiply(const Matrix& lhs, const Matrix& rhs) {
    if (lhs.cols != rhs.rows) {
        throw std::invalid_argument("matrix dimension mismatch");
    }
    Matrix out(lhs.rows, rhs.cols, 0.0);
    for (int row = 0; row < lhs.rows; ++row) {
        for (int inner = 0; inner < lhs.cols; ++inner) {
            const double value = lhs(row, inner);
            for (int col = 0; col < rhs.cols; ++col) {
                out(row, col) += value * rhs(inner, col);
            }
        }
    }
    return out;
}

Matrix inverse(Matrix matrix) {
    if (matrix.rows != matrix.cols) {
        throw std::invalid_argument("inverse requires square matrix");
    }
    const int n = matrix.rows;
    Matrix inv = identity(n);
    for (int pivot = 0; pivot < n; ++pivot) {
        int best_row = pivot;
        double best_value = std::abs(matrix(pivot, pivot));
        for (int row = pivot + 1; row < n; ++row) {
            const double candidate = std::abs(matrix(row, pivot));
            if (candidate > best_value) {
                best_value = candidate;
                best_row = row;
            }
        }
        if (best_value <= 1e-12) {
            throw std::runtime_error("matrix is singular");
        }
        if (best_row != pivot) {
            for (int col = 0; col < n; ++col) {
                std::swap(matrix(pivot, col), matrix(best_row, col));
                std::swap(inv(pivot, col), inv(best_row, col));
            }
        }
        const double diag = matrix(pivot, pivot);
        for (int col = 0; col < n; ++col) {
            matrix(pivot, col) /= diag;
            inv(pivot, col) /= diag;
        }
        for (int row = 0; row < n; ++row) {
            if (row == pivot) {
                continue;
            }
            const double factor = matrix(row, pivot);
            if (std::abs(factor) <= 1e-18) {
                continue;
            }
            for (int col = 0; col < n; ++col) {
                matrix(row, col) -= factor * matrix(pivot, col);
                inv(row, col) -= factor * inv(pivot, col);
            }
        }
    }
    return inv;
}

std::vector<Vector> split_train_test_indices(int n_items, double train_ratio, std::uint32_t seed) {
    std::vector<int> indices(static_cast<std::size_t>(n_items));
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(seed);
    std::shuffle(indices.begin(), indices.end(), rng);
    const int n_train = std::min(n_items - 1, std::max(1, static_cast<int>(std::round(train_ratio * static_cast<double>(n_items)))));
    Vector train;
    Vector test;
    for (int index = 0; index < n_items; ++index) {
        if (index < n_train) {
            train.push_back(static_cast<double>(indices[static_cast<std::size_t>(index)]));
        } else {
            test.push_back(static_cast<double>(indices[static_cast<std::size_t>(index)]));
        }
    }
    return {train, test};
}

void generate_exponents_recursive(int n_state, int total_degree, int idx, std::vector<int>& prefix, std::vector<std::vector<int>>& out) {
    if (idx == n_state - 1) {
        prefix[static_cast<std::size_t>(idx)] = total_degree;
        out.push_back(prefix);
        return;
    }
    for (int exponent = 0; exponent <= total_degree; ++exponent) {
        prefix[static_cast<std::size_t>(idx)] = exponent;
        generate_exponents_recursive(n_state, total_degree - exponent, idx + 1, prefix, out);
    }
}

std::vector<std::vector<int>> build_polynomial_exponents(int n_state, int degree) {
    std::vector<std::vector<int>> exponents;
    std::vector<int> prefix(static_cast<std::size_t>(n_state), 0);
    for (int total_degree = 0; total_degree <= degree; ++total_degree) {
        generate_exponents_recursive(n_state, total_degree, 0, prefix, exponents);
    }
    return exponents;
}

Vector get_row(const Matrix& matrix, int row) {
    Vector out(static_cast<std::size_t>(matrix.cols), 0.0);
    for (int col = 0; col < matrix.cols; ++col) {
        out[static_cast<std::size_t>(col)] = matrix(row, col);
    }
    return out;
}

void set_row(Matrix& matrix, int row, const Vector& values) {
    for (int col = 0; col < matrix.cols; ++col) {
        matrix(row, col) = values[static_cast<std::size_t>(col)];
    }
}

Vector polynomial_lift(const Vector& x, const std::vector<std::vector<int>>& exponents) {
    Vector psi(exponents.size(), 0.0);
    for (std::size_t term = 0; term < exponents.size(); ++term) {
        double prod = 1.0;
        for (std::size_t idx = 0; idx < x.size(); ++idx) {
            const int exponent = exponents[term][idx];
            if (exponent != 0) {
                prod *= std::pow(clip(x[idx], -10.0, 10.0), exponent);
            }
        }
        psi[term] = std::isfinite(prod) ? prod : 0.0;
    }
    return psi;
}

Vector tanh_lift(const Vector& x, const Matrix& weights, const Vector& bias) {
    Vector psi(x.size() + bias.size(), 0.0);
    for (std::size_t idx = 0; idx < x.size(); ++idx) {
        psi[idx] = x[idx];
    }
    for (int feature = 0; feature < weights.rows; ++feature) {
        double accum = bias[static_cast<std::size_t>(feature)];
        for (int col = 0; col < weights.cols; ++col) {
            accum += weights(feature, col) * x[static_cast<std::size_t>(col)];
        }
        psi[x.size() + static_cast<std::size_t>(feature)] = std::tanh(accum);
    }
    return psi;
}

Vector model_lift(const KoopmanModel& model, const Vector& x) {
    if (model.lift == "poly") {
        return polynomial_lift(x, model.exponents);
    }
    return tanh_lift(x, model.tanh_weights, model.tanh_bias);
}

Matrix model_lift_jacobian_internal(const KoopmanModel& model, const Vector& x) {
    if (model.lift == "poly") {
        Matrix J(model.n_psi, model.n_state, 0.0);
        for (int term = 0; term < model.n_psi; ++term) {
            const auto& exp = model.exponents[static_cast<std::size_t>(term)];
            for (int k = 0; k < model.n_state; ++k) {
                const int ek = exp[static_cast<std::size_t>(k)];
                if (ek == 0) {
                    continue;
                }
                double value = static_cast<double>(ek);
                for (int i = 0; i < model.n_state; ++i) {
                    const int ei = exp[static_cast<std::size_t>(i)];
                    if (ei == 0) {
                        continue;
                    }
                    if (i == k) {
                        if (ei > 1) {
                            value *= std::pow(x[static_cast<std::size_t>(i)], ei - 1);
                        }
                    } else {
                        value *= std::pow(x[static_cast<std::size_t>(i)], ei);
                    }
                }
                J(term, k) = value;
            }
        }
        return J;
    }

    Matrix J(model.n_psi, model.n_state, 0.0);
    for (int i = 0; i < model.n_state; ++i) {
        J(i, i) = 1.0;
    }
    for (int feature = 0; feature < model.tanh_weights.rows; ++feature) {
        double accum = model.tanh_bias[static_cast<std::size_t>(feature)];
        for (int col = 0; col < model.tanh_weights.cols; ++col) {
            accum += model.tanh_weights(feature, col) * x[static_cast<std::size_t>(col)];
        }
        const double deriv = 1.0 - std::pow(std::tanh(accum), 2.0);
        for (int col = 0; col < model.tanh_weights.cols; ++col) {
            J(model.n_state + feature, col) = deriv * model.tanh_weights(feature, col);
        }
    }
    return J;
}

Matrix build_psi_batch(const std::vector<Vector>& rows, const KoopmanModel& model) {
    Matrix out(static_cast<int>(rows.size()), model.n_psi, 0.0);
    for (int row = 0; row < static_cast<int>(rows.size()); ++row) {
        set_row(out, row, model_lift(model, rows[static_cast<std::size_t>(row)]));
    }
    return out;
}

KoopmanModel fit_koopman_model(const std::vector<Matrix>& segments_X, const std::vector<Matrix>& segments_U, KoopmanModel model, double reg) {
    std::vector<Vector> Xk_rows;
    std::vector<Vector> Xk1_rows;
    Vector Uk_values;
    for (std::size_t seg = 0; seg < segments_X.size(); ++seg) {
        const Matrix& Xseg = segments_X[seg];
        const Matrix& Useg = segments_U[seg];
        for (int row = 0; row < Xseg.rows - 1; ++row) {
            Xk_rows.push_back(get_row(Xseg, row));
            Xk1_rows.push_back(get_row(Xseg, row + 1));
            Uk_values.push_back(Useg(row, 0));
        }
    }

    const Matrix Psi_xk = build_psi_batch(Xk_rows, model);
    const Matrix Psi_xk1 = build_psi_batch(Xk1_rows, model);
    Matrix Z(Psi_xk.rows, model.n_psi + 1, 0.0);
    for (int row = 0; row < Z.rows; ++row) {
        for (int col = 0; col < model.n_psi; ++col) {
            Z(row, col) = Psi_xk(row, col);
        }
        Z(row, model.n_psi) = Uk_values[static_cast<std::size_t>(row)];
    }

    const Matrix Z_t = transpose(Z);
    Matrix G = multiply(Z_t, Z);
    for (int idx = 0; idx < G.rows; ++idx) {
        G(idx, idx) += reg;
    }
    const Matrix G_inv = inverse(G);

    const Matrix Psi_next_t = transpose(Psi_xk1);
    const Matrix K = multiply(multiply(Psi_next_t, Z), G_inv);

    model.Kx = Matrix(model.n_psi, model.n_psi, 0.0);
    model.Ku = Matrix(model.n_psi, 1, 0.0);
    for (int row = 0; row < model.n_psi; ++row) {
        for (int col = 0; col < model.n_psi; ++col) {
            model.Kx(row, col) = K(row, col);
        }
        model.Ku(row, 0) = K(row, model.n_psi);
    }

    Matrix Gpsi = multiply(transpose(Psi_xk), Psi_xk);
    for (int idx = 0; idx < Gpsi.rows; ++idx) {
        Gpsi(idx, idx) += reg;
    }
    const Matrix Gpsi_inv = inverse(Gpsi);

    Matrix Xk_mat(static_cast<int>(Xk_rows.size()), model.n_state, 0.0);
    for (int row = 0; row < Xk_mat.rows; ++row) {
        set_row(Xk_mat, row, Xk_rows[static_cast<std::size_t>(row)]);
    }
    model.C = multiply(multiply(transpose(Xk_mat), Psi_xk), Gpsi_inv);
    return model;
}

Vector predict_next_xn_internal(const KoopmanModel& model, const Vector& x_n, double u_n) {
    const Vector psi = model_lift(model, x_n);
    Vector psi_next(static_cast<std::size_t>(model.n_psi), 0.0);
    for (int row = 0; row < model.n_psi; ++row) {
        double accum = model.Ku(row, 0) * u_n;
        for (int col = 0; col < model.n_psi; ++col) {
            accum += model.Kx(row, col) * psi[static_cast<std::size_t>(col)];
        }
        psi_next[static_cast<std::size_t>(row)] = accum;
    }
    Vector x_out(static_cast<std::size_t>(model.n_state), 0.0);
    for (int row = 0; row < model.n_state; ++row) {
        for (int col = 0; col < model.n_psi; ++col) {
            x_out[static_cast<std::size_t>(row)] += model.C(row, col) * psi_next[static_cast<std::size_t>(col)];
        }
    }
    return x_out;
}

void simulate_nonlinear_trajectory(int T, double dt, std::uint32_t seed, Matrix& X, Matrix& U) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> lat(-1.0, 1.0);
    std::uniform_real_distribution<double> psi(-0.15, 0.15);
    std::uniform_real_distribution<double> kappa(-0.06, 0.06);
    std::uniform_real_distribution<double> kappa_dot(-0.08, 0.08);
    std::normal_distribution<double> normal(0.0, 1.0);

    X = Matrix(T, 4, 0.0);
    U = Matrix(T - 1, 1, 0.0);
    X(0, 0) = lat(rng);
    X(0, 1) = psi(rng);
    X(0, 2) = kappa(rng);
    X(0, 3) = kappa_dot(rng);

    Vector base_noise(static_cast<std::size_t>(T - 1), 0.0);
    for (double& value : base_noise) {
        value = normal(rng);
    }
    Vector smooth(base_noise.size(), 0.0);
    for (int idx = 0; idx < T - 1; ++idx) {
        double accum = 0.0;
        int count = 0;
        for (int offset = -3; offset <= 3; ++offset) {
            const int j = idx + offset;
            if (j >= 0 && j < T - 1) {
                accum += base_noise[static_cast<std::size_t>(j)];
                ++count;
            }
        }
        smooth[static_cast<std::size_t>(idx)] = accum / static_cast<double>(count);
    }

    for (int k = 0; k < T - 1; ++k) {
        const double lat_c = clip(X(k, 0), -6.0, 6.0);
        const double psi_c = clip(X(k, 1), -1.2, 1.2);
        const double kappa_c = clip(X(k, 2), -0.6, 0.6);
        const double kappa_dot_c = clip(X(k, 3), -1.0, 1.0);

        const double control = clip(0.18 * std::sin(0.08 * k + 0.7) + 0.12 * std::sin(0.027 * std::pow(static_cast<double>(k), 1.15)) + 0.08 * smooth[static_cast<std::size_t>(k)], -0.5, 0.5);
        U(k, 0) = control;
        const double v = clip(28.0 + 6.0 * std::sin(0.011 * k) + 2.0 * std::cos(0.023 * k), 18.0, 38.0);

        double lat_next = lat_c + dt * (v * std::tanh(psi_c) + 0.35 * std::tanh(1.5 * kappa_c) - 0.03 * lat_c);
        double psi_next = psi_c + dt * (v * (kappa_c + 0.08 * std::sin(1.2 * lat_c)) + 0.08 * std::sin(1.8 * psi_c) - 0.15 * psi_c);
        double kappa_next = kappa_c + dt * (kappa_dot_c + 0.20 * std::sin(clip(kappa_c * lat_c, -8.0, 8.0)) - 0.02 * std::pow(psi_c, 3.0) - 0.18 * kappa_c);
        double kappa_dot_next = kappa_dot_c + dt * (control + 0.12 * std::tanh(2.0 * kappa_dot_c) - 0.04 * std::sin(psi_c) * kappa_c - 0.25 * kappa_dot_c);

        lat_next += normal(rng) * 0.001;
        psi_next += normal(rng) * 0.0006;
        kappa_next += normal(rng) * 0.00035;
        kappa_dot_next += normal(rng) * 0.00035;

        X(k + 1, 0) = clip(lat_next, -8.0, 8.0);
        X(k + 1, 1) = clip(psi_next, -0.8, 0.8);
        X(k + 1, 2) = clip(kappa_next, -0.5, 0.5);
        X(k + 1, 3) = clip(kappa_dot_next, -0.8, 0.8);
    }
}

std::vector<double> compute_mean(const std::vector<Matrix>& trajectories) {
    std::vector<double> mean(static_cast<std::size_t>(trajectories.front().cols), 0.0);
    double count = 0.0;
    for (const Matrix& traj : trajectories) {
        for (int row = 0; row < traj.rows; ++row) {
            for (int col = 0; col < traj.cols; ++col) {
                mean[static_cast<std::size_t>(col)] += traj(row, col);
            }
            count += 1.0;
        }
    }
    for (double& value : mean) {
        value /= std::max(1.0, count);
    }
    return mean;
}

std::vector<double> compute_std(const std::vector<Matrix>& trajectories, const std::vector<double>& mean) {
    std::vector<double> var(mean.size(), 0.0);
    double count = 0.0;
    for (const Matrix& traj : trajectories) {
        for (int row = 0; row < traj.rows; ++row) {
            for (int col = 0; col < traj.cols; ++col) {
                const double diff = traj(row, col) - mean[static_cast<std::size_t>(col)];
                var[static_cast<std::size_t>(col)] += diff * diff;
            }
            count += 1.0;
        }
    }
    for (double& value : var) {
        value = std::sqrt(value / std::max(1.0, count)) + 1e-9;
    }
    return var;
}

Matrix normalize_matrix(const Matrix& matrix, const std::vector<double>& mean, const std::vector<double>& stddev) {
    Matrix out(matrix.rows, matrix.cols, 0.0);
    for (int row = 0; row < matrix.rows; ++row) {
        for (int col = 0; col < matrix.cols; ++col) {
            out(row, col) = (matrix(row, col) - mean[static_cast<std::size_t>(col)]) / stddev[static_cast<std::size_t>(col)];
        }
    }
    return out;
}

double mean_value(const std::vector<double>& values) {
    if (values.empty()) {
        return 0.0;
    }
    return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
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

struct TrajectoryRaw {
    std::string id;
    Matrix X;
    Matrix U;
    std::vector<double> V;
    std::vector<double> ref_orientation;
    std::vector<double> ref_curvature;
    std::vector<double> time_s;
};

}  // namespace

Matrix::Matrix(int rows_in, int cols_in, double value) : rows(rows_in), cols(cols_in), data(static_cast<std::size_t>(rows_in * cols_in), value) {}

double& Matrix::operator()(int row, int col) {
    return data[static_cast<std::size_t>(row * cols + col)];
}

double Matrix::operator()(int row, int col) const {
    return data[static_cast<std::size_t>(row * cols + col)];
}

TrajectoryDataset build_synthetic_validation_dataset(const DynamicsValidationConfig& config) {
    std::vector<Matrix> X_all;
    std::vector<Matrix> U_all;
    for (int idx = 0; idx < config.n_traj; ++idx) {
        Matrix X;
        Matrix U;
        simulate_nonlinear_trajectory(config.horizon, config.dt, config.seed + static_cast<std::uint32_t>(idx), X, U);
        X_all.push_back(std::move(X));
        U_all.push_back(std::move(U));
    }
    const auto split = split_train_test_indices(config.n_traj, config.train_ratio, config.seed);

    TrajectoryDataset dataset{};
    for (double index_value : split[0]) {
        const int index = static_cast<int>(index_value);
        dataset.X_train.push_back(X_all[static_cast<std::size_t>(index)]);
        dataset.U_train.push_back(U_all[static_cast<std::size_t>(index)]);
    }
    for (double index_value : split[1]) {
        const int index = static_cast<int>(index_value);
        dataset.X_test.push_back(X_all[static_cast<std::size_t>(index)]);
        dataset.U_test.push_back(U_all[static_cast<std::size_t>(index)]);
    }
    dataset.X_mean = compute_mean(dataset.X_train);
    dataset.X_std = compute_std(dataset.X_train, dataset.X_mean);
    dataset.U_mean = compute_mean(dataset.U_train);
    dataset.U_std = compute_std(dataset.U_train, dataset.U_mean);

    for (Matrix& X : dataset.X_train) {
        X = normalize_matrix(X, dataset.X_mean, dataset.X_std);
    }
    for (Matrix& U : dataset.U_train) {
        U = normalize_matrix(U, dataset.U_mean, dataset.U_std);
    }
    for (Matrix& X : dataset.X_test) {
        X = normalize_matrix(X, dataset.X_mean, dataset.X_std);
    }
    for (Matrix& U : dataset.U_test) {
        U = normalize_matrix(U, dataset.U_mean, dataset.U_std);
    }
    return dataset;
}

SegmentDataset build_validation_segments(const TrajectoryDataset& dataset, const DynamicsValidationConfig& config) {
    auto build = [&](const std::vector<Matrix>& X_list, const std::vector<Matrix>& U_list, std::vector<Matrix>& out_X, std::vector<Matrix>& out_U) {
        for (std::size_t idx = 0; idx < X_list.size(); ++idx) {
            const Matrix& X = X_list[idx];
            const Matrix& U = U_list[idx];
            if (X.rows < config.seg_len) {
                continue;
            }
            for (int start = 0; start <= X.rows - config.seg_len; start += config.stride) {
                Matrix seg_X(config.seg_len, X.cols, 0.0);
                Matrix seg_U(config.seg_len - 1, U.cols, 0.0);
                for (int row = 0; row < config.seg_len; ++row) {
                    for (int col = 0; col < X.cols; ++col) {
                        seg_X(row, col) = X(start + row, col);
                    }
                }
                for (int row = 0; row < config.seg_len - 1; ++row) {
                    seg_U(row, 0) = U(start + row, 0);
                }
                out_X.push_back(std::move(seg_X));
                out_U.push_back(std::move(seg_U));
            }
        }
    };

    SegmentDataset segments{};
    build(dataset.X_train, dataset.U_train, segments.segments_X_train, segments.segments_U_train);
    build(dataset.X_test, dataset.U_test, segments.segments_X_test, segments.segments_U_test);
    return segments;
}

KoopmanModel learn_koopman_dynamics_poly(const std::vector<Matrix>& segments_X, const std::vector<Matrix>& segments_U, int n_state, int degree, double reg) {
    KoopmanModel model{};
    model.lift = "poly";
    model.n_state = n_state;
    model.degree = degree;
    model.exponents = build_polynomial_exponents(n_state, degree);
    model.n_psi = static_cast<int>(model.exponents.size());
    return fit_koopman_model(segments_X, segments_U, std::move(model), reg);
}

KoopmanModel learn_koopman_dynamics_tanh(const std::vector<Matrix>& segments_X, const std::vector<Matrix>& segments_U, int n_state, int tanh_features, std::uint32_t seed, double reg) {
    KoopmanModel model{};
    model.lift = "tanh";
    model.n_state = n_state;
    model.n_psi = n_state + tanh_features;
    model.tanh_weights = Matrix(tanh_features, n_state, 0.0);
    model.tanh_bias.assign(static_cast<std::size_t>(tanh_features), 0.0);
    std::mt19937 rng(seed);
    std::normal_distribution<double> normal(0.0, 0.7);
    for (int row = 0; row < tanh_features; ++row) {
        model.tanh_bias[static_cast<std::size_t>(row)] = normal(rng);
        for (int col = 0; col < n_state; ++col) {
            model.tanh_weights(row, col) = normal(rng);
        }
    }
    return fit_koopman_model(segments_X, segments_U, std::move(model), reg);
}

MetricSummary eval_koopman_one_step_mse(const std::vector<Matrix>& segments_X, const std::vector<Matrix>& segments_U, const KoopmanModel& model, int max_segments) {
    std::vector<double> errors;
    const int n_segments = max_segments > 0 ? std::min<int>(max_segments, static_cast<int>(segments_X.size())) : static_cast<int>(segments_X.size());
    for (int seg = 0; seg < n_segments; ++seg) {
        const Matrix& Xseg = segments_X[static_cast<std::size_t>(seg)];
        const Matrix& Useg = segments_U[static_cast<std::size_t>(seg)];
        for (int row = 0; row < Useg.rows; ++row) {
            const Vector x_next = predict_next_xn_internal(model, get_row(Xseg, row), Useg(row, 0));
            double mse = 0.0;
            for (int idx = 0; idx < model.n_state; ++idx) {
                const double diff = x_next[static_cast<std::size_t>(idx)] - Xseg(row + 1, idx);
                mse += diff * diff;
            }
            errors.push_back(mse / static_cast<double>(model.n_state));
        }
    }
    MetricSummary summary{};
    summary.n_segments = n_segments;
    summary.mse_mean = mean_value(errors);
    summary.rmse_mean = std::sqrt(std::max(0.0, summary.mse_mean));
    return summary;
}

MetricSummary eval_koopman_rollout_rmse(const std::vector<Matrix>& segments_X, const std::vector<Matrix>& segments_U, const KoopmanModel& model, int rollout_h, int max_segments) {
    std::vector<double> rmses;
    const int n_segments = max_segments > 0 ? std::min<int>(max_segments, static_cast<int>(segments_X.size())) : static_cast<int>(segments_X.size());
    for (int seg = 0; seg < n_segments; ++seg) {
        const Matrix& Xseg = segments_X[static_cast<std::size_t>(seg)];
        const Matrix& Useg = segments_U[static_cast<std::size_t>(seg)];
        Vector x_n = get_row(Xseg, 0);
        const int H = std::min(rollout_h, Useg.rows);
        double se_sum = 0.0;
        int count = 0;
        for (int k = 0; k < H; ++k) {
            x_n = predict_next_xn_internal(model, x_n, Useg(k, 0));
            for (int idx = 0; idx < model.n_state; ++idx) {
                const double diff = x_n[static_cast<std::size_t>(idx)] - Xseg(k + 1, idx);
                se_sum += diff * diff;
            }
            ++count;
        }
        if (count > 0) {
            rmses.push_back(std::sqrt(se_sum / static_cast<double>(count * model.n_state)));
        }
    }
    MetricSummary summary{};
    summary.n_segments = n_segments;
    summary.rmse_mean = mean_value(rmses);
    summary.mse_mean = summary.rmse_mean * summary.rmse_mean;
    return summary;
}

double eval_rollout_rmse_raw(const std::vector<Matrix>& segments_X, const std::vector<Matrix>& segments_U, const KoopmanModel& model, const std::vector<double>& X_mean, const std::vector<double>& X_std, int rollout_h) {
    std::vector<double> rmses;
    for (std::size_t seg = 0; seg < segments_X.size(); ++seg) {
        const Matrix& Xseg = segments_X[seg];
        const Matrix& Useg = segments_U[seg];
        Vector x_n = get_row(Xseg, 0);
        const int H = std::min(rollout_h, Useg.rows);
        double se_sum = 0.0;
        int count = 0;
        for (int k = 0; k < H; ++k) {
            x_n = predict_next_xn_internal(model, x_n, Useg(k, 0));
            for (int idx = 0; idx < model.n_state; ++idx) {
                const double x_hat = x_n[static_cast<std::size_t>(idx)] * X_std[static_cast<std::size_t>(idx)] + X_mean[static_cast<std::size_t>(idx)];
                const double x_true = Xseg(k + 1, idx) * X_std[static_cast<std::size_t>(idx)] + X_mean[static_cast<std::size_t>(idx)];
                const double diff = x_hat - x_true;
                se_sum += diff * diff;
            }
            ++count;
        }
        if (count > 0) {
            rmses.push_back(std::sqrt(se_sum / static_cast<double>(count * model.n_state)));
        }
    }
    return mean_value(rmses);
}

std::vector<double> koopman_predict_next_xn(const KoopmanModel& model, const std::vector<double>& x_n, double u_n) {
    return predict_next_xn_internal(model, x_n, u_n);
}

Matrix model_state_jacobian(const KoopmanModel& model, const std::vector<double>& x_n) {
    const Matrix Dpsi = model_lift_jacobian_internal(model, x_n);
    return multiply(multiply(model.C, model.Kx), Dpsi);
}

PreparedSegments prepare_train_test_segments_from_merged_csv(const std::string& merged_csv_path, int n_traj, int min_traj_len, double train_ratio, std::uint32_t split_seed, int seg_len, int segment_stride) {
    std::ifstream in(merged_csv_path);
    if (!in) {
        throw std::runtime_error("failed to open merged CSV");
    }
    std::string header_line;
    if (!std::getline(in, header_line)) {
        throw std::runtime_error("empty merged CSV");
    }
    const auto headers = split_csv_line(header_line);
    std::map<std::string, int> index;
    for (int col = 0; col < static_cast<int>(headers.size()); ++col) {
        index[headers[static_cast<std::size_t>(col)]] = col;
    }
    const auto require = [&](const std::string& name) {
        if (!index.count(name)) {
            throw std::runtime_error("missing CSV column: " + name);
        }
        return index[name];
    };
    const int file_id_col = index.count("lc_id_label") ? index["lc_id_label"] : (index.count("lc_id") ? index["lc_id"] : require("lc_source"));
    const int time_col = require("time_s");
    const int lat_col = require("lateral_offset_m_sg");
    const int psi_col = require("target_orientation_rad_sg");
    const int kappa_col = require("target_curvature_1pm_sg");
    const int kappa_dot_col = require("target_curvature_1pm_dot");
    const int u_col = require("target_curvature_1pm_ddot");
    const int ref_ori_col = require("reference_orientation_sg");
    const int ref_kappa_col = require("reference_curvature_sg");
    const int speed_col = index.count("target_speed_mps") ? index["target_speed_mps"] : -1;

    std::map<std::string, std::vector<std::vector<std::string>>> grouped;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        auto parts = split_csv_line(line);
        if (static_cast<int>(parts.size()) != static_cast<int>(headers.size())) {
            continue;
        }
        grouped[parts[static_cast<std::size_t>(file_id_col)]].push_back(std::move(parts));
    }

    std::vector<TrajectoryRaw> trajectories;
    trajectories.reserve(grouped.size());
    for (auto& entry : grouped) {
        auto& rows = entry.second;
        std::sort(rows.begin(), rows.end(), [&](const auto& lhs, const auto& rhs) {
            return std::stod(lhs[static_cast<std::size_t>(time_col)]) < std::stod(rhs[static_cast<std::size_t>(time_col)]);
        });
        if (static_cast<int>(rows.size()) < min_traj_len) {
            continue;
        }
        TrajectoryRaw traj{};
        traj.id = entry.first;
        traj.X = Matrix(static_cast<int>(rows.size()), 4, 0.0);
        traj.U = Matrix(static_cast<int>(rows.size()), 1, 0.0);
        for (int row = 0; row < static_cast<int>(rows.size()); ++row) {
            const auto& r = rows[static_cast<std::size_t>(row)];
            traj.time_s.push_back(std::stod(r[static_cast<std::size_t>(time_col)]));
            traj.X(row, 0) = std::stod(r[static_cast<std::size_t>(lat_col)]);
            traj.X(row, 1) = std::stod(r[static_cast<std::size_t>(psi_col)]);
            traj.X(row, 2) = std::stod(r[static_cast<std::size_t>(kappa_col)]);
            traj.X(row, 3) = std::stod(r[static_cast<std::size_t>(kappa_dot_col)]);
            traj.U(row, 0) = std::stod(r[static_cast<std::size_t>(u_col)]);
            traj.ref_orientation.push_back(std::stod(r[static_cast<std::size_t>(ref_ori_col)]));
            traj.ref_curvature.push_back(std::stod(r[static_cast<std::size_t>(ref_kappa_col)]));
            traj.V.push_back(speed_col >= 0 ? std::stod(r[static_cast<std::size_t>(speed_col)]) : 1.0);
        }
        trajectories.push_back(std::move(traj));
        if (static_cast<int>(trajectories.size()) >= n_traj) {
            break;
        }
    }
    if (trajectories.empty()) {
        throw std::runtime_error("no valid trajectories loaded from merged CSV");
    }

    const auto split = split_train_test_indices(static_cast<int>(trajectories.size()), train_ratio, split_seed);
    std::vector<TrajectoryRaw> train_traj;
    std::vector<TrajectoryRaw> test_traj;
    for (double idx : split[0]) {
        train_traj.push_back(trajectories[static_cast<std::size_t>(static_cast<int>(idx))]);
    }
    for (double idx : split[1]) {
        test_traj.push_back(trajectories[static_cast<std::size_t>(static_cast<int>(idx))]);
    }

    std::vector<Matrix> X_train_raw;
    std::vector<Matrix> U_train_raw;
    for (const auto& traj : train_traj) {
        X_train_raw.push_back(traj.X);
        Matrix Utrim(std::max(0, traj.U.rows - 1), 1, 0.0);
        for (int row = 0; row < Utrim.rows; ++row) {
            Utrim(row, 0) = traj.U(row, 0);
        }
        U_train_raw.push_back(std::move(Utrim));
    }
    const auto X_mean = compute_mean(X_train_raw);
    const auto X_std = compute_std(X_train_raw, X_mean);
    const auto U_mean = compute_mean(U_train_raw);
    const auto U_std = compute_std(U_train_raw, U_mean);

    PreparedSegments out{};
    out.X_mean = X_mean;
    out.X_std = X_std;
    out.U_mean = U_mean;
    out.U_std = U_std;
    out.segment_stride = segment_stride;

    auto fill_segments = [&](const std::vector<TrajectoryRaw>& source, bool is_train) {
        for (const auto& traj : source) {
            const Matrix Xn = normalize_matrix(traj.X, X_mean, X_std);
            Matrix Utrim(std::max(0, traj.U.rows - 1), 1, 0.0);
            for (int row = 0; row < Utrim.rows; ++row) {
                Utrim(row, 0) = traj.U(row, 0);
            }
            const Matrix Un = normalize_matrix(Utrim, U_mean, U_std);
            const double target_lat = traj.X(traj.X.rows - 1, 0);
            double local_dt = 0.1;
            if (traj.time_s.size() >= 2) {
                std::vector<double> diffs;
                for (std::size_t i = 1; i < traj.time_s.size(); ++i) {
                    diffs.push_back(traj.time_s[i] - traj.time_s[i - 1]);
                }
                std::sort(diffs.begin(), diffs.end());
                local_dt = diffs[diffs.size() / 2];
            }
            out.dt_s = local_dt;
            out.traj_X_raw.push_back(traj.X);
            out.traj_ids.push_back(traj.id);
            if (is_train) {
                out.ids_train.push_back(traj.id);
            }
            if (Xn.rows < seg_len) {
                continue;
            }
            for (int start = 0; start <= Xn.rows - seg_len; start += segment_stride) {
                Matrix Xseg(seg_len, Xn.cols, 0.0);
                Matrix Useg(seg_len - 1, 1, 0.0);
                std::vector<double> Vseg(static_cast<std::size_t>(seg_len), 0.0);
                std::vector<double> ref_ori(static_cast<std::size_t>(seg_len), 0.0);
                std::vector<double> ref_kappa(static_cast<std::size_t>(seg_len), 0.0);
                for (int row = 0; row < seg_len; ++row) {
                    for (int col = 0; col < Xn.cols; ++col) {
                        Xseg(row, col) = Xn(start + row, col);
                    }
                    Vseg[static_cast<std::size_t>(row)] = traj.V[static_cast<std::size_t>(start + row)];
                    ref_ori[static_cast<std::size_t>(row)] = traj.ref_orientation[static_cast<std::size_t>(start + row)];
                    ref_kappa[static_cast<std::size_t>(row)] = traj.ref_curvature[static_cast<std::size_t>(start + row)];
                    if (row < seg_len - 1) {
                        Useg(row, 0) = Un(start + row, 0);
                    }
                }
                if (is_train) {
                    out.segments_X.push_back(std::move(Xseg));
                    out.segments_U.push_back(std::move(Useg));
                    out.segments_V.push_back(std::move(Vseg));
                    out.segments_ref_orientation.push_back(std::move(ref_ori));
                    out.segments_ref_curvature.push_back(std::move(ref_kappa));
                    out.segments_target_lat_m.push_back(target_lat);
                    out.segments_dt_s.push_back(local_dt);
                } else {
                    out.segments_X_test.push_back(std::move(Xseg));
                    out.segments_U_test.push_back(std::move(Useg));
                    out.segments_V_test.push_back(std::move(Vseg));
                    out.segments_ref_orientation_test.push_back(std::move(ref_ori));
                    out.segments_ref_curvature_test.push_back(std::move(ref_kappa));
                    out.segments_target_lat_m_test.push_back(target_lat);
                    out.segments_dt_s_test.push_back(local_dt);
                }
            }
        }
    };

    fill_segments(train_traj, true);
    fill_segments(test_traj, false);
    if (out.segments_X.empty()) {
        throw std::runtime_error("no train segments built from merged CSV");
    }
    return out;
}

}  // namespace ddioc::ddioc