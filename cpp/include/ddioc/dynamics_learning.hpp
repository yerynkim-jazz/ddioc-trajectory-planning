#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ddioc::ddioc {

struct Matrix {
    int rows = 0;
    int cols = 0;
    std::vector<double> data;

    Matrix() = default;
    Matrix(int rows_in, int cols_in, double value = 0.0);

    double& operator()(int row, int col);
    double operator()(int row, int col) const;
};

struct DynamicsValidationConfig {
    std::uint32_t seed = 42;
    double dt = 0.05;
    int n_traj = 120;
    int horizon = 140;
    double train_ratio = 0.8;
    int poly_degree = 2;
    int seg_len = 20;
    int stride = 10;
    int tanh_features = 28;
};

struct TrajectoryDataset {
    std::vector<Matrix> X_train;
    std::vector<Matrix> U_train;
    std::vector<Matrix> X_test;
    std::vector<Matrix> U_test;
    std::vector<double> X_mean;
    std::vector<double> X_std;
    std::vector<double> U_mean;
    std::vector<double> U_std;
};

struct SegmentDataset {
    std::vector<Matrix> segments_X_train;
    std::vector<Matrix> segments_U_train;
    std::vector<Matrix> segments_X_test;
    std::vector<Matrix> segments_U_test;
};

struct PreparedSegments {
    std::vector<Matrix> traj_X_raw;
    std::vector<std::string> traj_ids;
    std::vector<Matrix> segments_X;
    std::vector<Matrix> segments_U;
    std::vector<std::vector<double>> segments_V;
    std::vector<std::vector<double>> segments_ref_orientation;
    std::vector<std::vector<double>> segments_ref_curvature;
    std::vector<double> segments_target_lat_m;
    std::vector<double> segments_dt_s;
    std::vector<Matrix> segments_X_test;
    std::vector<Matrix> segments_U_test;
    std::vector<std::vector<double>> segments_V_test;
    std::vector<std::vector<double>> segments_ref_orientation_test;
    std::vector<std::vector<double>> segments_ref_curvature_test;
    std::vector<double> segments_target_lat_m_test;
    std::vector<double> segments_dt_s_test;
    std::vector<std::string> ids_train;
    std::vector<double> X_mean;
    std::vector<double> X_std;
    std::vector<double> U_mean;
    std::vector<double> U_std;
    double dt_s = 0.1;
    int segment_stride = 0;
};

struct KoopmanModel {
    std::string lift;
    int n_state = 0;
    int n_psi = 0;
    int degree = 0;
    Matrix Kx;
    Matrix Ku;
    Matrix C;
    std::vector<std::vector<int>> exponents;
    Matrix tanh_weights;
    std::vector<double> tanh_bias;
};

struct MetricSummary {
    double mse_mean = 0.0;
    double rmse_mean = 0.0;
    int n_segments = 0;
};

TrajectoryDataset build_synthetic_validation_dataset(const DynamicsValidationConfig& config);
SegmentDataset build_validation_segments(const TrajectoryDataset& dataset, const DynamicsValidationConfig& config);
PreparedSegments prepare_train_test_segments_from_merged_csv(const std::string& merged_csv_path, int n_traj, int min_traj_len, double train_ratio, std::uint32_t split_seed, int seg_len, int segment_stride);
KoopmanModel learn_koopman_dynamics_poly(const std::vector<Matrix>& segments_X, const std::vector<Matrix>& segments_U, int n_state, int degree, double reg);
KoopmanModel learn_koopman_dynamics_tanh(const std::vector<Matrix>& segments_X, const std::vector<Matrix>& segments_U, int n_state, int tanh_features, std::uint32_t seed, double reg);
MetricSummary eval_koopman_one_step_mse(const std::vector<Matrix>& segments_X, const std::vector<Matrix>& segments_U, const KoopmanModel& model, int max_segments = 0);
MetricSummary eval_koopman_rollout_rmse(const std::vector<Matrix>& segments_X, const std::vector<Matrix>& segments_U, const KoopmanModel& model, int rollout_h, int max_segments = 0);
double eval_rollout_rmse_raw(const std::vector<Matrix>& segments_X, const std::vector<Matrix>& segments_U, const KoopmanModel& model, const std::vector<double>& X_mean, const std::vector<double>& X_std, int rollout_h);
std::vector<double> koopman_predict_next_xn(const KoopmanModel& model, const std::vector<double>& x_n, double u_n);
Matrix model_state_jacobian(const KoopmanModel& model, const std::vector<double>& x_n);

}  // namespace ddioc::ddioc