from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

try:
	import matplotlib.pyplot as plt
except Exception:
	plt = None

try:
	from scipy.optimize import minimize
except Exception:
	minimize = None


ArrayLike = Union[Sequence[float], np.ndarray]


_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_OUTPUT_DIR = _REPO_ROOT / "outputs" / "examples" / "synthetic_ioc_demo"


@dataclass(frozen=True)
class DemoGenerationConfig:
	dt_s: float = 0.1
	horizon: int = 40
	constant_speed_mps: float = 8.0
	control_limit: float = 0.25
	n_demos: int = 8
	seed: int = 7
	restarts: int = 4
	target_lat_offset_m: float = 3.5
	output_json: str = field(default_factory=lambda: str(_DEFAULT_OUTPUT_DIR / "synthetic_demos.json"))
	output_plot: str = field(default_factory=lambda: str(_DEFAULT_OUTPUT_DIR / "synthetic_demos.png"))
	output_comparison_plot: str = field(default_factory=lambda: str(_DEFAULT_OUTPUT_DIR / "method_comparison.png"))


@dataclass(frozen=True)
class SyntheticDemoTrajectory:
	x0: np.ndarray
	states: np.ndarray
	controls: np.ndarray
	velocity_mps: np.ndarray
	true_hlo_cost: float
	feature_sums: np.ndarray


@dataclass(frozen=True)
class LQRPlannerWeights:
	w_d: float = 1.0
	w_a1: float = 0.2
	w_a2: float = 0.1
	w_a3: float = 0.1
	w_a4: float = 0.05


@dataclass(frozen=True)
class LQRPlannerResult:
	states: np.ndarray
	controls: np.ndarray
	velocity_mps: np.ndarray
	gains: np.ndarray


@dataclass(frozen=True)
class OurMethodConfig:
	pref_samples_per_demo: int = 12
	preference_noise_std: float = 0.03
	margin: float = 0.02
	omega_reg: float = 2e-2
	omega_entropy_reg: float = 5e-3
	omega_min: float = 0.02
	omega_max: float = 0.55
	hlo_restarts: int = 4
	planner_restarts: int = 6
	planner_maxiter: int = 120


@dataclass(frozen=True)
class LearnedHLOResult:
	omega: np.ndarray
	feature_scales: np.ndarray


@dataclass(frozen=True)
class OurMethodResult:
	learned_hlo: LearnedHLOResult
	tuned_planner_weights: LQRPlannerWeights
	planned_trajectories: list[LQRPlannerResult]
	mean_learned_hlo_cost: float


@dataclass(frozen=True)
class ClassicalIOCResult:
	tuned_planner_weights: LQRPlannerWeights
	planned_trajectories: list[LQRPlannerResult]
	mean_tracking_sse: float


@dataclass(frozen=True)
class EvaluationSummary:
	our_method_mean_gt_hlo_cost: float
	classical_ioc_mean_gt_hlo_cost: float
	expert_mean_gt_hlo_cost: float
	our_method_std_gt_hlo_cost: float
	classical_ioc_std_gt_hlo_cost: float
	expert_std_gt_hlo_cost: float
	n_test: int


FEATURE_NAMES: tuple[str, ...] = (
	"lat_err_sq",
	"lat_rate_sq_v2",
	"lat_err_quartic",
	"lat_psi_cross_abs_v2",
	"kappa_dot_abs_sq_v4",
	"kappa_err_sq_v4",
	"kappa_err_quartic_v4",
	"delta_kappa_err_sq_v4",
	"u_sq_v4",
	"psi_err_sq_v2",
)


def _wrap_to_pi(angle: ArrayLike) -> np.ndarray:
	angle_np = np.asarray(angle, dtype=float)
	return np.arctan2(np.sin(angle_np), np.cos(angle_np))


def project_to_simplex(weights: ArrayLike, minimum: float = 1e-8) -> np.ndarray:
	weights_np = np.asarray(weights, dtype=float).reshape(-1)
	if weights_np.size == 0:
		raise ValueError("weights must be non-empty")
	weights_np = np.maximum(weights_np, float(minimum))
	return weights_np / np.sum(weights_np)


def project_to_bounded_simplex(
	weights: ArrayLike,
	*,
	minimum: float = 1e-8,
	maximum: Optional[float] = None,
	iterations: int = 12,
) -> np.ndarray:
	weights_np = project_to_simplex(weights, minimum=minimum)
	if maximum is None:
		return weights_np
	max_value = float(maximum)
	if not np.isfinite(max_value) or max_value <= 0.0:
		raise ValueError("maximum must be a positive finite float")
	n = int(weights_np.shape[0])
	min_value = float(max(minimum, 0.0))
	if min_value * n > 1.0:
		raise ValueError("minimum is too large for the simplex")
	if max_value * n < 1.0:
		raise ValueError("maximum is too small for the simplex")
	for _ in range(max(1, int(iterations))):
		weights_np = np.clip(weights_np, min_value, max_value)
		s = float(np.sum(weights_np))
		if abs(s - 1.0) <= 1e-10:
			break
		free = (weights_np > min_value + 1e-12) & (weights_np < max_value - 1e-12)
		if not np.any(free):
			weights_np = weights_np / max(s, 1e-12)
			weights_np = np.clip(weights_np, min_value, max_value)
			break
		weights_np[free] += (1.0 - s) / float(np.sum(free))
	weights_np = np.clip(weights_np, min_value, max_value)
	return weights_np / np.sum(weights_np)


def _run_multistart_optimization(
	*,
	objective,
	initial: np.ndarray,
	rng: np.random.Generator,
	restarts: int,
	noise_scale: float,
	bounds: list[tuple[float | None, float | None]] | None,
	maxiter: int,
) -> tuple[np.ndarray, float]:
	initial_np = np.asarray(initial, dtype=float).reshape(-1)
	best_x = initial_np.copy()
	best_value = float(objective(best_x))
	for restart_index in range(max(1, int(restarts))):
		if restart_index == 0:
			start = initial_np.copy()
		else:
			start = initial_np + float(noise_scale) * rng.normal(size=initial_np.shape)
			if bounds is not None:
				for i, (lower, upper) in enumerate(bounds):
					if lower is not None:
						start[i] = max(start[i], float(lower))
					if upper is not None:
						start[i] = min(start[i], float(upper))
		if minimize is None:
			candidate_x = start
			candidate_value = float(objective(candidate_x))
		else:
			result = minimize(
				objective,
				start,
				method="L-BFGS-B",
				bounds=bounds,
				options={"maxiter": int(maxiter)},
			)
			candidate_x = np.asarray(result.x, dtype=float).reshape(-1)
			candidate_value = float(result.fun)
		if candidate_value < best_value:
			best_x = candidate_x
			best_value = candidate_value
	return best_x, best_value


def _lqr_weights_to_array(weights: LQRPlannerWeights) -> np.ndarray:
	return np.array([weights.w_d, weights.w_a1, weights.w_a2, weights.w_a3, weights.w_a4], dtype=float)


def _array_to_lqr_weights(values: ArrayLike) -> LQRPlannerWeights:
	arr = np.asarray(values, dtype=float).reshape(-1)
	if arr.shape[0] != 5:
		raise ValueError("LQR weight array must have length 5")
	return LQRPlannerWeights(
		w_d=float(arr[0]),
		w_a1=float(arr[1]),
		w_a2=float(arr[2]),
		w_a3=float(arr[3]),
		w_a4=float(arr[4]),
	)


@dataclass(frozen=True)
class GroundTruthHLO:
	feature_names: tuple[str, ...]
	omega: np.ndarray

	def evaluate_features(
		self,
		*,
		states: ArrayLike,
		controls: ArrayLike,
		velocity_mps: ArrayLike,
		dt_s: float,
		target_lat_offset_m: float = 0.0,
		ref_orientation_rad: Optional[ArrayLike] = None,
		ref_curvature_1pm: Optional[ArrayLike] = None,
	) -> np.ndarray:
		states_np = np.asarray(states, dtype=float)
		controls_np = np.asarray(controls, dtype=float).reshape(-1)
		velocity_np = np.asarray(velocity_mps, dtype=float).reshape(-1)
		if states_np.ndim != 2 or states_np.shape[1] != 4:
			raise ValueError("states must have shape (T + 1, 4) with [d, psi, kappa, kappa_dot]")
		if controls_np.shape[0] != states_np.shape[0] - 1:
			raise ValueError("controls must have length T when states has shape (T + 1, 4)")
		if velocity_np.shape[0] != controls_np.shape[0]:
			raise ValueError("velocity_mps must have length T")
		dt = float(dt_s)
		if not np.isfinite(dt) or dt <= 0.0:
			raise ValueError("dt_s must be positive")

		if ref_orientation_rad is None:
			ref_psi_np = np.zeros_like(controls_np)
		else:
			ref_psi_np = np.asarray(ref_orientation_rad, dtype=float).reshape(-1)
			if ref_psi_np.shape[0] != controls_np.shape[0]:
				raise ValueError("ref_orientation_rad must have length T")

		if ref_curvature_1pm is None:
			ref_kappa_np = np.zeros_like(controls_np)
		else:
			ref_kappa_np = np.asarray(ref_curvature_1pm, dtype=float).reshape(-1)
			if ref_kappa_np.shape[0] != controls_np.shape[0]:
				raise ValueError("ref_curvature_1pm must have length T")

		d_next = states_np[1:, 0]
		d_prev = states_np[:-1, 0]
		psi_next = states_np[1:, 1]
		kappa_next = states_np[1:, 2]
		kappa_prev = states_np[:-1, 2]
		kappa_dot_next = states_np[1:, 3]
		v_eff = np.maximum(velocity_np, 1e-3)

		lat_err = d_next - float(target_lat_offset_m)
		lat_rate = (d_next - d_prev) / dt
		kappa_err = kappa_next - ref_kappa_np
		delta_kappa_err = kappa_err - (kappa_prev - ref_kappa_np)
		psi_err = _wrap_to_pi(psi_next - ref_psi_np)

		features = np.array(
			[
				np.sum(lat_err**2),
				np.sum((v_eff**2) * (lat_rate**2)),
				np.sum(lat_err**4),
				np.sum((v_eff**2) * np.abs(lat_err * psi_err)),
				np.sum((v_eff**4) * (kappa_dot_next**2)),
				np.sum((v_eff**4) * (kappa_err**2)),
				np.sum((v_eff**4) * (kappa_err**4)),
				np.sum((v_eff**4) * (delta_kappa_err**2)),
				np.sum((v_eff**4) * (controls_np**2)),
				np.sum((v_eff**2) * (psi_err**2)),
			],
			dtype=float,
		)
		return features

	def evaluate_cost(self, **kwargs: object) -> float:
		features = self.evaluate_features(**kwargs)
		return float(np.dot(self.omega, features))


def rollout_linearized_kinematics(
	*,
	x0: ArrayLike,
	controls: ArrayLike,
	dt_s: float,
	constant_speed_mps: float,
) -> tuple[np.ndarray, np.ndarray]:
	controls_np = np.asarray(controls, dtype=float).reshape(-1)
	state = np.asarray(x0, dtype=float).reshape(-1)
	if state.shape[0] != 4:
		raise ValueError("x0 must have shape (4,) with [d, psi, kappa, kappa_dot]")
	dt = float(dt_s)
	v = float(constant_speed_mps)
	states = np.zeros((controls_np.size + 1, 4), dtype=float)
	states[0] = state
	for index, control in enumerate(controls_np):
		d, psi, kappa, kappa_dot = states[index]
		next_state = np.array(
			[
				d + dt * v * psi,
				psi + dt * v * kappa,
				kappa + dt * kappa_dot,
				kappa_dot + dt * float(control),
			],
			dtype=float,
		)
		states[index + 1] = next_state
	velocity = np.full(controls_np.size, v, dtype=float)
	return states, velocity


def build_discrete_dynamics_matrices(*, dt_s: float, speed_mps: float) -> tuple[np.ndarray, np.ndarray]:
	dt = float(dt_s)
	v = float(speed_mps)
	A_d = np.array(
		[
			[1.0, v * dt, 0.5 * v**2 * dt**2, (1.0 / 6.0) * v**2 * dt**3],
			[0.0, 1.0, v * dt, 0.5 * v * dt**2],
			[0.0, 0.0, 1.0, dt],
			[0.0, 0.0, 0.0, 1.0],
		],
		dtype=float,
	)
	B_d = np.array(
		[
			[(1.0 / 24.0) * v**2 * dt**4],
			[(1.0 / 6.0) * v * dt**3],
			[0.5 * dt**2],
			[dt],
		],
		dtype=float,
	)
	return A_d, B_d


def build_discrete_disturbance_matrix(*, dt_s: float, speed_mps: float) -> np.ndarray:
	dt = float(dt_s)
	v = float(speed_mps)
	return np.array(
		[
			[dt, -v * dt],
			[0.0, 0.0],
			[0.0, 0.0],
			[0.0, 0.0],
		],
		dtype=float,
	)


def _resolve_reference_vector(
	ref: Optional[ArrayLike],
	*,
	horizon: int,
	default_value: float,
	name: str,
) -> np.ndarray:
	if ref is None:
		return np.full(int(horizon) + 1, float(default_value), dtype=float)
	ref_np = np.asarray(ref, dtype=float).reshape(-1)
	if ref_np.shape[0] != int(horizon) + 1:
		raise ValueError(f"{name} must have length horizon + 1")
	return ref_np


def build_lqr_cost_matrices(
	*,
	weights: LQRPlannerWeights,
	speed_mps: float,
	v_bar: float = 0.1,
	terminal_scale: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	v_eff = max(float(speed_mps), float(v_bar))
	v2 = v_eff * v_eff
	v4 = v2 * v2
	Q = np.diag(
		[
			float(weights.w_d),
			float(weights.w_a1) * v2,
			float(weights.w_a2) * v4,
			float(weights.w_a3) * v4,
		],
	)
	R = np.array([[float(weights.w_a4) * v4]], dtype=float)
	Q_terminal = float(terminal_scale) * Q
	return Q, R, Q_terminal


def plan_with_lqr(
	*,
	x0: ArrayLike,
	weights: LQRPlannerWeights,
	horizon: int,
	dt_s: float,
	constant_speed_mps: float,
	target_lat_offset_m: float = 0.0,
	ref_orientation_rad: Optional[ArrayLike] = None,
	ref_curvature_1pm: Optional[ArrayLike] = None,
	ref_curvature_dot_1pm2: Optional[ArrayLike] = None,
	control_limit: Optional[float] = None,
) -> LQRPlannerResult:
	A_d, B_d = build_discrete_dynamics_matrices(dt_s=dt_s, speed_mps=constant_speed_mps)
	D_d = build_discrete_disturbance_matrix(dt_s=dt_s, speed_mps=constant_speed_mps)
	Q, R, Q_terminal = build_lqr_cost_matrices(weights=weights, speed_mps=constant_speed_mps)
	state0 = np.asarray(x0, dtype=float).reshape(-1)
	if state0.shape[0] != 4:
		raise ValueError("x0 must have shape (4,) with [d, psi, kappa, kappa_dot]")
	horizon_i = int(horizon)
	ref_d = np.full(horizon_i + 1, float(target_lat_offset_m), dtype=float)
	ref_psi = _resolve_reference_vector(
		ref_orientation_rad,
		horizon=horizon_i,
		default_value=0.0,
		name="ref_orientation_rad",
	)
	ref_kappa = _resolve_reference_vector(
		ref_curvature_1pm,
		horizon=horizon_i,
		default_value=0.0,
		name="ref_curvature_1pm",
	)
	ref_kappa_dot = _resolve_reference_vector(
		ref_curvature_dot_1pm2,
		horizon=horizon_i,
		default_value=0.0,
		name="ref_curvature_dot_1pm2",
	)
	ref_states = np.column_stack([ref_d, ref_psi, ref_kappa, ref_kappa_dot]).astype(float)
	error0 = state0 - ref_states[0]
	P = Q_terminal.copy()
	gains_rev = []
	for _ in range(horizon_i):
		middle = R + B_d.T @ P @ B_d
		K = np.linalg.solve(middle, B_d.T @ P @ A_d)
		gains_rev.append(K)
		P = Q + A_d.T @ P @ (A_d - B_d @ K)
	gains = np.asarray(gains_rev[::-1], dtype=float).reshape(horizon_i, 1, 4)

	errors = np.zeros((horizon_i + 1, 4), dtype=float)
	states = np.zeros((horizon_i + 1, 4), dtype=float)
	controls = np.zeros(horizon_i, dtype=float)
	errors[0] = error0
	states[0] = state0
	for index in range(horizon_i):
		control = float(-(gains[index] @ errors[index].reshape(-1, 1)).item())
		if control_limit is not None:
			control = float(np.clip(control, -float(control_limit), float(control_limit)))
		controls[index] = control
		theta_r_bar = 0.5 * float(ref_psi[index] + ref_psi[index + 1])
		z_k = np.array([0.0, theta_r_bar], dtype=float)
		affine_term = A_d @ ref_states[index] + D_d @ z_k - ref_states[index + 1]
		errors[index + 1] = A_d @ errors[index] + B_d[:, 0] * control + affine_term
		states[index + 1] = errors[index + 1] + ref_states[index + 1]
	velocity = np.full(horizon_i, float(constant_speed_mps), dtype=float)
	return LQRPlannerResult(states=states, controls=controls, velocity_mps=velocity, gains=gains)


def sample_initial_states(rng: np.random.Generator, count: int) -> np.ndarray:
	count = int(count)
	x0 = np.zeros((count, 4), dtype=float)
	x0[:, 0] = rng.uniform(-0.75, 0.75, size=count)
	x0[:, 1] = rng.uniform(-0.15, 0.15, size=count)
	x0[:, 2] = rng.uniform(-0.03, 0.03, size=count)
	x0[:, 3] = rng.uniform(-0.03, 0.03, size=count)
	return x0


def generate_demo_for_state(
	*,
	x0: ArrayLike,
	hlo: GroundTruthHLO,
	config: DemoGenerationConfig,
	rng: np.random.Generator,
) -> SyntheticDemoTrajectory:
	horizon = int(config.horizon)
	control_limit = float(config.control_limit)
	initial_controls = np.zeros(horizon, dtype=float)
	best_controls = initial_controls.copy()
	best_states, best_velocity = rollout_linearized_kinematics(
		x0=x0,
		controls=best_controls,
		dt_s=config.dt_s,
		constant_speed_mps=config.constant_speed_mps,
	)
	best_cost = hlo.evaluate_cost(
		states=best_states,
		controls=best_controls,
		velocity_mps=best_velocity,
		dt_s=config.dt_s,
		target_lat_offset_m=config.target_lat_offset_m,
	)

	def objective(control_seq: np.ndarray) -> float:
		controls = np.clip(np.asarray(control_seq, dtype=float).reshape(-1), -control_limit, control_limit)
		states, velocity = rollout_linearized_kinematics(
			x0=x0,
			controls=controls,
			dt_s=config.dt_s,
			constant_speed_mps=config.constant_speed_mps,
		)
		return hlo.evaluate_cost(
			states=states,
			controls=controls,
			velocity_mps=velocity,
			dt_s=config.dt_s,
			target_lat_offset_m=config.target_lat_offset_m,
		)

	for _ in range(max(1, int(config.restarts))):
		start = best_controls + 0.05 * rng.normal(size=best_controls.shape)
		start = np.clip(start, -control_limit, control_limit)
		if minimize is None:
			candidate_controls = start
			candidate_cost = objective(candidate_controls)
		else:
			result = minimize(
				objective,
				start,
				method="L-BFGS-B",
				bounds=[(-control_limit, control_limit)] * horizon,
				options={"maxiter": 200},
			)
			candidate_controls = np.clip(np.asarray(result.x, dtype=float).reshape(-1), -control_limit, control_limit)
			candidate_cost = float(result.fun)
		candidate_states, candidate_velocity = rollout_linearized_kinematics(
			x0=x0,
			controls=candidate_controls,
			dt_s=config.dt_s,
			constant_speed_mps=config.constant_speed_mps,
		)
		if candidate_cost < best_cost:
			best_controls = candidate_controls
			best_states = candidate_states
			best_velocity = candidate_velocity
			best_cost = candidate_cost

	feature_sums = hlo.evaluate_features(
		states=best_states,
		controls=best_controls,
		velocity_mps=best_velocity,
		dt_s=config.dt_s,
		target_lat_offset_m=config.target_lat_offset_m,
	)
	return SyntheticDemoTrajectory(
		x0=np.asarray(x0, dtype=float).copy(),
		states=best_states,
		controls=best_controls,
		velocity_mps=best_velocity,
		true_hlo_cost=float(best_cost),
		feature_sums=feature_sums,
	)


def generate_synthetic_demonstrations(
	*,
	hlo: GroundTruthHLO,
	config: DemoGenerationConfig,
) -> list[SyntheticDemoTrajectory]:
	rng = np.random.default_rng(int(config.seed))
	initial_states = sample_initial_states(rng, int(config.n_demos))
	return [
		generate_demo_for_state(x0=x0, hlo=hlo, config=config, rng=rng)
		for x0 in initial_states
	]


def learn_feature_scales_from_demos(demos: Sequence[SyntheticDemoTrajectory]) -> np.ndarray:
	stacked = np.vstack([demo.feature_sums for demo in demos])
	return np.maximum(np.median(stacked, axis=0), 1e-6)


def learn_hlo_from_demos(
	*,
	demos: Sequence[SyntheticDemoTrajectory],
	basis_hlo: GroundTruthHLO,
	demo_config: DemoGenerationConfig,
	method_config: OurMethodConfig,
	rng: np.random.Generator,
) -> LearnedHLOResult:
	feature_scales = learn_feature_scales_from_demos(demos)
	deltas: list[np.ndarray] = []
	uniform = np.full(len(FEATURE_NAMES), 1.0 / len(FEATURE_NAMES), dtype=float)
	for demo in demos:
		expert = np.asarray(demo.feature_sums, dtype=float) / feature_scales
		for _ in range(int(method_config.pref_samples_per_demo)):
			candidate_controls = demo.controls + float(method_config.preference_noise_std) * rng.normal(size=demo.controls.shape)
			candidate_controls = np.clip(candidate_controls, -float(demo_config.control_limit), float(demo_config.control_limit))
			candidate_states, candidate_velocity = rollout_linearized_kinematics(
				x0=demo.x0,
				controls=candidate_controls,
				dt_s=demo_config.dt_s,
				constant_speed_mps=demo_config.constant_speed_mps,
			)
			candidate = basis_hlo.evaluate_features(
				states=candidate_states,
				controls=candidate_controls,
				velocity_mps=candidate_velocity,
				dt_s=demo_config.dt_s,
				target_lat_offset_m=demo_config.target_lat_offset_m,
			) / feature_scales
			delta = candidate - expert
			delta_norm = float(np.linalg.norm(delta))
			if delta_norm > 1e-10:
				deltas.append(delta / delta_norm)
	if not deltas:
		return LearnedHLOResult(
			omega=uniform,
			feature_scales=feature_scales,
		)

	delta_matrix = np.vstack(deltas)
	n_features = int(delta_matrix.shape[1])

	def objective(raw_weights: np.ndarray) -> float:
		weights = project_to_bounded_simplex(
			raw_weights,
			minimum=float(method_config.omega_min),
			maximum=float(method_config.omega_max),
		)
		scores = delta_matrix @ weights
		loss = np.log1p(np.exp(float(method_config.margin) - scores)).mean()
		prior_reg = float(method_config.omega_reg) * float(np.sum((weights - uniform) ** 2))
		entropy = -float(np.sum(weights * np.log(np.maximum(weights, 1e-12))))
		entropy_reg = -float(method_config.omega_entropy_reg) * entropy
		return float(loss + prior_reg + entropy_reg)

	initial = uniform.copy()
	bounds = [(float(method_config.omega_min), float(method_config.omega_max))] * n_features
	best_raw, _ = _run_multistart_optimization(
		objective=objective,
		initial=initial,
		rng=rng,
		restarts=int(method_config.hlo_restarts),
		noise_scale=0.2,
		bounds=bounds,
		maxiter=200,
	)
	return LearnedHLOResult(
		omega=project_to_bounded_simplex(
			best_raw,
			minimum=float(method_config.omega_min),
			maximum=float(method_config.omega_max),
		),
		feature_scales=feature_scales,
	)


def learned_hlo_cost_for_planner_result(
	*,
	planner_result: LQRPlannerResult,
	learned_hlo: LearnedHLOResult,
	basis_hlo: GroundTruthHLO,
	demo_config: DemoGenerationConfig,
) -> float:
	features = basis_hlo.evaluate_features(
		states=planner_result.states,
		controls=planner_result.controls,
		velocity_mps=planner_result.velocity_mps,
		dt_s=demo_config.dt_s,
		target_lat_offset_m=demo_config.target_lat_offset_m,
	)
	normalized = features / learned_hlo.feature_scales
	return float(np.dot(learned_hlo.omega, normalized))


def tune_lqr_weights_with_learned_hlo(
	*,
	demos: Sequence[SyntheticDemoTrajectory],
	learned_hlo: LearnedHLOResult,
	basis_hlo: GroundTruthHLO,
	demo_config: DemoGenerationConfig,
	method_config: OurMethodConfig,
	rng: np.random.Generator,
	initial_weights: Optional[LQRPlannerWeights] = None,
) -> LQRPlannerWeights:
	base = initial_weights if initial_weights is not None else get_default_lqr_planner_weights()
	initial_log = np.log(np.maximum(_lqr_weights_to_array(base), 1e-8))
	bounds = [(-8.0, 4.0)] * 5

	def objective(log_theta: np.ndarray) -> float:
		weights = _array_to_lqr_weights(np.exp(np.asarray(log_theta, dtype=float)))
		costs = []
		for demo in demos:
			planner_result = plan_with_lqr(
				x0=demo.x0,
				weights=weights,
				horizon=demo_config.horizon,
				dt_s=demo_config.dt_s,
				constant_speed_mps=demo_config.constant_speed_mps,
				target_lat_offset_m=demo_config.target_lat_offset_m,
				control_limit=demo_config.control_limit,
			)
			costs.append(
				learned_hlo_cost_for_planner_result(
					planner_result=planner_result,
					learned_hlo=learned_hlo,
					basis_hlo=basis_hlo,
					demo_config=demo_config,
				)
			)
		return float(np.mean(costs))

	best_log, _ = _run_multistart_optimization(
		objective=objective,
		initial=initial_log,
		rng=rng,
		restarts=int(method_config.planner_restarts),
		noise_scale=0.35,
		bounds=bounds,
		maxiter=int(method_config.planner_maxiter),
	)
	return _array_to_lqr_weights(np.exp(best_log))


def run_our_method(
	*,
	demos: Sequence[SyntheticDemoTrajectory],
	basis_hlo: GroundTruthHLO,
	demo_config: DemoGenerationConfig,
	method_config: Optional[OurMethodConfig] = None,
	initial_planner_weights: Optional[LQRPlannerWeights] = None,
) -> OurMethodResult:
	config = method_config if method_config is not None else OurMethodConfig()
	rng = np.random.default_rng(int(demo_config.seed) + 101)
	learned_hlo = learn_hlo_from_demos(
		demos=demos,
		basis_hlo=basis_hlo,
		demo_config=demo_config,
		method_config=config,
		rng=rng,
	)
	tuned_weights = tune_lqr_weights_with_learned_hlo(
		demos=demos,
		learned_hlo=learned_hlo,
		basis_hlo=basis_hlo,
		demo_config=demo_config,
		method_config=config,
		rng=rng,
		initial_weights=initial_planner_weights,
	)
	planned_trajectories = [
		plan_with_lqr(
			x0=demo.x0,
			weights=tuned_weights,
			horizon=demo_config.horizon,
			dt_s=demo_config.dt_s,
			constant_speed_mps=demo_config.constant_speed_mps,
			target_lat_offset_m=demo_config.target_lat_offset_m,
			control_limit=demo_config.control_limit,
		)
		for demo in demos
	]
	learned_costs = [
		learned_hlo_cost_for_planner_result(
			planner_result=planner_result,
			learned_hlo=learned_hlo,
			basis_hlo=basis_hlo,
			demo_config=demo_config,
		)
		for planner_result in planned_trajectories
	]
	return OurMethodResult(
		learned_hlo=learned_hlo,
		tuned_planner_weights=tuned_weights,
		planned_trajectories=planned_trajectories,
		mean_learned_hlo_cost=float(np.mean(learned_costs)),
	)


def tracking_sse_for_planner_result(
	*,
	planner_result: LQRPlannerResult,
	demo: SyntheticDemoTrajectory,
	control_weight: float = 0.1,
) -> float:
	state_error = float(np.mean((planner_result.states - demo.states) ** 2))
	control_error = float(np.mean((planner_result.controls - demo.controls) ** 2))
	return float(state_error + float(control_weight) * control_error)


def tune_lqr_weights_with_tracking_sse(
	*,
	demos: Sequence[SyntheticDemoTrajectory],
	demo_config: DemoGenerationConfig,
	rng: np.random.Generator,
	planner_restarts: int,
	planner_maxiter: int,
	initial_weights: Optional[LQRPlannerWeights] = None,
	control_weight: float = 0.1,
) -> LQRPlannerWeights:
	base = initial_weights if initial_weights is not None else get_default_lqr_planner_weights()
	initial_log = np.log(np.maximum(_lqr_weights_to_array(base), 1e-8))
	bounds = [(-8.0, 4.0)] * 5

	def objective(log_theta: np.ndarray) -> float:
		weights = _array_to_lqr_weights(np.exp(np.asarray(log_theta, dtype=float)))
		errors = []
		for demo in demos:
			planner_result = plan_with_lqr(
				x0=demo.x0,
				weights=weights,
				horizon=demo_config.horizon,
				dt_s=demo_config.dt_s,
				constant_speed_mps=demo_config.constant_speed_mps,
				target_lat_offset_m=demo_config.target_lat_offset_m,
				control_limit=demo_config.control_limit,
			)
			errors.append(
				tracking_sse_for_planner_result(
					planner_result=planner_result,
					demo=demo,
					control_weight=control_weight,
				)
			)
		return float(np.mean(errors))

	best_log, _ = _run_multistart_optimization(
		objective=objective,
		initial=initial_log,
		rng=rng,
		restarts=int(planner_restarts),
		noise_scale=0.35,
		bounds=bounds,
		maxiter=int(planner_maxiter),
	)
	return _array_to_lqr_weights(np.exp(best_log))


def run_classical_ioc_benchmark(
	*,
	demos: Sequence[SyntheticDemoTrajectory],
	demo_config: DemoGenerationConfig,
	method_config: Optional[OurMethodConfig] = None,
	initial_planner_weights: Optional[LQRPlannerWeights] = None,
) -> ClassicalIOCResult:
	config = method_config if method_config is not None else OurMethodConfig()
	rng = np.random.default_rng(int(demo_config.seed) + 202)
	tuned_weights = tune_lqr_weights_with_tracking_sse(
		demos=demos,
		demo_config=demo_config,
		rng=rng,
		planner_restarts=int(config.planner_restarts),
		planner_maxiter=int(config.planner_maxiter),
		initial_weights=initial_planner_weights,
	)
	planned_trajectories = [
		plan_with_lqr(
			x0=demo.x0,
			weights=tuned_weights,
			horizon=demo_config.horizon,
			dt_s=demo_config.dt_s,
			constant_speed_mps=demo_config.constant_speed_mps,
			target_lat_offset_m=demo_config.target_lat_offset_m,
			control_limit=demo_config.control_limit,
		)
		for demo in demos
	]
	tracking_errors = [
		tracking_sse_for_planner_result(planner_result=planner_result, demo=demo)
		for planner_result, demo in zip(planned_trajectories, demos)
	]
	return ClassicalIOCResult(
		tuned_planner_weights=tuned_weights,
		planned_trajectories=planned_trajectories,
		mean_tracking_sse=float(np.mean(tracking_errors)),
	)


def evaluate_on_unseen_initial_states(
	*,
	hlo: GroundTruthHLO,
	demo_config: DemoGenerationConfig,
	our_method: OurMethodResult,
	classical_ioc: ClassicalIOCResult,
	n_test: int = 24,
	seed_offset: int = 1000,
) -> EvaluationSummary:
	rng = np.random.default_rng(int(demo_config.seed) + int(seed_offset))
	test_x0 = sample_initial_states(rng, int(n_test))
	out_our: list[float] = []
	out_ioc: list[float] = []
	out_expert: list[float] = []
	for x0 in test_x0:
		our_plan = plan_with_lqr(
			x0=x0,
			weights=our_method.tuned_planner_weights,
			horizon=demo_config.horizon,
			dt_s=demo_config.dt_s,
			constant_speed_mps=demo_config.constant_speed_mps,
			target_lat_offset_m=demo_config.target_lat_offset_m,
			control_limit=demo_config.control_limit,
		)
		ioc_plan = plan_with_lqr(
			x0=x0,
			weights=classical_ioc.tuned_planner_weights,
			horizon=demo_config.horizon,
			dt_s=demo_config.dt_s,
			constant_speed_mps=demo_config.constant_speed_mps,
			target_lat_offset_m=demo_config.target_lat_offset_m,
			control_limit=demo_config.control_limit,
		)
		expert_demo = generate_demo_for_state(
			x0=x0,
			hlo=hlo,
			config=demo_config,
			rng=rng,
		)
		out_our.append(
			hlo.evaluate_cost(
				states=our_plan.states,
				controls=our_plan.controls,
				velocity_mps=our_plan.velocity_mps,
				dt_s=demo_config.dt_s,
				target_lat_offset_m=demo_config.target_lat_offset_m,
			)
		)
		out_ioc.append(
			hlo.evaluate_cost(
				states=ioc_plan.states,
				controls=ioc_plan.controls,
				velocity_mps=ioc_plan.velocity_mps,
				dt_s=demo_config.dt_s,
				target_lat_offset_m=demo_config.target_lat_offset_m,
			)
		)
		out_expert.append(float(expert_demo.true_hlo_cost))
	return EvaluationSummary(
		our_method_mean_gt_hlo_cost=float(np.mean(out_our)),
		classical_ioc_mean_gt_hlo_cost=float(np.mean(out_ioc)),
		expert_mean_gt_hlo_cost=float(np.mean(out_expert)),
		our_method_std_gt_hlo_cost=float(np.std(out_our)),
		classical_ioc_std_gt_hlo_cost=float(np.std(out_ioc)),
		expert_std_gt_hlo_cost=float(np.std(out_expert)),
		n_test=int(n_test),
	)


def save_synthetic_demonstrations(
	*,
	demos: Sequence[SyntheticDemoTrajectory],
	hlo: GroundTruthHLO,
	config: DemoGenerationConfig,
	path: Union[str, Path],
) -> Path:
	output_path = Path(path)
	payload = {
		"feature_names": list(hlo.feature_names),
		"omega": [float(x) for x in hlo.omega.tolist()],
		"config": {
			"dt_s": float(config.dt_s),
			"horizon": int(config.horizon),
			"constant_speed_mps": float(config.constant_speed_mps),
			"control_limit": float(config.control_limit),
			"n_demos": int(config.n_demos),
			"seed": int(config.seed),
			"restarts": int(config.restarts),
			"target_lat_offset_m": float(config.target_lat_offset_m),
		},
		"demos": [
			{
				"x0": [float(x) for x in demo.x0.tolist()],
				"states": [[float(v) for v in row] for row in demo.states.tolist()],
				"controls": [float(x) for x in demo.controls.tolist()],
				"velocity_mps": [float(x) for x in demo.velocity_mps.tolist()],
				"true_hlo_cost": float(demo.true_hlo_cost),
				"feature_sums": [float(x) for x in demo.feature_sums.tolist()],
			}
			for demo in demos
		],
	}
	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
	return output_path


def plot_synthetic_demonstrations(
	*,
	demos: Sequence[SyntheticDemoTrajectory],
	config: DemoGenerationConfig,
	path: Union[str, Path],
) -> Optional[Path]:
	if plt is None:
		return None
	output_path = Path(path)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
	for index, demo in enumerate(demos):
		time_states = np.arange(demo.states.shape[0], dtype=float) * float(config.dt_s)
		time_controls = np.arange(demo.controls.shape[0], dtype=float) * float(config.dt_s)
		label = f"demo_{index}" if index < 5 else None
		axes[0].plot(time_states, demo.states[:, 0], alpha=0.8, linewidth=1.5, label=label)
		axes[1].plot(time_controls, demo.controls, alpha=0.8, linewidth=1.2)
	axes[0].axhline(float(config.target_lat_offset_m), color="black", linestyle="--", linewidth=1.0, label="target")
	axes[0].set_ylabel("lateral offset d [m]")
	axes[1].set_ylabel("control")
	axes[1].set_xlabel("time [s]")
	axes[0].set_title("Synthetic demonstrations from GT HLO")
	axes[0].legend(loc="best")
	fig.tight_layout()
	fig.savefig(output_path, dpi=160)
	plt.close(fig)
	return output_path


def plot_method_comparison_on_unseen_initial_states(
	*,
	hlo: GroundTruthHLO,
	demo_config: DemoGenerationConfig,
	our_method: OurMethodResult,
	classical_ioc: ClassicalIOCResult,
	path: Union[str, Path],
	n_plot: int = 4,
	seed_offset: int = 1000,
) -> Optional[Path]:
	if plt is None:
		return None
	rng = np.random.default_rng(int(demo_config.seed) + int(seed_offset))
	test_x0 = sample_initial_states(rng, int(n_plot))
	output_path = Path(path)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
	for index, x0 in enumerate(test_x0):
		expert_demo = generate_demo_for_state(x0=x0, hlo=hlo, config=demo_config, rng=rng)
		our_plan = plan_with_lqr(
			x0=x0,
			weights=our_method.tuned_planner_weights,
			horizon=demo_config.horizon,
			dt_s=demo_config.dt_s,
			constant_speed_mps=demo_config.constant_speed_mps,
			target_lat_offset_m=demo_config.target_lat_offset_m,
			control_limit=demo_config.control_limit,
		)
		ioc_plan = plan_with_lqr(
			x0=x0,
			weights=classical_ioc.tuned_planner_weights,
			horizon=demo_config.horizon,
			dt_s=demo_config.dt_s,
			constant_speed_mps=demo_config.constant_speed_mps,
			target_lat_offset_m=demo_config.target_lat_offset_m,
			control_limit=demo_config.control_limit,
		)
		time_states = np.arange(expert_demo.states.shape[0], dtype=float) * float(demo_config.dt_s)
		time_controls = np.arange(expert_demo.controls.shape[0], dtype=float) * float(demo_config.dt_s)
		expert_label = "expert" if index == 0 else None
		our_label = "our method" if index == 0 else None
		ioc_label = "classical IOC" if index == 0 else None
		axes[0].plot(time_states, expert_demo.states[:, 0], color="#222222", linestyle="--", linewidth=1.4, alpha=0.9, label=expert_label)
		axes[0].plot(time_states, our_plan.states[:, 0], color="#0b6e4f", linewidth=1.6, alpha=0.85, label=our_label)
		axes[0].plot(time_states, ioc_plan.states[:, 0], color="#b02e0c", linewidth=1.4, alpha=0.8, label=ioc_label)
		axes[1].plot(time_controls, expert_demo.controls, color="#222222", linestyle="--", linewidth=1.2, alpha=0.9)
		axes[1].plot(time_controls, our_plan.controls, color="#0b6e4f", linewidth=1.4, alpha=0.85)
		axes[1].plot(time_controls, ioc_plan.controls, color="#b02e0c", linewidth=1.2, alpha=0.8)
	axes[0].axhline(float(demo_config.target_lat_offset_m), color="#666666", linestyle=":", linewidth=1.0, label="target")
	axes[0].set_ylabel("lateral offset d [m]")
	axes[1].set_ylabel("control")
	axes[1].set_xlabel("time [s]")
	axes[0].set_title("Trajectory comparison on unseen initial states")
	axes[0].legend(loc="best")
	fig.tight_layout()
	fig.savefig(output_path, dpi=160)
	plt.close(fig)
	return output_path


GT_HLO = GroundTruthHLO(
	feature_names=FEATURE_NAMES,
	omega=project_to_simplex(
		np.array([
			0.14,
			0.08,
			0.15,
			0.10,
			0.11,
			0.08,
			0.10,
			0.09,
			0.07,
			0.08,
		], dtype=float)
	),
)


def get_ground_truth_hlo() -> GroundTruthHLO:
	return GT_HLO


def get_default_lqr_planner_weights() -> LQRPlannerWeights:
	return LQRPlannerWeights()


def main() -> None:
	hlo = get_ground_truth_hlo()
	print("GT HLO")
	for name, weight in zip(hlo.feature_names, hlo.omega):
		print(f"- {name}: {float(weight):.6f}")
	planner_weights = get_default_lqr_planner_weights()
	print("\nLQR planner weights")
	print(f"- w_d: {planner_weights.w_d:.6f}")
	print(f"- w_a1: {planner_weights.w_a1:.6f}")
	print(f"- w_a2: {planner_weights.w_a2:.6f}")
	print(f"- w_a3: {planner_weights.w_a3:.6f}")
	print(f"- w_a4: {planner_weights.w_a4:.6f}")
	config = DemoGenerationConfig()
	demos = generate_synthetic_demonstrations(hlo=hlo, config=config)
	our_method = run_our_method(
		demos=demos,
		basis_hlo=hlo,
		demo_config=config,
		initial_planner_weights=planner_weights,
	)
	classical_ioc = run_classical_ioc_benchmark(
		demos=demos,
		demo_config=config,
		initial_planner_weights=planner_weights,
	)
	evaluation = evaluate_on_unseen_initial_states(
		hlo=hlo,
		demo_config=config,
		our_method=our_method,
		classical_ioc=classical_ioc,
	)
	print("\nSynthetic demonstrations")
	print(f"- count: {len(demos)}")
	if demos:
		first_demo = demos[0]
		print(f"- first demo x0: {first_demo.x0.tolist()}")
		print(f"- first demo cost: {first_demo.true_hlo_cost:.6f}")
		print(f"- first final state: {first_demo.states[-1].tolist()}")
		planner_result = plan_with_lqr(
			x0=first_demo.x0,
			weights=planner_weights,
			horizon=config.horizon,
			dt_s=config.dt_s,
			constant_speed_mps=config.constant_speed_mps,
			target_lat_offset_m=config.target_lat_offset_m,
			control_limit=config.control_limit,
		)
		print(f"- first planner final state: {planner_result.states[-1].tolist()}")
		print(f"- first planner first control: {float(planner_result.controls[0]):.6f}")
	print("\nOur method")
	print(f"- learned omega: {our_method.learned_hlo.omega.tolist()}")
	print(f"- feature scales: {our_method.learned_hlo.feature_scales.tolist()}")
	print(f"- tuned w_d: {our_method.tuned_planner_weights.w_d:.6f}")
	print(f"- tuned w_a1: {our_method.tuned_planner_weights.w_a1:.6f}")
	print(f"- tuned w_a2: {our_method.tuned_planner_weights.w_a2:.6f}")
	print(f"- tuned w_a3: {our_method.tuned_planner_weights.w_a3:.6f}")
	print(f"- tuned w_a4: {our_method.tuned_planner_weights.w_a4:.6f}")
	print(f"- mean learned HLO cost: {our_method.mean_learned_hlo_cost:.6f}")
	print("\nClassical IOC benchmark")
	print(f"- tuned w_d: {classical_ioc.tuned_planner_weights.w_d:.6f}")
	print(f"- tuned w_a1: {classical_ioc.tuned_planner_weights.w_a1:.6f}")
	print(f"- tuned w_a2: {classical_ioc.tuned_planner_weights.w_a2:.6f}")
	print(f"- tuned w_a3: {classical_ioc.tuned_planner_weights.w_a3:.6f}")
	print(f"- tuned w_a4: {classical_ioc.tuned_planner_weights.w_a4:.6f}")
	print(f"- mean tracking SSE: {classical_ioc.mean_tracking_sse:.6f}")
	print("\nEvaluation on unseen initial states")
	print(f"- n_test: {evaluation.n_test}")
	print(f"- our method mean GT HLO cost: {evaluation.our_method_mean_gt_hlo_cost:.6f}")
	print(f"- our method std GT HLO cost: {evaluation.our_method_std_gt_hlo_cost:.6f}")
	print(f"- classical IOC mean GT HLO cost: {evaluation.classical_ioc_mean_gt_hlo_cost:.6f}")
	print(f"- classical IOC std GT HLO cost: {evaluation.classical_ioc_std_gt_hlo_cost:.6f}")
	print(f"- expert mean GT HLO cost: {evaluation.expert_mean_gt_hlo_cost:.6f}")
	print(f"- expert std GT HLO cost: {evaluation.expert_std_gt_hlo_cost:.6f}")
	json_path = save_synthetic_demonstrations(demos=demos, hlo=hlo, config=config, path=config.output_json)
	print(f"- saved demos: {json_path}")
	plot_path = plot_synthetic_demonstrations(demos=demos, config=config, path=config.output_plot)
	if plot_path is None:
		print("- plot skipped: matplotlib is not available")
	else:
		print(f"- saved plot: {plot_path}")
	comparison_plot_path = plot_method_comparison_on_unseen_initial_states(
		hlo=hlo,
		demo_config=config,
		our_method=our_method,
		classical_ioc=classical_ioc,
		path=config.output_comparison_plot,
	)
	if comparison_plot_path is None:
		print("- comparison plot skipped: matplotlib is not available")
	else:
		print(f"- saved comparison plot: {comparison_plot_path}")


if __name__ == "__main__":
	main()
