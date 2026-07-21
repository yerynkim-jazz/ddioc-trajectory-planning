from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Callable

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(_REPO_ROOT))

try:
	from .dynamics_learning import (  # type: ignore
		eval_koopman_one_step_mse,
		eval_koopman_rollout_rmse,
		koopman_predict_next_xn,
		learn_koopman_dynamics,
	)
except Exception:
	from dynamics_learning import (  # type: ignore
		eval_koopman_one_step_mse,
		eval_koopman_rollout_rmse,
		koopman_predict_next_xn,
		learn_koopman_dynamics,
	)


def _simulate_nonlinear_trajectory(*, T: int, dt: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
	rng = np.random.default_rng(int(seed))

	x = np.zeros((int(T), 4), dtype=float)
	u = np.zeros((int(T) - 1, 1), dtype=float)

	x[0] = np.array(
		[
			rng.uniform(-1.0, 1.0),
			rng.uniform(-0.15, 0.15),
			rng.uniform(-0.06, 0.06),
			rng.uniform(-0.08, 0.08),
		],
		dtype=float,
	)

	base_noise = rng.normal(0.0, 1.0, size=int(T) - 1)
	smooth = np.convolve(base_noise, np.ones(7) / 7.0, mode="same")

	for k in range(int(T) - 1):
		lat, psi, kappa, kappa_dot = x[k]

		lat_c = float(np.clip(lat, -6.0, 6.0))
		psi_c = float(np.clip(psi, -1.2, 1.2))
		kappa_c = float(np.clip(kappa, -0.6, 0.6))
		kappa_dot_c = float(np.clip(kappa_dot, -1.0, 1.0))

		control = (
			0.18 * np.sin(0.08 * k + 0.7)
			+ 0.12 * np.sin(0.027 * (k**1.15))
			+ 0.08 * smooth[k]
		)
		u[k, 0] = float(np.clip(control, -0.5, 0.5))

		v = float(np.clip(28.0 + 6.0 * np.sin(0.011 * k) + 2.0 * np.cos(0.023 * k), 18.0, 38.0))

		lat_next = lat_c + dt * (
			v * np.tanh(psi_c)
			+ 0.35 * np.tanh(1.5 * kappa_c)
			- 0.03 * lat_c
		)
		psi_next = psi_c + dt * (
			v * (kappa_c + 0.08 * np.sin(1.2 * lat_c))
			+ 0.08 * np.sin(1.8 * psi_c)
			- 0.15 * psi_c
		)
		kappa_next = kappa_c + dt * (
			kappa_dot_c
			+ 0.20 * np.sin(np.clip(kappa_c * lat_c, -8.0, 8.0))
			- 0.02 * (psi_c**3)
			- 0.18 * kappa_c
		)
		kappa_dot_next = kappa_dot_c + dt * (
			u[k, 0]
			+ 0.12 * np.tanh(2.0 * kappa_dot_c)
			- 0.04 * np.sin(psi_c) * kappa_c
			- 0.25 * kappa_dot_c
		)

		noise = rng.normal(0.0, [0.001, 0.0006, 0.00035, 0.00035], size=4)
		x_next = np.array([lat_next, psi_next, kappa_next, kappa_dot_next], dtype=float) + noise
		if not np.isfinite(x_next).all():
			x_next = np.array([lat_c, psi_c, kappa_c, kappa_dot_c], dtype=float)
		x[k + 1] = x_next

	x[:, 0] = np.clip(x[:, 0], -8.0, 8.0)
	x[:, 1] = np.clip(x[:, 1], -0.8, 0.8)
	x[:, 2] = np.clip(x[:, 2], -0.5, 0.5)
	x[:, 3] = np.clip(x[:, 3], -0.8, 0.8)
	return x, u


def _build_dataset(*, n_traj: int, T: int, dt: float, seed: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
	X_list: list[np.ndarray] = []
	U_list: list[np.ndarray] = []
	for i in range(int(n_traj)):
		x, u = _simulate_nonlinear_trajectory(T=int(T), dt=float(dt), seed=int(seed) + i)
		X_list.append(x)
		U_list.append(u)
	return X_list, U_list


def _split_train_test(
	X_list: list[np.ndarray],
	U_list: list[np.ndarray],
	*,
	train_ratio: float,
	seed: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
	n = len(X_list)
	if n != len(U_list):
		raise ValueError("X_list and U_list length mismatch")
	rng = np.random.default_rng(int(seed))
	perm = rng.permutation(n)
	n_train = max(1, int(round(float(train_ratio) * n)))
	n_train = min(n - 1, n_train)
	tr_idx = perm[:n_train]
	te_idx = perm[n_train:]
	return (
		[X_list[i] for i in tr_idx],
		[U_list[i] for i in tr_idx],
		[X_list[i] for i in te_idx],
		[U_list[i] for i in te_idx],
	)


def _normalize_from_train(
	X_train: list[np.ndarray],
	U_train: list[np.ndarray],
	X_test: list[np.ndarray],
	U_test: list[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	X_all = np.vstack(X_train)
	U_all = np.vstack(U_train)

	X_mean = X_all.mean(axis=0)
	X_std = X_all.std(axis=0) + 1e-9
	U_mean = U_all.mean(axis=0)
	U_std = U_all.std(axis=0) + 1e-9

	X_train_n = [(X - X_mean) / X_std for X in X_train]
	U_train_n = [(U - U_mean) / U_std for U in U_train]
	X_test_n = [(X - X_mean) / X_std for X in X_test]
	U_test_n = [(U - U_mean) / U_std for U in U_test]

	return X_train_n, U_train_n, X_test_n, U_test_n, X_mean, X_std, U_mean, U_std


def _build_segments(
	X_list: list[np.ndarray],
	U_list: list[np.ndarray],
	*,
	seg_len: int,
	stride: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
	segments_X: list[np.ndarray] = []
	segments_U: list[np.ndarray] = []

	seg_len = int(seg_len)
	stride = int(stride)
	if seg_len < 3:
		raise ValueError("seg_len must be >= 3")
	if stride <= 0:
		raise ValueError("stride must be positive")

	for X, U in zip(X_list, U_list):
		n = len(X)
		if n < seg_len:
			continue
		for start in range(0, n - seg_len + 1, stride):
			end = start + seg_len
			segments_X.append(X[start:end].copy())
			segments_U.append(U[start : end - 1].copy())

	if not segments_X:
		raise RuntimeError("No segments built")
	return segments_X, segments_U


def _rollout_rmse_raw(
	segments_Xn: list[np.ndarray],
	segments_Un: list[np.ndarray],
	*,
	Kx: np.ndarray,
	Ku: np.ndarray,
	C: np.ndarray,
	lift_np: Callable[[np.ndarray], np.ndarray],
	X_mean: np.ndarray,
	X_std: np.ndarray,
	rollout_h: int,
) -> float:
	se_sum = 0.0
	n_sum = 0
	for Xseg_n, Useg_n in zip(segments_Xn, segments_Un):
		x_n = np.asarray(Xseg_n[0], dtype=float).copy()
		H = min(int(rollout_h), len(Useg_n), len(Xseg_n) - 1)
		for k in range(H):
			x_n = koopman_predict_next_xn(x_n, Useg_n[k], Kx=Kx, Ku=Ku, C=C, lift_np=lift_np)
			x_hat = x_n * X_std + X_mean
			x_true = np.asarray(Xseg_n[k + 1], dtype=float) * X_std + X_mean
			err = x_hat - x_true
			se_sum += float(np.sum(err**2))
			n_sum += 1
	if n_sum == 0:
		return float("nan")
	return float(np.sqrt(se_sum / n_sum))


def _rollout_full_trajectory_raw(
	X0_raw: np.ndarray,
	U_raw: np.ndarray,
	*,
	Kx: np.ndarray,
	Ku: np.ndarray,
	C: np.ndarray,
	lift_np: Callable[[np.ndarray], np.ndarray],
	X_mean: np.ndarray,
	X_std: np.ndarray,
	U_mean: np.ndarray,
	U_std: np.ndarray,
) -> np.ndarray:
	x_n = (np.asarray(X0_raw, dtype=float) - X_mean) / X_std
	U_n = (np.asarray(U_raw, dtype=float) - U_mean.reshape(1, -1)) / U_std.reshape(1, -1)

	pred = np.zeros((len(U_raw) + 1, 4), dtype=float)
	pred[0] = np.asarray(X0_raw, dtype=float)

	for k in range(len(U_raw)):
		x_n = koopman_predict_next_xn(x_n, U_n[k], Kx=Kx, Ku=Ku, C=C, lift_np=lift_np)
		pred[k + 1] = x_n * X_std + X_mean
	return pred


def _avg_rmse_over_test_trajectories_raw(
	X_test: list[np.ndarray],
	U_test: list[np.ndarray],
	*,
	Kx: np.ndarray,
	Ku: np.ndarray,
	C: np.ndarray,
	lift_np: Callable[[np.ndarray], np.ndarray],
	X_mean: np.ndarray,
	X_std: np.ndarray,
	U_mean: np.ndarray,
	U_std: np.ndarray,
) -> tuple[float, float, int]:
	rmses: list[float] = []
	for X_true, U_true in zip(X_test, U_test):
		pred = _rollout_full_trajectory_raw(
			X_true[0],
			U_true,
			Kx=Kx,
			Ku=Ku,
			C=C,
			lift_np=lift_np,
			X_mean=X_mean,
			X_std=X_std,
			U_mean=U_mean,
			U_std=U_std,
		)
		err = np.asarray(pred, dtype=float) - np.asarray(X_true, dtype=float)
		rmse = float(np.sqrt(np.mean(np.sum(err**2, axis=1))))
		rmses.append(rmse)

	if not rmses:
		return float("nan"), float("nan"), 0
	arr = np.asarray(rmses, dtype=float)
	return float(np.mean(arr)), float(np.std(arr)), int(arr.shape[0])


def _plot_trajectory_overlay(
	X_list: list[np.ndarray],
	*,
	dt: float,
	title: str,
	out_path: Path,
) -> None:
	state_names = ["lateral_offset_m", "heading_rad", "curvature_1pm", "curvature_dot_1pmps"]
	fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
	axes = axes.ravel()

	for i, ax in enumerate(axes):
		for X in X_list:
			t = np.arange(len(X), dtype=float) * float(dt)
			ax.plot(t, X[:, i], color="tab:blue", alpha=0.18, linewidth=0.9)
		ax.set_ylabel(state_names[i])
		ax.grid(alpha=0.25)

	axes[2].set_xlabel("time [s]")
	axes[3].set_xlabel("time [s]")
	fig.suptitle(title)
	fig.tight_layout(rect=[0, 0, 1, 0.97])
	fig.savefig(out_path, dpi=150)
	plt.close(fig)


def _load_optional_validate_dnn_settings() -> tuple[str, dict]:
	candidates: list[Path] = []
	config_override = str(os.environ.get("IOC_VALIDATE_DNN_CONFIG", "")).strip()
	if config_override:
		candidates.append(Path(config_override))
	candidates.append(_REPO_ROOT / "config" / "validate_dnn.json")

	for path in candidates:
		if path.exists():
			with open(path, encoding="utf-8") as f:
				return str(path), json.load(f)
	return "", {}


def main() -> None:
	config_path, config = _load_optional_validate_dnn_settings()
	profiles = config.get("profiles", config.get("bilevel_profiles", {}))
	defaults = config.get("defaults", config.get("bilevel", {}))
	active_profile = str(
		os.environ.get(
			"IOC_VALIDATE_DNN_PROFILE",
			config.get("active_profile", config.get("bilevel_active_profile", "default")),
		)
	).strip()
	if active_profile in profiles:
		settings = profiles[active_profile]
	else:
		settings = defaults

	# Extract parameters from optional config settings (with defaults)
	dyn_degree = int(settings.get("dyn_degree", "2"))
	dyn_seg_len = int(settings.get("dyn_seg_len", "20"))
	dyn_stride = int(settings.get("dyn_stride", "10"))
	hlo_dnn_pretrain_steps = int(settings.get("hlo_dnn_pretrain_steps", "200"))
	hlo_lift = settings.get("hlo_lift", "dnn")

	if config_path:
		print(f"[validate_dnn] Loaded settings from {config_path} (profile '{active_profile}'):")
	else:
		print("[validate_dnn] Using built-in default settings:")
	print(f"  dyn_degree={dyn_degree}")
	print(f"  dyn_seg_len={dyn_seg_len}")
	print(f"  dyn_stride={dyn_stride}")
	print(f"  hlo_dnn_pretrain_steps={hlo_dnn_pretrain_steps}")
	print(f"  hlo_lift={hlo_lift}")
	print()

	# Hardcoded synthetic data parameters
	seed = 42
	dt = 0.05
	n_traj = 120
	T = 140
	train_ratio = 0.8

	X_all, U_all = _build_dataset(n_traj=n_traj, T=T, dt=dt, seed=seed)

	X_train, U_train, X_test, U_test = _split_train_test(
		X_all,
		U_all,
		train_ratio=train_ratio,
		seed=seed,
	)

	figure_dir = _REPO_ROOT / "outputs" / "figures" / "validate_dnn"
	figure_dir.mkdir(parents=True, exist_ok=True)
	train_overlay_path = figure_dir / "train_trajectories_overlay.png"
	test_overlay_path = figure_dir / "test_trajectories_overlay.png"
	_plot_trajectory_overlay(
		X_train,
		dt=dt,
		title="Overlay of all TRAIN trajectories (synthetic)",
		out_path=train_overlay_path,
	)
	_plot_trajectory_overlay(
		X_test,
		dt=dt,
		title="Overlay of all TEST trajectories (synthetic)",
		out_path=test_overlay_path,
	)

	X_train_n, U_train_n, X_test_n, U_test_n, X_mean, X_std, U_mean, U_std = _normalize_from_train(
		X_train,
		U_train,
		X_test,
		U_test,
	)

	seg_X_train, seg_U_train = _build_segments(X_train_n, U_train_n, seg_len=dyn_seg_len, stride=dyn_stride)
	seg_X_test, seg_U_test = _build_segments(X_test_n, U_test_n, seg_len=dyn_seg_len, stride=dyn_stride)

	poly = learn_koopman_dynamics(
		segments_X=seg_X_train,
		segments_U=seg_U_train,
		lift="poly",
		n_state=4,
		degree=dyn_degree,
		dnn_psi_dim=20,
		dnn_hidden_dim=64,
		dnn_hidden_layers=3,
		dnn_pretrain_steps=0,
		dnn_pretrain_segments=0,
		dnn_lr=1e-3,
		dnn_batch_segments=8,
		dnn_refit_every=25,
		split_seed=seed,
		reg=1e-6,
	)

	dnn = learn_koopman_dynamics(
		segments_X=seg_X_train,
		segments_U=seg_U_train,
		lift="dnn",
		n_state=4,
		degree=dyn_degree,
		dnn_psi_dim=28,
		dnn_hidden_dim=96,
		dnn_hidden_layers=4,
		dnn_pretrain_steps=hlo_dnn_pretrain_steps,
		dnn_pretrain_segments=min(120, len(seg_X_train)),
		dnn_lr=1e-3,
		dnn_batch_segments=12,
		dnn_refit_every=30,
		split_seed=seed,
		reg=1e-6,
	)

	poly_1step = eval_koopman_one_step_mse(
		seg_X_test,
		seg_U_test,
		Kx=poly["Kx"],
		Ku=poly["Ku"],
		C=poly["C"],
		lift_np=poly["lift_np"],
	)
	dnn_1step = eval_koopman_one_step_mse(
		seg_X_test,
		seg_U_test,
		Kx=dnn["Kx"],
		Ku=dnn["Ku"],
		C=dnn["C"],
		lift_np=dnn["lift_np"],
	)

	poly_roll = eval_koopman_rollout_rmse(
		seg_X_test,
		seg_U_test,
		Kx=poly["Kx"],
		Ku=poly["Ku"],
		C=poly["C"],
		lift_np=poly["lift_np"],
		rollout_h=30,
		max_segments=300,
	)
	dnn_roll = eval_koopman_rollout_rmse(
		seg_X_test,
		seg_U_test,
		Kx=dnn["Kx"],
		Ku=dnn["Ku"],
		C=dnn["C"],
		lift_np=dnn["lift_np"],
		rollout_h=30,
		max_segments=300,
	)

	poly_roll_raw = _rollout_rmse_raw(
		seg_X_test,
		seg_U_test,
		Kx=poly["Kx"],
		Ku=poly["Ku"],
		C=poly["C"],
		lift_np=poly["lift_np"],
		X_mean=X_mean,
		X_std=X_std,
		rollout_h=30,
	)
	dnn_roll_raw = _rollout_rmse_raw(
		seg_X_test,
		seg_U_test,
		Kx=dnn["Kx"],
		Ku=dnn["Ku"],
		C=dnn["C"],
		lift_np=dnn["lift_np"],
		X_mean=X_mean,
		X_std=X_std,
		rollout_h=30,
	)

	print("=== Test one-step MSE (normalized states) ===")
	print(f"Poly: {poly_1step['mse_mean']:.6e}")
	print(f"DNN : {dnn_1step['mse_mean']:.6e}")

	print("\n=== Test rollout RMSE (normalized states, horizon=30) ===")
	print(f"Poly: {poly_roll['rmse_mean']:.6e}")
	print(f"DNN : {dnn_roll['rmse_mean']:.6e}")

	print("\n=== Test rollout RMSE (raw states, horizon=30) ===")
	print(f"Poly: {poly_roll_raw:.6e}")
	print(f"DNN : {dnn_roll_raw:.6e}")

	poly_avg_rmse, poly_std_rmse, n_test_eval = _avg_rmse_over_test_trajectories_raw(
		X_test,
		U_test,
		Kx=poly["Kx"],
		Ku=poly["Ku"],
		C=poly["C"],
		lift_np=poly["lift_np"],
		X_mean=X_mean,
		X_std=X_std,
		U_mean=U_mean,
		U_std=U_std,
	)
	dnn_avg_rmse, dnn_std_rmse, _ = _avg_rmse_over_test_trajectories_raw(
		X_test,
		U_test,
		Kx=dnn["Kx"],
		Ku=dnn["Ku"],
		C=dnn["C"],
		lift_np=dnn["lift_np"],
		X_mean=X_mean,
		X_std=X_std,
		U_mean=U_mean,
		U_std=U_std,
	)
	print("\n=== Average trajectory RMSE over ALL test trajectories (raw states) ===")
	print(f"n_test_trajectories: {n_test_eval}")
	print(f"Poly mean±std: {poly_avg_rmse:.6e} ± {poly_std_rmse:.6e}")
	print(f"DNN  mean±std: {dnn_avg_rmse:.6e} ± {dnn_std_rmse:.6e}")
	if np.isfinite(poly_roll_raw) and np.isfinite(dnn_roll_raw):
		if dnn_roll_raw < poly_roll_raw:
			improvement = 100.0 * (poly_roll_raw - dnn_roll_raw) / max(poly_roll_raw, 1e-12)
			print(f"DNN improves rollout RMSE by {improvement:.2f}% (raw units).")
		else:
			print("DNN did not outperform polynomial in this run; increase dnn_pretrain_steps or lower polynomial degree.")

	x_true = X_test[0]
	u_true = U_test[0]
	pred_poly = _rollout_full_trajectory_raw(
		x_true[0],
		u_true,
		Kx=poly["Kx"],
		Ku=poly["Ku"],
		C=poly["C"],
		lift_np=poly["lift_np"],
		X_mean=X_mean,
		X_std=X_std,
		U_mean=U_mean,
		U_std=U_std,
	)
	pred_dnn = _rollout_full_trajectory_raw(
		x_true[0],
		u_true,
		Kx=dnn["Kx"],
		Ku=dnn["Ku"],
		C=dnn["C"],
		lift_np=dnn["lift_np"],
		X_mean=X_mean,
		X_std=X_std,
		U_mean=U_mean,
		U_std=U_std,
	)

	t = np.arange(len(x_true), dtype=float) * dt
	state_names = ["lateral_offset_m", "heading_rad", "curvature_1pm", "curvature_dot_1pmps"]

	fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
	axes = axes.ravel()
	for i, ax in enumerate(axes):
		ax.plot(t, x_true[:, i], color="k", linewidth=2.0, label="test")
		ax.plot(t, pred_poly[:, i], color="tab:orange", linewidth=1.4, label="poly")
		ax.plot(t, pred_dnn[:, i], color="tab:blue", linewidth=1.4, label="dnn")
		ax.set_ylabel(state_names[i])
		ax.grid(alpha=0.25)
	axes[2].set_xlabel("time [s]")
	axes[3].set_xlabel("time [s]")
	axes[0].legend(loc="best")
	fig.tight_layout()

	out_path = figure_dir / "validate_dnn_overlay.png"
	fig.savefig(out_path, dpi=150)
	plt.close(fig)
	print(f"\nSaved overlay figure: {out_path}")
	print(f"Saved train trajectory overlay: {train_overlay_path}")
	print(f"Saved test trajectory overlay: {test_overlay_path}")


if __name__ == "__main__":
	main()
