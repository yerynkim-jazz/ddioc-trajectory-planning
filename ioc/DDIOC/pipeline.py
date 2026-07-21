"""pipeline.py"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]

try:
	from .dynamics_learning import (
		Entities,
		extract_and_merge_from_mcap_roots,
		prepare_train_test_segments_from_merged_csv,
		learn_koopman_dynamics,
		eval_koopman_one_step_mse,
		eval_koopman_rollout_rmse,
	)
	from .hlo_learning import (
		build_F_big_sharedomega,
		compute_feature_scales,
		eval_ioc_residual,
		learn_hlo_omega,
	)
except Exception:  # pragma: no cover
	from dynamics_learning import (  # type: ignore
		Entities,
		extract_and_merge_from_mcap_roots,
		prepare_train_test_segments_from_merged_csv,
		learn_koopman_dynamics,
		eval_koopman_one_step_mse,
		eval_koopman_rollout_rmse,
	)
	from hlo_learning import (  # type: ignore
		build_F_big_sharedomega,
		compute_feature_scales,
		eval_ioc_residual,
		learn_hlo_omega,
	)


def _write_omega_history_csv(path: Path, *, omega_history: list[dict], feature_names: list[str]) -> None:
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	fieldnames = ["stage", "solve_index", "segment_index", "buffer_size", "omega_sum"] + list(feature_names)
	with path.open("w", encoding="utf-8", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		for entry in omega_history:
			row = {
				"stage": entry.get("stage"),
				"solve_index": entry.get("solve_index"),
				"segment_index": entry.get("segment_index"),
				"buffer_size": entry.get("buffer_size"),
				"omega_sum": entry.get("omega_sum"),
			}
			omega = list(entry.get("omega", []))
			for i, name in enumerate(feature_names):
				row[name] = float(omega[i]) if i < len(omega) else None
			writer.writerow(row)


def _write_omega_convergence_plot(path: Path, *, omega_history: list[dict], feature_names: list[str]) -> None:
	"""Save a non-interactive PNG showing omega per IOC update."""
	try:
		import matplotlib

		matplotlib.use("Agg", force=True)
		import matplotlib.pyplot as plt
	except Exception as e:
		raise RuntimeError(
			"Plotting requires matplotlib. Install it in your environment (e.g. 'pip install matplotlib')."
		) from e

	if not omega_history:
		raise RuntimeError("Cannot plot omega convergence without history")

	solve_hist = [entry for entry in omega_history if str(entry.get("stage")) != "init"]
	plot_hist = solve_hist if solve_hist else list(omega_history)
	x = np.arange(len(plot_hist), dtype=int)
	W = np.asarray([entry.get("omega", []) for entry in plot_hist], dtype=float)
	if W.ndim != 2 or W.shape[0] == 0:
		raise RuntimeError("Invalid omega history for plotting")

	fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
	for j in range(W.shape[1]):
		label = str(feature_names[j]) if j < len(feature_names) else f"omega_{j}"
		ax.plot(x, W[:, j], marker="o", linewidth=1.8, markersize=3.5, label=label)

	if plot_hist and str(plot_hist[0].get("stage")) == "init":
		ax.set_xlabel("history index")
	else:
		ax.set_xlabel("IOC solve index")
	ax.set_ylabel("omega")
	ax.set_title("Omega convergence")
	ax.grid(True, alpha=0.3)
	ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
	fig.tight_layout()
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(path, dpi=150, bbox_inches="tight")
	plt.close(fig)


def learn_dynamics_and_weights(
	*,
	merged_csv: Path,
	output_root: Path,
	n_traj: int,
	min_traj_len: int,
	train_ratio: float,
	split_seed: int,
	lift: str,
	degree: int,
	dnn_psi_dim: int,
	dnn_hidden_dim: int,
	dnn_hidden_layers: int,
	dnn_pretrain_steps: int,
	dnn_pretrain_segments: int,
	dnn_lr: float,
	dnn_batch_segments: int,
	dnn_refit_every: int,
	seg_len: int,
	segment_stride: Optional[int],
	reg: float,
	ioc_every: int,
	window: int,
	use_reference_features: bool | None = None,
	verbose: bool = False,
) -> dict:
	if use_reference_features is False:
		raise ValueError("Reference-aware features are mandatory")

	if bool(verbose):
		print(f"[PIPELINE] loading segments from merged_csv={merged_csv}", file=sys.stderr, flush=True)

	data = prepare_train_test_segments_from_merged_csv(
		merged_csv=Path(merged_csv),
		n_traj=int(n_traj),
		min_traj_len=int(min_traj_len),
		train_ratio=float(train_ratio),
		split_seed=int(split_seed),
		seg_len=int(seg_len),
		segment_stride=segment_stride,
	)
	if bool(verbose):
		print(
			f"[PIPELINE] segments: train={len(data.get('segments_X', []))} test={len(data.get('segments_X_test', []))} dt_s={float(data.get('dt_s', float('nan'))):.3f}",
			file=sys.stderr,
			flush=True,
		)

	segments_X = data["segments_X"]
	segments_U = data["segments_U"]
	segments_V = data["segments_V"]
	segments_target_lat_m = data["segments_target_lat_m"]
	segments_dt_s = data["segments_dt_s"]
	segments_ref_orientation = data["segments_ref_orientation"]
	segments_ref_curvature = data["segments_ref_curvature"]

	segments_X_test = data["segments_X_test"]
	segments_U_test = data["segments_U_test"]
	segments_V_test = data["segments_V_test"]
	segments_target_lat_m_test = data["segments_target_lat_m_test"]
	segments_dt_s_test = data["segments_dt_s_test"]
	segments_ref_orientation_test = data["segments_ref_orientation_test"]
	segments_ref_curvature_test = data["segments_ref_curvature_test"]

	X_mean = data["X_mean"]
	X_std = data["X_std"]
	U_mean = data["U_mean"]
	U_std = data["U_std"]
	dt_s = float(data["dt_s"])

	n_state = 4
	n_cost = 9
	feature_names = [
		"lat_err_sq",
		"lat_rate_sq_v2",
		"kappa_abs_sq",
		"kappa_dot_abs_sq",
		"kappa_err_sq",
		"delta_kappa_err_sq",
		"delta_kappa_dot_sq",
		"u_sq",
		"psi_err_sq",
	]
	feature_scale_method = "median"

	feature_scales, feature_stats_train = compute_feature_scales(
		segments_X=segments_X,
		segments_U=segments_U,
		segments_V=segments_V,
		segments_target_lat_m=segments_target_lat_m,
		segments_ref_orientation=segments_ref_orientation,
		segments_ref_curvature=segments_ref_curvature,
		dt_s=float(dt_s),
		X_mean=X_mean,
		X_std=X_std,
		U_mean=U_mean,
		U_std=U_std,
		softness_m=0.2,
		method=feature_scale_method,
		eps=1e-6,
	)
	if bool(verbose):
		print("[PIPELINE] learning Koopman dynamics...", file=sys.stderr, flush=True)

	dyn = learn_koopman_dynamics(
		segments_X=segments_X,
		segments_U=segments_U,
		lift=str(lift),
		n_state=int(n_state),
		degree=int(degree),
		dnn_psi_dim=int(dnn_psi_dim),
		dnn_hidden_dim=int(dnn_hidden_dim),
		dnn_hidden_layers=int(dnn_hidden_layers),
		dnn_pretrain_steps=int(dnn_pretrain_steps),
		dnn_pretrain_segments=int(dnn_pretrain_segments),
		dnn_lr=float(dnn_lr),
		dnn_batch_segments=int(dnn_batch_segments),
		dnn_refit_every=int(dnn_refit_every),
		split_seed=int(split_seed),
		reg=float(reg),
	)
	if bool(verbose):
		print("[PIPELINE] learning HLO omega...", file=sys.stderr, flush=True)

	Kx = dyn["Kx"]
	Ku = dyn["Ku"]
	C = dyn["C"]
	lift_np = dyn["lift_np"]
	dpsi_dx_np = dyn["dpsi_dx_np"]

	omega, omega_history = learn_hlo_omega(
		segments_X=segments_X,
		segments_U=segments_U,
		segments_V=segments_V,
		segments_target_lat_m=segments_target_lat_m,
		segments_dt_s=segments_dt_s,
		segments_ref_orientation=segments_ref_orientation,
		segments_ref_curvature=segments_ref_curvature,
		Kx=Kx,
		Ku=Ku,
		C=C,
		dpsi_dx_np=dpsi_dx_np,
		X_mean=X_mean,
		X_std=X_std,
		U_mean=U_mean,
		U_std=U_std,
		feature_scales=feature_scales,
		feature_scale_eps=1e-6,
		ioc_every=int(ioc_every),
		window=int(window),
		softness_m=0.2,
		return_history=True,
		verbose=bool(verbose),
	)
	omega = np.asarray(omega, dtype=float).reshape(-1)
	if (not np.isfinite(omega).all()) or np.any(omega < 0):
		raise ValueError(f"Invalid omega returned from HLO learning: {omega}")
	if bool(verbose):
		print("[PIPELINE] evaluating metrics + writing outputs...", file=sys.stderr, flush=True)

	train_one_step = eval_koopman_one_step_mse(segments_X, segments_U, Kx=Kx, Ku=Ku, C=C, lift_np=lift_np, max_segments=0)
	test_one_step = eval_koopman_one_step_mse(segments_X_test, segments_U_test, Kx=Kx, Ku=Ku, C=C, lift_np=lift_np, max_segments=0)
	train_rollout = eval_koopman_rollout_rmse(segments_X, segments_U, Kx=Kx, Ku=Ku, C=C, lift_np=lift_np, rollout_h=10, max_segments=200)
	test_rollout = eval_koopman_rollout_rmse(segments_X_test, segments_U_test, Kx=Kx, Ku=Ku, C=C, lift_np=lift_np, rollout_h=10, max_segments=200)

	WINDOW = max(1, int(window))
	F_train, off_train = build_F_big_sharedomega(
		segments_X[-WINDOW:],
		segments_U[-WINDOW:],
		segments_V[-WINDOW:],
		Kx,
		Ku,
		C,
		dpsi_dx_np,
		n_state=n_state,
		n_cost=n_cost,
		buffer_target_lat_offset_m=segments_target_lat_m[-WINDOW:],
		buffer_ref_orientation_rad=segments_ref_orientation[-WINDOW:],
		buffer_ref_curvature_1pm=segments_ref_curvature[-WINDOW:],
		dt_s=float(np.median(segments_dt_s[-WINDOW:])) if segments_dt_s else float(dt_s),
		softness_m=0.2,
		x_mean=X_mean,
		x_std=X_std,
		u_mean=U_mean,
		u_std=U_std,
		feature_scales=feature_scales,
		feature_scale_eps=1e-6,
	)
	ioc_train_resid = eval_ioc_residual(F_train, off_train, omega, n_cost=n_cost)

	if len(segments_X_test) >= 2:
		wN = min(WINDOW, len(segments_X_test))
		F_test, off_test = build_F_big_sharedomega(
			segments_X_test[-wN:],
			segments_U_test[-wN:],
			segments_V_test[-wN:],
			Kx,
			Ku,
			C,
			dpsi_dx_np,
			n_state=n_state,
			n_cost=n_cost,
			buffer_target_lat_offset_m=segments_target_lat_m_test[-wN:],
			buffer_ref_orientation_rad=segments_ref_orientation_test[-wN:],
			buffer_ref_curvature_1pm=segments_ref_curvature_test[-wN:],
			dt_s=float(np.median(segments_dt_s_test[-wN:])) if segments_dt_s_test else float(dt_s),
			softness_m=0.2,
			x_mean=X_mean,
			x_std=X_std,
			u_mean=U_mean,
			u_std=U_std,
			feature_scales=feature_scales,
			feature_scale_eps=1e-6,
		)
		ioc_test_resid = eval_ioc_residual(F_test, off_test, omega, n_cost=n_cost)
	else:
		ioc_test_resid = {"residual_l2": None, "rows": 0}

	output_root = Path(output_root)
	output_root.mkdir(parents=True, exist_ok=True)
	learned_objective_json_path = output_root / "learned_objective.json"
	learned_objective_txt_path = output_root / "learned_objective.txt"
	omega_history_json_path = output_root / "omega_history.json"
	omega_history_csv_path = output_root / "omega_history.csv"
	omega_plot_path = output_root / "omega_convergence.png"
	omega_plot_error_path = output_root / "omega_convergence_error.txt"
	omega_npy_path = output_root / "omega.npy"
	koopman_npz_path = output_root / "koopman_model.npz"

	result = {
		"merged_csv": str(merged_csv),
		"n_traj_loaded": int(len(data["traj_X_raw"])),
		"train_ids": data["ids_train"],
		"lift": str(lift),
		"degree": int(degree) if str(lift).lower() == "poly" else None,
		"seg_len": int(seg_len),
		"segment_stride": int(data["segment_stride"]),
		"reg": float(reg),
		"ioc_every": int(ioc_every),
		"window": int(window),
		"omega": [float(x) for x in np.asarray(omega, dtype=float).reshape(-1).tolist()],
		"omega_history": omega_history,
		"omega_sum": float(np.sum(omega)),
		"feature_names": list(feature_names),
		"feature_scales": [float(x) for x in np.asarray(feature_scales, dtype=float).reshape(-1).tolist()],
		"feature_scale_method": str(feature_scale_method),
		"feature_scale_eps": 1e-6,
		"use_reference_features": True,
		"normalization": {
			"X_mean": np.asarray(X_mean, dtype=float).reshape(-1).tolist(),
			"X_std": np.asarray(X_std, dtype=float).reshape(-1).tolist(),
			"U_mean": np.asarray(U_mean, dtype=float).reshape(-1).tolist(),
			"U_std": np.asarray(U_std, dtype=float).reshape(-1).tolist(),
		},
		"metrics": {
			"koopman_one_step": {"train": train_one_step, "test": test_one_step},
			"koopman_rollout10": {"train": train_rollout, "test": test_rollout},
			"ioc_residual": {"train": ioc_train_resid, "test": ioc_test_resid},
		},
		"feature_magnitudes_raw": {"train": feature_stats_train},
		"artifacts": {
			"learned_objective_json": str(learned_objective_json_path),
			"learned_objective_txt": str(learned_objective_txt_path),
			"omega_history_json": str(omega_history_json_path),
			"omega_history_csv": str(omega_history_csv_path),
			"omega_convergence_png": str(omega_plot_path),
			"omega_npy": str(omega_npy_path),
			"koopman_model_npz": str(koopman_npz_path),
		},
	}

	learned_objective_json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
	learned_objective_txt_path.write_text("omega=" + ",".join([str(float(x)) for x in omega]) + "\n", encoding="utf-8")
	omega_history_json_path.write_text(json.dumps(omega_history, indent=2) + "\n", encoding="utf-8")
	_write_omega_history_csv(omega_history_csv_path, omega_history=omega_history, feature_names=feature_names)
	try:
		_write_omega_convergence_plot(
			omega_plot_path,
			omega_history=omega_history,
			feature_names=feature_names,
		)
		if omega_plot_error_path.exists():
			omega_plot_error_path.unlink()
		if bool(verbose):
			print(f"[PIPELINE] wrote omega convergence plot: {omega_plot_path}", file=sys.stderr, flush=True)
	except Exception as e:
		err_msg = f"Failed to write omega convergence plot at {omega_plot_path}: {e}"
		omega_plot_error_path.write_text(err_msg + "\n", encoding="utf-8")
		print(f"[PIPELINE] warning: {err_msg}", file=sys.stderr, flush=True)
	np.save(omega_npy_path, np.asarray(omega, dtype=float))

	np.savez(
		koopman_npz_path,
		Kx=Kx,
		Ku=Ku,
		C=C,
		X_mean=X_mean,
		X_std=X_std,
		U_mean=U_mean,
		U_std=U_std,
	)
	if bool(verbose):
		print(f"[PIPELINE] outputs written under: {output_root}", file=sys.stderr, flush=True)

	return result


def main(argv: Optional[list[str]] = None) -> int:
	ap = argparse.ArgumentParser(description="DDIOC pipeline")
	ap.add_argument("--merged_csv", default=None)
	ap.add_argument("--csv", dest="merged_csv", default=None)
	ap.add_argument("--src_root", action="append", default=None)
	ap.add_argument("--tag", action="append", default=None)
	ap.add_argument("--output_root", default=str(_REPO_ROOT / "Datasets" / "Output" / "PipelineRun"))

	ap.add_argument("--entities", choices=["ego", "agents", "both"], default="agents")
	ap.add_argument("--entities_by_tag", action="append", default=None)
	ap.add_argument("--unzip", action="store_true")
	ap.add_argument("--overwrite_zip", action="store_true")
	ap.add_argument("--max_files_per_root", type=int, default=0)
	ap.add_argument("--verbose", action="store_true")
	ap.add_argument("--write_pkl", action="store_true")
	ap.add_argument("--clean_output", action="store_true")
	ap.add_argument("--ts", type=float, default=0.1)
	ap.add_argument("--rel_opt_horizon", type=int, default=31)
	ap.add_argument("--max_interpolation_gap_s", type=float, default=2.0)
	ap.add_argument("--leave_lane_threshold_m", type=float, default=0.25)
	ap.add_argument("--leave_lane_padding_s", type=float, default=1.0)
	ap.add_argument("--state_estimation_padding_s", type=float, default=1.0)
	ap.add_argument("--min_velocity", type=float, default=5 / 3.6)

	ap.add_argument("--n_traj", type=int, default=200)
	ap.add_argument("--min_traj_len", type=int, default=30)
	ap.add_argument("--train_ratio", type=float, default=0.8)
	ap.add_argument("--split_seed", type=int, default=0)
	ap.add_argument("--lift", choices=["poly", "dnn"], default="poly")
	ap.add_argument("--degree", type=int, default=2)
	ap.add_argument("--dnn_psi_dim", type=int, default=20)
	ap.add_argument("--dnn_hidden_dim", type=int, default=64)
	ap.add_argument("--dnn_hidden_layers", type=int, default=4)
	ap.add_argument("--dnn_pretrain_steps", type=int, default=0)
	ap.add_argument("--dnn_pretrain_segments", type=int, default=32)
	ap.add_argument("--dnn_lr", type=float, default=1e-3)
	ap.add_argument("--dnn_batch_segments", type=int, default=8)
	ap.add_argument("--dnn_refit_every", type=int, default=50)
	ap.add_argument("--seg_len", type=int, default=31)
	ap.add_argument("--segment_stride", type=int, default=None)
	ap.add_argument("--reg", type=float, default=1e-6)
	ap.add_argument("--ioc_every", type=int, default=20)
	ap.add_argument("--window", type=int, default=200)
	ap.add_argument("--use_reference_features", action="store_true", default=True)
	ap.add_argument("--tune_theta", action="store_true", help="Run theta tuning via tune_qp_planner.py")
	ap.add_argument("--tune_args", nargs=argparse.REMAINDER, default=None)

	args = ap.parse_args(list(argv) if argv is not None else None)

	entities_by_tag: Optional[dict[str, Entities]] = None
	if args.entities_by_tag:
		entities_by_tag = {}
		for spec in args.entities_by_tag:
			if "=" not in spec:
				raise SystemExit("--entities_by_tag must be <tag>=<ego|agents|both>")
			tag, ent = spec.split("=", 1)
			ent = ent.strip()
			if ent not in ("ego", "agents", "both"):
				raise SystemExit("Invalid entities")
			entities_by_tag[tag.strip()] = ent  # type: ignore[assignment]

	output_root = Path(args.output_root)
	output_root.mkdir(parents=True, exist_ok=True)

	merged_csv: Optional[Path] = Path(args.merged_csv) if args.merged_csv else None
	if merged_csv is None:
		if not args.src_root:
			raise SystemExit("Provide --merged_csv or --src_root")
		merged_csv = extract_and_merge_from_mcap_roots(
			src_roots=[Path(p) for p in args.src_root],
			tags=args.tag,
			output_root=output_root / "Extraction",
			entities=args.entities,
			entities_by_tag=entities_by_tag,
			unzip=bool(args.unzip),
			overwrite_zip=bool(args.overwrite_zip),
			max_files_per_root=int(args.max_files_per_root),
			verbose=bool(args.verbose),
			ts=float(args.ts),
			rel_opt_horizon=int(args.rel_opt_horizon),
			max_interpolation_gap_s=float(args.max_interpolation_gap_s),
			leave_lane_threshold_m=float(args.leave_lane_threshold_m),
			leave_lane_padding_s=float(args.leave_lane_padding_s),
			state_estimation_padding_s=float(args.state_estimation_padding_s),
			min_velocity=float(args.min_velocity),
			write_pkl=bool(args.write_pkl),
			clean_output=bool(args.clean_output),
		)

	learn_dynamics_and_weights(
		merged_csv=Path(merged_csv),
		output_root=output_root / "Learned",
		n_traj=int(args.n_traj),
		min_traj_len=int(args.min_traj_len),
		train_ratio=float(args.train_ratio),
		split_seed=int(args.split_seed),
		lift=str(args.lift),
		degree=int(args.degree),
		dnn_psi_dim=int(args.dnn_psi_dim),
		dnn_hidden_dim=int(args.dnn_hidden_dim),
		dnn_hidden_layers=int(args.dnn_hidden_layers),
		dnn_pretrain_steps=int(args.dnn_pretrain_steps),
		dnn_pretrain_segments=int(args.dnn_pretrain_segments),
		dnn_lr=float(args.dnn_lr),
		dnn_batch_segments=int(args.dnn_batch_segments),
		dnn_refit_every=int(args.dnn_refit_every),
		seg_len=int(args.seg_len),
		segment_stride=args.segment_stride,
		reg=float(args.reg),
		ioc_every=int(args.ioc_every),
		window=int(args.window),
		use_reference_features=True,
		verbose=bool(args.verbose),
	)
	if bool(args.tune_theta):
		try:
			# Lazy import to keep pipeline startup fast when tuning is not used.
			from . import tune_qp_planner as tune_qp_planner_mod
		except Exception:  # pragma: no cover
			import tune_qp_planner as tune_qp_planner_mod  # type: ignore
		tune_argv = list(args.tune_args) if args.tune_args else []
		tune_qp_planner_mod.main(tune_argv)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
