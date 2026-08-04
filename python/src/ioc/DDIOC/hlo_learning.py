from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import numpy as np

try:
	from scipy.optimize import minimize
except Exception:  # pragma: no cover
	minimize = None

# Allow running this module from outside the repo root.
def _find_repo_root() -> Path:
	current = Path(__file__).resolve()
	for parent in current.parents:
		if (parent / "README.md").is_file() and (parent / "examples").is_dir():
			return parent
	raise RuntimeError("Unable to locate repository root from hlo_learning.py")


_REPO_ROOT = _find_repo_root()
if str(_REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(_REPO_ROOT))


def _parse_floats_csv(s: str) -> List[float]:
	parts = [p.strip() for p in str(s).split(",") if p.strip()]
	if not parts:
		return []
	return [float(p) for p in parts]


def compute_hlo_omega_via_pipeline(
	*,
	merged_csv: Path,
	n_traj: int,
	seg_len: int,
	segment_stride: int | None = None,
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
	pipeline_output_root: Path | None = None,
	return_meta: bool = False,
	verbose: bool = False,
) -> Any:
	"""Compute omega by calling `Pipeline.pipeline` as a Python module.

	This avoids parsing stdout and gives a stable API for other scripts.
	"""

	merged_csv = Path(merged_csv)
	if not merged_csv.exists():
		raise FileNotFoundError(f"Merged CSV for Pipeline.pipeline not found: {merged_csv}")

	if bool(verbose):
		print(
			f"[HLO] via_pipeline: merged_csv={merged_csv} lift={lift} seg_len={int(seg_len)} stride={segment_stride} n_traj={int(n_traj)}",
			file=sys.stderr,
			flush=True,
		)

	# Lazy import to keep startup fast.
	try:
		# When executed as a module: `python -m ioc.DDIOC.hlo_learning`
		from . import pipeline as pipeline_mod  # type: ignore
	except Exception:  # pragma: no cover
		# When executed as a script from this folder: `python hlo_learning.py`
		import pipeline as pipeline_mod  # type: ignore
	if bool(verbose):
		print("[HLO] via_pipeline: imported pipeline module", file=sys.stderr, flush=True)

	# Match Pipeline.pipeline defaults for args we don't expose here.
	min_traj_len = 30
	train_ratio = 0.8
	split_seed = 0
	# If not provided, Pipeline defaults stride to seg_len.
	segment_stride = None if segment_stride is None else int(segment_stride)
	reg = 1e-6
	ioc_every = 20
	window = 200

	def _run_pipeline_once(out_root: Path) -> Dict[str, Any]:
		if bool(verbose):
			print(f"[HLO] via_pipeline: output_root={out_root}", file=sys.stderr, flush=True)
		return pipeline_mod.learn_dynamics_and_weights(
			merged_csv=merged_csv,
			output_root=out_root,
			n_traj=int(n_traj),
			min_traj_len=int(min_traj_len),
			train_ratio=float(train_ratio),
			split_seed=int(split_seed),
			lift=str(lift),
			degree=int(degree),
			dnn_psi_dim=int(dnn_psi_dim),
			dnn_hidden_dim=int(dnn_hidden_dim),
			dnn_hidden_layers=int(dnn_hidden_layers),
			dnn_pretrain_steps=int(dnn_pretrain_steps),
			dnn_pretrain_segments=int(dnn_pretrain_segments),
			dnn_lr=float(dnn_lr),
			dnn_batch_segments=int(dnn_batch_segments),
			dnn_refit_every=int(dnn_refit_every),
			seg_len=int(seg_len),
			segment_stride=segment_stride,
			reg=float(reg),
			ioc_every=int(ioc_every),
			window=int(window),
			verbose=bool(verbose),
		)

	if pipeline_output_root is not None:
		output_root = Path(pipeline_output_root)
		output_root.mkdir(parents=True, exist_ok=True)
		result = _run_pipeline_once(output_root)
	else:
		with tempfile.TemporaryDirectory(prefix="learn_hlo_pipeline_") as tmp:
			result = _run_pipeline_once(Path(tmp))

	if bool(verbose):
		print("[HLO] via_pipeline: pipeline finished", file=sys.stderr, flush=True)

	omega = result.get("omega", None)
	if omega is None:
		raise RuntimeError("Pipeline.pipeline.learn_dynamics_and_weights did not return an 'omega' field.")
	om = np.asarray(list(omega), dtype=float).reshape(-1)
	if (not np.isfinite(om).all()):
		raise RuntimeError(f"Invalid omega returned from Pipeline.pipeline: {om}")
	if om.size not in (8, 9):
		raise RuntimeError(f"Expected 8D or 9D omega from Pipeline.pipeline, got {om.size}: {om}")
	omega_out = [float(x) for x in om]
	if not bool(return_meta):
		return omega_out
	meta = {
		"feature_scales": result.get("feature_scales", None),
		"feature_scale_eps": result.get("feature_scale_eps", None),
		"feature_scale_method": result.get("feature_scale_method", None),
		"artifacts": result.get("artifacts", None),
	}
	return omega_out, meta


def load_hlo_omega_from_learned_objective_json(path: Path) -> List[float]:
	path = Path(path)
	data = json.loads(path.read_text(encoding="utf-8"))
	if not isinstance(data, dict):
		raise ValueError(f"Expected dict/object in learned objective JSON, got {type(data).__name__}")
	omega = data.get("omega")
	if omega is None:
		raise ValueError(f"Missing 'omega' field in learned objective JSON: {path}")
	om = np.asarray(list(omega), dtype=float).reshape(-1)
	if om.size not in (4, 5, 6, 7, 8, 9) or (not np.isfinite(om).all()):
		raise ValueError(f"Invalid omega in {path}: {om}")
	# Backward compatibility: if the JSON contains feature names and the first feature
	# is the legacy duration term, drop it.
	names = data.get("feature_names", None)
	if isinstance(names, list) and len(names) == int(om.size):
		try:
			first = str(names[0])
		except Exception:
			first = ""
		if first == "duration_dt_sigmoid":
			om = om[1:]
	return [float(x) for x in om.tolist()]


def load_hlo_feature_scales_from_learned_objective_json(path: Path) -> dict | None:
	"""Load feature-normalization scales from a Pipeline learned_objective.json.

	Expected schema (written by method/DDIOC/pipeline.py):
		feature_scales: [s0..s7]
		feature_scale_eps: float (optional)
		feature_scale_method: str (optional)

	Returns None if the JSON does not contain feature scales.
	"""
	path = Path(path)
	data = json.loads(path.read_text(encoding="utf-8"))
	if not isinstance(data, dict):
		raise ValueError(f"Expected dict/object in learned objective JSON, got {type(data).__name__}")
	scales = data.get("feature_scales", None)
	if scales is None:
		return None
	fs = np.asarray(list(scales), dtype=float).reshape(-1)
	if fs.size not in (7, 8, 9) or (not np.isfinite(fs).all()):
		raise ValueError(f"Invalid feature_scales in {path}: {fs}")
	if np.any(fs < 0):
		raise ValueError(f"feature_scales must be >= 0 in {path}: {fs}")
	# Backward compatibility: if feature_names indicates a leading duration feature, drop it.
	names = data.get("feature_names", None)
	if isinstance(names, list) and len(names) == int(fs.size):
		try:
			first = str(names[0])
		except Exception:
			first = ""
		if first == "duration_dt_sigmoid":
			fs = fs[1:]
	eps = float(data.get("feature_scale_eps", 1e-6))
	if not np.isfinite(eps) or eps < 0:
		raise ValueError(f"Invalid feature_scale_eps in {path}: {eps}")
	method = data.get("feature_scale_method", None)
	return {
		"feature_scales": [float(x) for x in fs.tolist()],
		"feature_scale_eps": float(eps),
		"feature_scale_method": str(method) if method is not None else None,
	}


def save_learned_objective_json(path: Path, *, omega: Sequence[float], extra: Optional[Dict[str, Any]] = None) -> None:
	path = Path(path)
	# Allow passing a directory (e.g. '.' from the GUI) and write the default
	# filename inside it.
	try:
		if path.exists() and path.is_dir():
			path = path / "learned_objective.json"
	except Exception:
		# If we can't stat the path, fall back to suffix heuristic below.
		pass
	# If the user passed something without a .json suffix, treat it like a directory.
	# This avoids writing to a directory path (PermissionError) and makes CLI usage
	# more forgiving.
	if path.suffix.lower() != ".json":
		path = path / "learned_objective.json"
	path.parent.mkdir(parents=True, exist_ok=True)
	om = np.asarray(list(omega), dtype=float).reshape(-1)
	if om.size not in (4, 5, 6, 7, 8, 9) or (not np.isfinite(om).all()):
		raise ValueError(f"omega must be 4, 5, 6, 7, 8, or 9 finite floats, got: {om}")
	payload: Dict[str, Any] = {"omega": [float(x) for x in om.tolist()]}
	if extra:
		payload.update(extra)
	path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
	parser = argparse.ArgumentParser(
		description="Learn/resolve HLO omega (weights) either via Pipeline.pipeline or from learned_objective.json."
	)
	parser.add_argument(
		"--via_pipeline",
		action="store_true",
		help="Compute omega by running Pipeline.pipeline logic (requires --merged_csv).",
	)
	parser.add_argument("--merged_csv", type=Path, default=None, help="Merged IOC CSV used when computing omega via Pipeline.pipeline.")
	parser.add_argument("--n_traj", type=int, default=200)
	parser.add_argument("--seg_len", type=int, default=20)
	parser.add_argument(
		"--segment_stride",
		type=int,
		default=None,
		help="Stride between segments when learning omega via Pipeline (default: seg_len).",
	)
	parser.add_argument("--lift", choices=["poly", "dnn"], default="poly")
	parser.add_argument("--degree", type=int, default=2)
	parser.add_argument("--dnn_psi_dim", type=int, default=20)
	parser.add_argument("--dnn_hidden_dim", type=int, default=64)
	parser.add_argument("--dnn_hidden_layers", type=int, default=4)
	parser.add_argument("--dnn_pretrain_steps", type=int, default=0)
	parser.add_argument("--dnn_pretrain_segments", type=int, default=32)
	parser.add_argument("--dnn_lr", type=float, default=1e-4)
	parser.add_argument("--dnn_batch_segments", type=int, default=8)
	parser.add_argument("--dnn_refit_every", type=int, default=50)
	parser.add_argument(
		"--pipeline_output_root",
		type=Path,
		default=None,
		help=(
			"Output folder for intermediate/final pipeline artifacts when using --via_pipeline "
			"(omega history, convergence plot, Koopman model, etc.). "
			"Default: parent folder of --out_json if provided; otherwise a temporary folder."
		),
	)

	_default_obj_json = _REPO_ROOT / "Datasets" / "Output" / "PipelineRun" / "Learned" / "learned_objective.json"
	parser.add_argument(
		"--objective_json",
		type=Path,
		default=_default_obj_json if _default_obj_json.exists() else None,
		help="Load omega from an existing learned_objective.json.",
	)
	parser.add_argument(
		"--out_json",
		type=Path,
		default=None,
		help="Optional: write a learned_objective.json containing the resolved omega.",
	)
	parser.add_argument(
		"--print",
		action="store_true",
		help="Print omega as comma-separated floats to stdout.",
	)
	parser.add_argument(
		"--omega",
		type=str,
		default=None,
		help="Explicit omega as 4, 5, 6, 7, or 8 comma-separated floats (overrides other sources).",
	)

	args = parser.parse_args(list(argv) if argv is not None else None)

	omega: Optional[List[float]] = None
	meta: Optional[Dict[str, Any]] = None
	if args.omega is not None:
		omega = _parse_floats_csv(args.omega)
		if len(omega) not in (4, 5, 6, 7, 8, 9):
			raise SystemExit(f"--omega must have 4, 5, 6, 7, 8 or 9 floats, got {len(omega)}")
	elif bool(args.via_pipeline):
		if args.merged_csv is None:
			raise SystemExit("--via_pipeline requires --merged_csv")
		# Ensure the GUI shows progress for long-running pipeline steps.
		print("[HLO] computing omega via pipeline...", file=sys.stderr, flush=True)
		# Save the objective JSON by default when running via_pipeline,
		# so it can be reused later for theta tuning.
		if args.out_json is None:
			args.out_json = Path("learned_objective.json")
		pipeline_output_root = (
			Path(args.pipeline_output_root)
			if args.pipeline_output_root is not None
			else (Path(args.out_json).parent if args.out_json is not None else None)
		)
		omega, meta = compute_hlo_omega_via_pipeline(
			merged_csv=Path(args.merged_csv),
			n_traj=int(args.n_traj),
			seg_len=int(args.seg_len),
			segment_stride=args.segment_stride,
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
			pipeline_output_root=pipeline_output_root,
			return_meta=True,
			verbose=True,
		)
	else:
		if args.objective_json is None:
			raise SystemExit("Provide one of: --omega, --via_pipeline --merged_csv ..., or --objective_json")
		omega = load_hlo_omega_from_learned_objective_json(Path(args.objective_json))

	assert omega is not None
	if args.out_json is not None:
		extra: Dict[str, Any] = {}
		if bool(args.via_pipeline):
			# Helpful provenance for later reuse (theta tuning, experiments).
			extra = meta.copy() if meta is not None else {}
			extra.update({
				"source": "ioc.DDIOC.hlo_learning:via_pipeline",
				"merged_csv": str(args.merged_csv) if args.merged_csv is not None else None,
				"n_traj": int(args.n_traj),
				"seg_len": int(args.seg_len),
				"segment_stride": int(args.segment_stride) if args.segment_stride is not None else None,
				"lift": str(args.lift),
				"degree": int(args.degree) if str(args.lift) == "poly" else None,
				"dnn_pretrain_steps": int(args.dnn_pretrain_steps) if str(args.lift) == "dnn" else None,
			})
		save_learned_objective_json(Path(args.out_json), omega=omega, extra=extra)
	if bool(args.print) or args.out_json is None:
		# By default print if no output file is requested.
		print(",".join([str(float(x)) for x in omega]))
	return 0


# -----------------------------
# Paper-style IOC omega solver
# -----------------------------


def build_F_big_sharedomega(
	buffer_X,
	buffer_U,
	buffer_V,
	Kx,
	Ku,
	C,
	dpsi_dx_np,
	n_state: int,
	n_cost: int,
	*,
	buffer_target_lat_offset_m=None,
	buffer_ref_lat_offset_m=None,
	buffer_ref_orientation_rad=None,
	buffer_ref_curvature_1pm=None,
	dt_s: float = 1.0,
	completion_tolerance_m: float = 0.5,
	softness_m: float = 0.2,
	x_mean: np.ndarray | None = None,
	x_std: np.ndarray | None = None,
	u_mean: np.ndarray | None = None,
	u_std: np.ndarray | None = None,
	feature_scales: np.ndarray | None = None,
	feature_scale_eps: float = 1e-6,
):
	"""Build IOC matrix F_big with shared omega (paper core).

	Paper-style stacking across multiple segments:
	  columns: [lambda_1 | ... | lambda_M | omega | mu_1 | ... | mu_M]
	  rows   : stacked KKT blocks for each segment

	IMPORTANT: Keeps the original convention Dpsi = dpsi_dx_np(X[k+1]).

	Lateral offset convention in this repo:
	- The lateral objective is expressed in a lane-center frame.
	- The lateral target is fixed to 0 m for all segments.
	- A separate lateral reference signal is not used (any provided
	  buffer_ref_lat_offset_m / buffer_target_lat_offset_m inputs are ignored).
	"""

	if int(n_state) != 4:
		raise ValueError(f"This objective expects n_state=4, got {n_state}")

	n_cost = int(n_cost)
	if n_cost not in (8, 9):
		raise ValueError(f"This objective uses 8 or 9 features (n_cost=8 or 9), got n_cost={n_cost}.")

	v_eps = 1e-3
	completion_tolerance_m = float(completion_tolerance_m)
	if not np.isfinite(completion_tolerance_m) or completion_tolerance_m < 0:
		raise ValueError("completion_tolerance_m must be finite and >= 0")
	softness_m = float(softness_m)
	if softness_m <= 0:
		raise ValueError("softness_m must be positive")

	dt_s = float(dt_s)
	if not np.isfinite(dt_s) or dt_s <= 0:
		raise ValueError("dt_s must be positive")

	# Normalization scalings (needed to compute gradients w.r.t. normalized x/u)
	if x_mean is None or x_std is None or u_mean is None or u_std is None:
		raise ValueError("x_mean/x_std/u_mean/u_std must be provided to compute physical-feature gradients.")
	X_mean = np.asarray(x_mean, dtype=float).reshape(-1)
	X_std = np.asarray(x_std, dtype=float).reshape(-1)
	U_mean = np.asarray(u_mean, dtype=float).reshape(-1)
	U_std = np.asarray(u_std, dtype=float).reshape(-1)
	if X_mean.shape[0] != 4 or X_std.shape[0] != 4:
		raise ValueError("x_mean/x_std must be shape (4,)")
	if U_mean.shape[0] < 1 or U_std.shape[0] < 1:
		raise ValueError("u_mean/u_std must have at least one element")

	# Feature normalization: if provided, scale each feature by 1/(s_i + eps)
	# which equivalently scales its gradients in the IOC stationarity equations.
	feature_scale_eps = float(feature_scale_eps)
	if not np.isfinite(feature_scale_eps) or feature_scale_eps < 0:
		raise ValueError("feature_scale_eps must be finite and >= 0")
	inv_feature_scales = np.ones(int(n_cost), dtype=float)
	if feature_scales is not None:
		fs = np.asarray(feature_scales, dtype=float).reshape(-1)
		if fs.size != int(n_cost) or (not np.isfinite(fs).all()):
			raise ValueError(f"feature_scales must be length {int(n_cost)} finite floats")
		if np.any(fs < 0):
			raise ValueError("feature_scales must be >= 0")
		inv_feature_scales = 1.0 / (fs + feature_scale_eps)

	def _sigmoid(z: float) -> float:
		# numerically safe sigmoid
		if z >= 0:
			e = np.exp(-z)
			return float(1.0 / (1.0 + e))
		e = np.exp(z)
		return float(e / (1.0 + e))

	def _wrap_to_pi(a: float) -> float:
		# smooth-ish wrapping via atan2(sin,cos)
		return float(np.arctan2(np.sin(a), np.cos(a)))

	def dphi_dx_pair(
		xk_norm: np.ndarray,
		xkp1_norm: np.ndarray,
		v: float,
		target_lat_offset_m: float,
		*,
		ref_psi_k: float | None = None,
		ref_psi_kp1: float | None = None,
		ref_kappa_k: float | None = None,
		ref_kappa_kp1: float | None = None,
	) -> np.ndarray:
		"""
		Return d(phi)/d(x_{k+1}_norm) as (n_cost, n_state).

		Feature basis (LaTeX-aligned), evaluated at k+1:
		0) phi_d: (d - d_tgt)^2
		1) phi_drate: v^2 * (d_dot)^2  where d_dot ≈ (d_{k+1}-d_k)/dt
		2) phi_kabs: v^4 * kappa^2
		3) phi_kdotabs: v^4 * (kappa_dot)^2
		4) phi_kappa: v^4 * (kappa - kappa_ref)^2
		5) phi_Dkappa: v^4 * ((kappa-kref) - (kappa_prev-kref_prev))^2
		6) phi_Dkdot: v^4 * (kappa_dot - kappa_dot_prev)^2
		7) phi_u: v^4 * u^2  (handled in dphi_du)
		8) phi_psi: v^2 * wrap_to_pi(psi - psi_ref)^2  (only when n_cost==9)
		"""
		xk_norm = np.asarray(xk_norm, dtype=float).reshape(-1)
		xkp1_norm = np.asarray(xkp1_norm, dtype=float).reshape(-1)
		if xk_norm.shape[0] != 4 or xkp1_norm.shape[0] != 4:
			raise ValueError("xk/xkp1 must be shape (4,)")

		# v_eff = max(float(v), v_eps)
		v_eff = max(v, v_eps) 

		# Denormalize
		xk_raw = xk_norm * X_std + X_mean
		xkp1_raw = xkp1_norm * X_std + X_mean

		J = np.zeros((n_cost, 4), dtype=float)

		# (0) lateral tracking error (d - d_tgt)^2
		e_d = float(xkp1_raw[0] - float(target_lat_offset_m))
		J[0, 0] = float(2.0 * e_d * float(X_std[0]))

		# (1) speed-weighted lateral-rate penalty v^2 * ((d_{k+1}-d_k)/dt)^2
		# Gradient w.r.t. d_{k+1} only (paper convention uses dpsi_dx(X[k+1]) in stationarity).
		dk = float(xkp1_raw[0] - xk_raw[0])
		inv_dt = 1.0 / float(max(dt_s, 1e-9))
		d_dot = dk * inv_dt
		J[1, 0] = float(2.0 * (v_eff**2) * d_dot * inv_dt * float(X_std[0]))

		# (2) absolute curvature penalty v^4 * kappa^2
		kappa_kp1 = float(xkp1_raw[2])
		dkappa_dknorm = float(X_std[2])
		J[2, 2] = float(2.0 * (v_eff**4) * kappa_kp1 * dkappa_dknorm)

		# (3) absolute curvature-rate penalty v^4 * (kappa_dot)^2
		kdot_kp1 = float(xkp1_raw[3])
		dkdot_dknorm = float(X_std[3])
		J[3, 3] = float(2.0 * (v_eff**4) * kdot_kp1 * dkdot_dknorm)

		# (4) curvature tracking penalty v^4 * (kappa - kappa_ref)^2
		kref_kp1 = float(ref_kappa_kp1) if (ref_kappa_kp1 is not None) else 0.0
		e_kp1 = kappa_kp1 - kref_kp1
		J[4, 2] = float(2.0 * (v_eff**4) * e_kp1 * dkappa_dknorm)

		# (5) curvature smoothness penalty v^4 * (Δ(kappa-kref))^2
		kappa_k = float(xk_raw[2])
		kref_k = float(ref_kappa_k) if (ref_kappa_k is not None) else 0.0
		e_k = kappa_k - kref_k
		delta_e = e_kp1 - e_k
		J[5, 2] = float(2.0 * (v_eff**4) * delta_e * dkappa_dknorm)

		# (6) v^4 * (Δkappa_dot)^2 where Δkappa_dot = kdot_{k+1} - kdot_k
		kdot_k = float(xk_raw[3])
		dkdot = kdot_kp1 - kdot_k
		J[6, 3] = float(2.0 * (v_eff**4) * dkdot * dkdot_dknorm)

		# (8) orientation tracking : v^2 * wrap_to_pi(psi - psi_ref)^2 (optional)
		if n_cost == 9:
			psi_kp1 = float(xkp1_raw[1])
			psi_ref_kp1 = float(ref_psi_kp1) if (ref_psi_kp1 is not None) else 0.0
			e_psi = _wrap_to_pi(psi_kp1 - psi_ref_kp1)
			dpsi_dpsinorm = float(X_std[1])
			J[8, 1] = float(2.0 * (v_eff**2) * e_psi * dpsi_dpsinorm)

		return J


	def dphi_du(u_norm: np.ndarray, v: float) -> np.ndarray:
		"""Return d(phi)/d(u_k_norm) as (m, n_cost)."""
		u_norm = np.asarray(u_norm, dtype=float).reshape(-1)
		m_local = int(u_norm.shape[0])
		if m_local < 1:
			return np.zeros((0, n_cost), dtype=float)
		if U_mean.shape[0] < m_local or U_std.shape[0] < m_local:
			raise ValueError("u_mean/u_std must have at least m elements")

		v_eff = max(float(v), v_eps)

		u_raw = u_norm[:m_local] * U_std[:m_local] + U_mean[:m_local]
		G = np.zeros((m_local, n_cost), dtype=float)
		# (7) control effort phi_u = v^4 * u^2 (per-step)
		# d(v^4*u_raw^2)/d(u_norm) = v^4 * 2*u_raw*du_raw/du_norm = v^4 * 2*u_raw*U_std
		G[:, 7] = (v_eff**4) * 2.0 * u_raw * U_std[:m_local]
		return G



	# ∂f/∂u = C Ku  (n_state x m) ; typically m=1
	DfDu = C @ Ku
	m = int(DfDu.shape[1])
	DfDu_T = DfDu.T  # (m x n_state)

	# ---- Build per-segment blocks first
	seg_blocks = []  # (A, Phi_x, V, B, Phi_u, nl, nr_top, nr_bot)
	lambda_sizes = []
	mu_sizes = []
	row_sizes = []

	# Lateral target/reference handling:
	# This repo's current objective uses a fixed lateral target of 0 m (lane-center frame)
	# and does not use a separate lateral reference signal.
	# Keep the function arguments for backward compatibility, but ignore their values.
	_ = buffer_target_lat_offset_m
	_ = buffer_ref_lat_offset_m
	buffer_target_lat_offset_m = [0.0 for _ in buffer_X]

	# reference arrays optional; when not provided, use None per segment
	if buffer_ref_orientation_rad is None:
		buffer_ref_orientation_rad = [None for _ in buffer_X]
	if buffer_ref_curvature_1pm is None:
		buffer_ref_curvature_1pm = [None for _ in buffer_X]
	if len(buffer_ref_orientation_rad) != len(buffer_X) or len(buffer_ref_curvature_1pm) != len(buffer_X):
		raise ValueError("buffer_ref_orientation_rad / buffer_ref_curvature_1pm must match buffer_X length")

	for X, U, Vtraj, target_lat_m, ref_psi_traj, ref_kappa_traj in zip(
		buffer_X,
		buffer_U,
		buffer_V,
		buffer_target_lat_offset_m,
		buffer_ref_orientation_rad,
		buffer_ref_curvature_1pm,
	):
		Nst = len(X) - 1
		if Nst < 2:
			continue

		nl = n_state * Nst  # lambda length for this segment
		nr_top = nl
		nr_bot = m * Nst
		nr = nr_top + nr_bot

		A = np.eye(nl)
		V = np.zeros((nl, n_state))
		Phi_x = np.zeros((nl, n_cost))

		B = np.zeros((nr_bot, nl))
		Phi_u = np.zeros((nr_bot, n_cost))

		for k in range(Nst):
			# same v convention as the original code
			v_kp1 = float(Vtraj[k + 1]) if (k + 1) < len(Vtraj) else float(Vtraj[-1])

			ref_psi_k = None
			ref_psi_kp1 = None
			if ref_psi_traj is not None:
				ref_psi_k = float(np.asarray(ref_psi_traj)[k]) if k < len(ref_psi_traj) else float(np.asarray(ref_psi_traj)[-1])
				ref_psi_kp1 = float(np.asarray(ref_psi_traj)[k + 1]) if (k + 1) < len(ref_psi_traj) else float(np.asarray(ref_psi_traj)[-1])

			ref_kappa_k = None
			ref_kappa_kp1 = None
			if ref_kappa_traj is not None:
				ref_kappa_k = float(np.asarray(ref_kappa_traj)[k]) if k < len(ref_kappa_traj) else float(np.asarray(ref_kappa_traj)[-1])
				ref_kappa_kp1 = float(np.asarray(ref_kappa_traj)[k + 1]) if (k + 1) < len(ref_kappa_traj) else float(np.asarray(ref_kappa_traj)[-1])

			# cost gradients at x_{k+1}
			Jx = dphi_dx_pair(
				X[k],
				X[k + 1],
				v_kp1,
				float(target_lat_m),
				ref_psi_k=ref_psi_k,
				ref_psi_kp1=ref_psi_kp1,
				ref_kappa_k=ref_kappa_k,
				ref_kappa_kp1=ref_kappa_kp1,
			).T
			Jx = Jx * inv_feature_scales.reshape(1, -1)
			Phi_x[k * n_state : (k + 1) * n_state, :] = Jx
			Ju = dphi_du(U[k], v_kp1)
			Ju = Ju * inv_feature_scales.reshape(1, -1)
			Phi_u[k * m : (k + 1) * m, :] = Ju

			Dpsi = dpsi_dx_np(X[k + 1])  # (npsi x n_state)
			DfDx = C @ Kx @ Dpsi  # (n_state x n_state)

			if k < Nst - 1:
				A[
					k * n_state : (k + 1) * n_state,
					(k + 1) * n_state : (k + 2) * n_state,
				] = -DfDx.T
			else:
				# terminal coupling
				V[k * n_state : (k + 1) * n_state, :] = DfDx.T

			# control stationarity rows
			B[k * m : (k + 1) * m, k * n_state : (k + 1) * n_state] = DfDu_T

		seg_blocks.append((A, Phi_x, V, B, Phi_u, nl, nr_top, nr_bot))
		lambda_sizes.append(nl)
		mu_sizes.append(n_state)
		row_sizes.append(nr)

	if not seg_blocks:
		raise RuntimeError("No valid segments to build IOC system.")

	# ---- Global column layout: [lambda_1..lambda_M | omega | mu_1..mu_M]
	lambda_offsets = np.cumsum([0] + lambda_sizes[:-1]).tolist()
	omega_offset = int(sum(lambda_sizes))
	mu_offsets = (omega_offset + n_cost + np.cumsum([0] + mu_sizes[:-1])).tolist()

	total_cols = omega_offset + n_cost + sum(mu_sizes)
	total_rows = int(sum(row_sizes))

	F_big = np.zeros((total_rows, total_cols), dtype=float)

	# ---- Fill global matrix
	r0 = 0
	for i, (A, Phi_x, V, B, Phi_u, nl, nr_top, nr_bot) in enumerate(seg_blocks):
		cL = lambda_offsets[i]
		cM = mu_offsets[i]

		r1 = r0 + nr_top
		r2 = r1 + nr_bot

		# Top block: [A | -Phi_x | -V]
		F_big[r0:r1, cL : cL + nl] = A
		F_big[r0:r1, omega_offset : omega_offset + n_cost] = -Phi_x
		F_big[r0:r1, cM : cM + n_state] = -V

		# Bottom block: [B | Phi_u | 0]
		F_big[r1:r2, cL : cL + nl] = B
		F_big[r1:r2, omega_offset : omega_offset + n_cost] = Phi_u

		r0 = r2

	return F_big, omega_offset


def _project_omega_with_lower_bounds(
	omega: np.ndarray,
	*,
	lower_bounds: np.ndarray | None = None,
	default_min: float = 1e-5,
) -> np.ndarray:
	"""Project omega onto the simplex with per-index lower bounds."""
	omega = np.asarray(omega, dtype=float).reshape(-1)
	n = int(omega.shape[0])
	if n <= 0:
		raise ValueError("omega must be non-empty")

	if lower_bounds is None:
		lb = np.full(n, float(default_min), dtype=float)
	else:
		lb = np.asarray(lower_bounds, dtype=float).reshape(-1)
		if lb.shape[0] != n:
			raise ValueError(f"lower_bounds has wrong shape: expected {n}, got {lb.shape[0]}")
		lb = lb.copy()
		lb = np.maximum(lb, float(default_min))

	if (not np.isfinite(lb).all()) or np.any(lb < 0.0):
		raise ValueError("lower_bounds must be finite and >= 0")

	rem = 1.0 - float(np.sum(lb))
	if not np.isfinite(rem) or rem < 0.0:
		raise ValueError(f"Invalid omega lower bounds: sum={float(np.sum(lb)):.6g}")

	z = omega - lb
	z[z < 0.0] = 0.0
	s = float(np.sum(z))
	if s > 1e-12:
		z = z * (rem / s)
	elif n > 0:
		z[:] = rem / float(n)
	return z + lb


def _build_hlo_omega_lower_bounds(n_cost: int) -> np.ndarray:
	"""Lower bounds for learned HLO omega entries.

	Feature indices 3, 6, and 7 are clamped from the first IOC iteration.
	"""
	n_cost = int(n_cost)
	lower_bounds = np.full(n_cost, 1e-5, dtype=float)
	if n_cost > 3:
		lower_bounds[3] = 0.02
	if n_cost > 6:
		lower_bounds[6] = 0.02
	if n_cost > 7:
		lower_bounds[7] = 0.01
	return lower_bounds


def _make_omega_history_entry(
	*,
	stage: str,
	solve_index: int,
	segment_index: int,
	buffer_size: int,
	omega: np.ndarray,
) -> Dict[str, Any]:
	"""Create a serializable omega-history snapshot."""
	w = np.asarray(omega, dtype=float).reshape(-1)
	return {
		"stage": str(stage),
		"solve_index": int(solve_index),
		"segment_index": int(segment_index),
		"buffer_size": int(buffer_size),
		"omega": [float(x) for x in w.tolist()],
		"omega_sum": float(np.sum(w)),
	}


def solve_omega_from_F(
	F_big,
	omega_offset: int,
	n_cost: int,
	nu0=None,
	maxiter: int = 800,
	alpha_omega: float = 1e-3,
	omega_lower_bounds: np.ndarray | None = None,
):
	"""Solve omega from F_big with warm-start + tiny regularization.

	min ||F nu||^2 + alpha * ||omega - uniform||^2
	s.t. sum(omega)=1, omega>=0
	"""
	if minimize is None:
		raise RuntimeError("scipy is required for solve_omega_from_F (missing scipy.optimize.minimize)")

	omega_lower_bounds = _build_hlo_omega_lower_bounds(int(n_cost)) if omega_lower_bounds is None else np.asarray(omega_lower_bounds, dtype=float).reshape(-1)
	if omega_lower_bounds.shape[0] != int(n_cost):
		raise ValueError("omega_lower_bounds has wrong shape")

	col_norm = np.linalg.norm(F_big, axis=0)
	col_norm[col_norm == 0] = 1.0
	F_scaled = F_big / col_norm

	nu_dim = F_scaled.shape[1]
	Q = F_scaled.T @ F_scaled

	w_uniform = np.ones(n_cost) / n_cost

	def obj(nu):
		base = float(nu.T @ Q @ nu)
		w = nu[omega_offset : omega_offset + n_cost]
		reg = alpha_omega * float(np.sum((w - w_uniform) ** 2))
		return base + reg

	def cons_sum_omega(nu):
		return float(np.sum(nu[omega_offset : omega_offset + n_cost]) - 1.0)

	bounds = [(None, None)] * nu_dim
	for j in range(omega_offset, omega_offset + n_cost):
		bounds[j] = (float(omega_lower_bounds[j - omega_offset]), None)

	if nu0 is None or (len(nu0) != nu_dim):
		nu0 = np.zeros(nu_dim)
		nu0[omega_offset : omega_offset + n_cost] = _project_omega_with_lower_bounds(
			w_uniform,
			lower_bounds=omega_lower_bounds,
		)
	else:
		# Ensure warm-start respects lower bounds to avoid SLSQP struggling at init.
		nu0 = np.asarray(nu0, dtype=float).reshape(-1)
		if nu0.shape[0] != int(nu_dim):
			raise ValueError("nu0 has wrong shape")
		w0 = nu0[omega_offset : omega_offset + n_cost].copy()
		nu0[omega_offset : omega_offset + n_cost] = _project_omega_with_lower_bounds(
			w0,
			lower_bounds=omega_lower_bounds,
		)

	res = minimize(
		obj,
		nu0,
		method="SLSQP",
		constraints={"type": "eq", "fun": cons_sum_omega},
		bounds=bounds,
		options={"maxiter": maxiter, "ftol": 1e-12, "disp": False},
	)

	if not bool(res.success):
		print("[IOC] Warning: omega solve not fully converged:", res.message)

	nu = res.x.copy()
	omega = nu[omega_offset : omega_offset + n_cost].copy()
	omega = _project_omega_with_lower_bounds(omega, lower_bounds=omega_lower_bounds)
	nu[omega_offset : omega_offset + n_cost] = omega
	return omega, nu


def solve_omega_eliminate_costates(
	F_big: np.ndarray,
	omega_offset: int,
	n_cost: int,
	ridge: float = 1e-8,
	alpha_omega: float = 1e-6,
	w0: np.ndarray | None = None,
	maxiter: int = 200,
	omega_lower_bounds: np.ndarray | None = None,
):
	"""Fast omega solve by eliminating [lambda, mu] from paper-style stacked F.

	Solves:
	  min_{omega} || (I - P) Fw omega ||^2 + alpha||omega - uniform||^2
	  s.t. omega >= 0, sum omega = 1
	"""
	if minimize is None:
		raise RuntimeError("scipy is required for solve_omega_eliminate_costates (missing scipy.optimize.minimize)")

	omega_lower_bounds = _build_hlo_omega_lower_bounds(int(n_cost)) if omega_lower_bounds is None else np.asarray(omega_lower_bounds, dtype=float).reshape(-1)
	if omega_lower_bounds.shape[0] != int(n_cost):
		raise ValueError("omega_lower_bounds has wrong shape")

	# Split columns: [Fx | Fw]
	Fx = np.hstack(
		[
			F_big[:, :omega_offset],  # all lambdas
			F_big[:, omega_offset + n_cost :],  # all mus
		]
	)
	Fw = F_big[:, omega_offset : omega_offset + n_cost]

	# Column scaling (helps conditioning)
	cn_fx = np.linalg.norm(Fx, axis=0)
	cn_fx[cn_fx == 0] = 1.0
	Fx_s = Fx / cn_fx

	cn_fw = np.linalg.norm(Fw, axis=0)
	cn_fw[cn_fw == 0] = 1.0
	Fw_s = Fw / cn_fw

	# Compute projector residual operator R*v = v - Fx * argmin_x ||Fx x - v||
	# Use normal equations with ridge to avoid huge SVD.
	G = Fx_s.T @ Fx_s
	G_reg = G + ridge * np.eye(G.shape[0])

	# Compute R(Fw) efficiently: R(Fw) = Fw - Fx * (G_inv * Fx^T * Fw)
	T = Fx_s.T @ Fw_s  # (nx x n_cost)
	Xhat = np.linalg.solve(G_reg, T)
	R_Fw = Fw_s - Fx_s @ Xhat  # (nrows x n_cost)

	H = R_Fw.T @ R_Fw  # (n_cost x n_cost)

	w_uniform = np.ones(n_cost) / n_cost

	def obj(w):
		w = np.asarray(w, dtype=float)
		base = float(w.T @ H @ w)
		reg = alpha_omega * float(np.sum((w - w_uniform) ** 2))
		return base + reg

	def cons_sum(w):
		return float(np.sum(w) - 1.0)

	bounds = [(0.0, None)] * n_cost
	for i in range(int(n_cost)):
		bounds[i] = (float(omega_lower_bounds[i]), None)
	if w0 is None or len(w0) != n_cost:
		w0 = _project_omega_with_lower_bounds(w_uniform, lower_bounds=omega_lower_bounds)
	else:
		w0 = _project_omega_with_lower_bounds(np.asarray(w0, dtype=float), lower_bounds=omega_lower_bounds)

	res = minimize(
		obj,
		w0,
		method="SLSQP",
		constraints={"type": "eq", "fun": cons_sum},
		bounds=bounds,
		options={"maxiter": maxiter, "ftol": 1e-12, "disp": False},
	)

	if not res.success:
		print("[IOC] Warning: omega solve not fully converged:", res.message)

	omega = res.x.copy()
	omega = _project_omega_with_lower_bounds(omega, lower_bounds=omega_lower_bounds)
	return omega, res


def eval_ioc_residual(
	F_big: np.ndarray,
	omega_offset: int,
	omega: np.ndarray,
	*,
	n_cost: int,
	ridge: float = 1e-8,
) -> dict:
	omega = np.asarray(omega, dtype=float).reshape(-1)
	if omega.shape[0] != int(n_cost):
		raise ValueError("omega has wrong shape")
	Fx = np.hstack([F_big[:, :omega_offset], F_big[:, omega_offset + n_cost :]])
	Fw = F_big[:, omega_offset : omega_offset + n_cost]

	cn_fx = np.linalg.norm(Fx, axis=0)
	cn_fx[cn_fx == 0] = 1.0
	Fx_s = Fx / cn_fx

	cn_fw = np.linalg.norm(Fw, axis=0)
	cn_fw[cn_fw == 0] = 1.0
	Fw_s = Fw / cn_fw

	G = Fx_s.T @ Fx_s
	G_reg = G + float(ridge) * np.eye(G.shape[0])

	T = Fx_s.T @ Fw_s
	Xhat = np.linalg.solve(G_reg, T)
	R_Fw = Fw_s - Fx_s @ Xhat

	r = R_Fw @ omega.reshape(-1, 1)
	resid = float(np.linalg.norm(r))
	return {"residual_l2": resid, "rows": int(F_big.shape[0])}


def summarize_feature_magnitudes(
	segments_X: List[np.ndarray],
	segments_U: List[np.ndarray],
	segments_V: List[np.ndarray],
	segments_target_lat_m: List[float],
	segments_ref_orientation: Optional[List[np.ndarray]],
	segments_ref_curvature: Optional[List[np.ndarray]],
	*,
	dt_s: float,
	softness_m: float,
	X_mean: np.ndarray,
	X_std: np.ndarray,
	U_mean: np.ndarray,
	U_std: np.ndarray,
	max_segments: int = 0,
) -> dict:
	X_mean = np.asarray(X_mean, dtype=float).reshape(-1)
	X_std = np.asarray(X_std, dtype=float).reshape(-1)
	U_mean = np.asarray(U_mean, dtype=float).reshape(-1)
	U_std = np.asarray(U_std, dtype=float).reshape(-1)
	dt_s = float(dt_s)
	softness_m = float(softness_m)

	def sigmoid(z: float) -> float:
		if z >= 0:
			e = np.exp(-z)
			return float(1.0 / (1.0 + e))
		e = np.exp(z)
		return float(e / (1.0 + e))

	def wrap_to_pi(a: float) -> float:
		return float(np.arctan2(np.sin(a), np.cos(a)))

	f0_vals: List[float] = []  # lat_err_sq
	f1_vals: List[float] = []  # lat_rate_sq_v2
	f2_vals: List[float] = []  # kappa_abs_sq
	f3_vals: List[float] = []  # kappa_dot_abs_sq
	f4_vals: List[float] = []  # kappa_err_sq
	f5_vals: List[float] = []  # delta_kappa_err_sq
	f6_vals: List[float] = []  # delta_kappa_dot_sq
	f7_vals: List[float] = []  # u_sq
	f8_vals: List[float] = []  # psi_err_sq

	n_seg_used = 0
	if segments_ref_orientation is None:
		segments_ref_orientation = [None for _ in segments_X]
	if segments_ref_curvature is None:
		segments_ref_curvature = [None for _ in segments_X]

	for Xseg_n, Useg_n, Vseg, target_lat, psiref_seg, kref_seg in zip(
		segments_X,
		segments_U,
		segments_V,
		segments_target_lat_m,
		segments_ref_orientation,
		segments_ref_curvature,
	):
		N = len(Xseg_n)
		if N < 2:
			continue
		Xseg_raw = np.asarray(Xseg_n) * X_std + X_mean
		Useg_raw = np.asarray(Useg_n) * U_std + U_mean

		kref_raw = np.asarray(kref_seg, dtype=float).reshape(-1) if kref_seg is not None else None
		psiref_raw = np.asarray(psiref_seg, dtype=float).reshape(-1) if psiref_seg is not None else None

		for k in range(N - 1):
			xk_raw = Xseg_raw[k]
			xkp1_raw = Xseg_raw[k + 1]
			lat_kp1 = float(xkp1_raw[0])
			lat_k = float(xk_raw[0])
			kappa_k = float(xk_raw[2])
			kappa_kp1 = float(xkp1_raw[2])
			kdot_k = float(xk_raw[3])
			kdot_kp1 = float(xkp1_raw[3])

			u_vec = np.asarray(Useg_raw[k], dtype=float).reshape(-1)
			u_k = float(u_vec[0]) if u_vec.size > 0 else 0.0

			v_kp1 = float(Vseg[k + 1]) if (k + 1) < len(Vseg) else float(Vseg[-1])
			v_eps = 1e-3
			v_eff = max(v_kp1, v_eps)
			v2 = v_eff * v_eff
			v4 = v2 * v2

			e_d = lat_kp1 - float(target_lat)
			f0_vals.append(e_d * e_d)
			d_dot = (lat_kp1 - lat_k) / float(max(dt_s, 1e-9))
			f1_vals.append(v2 * (d_dot * d_dot))

			kref_k = float(kref_raw[k]) if (kref_raw is not None and k < len(kref_raw)) else 0.0
			kref_kp1 = float(kref_raw[k + 1]) if (kref_raw is not None and (k + 1) < len(kref_raw)) else 0.0
			e_kappa_k = kappa_k - kref_k
			e_kappa_kp1 = kappa_kp1 - kref_kp1

			f2_vals.append(v4 * (kappa_kp1 * kappa_kp1))
			f3_vals.append(v4 * (kdot_kp1 * kdot_kp1))
			f4_vals.append(v4 * (e_kappa_kp1 * e_kappa_kp1))
			de = e_kappa_kp1 - e_kappa_k
			f5_vals.append(v4 * (de * de))
			dkdot = kdot_kp1 - kdot_k
			f6_vals.append(v4 * (dkdot * dkdot))
			f7_vals.append(v4 * (u_k * u_k))

			psi_ref_kp1 = float(psiref_raw[k + 1]) if (psiref_raw is not None and (k + 1) < len(psiref_raw)) else 0.0
			psi_kp1 = float(xkp1_raw[1])
			e_psi = wrap_to_pi(psi_kp1 - psi_ref_kp1)
			f8_vals.append(v2 * (e_psi * e_psi))

		n_seg_used += 1
		if max_segments and n_seg_used >= int(max_segments):
			break

	def stats(arr: List[float]) -> dict:
		if not arr:
			return {"n": 0}
		a = np.asarray(arr, dtype=float)
		return {
			"n": int(a.size),
			"mean": float(np.mean(a)),
			"median": float(np.median(a)),
			"p10": float(np.percentile(a, 10)),
			"p90": float(np.percentile(a, 90)),
			"min": float(np.min(a)),
			"max": float(np.max(a)),
		}

	return {
		"n_segments_used": int(n_seg_used),
		"lat_err_sq": stats(f0_vals),
		"lat_rate_sq_v2": stats(f1_vals),
		"kappa_abs_sq": stats(f2_vals),
		"kappa_dot_abs_sq": stats(f3_vals),
		"kappa_err_sq": stats(f4_vals),
		"delta_kappa_err_sq": stats(f5_vals),
		"delta_kappa_dot_sq": stats(f6_vals),
		"u_sq": stats(f7_vals),
		"psi_err_sq": stats(f8_vals),
	}


def compute_feature_scales(
	*,
	segments_X: List[np.ndarray],
	segments_U: List[np.ndarray],
	segments_V: List[np.ndarray],
	segments_target_lat_m: List[float],
	segments_ref_orientation: List[np.ndarray],
	segments_ref_curvature: List[np.ndarray],
	dt_s: float,
	X_mean: np.ndarray,
	X_std: np.ndarray,
	U_mean: np.ndarray,
	U_std: np.ndarray,
	softness_m: float = 0.2,
	method: str = "median",
	eps: float = 1e-6,
) -> tuple[np.ndarray, dict]:
	stats_dict = summarize_feature_magnitudes(
		segments_X,
		segments_U,
		segments_V,
		segments_target_lat_m,
		segments_ref_orientation,
		segments_ref_curvature,
		dt_s=float(dt_s),
		softness_m=float(softness_m),
		X_mean=X_mean,
		X_std=X_std,
		U_mean=U_mean,
		U_std=U_std,
	)

	method = str(method)
	eps = float(eps)

	def pick_scale(d: dict) -> float:
		if not isinstance(d, dict):
			return eps
		cands = []
		if method in d:
			cands.append(d.get(method))
		for k in ["p90", "median", "mean", "max"]:
			if k in d:
				cands.append(d.get(k))
		for c in cands:
			try:
				c = float(c)
			except Exception:
				continue
			if np.isfinite(c) and c > 0:
				return c
		return eps

	feature_scales = np.asarray(
		[
			pick_scale(stats_dict.get("lat_err_sq", {})),
			pick_scale(stats_dict.get("lat_rate_sq_v2", {})),
			pick_scale(stats_dict.get("kappa_abs_sq", {})),
			pick_scale(stats_dict.get("kappa_dot_abs_sq", {})),
			pick_scale(stats_dict.get("kappa_err_sq", {})),
			pick_scale(stats_dict.get("delta_kappa_err_sq", {})),
			pick_scale(stats_dict.get("delta_kappa_dot_sq", {})),
			pick_scale(stats_dict.get("u_sq", {})),
			pick_scale(stats_dict.get("psi_err_sq", {})),
		],
		dtype=float,
	)
	if (not np.isfinite(feature_scales).all()) or feature_scales.shape != (9,):
		raise RuntimeError("Invalid feature_scales")
	return feature_scales, stats_dict


def learn_hlo_omega(
	*,
	segments_X: List[np.ndarray],
	segments_U: List[np.ndarray],
	segments_V: List[np.ndarray],
	segments_target_lat_m: List[float],
	segments_dt_s: List[float],
	segments_ref_orientation: List[np.ndarray],
	segments_ref_curvature: List[np.ndarray],
	Kx: np.ndarray,
	Ku: np.ndarray,
	C: np.ndarray,
	dpsi_dx_np,
	X_mean: np.ndarray,
	X_std: np.ndarray,
	U_mean: np.ndarray,
	U_std: np.ndarray,
	feature_scales: np.ndarray,
	feature_scale_eps: float = 1e-6,
	ioc_every: int = 20,
	window: int = 200,
	softness_m: float = 0.2,
	return_history: bool = False,
	verbose: bool = False,
) -> Any:
	n_state = 4
	n_cost = 9
	omega_lower_bounds = _build_hlo_omega_lower_bounds(n_cost)
	omega = _project_omega_with_lower_bounds(
		np.ones(n_cost, dtype=float) / float(n_cost),
		lower_bounds=omega_lower_bounds,
	)
	omega_history: List[Dict[str, Any]] = [
		_make_omega_history_entry(
			stage="init",
			solve_index=0,
			segment_index=-1,
			buffer_size=0,
			omega=omega,
		)
	]
	solve_index = 0
	IOC_EVERY = max(1, int(ioc_every))
	WINDOW = max(1, int(window))
	if bool(verbose):
		print(
			f"[HLO] learn_hlo_omega: segments={len(segments_X)} ioc_every={IOC_EVERY} window={WINDOW}",
			file=sys.stderr,
			flush=True,
		)

	buffer_X: List[np.ndarray] = []
	buffer_U: List[np.ndarray] = []
	buffer_V: List[np.ndarray] = []
	buffer_target_lat: List[float] = []
	buffer_dt: List[float] = []
	buffer_ref_ori: List[np.ndarray] = []
	buffer_ref_kappa: List[np.ndarray] = []

	start_t = time.perf_counter()
	last_log_t = start_t
	for it in range(len(segments_X)):
		buffer_X.append(segments_X[it])
		buffer_U.append(segments_U[it])
		buffer_V.append(segments_V[it])
		buffer_target_lat.append(float(segments_target_lat_m[it]))
		buffer_dt.append(float(segments_dt_s[it]))
		buffer_ref_ori.append(segments_ref_orientation[it])
		buffer_ref_kappa.append(segments_ref_curvature[it])

		if len(buffer_X) > WINDOW:
			buffer_X = buffer_X[-WINDOW:]
			buffer_U = buffer_U[-WINDOW:]
			buffer_V = buffer_V[-WINDOW:]
			buffer_target_lat = buffer_target_lat[-WINDOW:]
			buffer_dt = buffer_dt[-WINDOW:]
			buffer_ref_ori = buffer_ref_ori[-WINDOW:]
			buffer_ref_kappa = buffer_ref_kappa[-WINDOW:]

		if (it % IOC_EVERY == 0) or (it == len(segments_X) - 1):
			if bool(verbose):
				elapsed = time.perf_counter() - start_t
				print(
					f"[HLO] IOC solve @ segment {it+1}/{len(segments_X)} (buffer={len(buffer_X)}) elapsed={elapsed:.1f}s",
					file=sys.stderr,
					flush=True,
				)
				t_build0 = time.perf_counter()
			F_big, omega_offset = build_F_big_sharedomega(
				buffer_X,
				buffer_U,
				buffer_V,
				Kx,
				Ku,
				C,
				dpsi_dx_np,
				n_state=n_state,
				n_cost=n_cost,
				buffer_target_lat_offset_m=buffer_target_lat,
				buffer_ref_orientation_rad=buffer_ref_ori,
				buffer_ref_curvature_1pm=buffer_ref_kappa,
				dt_s=float(np.median(buffer_dt)) if buffer_dt else 1.0,
				softness_m=float(softness_m),
				x_mean=X_mean,
				x_std=X_std,
				u_mean=U_mean,
				u_std=U_std,
				feature_scales=np.asarray(feature_scales, dtype=float),
				feature_scale_eps=float(feature_scale_eps),
			)
			if bool(verbose):
				t_build = time.perf_counter() - t_build0
				print(
					f"[HLO] built F_big shape={tuple(F_big.shape)} build_time={t_build:.1f}s",
					file=sys.stderr,
					flush=True,
				)
				t_solve0 = time.perf_counter()
			omega, _ = solve_omega_eliminate_costates(
				F_big,
				omega_offset,
				n_cost=n_cost,
				ridge=1e-8,
				alpha_omega=1e-3,
				w0=omega,
				maxiter=200,
				omega_lower_bounds=omega_lower_bounds,
			)
			solve_index += 1
			omega_history.append(
				_make_omega_history_entry(
					stage="ioc_update",
					solve_index=solve_index,
					segment_index=int(it),
					buffer_size=len(buffer_X),
					omega=omega,
				)
			)
			if bool(verbose):
				t_solve = time.perf_counter() - t_solve0
				w = np.asarray(omega, dtype=float).reshape(-1)
				w_txt = ",".join([f"{float(x):.6g}" for x in w.tolist()])
				print(
					f"[HLO] omega updated (solve_time={t_solve:.1f}s) omega=[{w_txt}] sum={float(np.sum(w)):.6g}",
					file=sys.stderr,
					flush=True,
				)
				last_log_t = time.perf_counter()
		elif bool(verbose) and (time.perf_counter() - last_log_t) > 30.0:
			# Heartbeat in case build_F_big/solve are slow on some machines.
			elapsed = time.perf_counter() - start_t
			print(
				f"[HLO] learn_hlo_omega heartbeat: segment {it+1}/{len(segments_X)} elapsed={elapsed:.1f}s",
				file=sys.stderr,
				flush=True,
			)
			last_log_t = time.perf_counter()

	omega = np.asarray(omega, dtype=float).reshape(-1)
	omega = _project_omega_with_lower_bounds(omega, lower_bounds=omega_lower_bounds)
	if omega_history:
		omega_history[-1]["omega"] = [float(x) for x in omega.tolist()]
		omega_history[-1]["omega_sum"] = float(np.sum(omega))
	if bool(return_history):
		return omega, omega_history
	return omega


if __name__ == "__main__":
	raise SystemExit(main())

