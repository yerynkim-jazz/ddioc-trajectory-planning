"""Theta learning via bilevel optimization.

1) Start with an arbitrary QP planner parameter vector `theta`.
2) Insert `theta` into a QP planner JSON config and generate an (inner) optimal trajectory.
3) Evaluate that trajectory in a high-level objective (HLO) and obtain a scalar cost.
4) Find the `theta` that minimizes the HLO cost (outer optimization).

Because the concrete QP planner / HLO executables differ across setups, this module is
implemented with *hooks*:

- Update a JSON file given a list of key-paths.
- Run an external planner command (optional) to produce a trajectory file (CSV or JSON trace).
- Evaluate cost either by:
  (a) running an external HLO command and parsing the output, or
	(b) computing a simple HLO cost directly from a trajectory file.

The goal is to provide a practical, runnable implementation without hard-coding
project-specific binaries.

HLO omega (objective weight) learning/loading is implemented in
`learn_high_level_objective.py` and imported here.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
	# When executed as a module: `python -m Pipeline.learn_theta`
	from .hlo_learning import (
		compute_hlo_omega_via_pipeline,
		load_hlo_feature_scales_from_learned_objective_json,
		load_hlo_omega_from_learned_objective_json,
	)
except Exception:  # pragma: no cover
	# When executed as a script from this folder: `python learn_theta.py`
	from hlo_learning import (
		compute_hlo_omega_via_pipeline,
		load_hlo_feature_scales_from_learned_objective_json,
		load_hlo_omega_from_learned_objective_json,
	)

# Allow running this module from outside the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(_REPO_ROOT))

try:
	from scipy.optimize import minimize
except Exception:  # pragma: no cover
	minimize = None


def _normalize_scipy_minimize_method(method: str) -> str:
	"""Normalize common aliases for scipy.optimize.minimize(method=...).

	This keeps CLI inputs forgiving (e.g. 'nelder_mead', 'nelder mead').
	If the method is unknown, return it unchanged and let SciPy validate.
	"""
	m = str(method).strip()
	if not m:
		return "Powell"
	key = re.sub(r"[\s_\-]+", "", m.lower())
	if key == "powell":
		return "Powell"
	if key in {"neldermead", "nm"}:
		return "Nelder-Mead"
	return m


def _import_skopt():  # pragma: no cover
	"""Lazy import for scikit-optimize (used for Bayesian optimization)."""
	try:
		from skopt import gp_minimize  # type: ignore
		from skopt.space import Real  # type: ignore
		return gp_minimize, Real
	except Exception as e:
		raise RuntimeError(
			"Bayesian optimization requires 'scikit-optimize'. Install it in your venv: 'pip install scikit-optimize'."
		) from e


def _import_pymoo():  # pragma: no cover
	"""Lazy import for pymoo (used for multi-objective evolutionary optimization)."""
	try:
		from pymoo.algorithms.moo.nsga2 import NSGA2  # type: ignore
		from pymoo.core.problem import ElementwiseProblem  # type: ignore
		from pymoo.operators.crossover.sbx import SBX  # type: ignore
		from pymoo.operators.mutation.pm import PM  # type: ignore
		from pymoo.operators.sampling.rnd import FloatRandomSampling  # type: ignore
		from pymoo.optimize import minimize as pymoo_minimize  # type: ignore
		from pymoo.termination import get_termination  # type: ignore
		return {
			"NSGA2": NSGA2,
			"ElementwiseProblem": ElementwiseProblem,
			"SBX": SBX,
			"PM": PM,
			"FloatRandomSampling": FloatRandomSampling,
			"pymoo_minimize": pymoo_minimize,
			"get_termination": get_termination,
		}
	except Exception as e:
		raise RuntimeError(
			"Multi-objective evolutionary optimization requires 'pymoo'. Install it in your venv: 'pip install pymoo'."
		) from e


# -----------------------------------------------------------------------------
# Optimization parameterization helpers


def _sigmoid(x: np.ndarray) -> np.ndarray:
	"""Numerically stable sigmoid."""
	x = np.asarray(x, dtype=float)
	# Avoid np.where here: it evaluates both branches and can overflow.
	out = np.empty_like(x, dtype=float)
	pos = x >= 0
	if np.any(pos):
		out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
	neg = ~pos
	if np.any(neg):
		ex = np.exp(x[neg])  # underflows safely for large negative x
		out[neg] = ex / (1.0 + ex)
	return out


def _logit(p: np.ndarray) -> np.ndarray:
	"""Inverse sigmoid; expects p in (0,1)."""
	p = np.asarray(p, dtype=float)
	return np.log(p) - np.log1p(-p)


def _softplus(x: np.ndarray) -> np.ndarray:
	"""Numerically stable softplus."""
	x = np.asarray(x, dtype=float)
	# Use logaddexp for true numerical stability.
	# NOTE: np.where evaluates both branches, so the previous implementation could
	# still overflow when exp(x) is computed for large x.
	return np.logaddexp(0.0, x)


def _inv_softplus(y: np.ndarray) -> np.ndarray:
	"""Approximate inverse of softplus for y>0."""
	y = np.asarray(y, dtype=float)
	# For small y, expm1(y) is stable.
	return np.where(y > 20, y, np.log(np.expm1(np.maximum(y, 1e-12))))


def _broadcast_param(x: Optional[Sequence[float]], *, n: int, default: float) -> np.ndarray:
	if x is None:
		return np.full((n,), float(default), dtype=float)
	arr = np.asarray(list(x), dtype=float).reshape(-1)
	if arr.size == 1:
		return np.full((n,), float(arr.item()), dtype=float)
	if arr.size != n:
		raise ValueError(f"Expected 1 or {n} values, got {arr.size}.")
	return arr


def _theta_from_opt_vars(
	x: np.ndarray,
	*,
	transform: str,
	theta_min: Optional[Sequence[float]],
	theta_max: Optional[Sequence[float]],
) -> np.ndarray:
	"""Map unconstrained optimizer variables to the physical theta used for planning/HLO."""
	x = np.asarray(x, dtype=float).reshape(-1)
	transform = str(transform)
	if transform == "identity":
		return x
	if transform == "softplus":
		return _softplus(x)
	if transform == "sigmoid":
		lo = _broadcast_param(theta_min, n=x.size, default=0.0)
		hi = _broadcast_param(theta_max, n=x.size, default=1.0)
		if not np.isfinite(lo).all() or not np.isfinite(hi).all():
			raise ValueError("theta_min/theta_max must be finite")
		if np.any(hi <= lo):
			raise ValueError("theta_max must be > theta_min (elementwise)")
		p = _sigmoid(x)
		return lo + (hi - lo) * p
	raise ValueError(f"Unknown theta transform: {transform!r}")


def _opt_vars_from_theta0(
	theta0: np.ndarray,
	*,
	transform: str,
	theta_min: Optional[Sequence[float]],
	theta_max: Optional[Sequence[float]],
) -> np.ndarray:
	"""Initialize optimizer variables so that the physical theta starts at theta0."""
	theta0 = np.asarray(theta0, dtype=float).reshape(-1)
	transform = str(transform)
	if transform == "identity":
		return theta0
	if transform == "softplus":
		return _inv_softplus(np.maximum(theta0, 0.0))
	if transform == "sigmoid":
		lo = _broadcast_param(theta_min, n=theta0.size, default=0.0)
		hi = _broadcast_param(theta_max, n=theta0.size, default=1.0)
		# Clamp into open interval to avoid infinities.
		eps = 1e-9
		y = (theta0 - lo) / (hi - lo)
		y = np.clip(y, eps, 1.0 - eps)
		return _logit(y)
	raise ValueError(f"Unknown theta transform: {transform!r}")


# -----------------------------------------------------------------------------
# JSON path utilities


_INDEX_RE = re.compile(r"^(?P<key>[^\[]+)(?:\[(?P<idx>-?\d+)\])?$")


def _parse_path(path: str) -> List[Tuple[str, Optional[int]]]:
	"""Parse a dotted path with optional list indices.

	Examples:
		"a.b[0].c" -> [("a", None), ("b", 0), ("c", None)]
	"""
	parts: List[Tuple[str, Optional[int]]] = []
	for raw in path.split("."):
		m = _INDEX_RE.match(raw)
		if not m:
			raise ValueError(f"Invalid key path segment: {raw!r} in {path!r}")
		key = m.group("key")
		idx = m.group("idx")
		parts.append((key, int(idx) if idx is not None else None))
	return parts


def json_set(obj: Any, path: str, value: Any) -> None:
	"""Set `obj[path] = value` where `path` is dot-separated with optional indices."""
	parts = _parse_path(path)
	cur = obj
	for i, (key, idx) in enumerate(parts):
		is_last = i == len(parts) - 1
		if not isinstance(cur, dict):
			raise TypeError(f"While setting {path!r}: expected dict at segment {key!r}, got {type(cur).__name__}")
		if key not in cur:
			# Create intermediate containers.
			cur[key] = [] if idx is not None else {}
		if idx is None:
			if is_last:
				cur[key] = value
				return
			cur = cur[key]
			continue

		# idx is not None => list indexing
		lst = cur[key]
		if not isinstance(lst, list):
			raise TypeError(
				f"While setting {path!r}: expected list at {key!r}, got {type(lst).__name__}"
			)
		# Ensure list has sufficient length.
		need = idx + 1 if idx >= 0 else 0
		if idx >= 0 and len(lst) < need:
			lst.extend([None] * (need - len(lst)))
		if is_last:
			lst[idx] = value
			return
		# Create intermediate dict if needed.
		if lst[idx] is None:
			lst[idx] = {}
		cur = lst[idx]


def load_json(path: Path) -> Any:
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)


def save_json_atomic(path: Path, data: Any) -> None:
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), encoding="utf-8") as tf:
		json.dump(data, tf, indent=2, sort_keys=True)
		tf.write("\n")
		tmp = Path(tf.name)
	os.replace(tmp, path)


# -----------------------------------------------------------------------------
# Planner overwrite-parameter JSON helpers


PLANNER_CHANGE_LANE_PARAM_SPECS: List[Tuple[str, str, int]] = [
	("tpl_change_lane_weight_lateral_acceleration", "1", 0),
	("tpl_change_lane_weight_lateral_jerk", "2", 1),
	("tpl_change_lane_weight_lateral_offset_fast", "3", 2),
	("tpl_change_lane_weight_lateral_offset_slow", "4", 3),
	("tpl_change_lane_weight_lateral_snap", "5", 4),
	("tpl_change_lane_weight_lateral_velocity_fast", "6", 5),
	("tpl_change_lane_weight_lateral_velocity_slow", "7", 6),
]


_LANE_CHANGE_THETA_KEYS: List[str] = [name for (name, _desc, _idx) in PLANNER_CHANGE_LANE_PARAM_SPECS]


def _lane_change_expand_theta5_to_theta7(theta5: Sequence[float]) -> np.ndarray:
	"""Expand 5 tied parameters into the 7 lane-change planner weights.

	Parameter order (5D):
		[accel, jerk, offset, snap, velocity]

	Expanded order (7D):
		[accel, jerk, offset_fast, offset_slow, snap, vel_fast, vel_slow]
	"""
	arr = np.asarray(list(theta5), dtype=float).reshape(-1)
	if arr.size != 5:
		raise ValueError(f"Expected 5 parameters for tied lane-change theta, got {arr.size}")
	acc, jerk, off, snap, vel = [float(x) for x in arr.tolist()]
	return np.asarray([acc, jerk, off, off, snap, vel, vel], dtype=float)


def _lane_change_reduce_theta7_to_theta5(theta7: Sequence[float]) -> np.ndarray:
	"""Reduce 7 lane-change weights to 5 by averaging the tied pairs."""
	arr = np.asarray(list(theta7), dtype=float).reshape(-1)
	if arr.size != 7:
		raise ValueError(f"Expected 7 parameters for lane-change theta, got {arr.size}")
	acc = float(arr[0])
	jerk = float(arr[1])
	off = 0.5 * (float(arr[2]) + float(arr[3]))
	snap = float(arr[4])
	vel = 0.5 * (float(arr[5]) + float(arr[6]))
	return np.asarray([acc, jerk, off, snap, vel], dtype=float)


def _repeat_value(value: float, *, n: int = 10) -> List[float]:
	value_f = float(value)
	if not np.isfinite(value_f):
		raise ValueError(f"initValue must be finite, got {value!r}")
	if int(n) <= 0:
		raise ValueError("n must be positive")
	return [value_f] * int(n)


def upsert_planner_overwrite_parameters_from_theta(
	data: Dict[str, Any],
	*,
	theta: Sequence[float],
	n_repeat: int = 10,
	param_specs: Sequence[Tuple[str, str, int]] = PLANNER_CHANGE_LANE_PARAM_SPECS,
) -> Dict[str, Any]:
	"""Upsert planner overwrite-parameter entries for a theta vector.

	This updates/creates entries in a JSON structure of the form:
		{"parameters": [{"name": ..., "description": ..., "initValue": ...}, ...]}

	For each spec (name, description, theta_index):
		- Ensures the parameter exists.
		- Sets "initValue" to a length-n_repeat list containing the same value.
		- Sets "description" to the provided description string.

	Returns the same dict instance (mutated) for convenience.
	"""
	if not isinstance(data, dict):
		raise TypeError(f"data must be a dict, got {type(data).__name__}")

	theta_arr = np.asarray(list(theta), dtype=float).reshape(-1)
	if theta_arr.size == 0:
		raise ValueError("theta must be non-empty")
	if not np.isfinite(theta_arr).all():
		raise ValueError("theta must be finite")

	params = data.get("parameters")
	if params is None:
		params = []
		data["parameters"] = params
	if not isinstance(params, list):
		raise TypeError(f"data['parameters'] must be a list, got {type(params).__name__}")

	# Build name -> object index map (first occurrence wins).
	name_to_idx: Dict[str, int] = {}
	for i, item in enumerate(params):
		if not isinstance(item, dict):
			continue
		name = item.get("name")
		if isinstance(name, str) and name not in name_to_idx:
			name_to_idx[name] = i

	for name, desc, theta_idx in param_specs:
		i = int(theta_idx)
		if i < 0 or i >= theta_arr.size:
			raise ValueError(
				f"theta index out of range for {name!r}: idx={i}, theta has size {theta_arr.size}"
			)
		init_value = _repeat_value(theta_arr[i], n=int(n_repeat))

		if name in name_to_idx:
			obj = params[name_to_idx[name]]
			if not isinstance(obj, dict):
				obj = {"name": name}
				params[name_to_idx[name]] = obj
		else:
			obj = {"name": name}
			params.append(obj)
			name_to_idx[name] = len(params) - 1

		obj["name"] = name
		obj["description"] = str(desc)
		obj["initValue"] = init_value

	return data


def write_planner_overwrite_parameters_from_theta(
	*,
	input_json_path: Path,
	output_json_path: Path,
	theta: Sequence[float],
	n_repeat: int = 10,
	param_specs: Sequence[Tuple[str, str, int]] = PLANNER_CHANGE_LANE_PARAM_SPECS,
) -> None:
	"""Read a planner overwrite JSON, upsert parameters from theta, write to output.

	This intentionally does NOT modify the input file unless you pass the same path as
	input_json_path and output_json_path.
	"""
	input_json_path = Path(input_json_path)
	output_json_path = Path(output_json_path)
	data = load_json(input_json_path)
	if not isinstance(data, dict):
		raise TypeError(f"Top-level JSON must be an object/dict, got {type(data).__name__}")
	upsert_planner_overwrite_parameters_from_theta(
		data,
		theta=theta,
		n_repeat=n_repeat,
		param_specs=param_specs,
	)
	save_json_atomic(output_json_path, data)


def set_planner_overwrite_parameter_value(
	data: Dict[str, Any],
	*,
	name: str,
	value: float,
	n_repeat: int = 10,
	description: Optional[str] = None,
) -> None:
	"""Set one parameter in a planner overwrite JSON (top-level "parameters" list).

	- Upserts an entry with matching "name".
	- Sets "initValue" to a list of length n_repeat with the same float.
	- If `description` is provided, overwrites it; otherwise preserves existing.
	"""
	if not isinstance(data, dict):
		raise TypeError(f"data must be a dict, got {type(data).__name__}")
	if not isinstance(name, str) or not name:
		raise ValueError("name must be a non-empty string")

	params = data.get("parameters")
	if params is None:
		params = []
		data["parameters"] = params
	if not isinstance(params, list):
		raise TypeError(f"data['parameters'] must be a list, got {type(params).__name__}")

	idx: Optional[int] = None
	for i, item in enumerate(params):
		if isinstance(item, dict) and item.get("name") == name:
			idx = i
			break

	entry: Dict[str, Any]
	if idx is None:
		entry = {"name": name}
		params.append(entry)
	else:
		obj = params[idx]
		entry = obj if isinstance(obj, dict) else {"name": name}
		params[idx] = entry

	entry["name"] = name
	if description is not None:
		entry["description"] = str(description)
	elif "description" not in entry:
		# Keep file minimal; don't invent descriptions.
		pass
	entry["initValue"] = _repeat_value(float(value), n=int(n_repeat))


def apply_theta_to_qp_json(
	data: Dict[str, Any],
	*,
	theta_keys: Sequence[str],
	theta: Sequence[float],
	qp_repeat: int = 10,
) -> None:
	"""Apply theta to a QP config JSON.

	Two supported key styles in `theta_keys`:
	1) Dotted JSON paths (existing behavior): e.g. "weights.lat" or "a.b[0].c".
	2) Planner overwrite parameter names (for JSON shaped like {"parameters": [...]}):
	   e.g. "tpl_change_lane_weight_lateral_jerk".

	Heuristic: if the key contains '.' or '[' we treat it as a JSON path; otherwise we
	treat it as a planner overwrite parameter name.
	"""
	keys = list(theta_keys)
	vals = np.asarray(list(theta), dtype=float).reshape(-1)
	if len(keys) != int(vals.size):
		raise ValueError(f"theta_keys length ({len(keys)}) must match theta length ({int(vals.size)})")
	for k, v in zip(keys, vals.tolist()):
		k = str(k)
		if "." in k or "[" in k:
			json_set(data, k, float(v))
		else:
			set_planner_overwrite_parameter_value(data, name=k, value=float(v), n_repeat=int(qp_repeat))


# -----------------------------------------------------------------------------
# Trajectory / cost evaluation


_TRAJ_REQUIRED_COLS: Tuple[str, ...] = (
	# Required columns (SG-filtered). If missing, we compute them from raw columns.
	"lateral_offset_m_sg",
	"target_orientation_rad_sg",
	"target_curvature_1pm_sg",
)


def _require_scipy_savgol_filter():
	"""Return scipy.signal.savgol_filter or raise a helpful error."""
	try:
		from scipy.signal import savgol_filter  # type: ignore
	except Exception as e:  # pragma: no cover
		raise ModuleNotFoundError(
			"SciPy is required to compute SG-filtered (*_sg) trajectory columns. "
			"Install it with `pip install scipy` or ensure the planner writes *_sg columns."
		) from e
	return savgol_filter


def _wrap_to_pi_np(a: np.ndarray) -> np.ndarray:
	return np.arctan2(np.sin(a), np.cos(a))


def _fill_nans_linear(y: np.ndarray) -> np.ndarray:
	"""Fill NaNs by linear interpolation (edge-filled)."""
	y = np.asarray(y, dtype=float).reshape(-1)
	if y.size == 0:
		return y
	mask = np.isfinite(y)
	if mask.all():
		return y
	idx = np.arange(y.size)
	if not mask.any():
		# Degenerate: all NaN/inf
		return np.zeros_like(y)
	# Edge fill then interpolate interior.
	y0 = y.copy()
	first = int(idx[mask][0])
	last = int(idx[mask][-1])
	y0[:first] = y0[first]
	y0[last + 1 :] = y0[last]
	mask2 = np.isfinite(y0)
	if mask2.all():
		return y0
	good = idx[mask2]
	bad = idx[~mask2]
	y0[bad] = np.interp(bad.astype(float), good.astype(float), y0[good])
	return y0


def _ensure_sg_column(
	df: "pd.DataFrame",
	*,
	raw_col: str,
	sg_col: str,
	is_angle: bool = False,
	window_length: int = 11,
	polyorder: int = 3,
) -> None:
	"""Ensure `sg_col` exists in df, computing it from `raw_col` if needed."""
	if sg_col in df.columns:
		return
	if raw_col not in df.columns:
		raise ValueError(
			f"Trajectory CSV missing required SG column {sg_col!r} and raw source {raw_col!r}. "
			"Either provide *_sg columns or include the raw columns for SG computation."
		)

	y = _fill_nans_linear(df[raw_col].to_numpy(dtype=float))
	if is_angle:
		# Unwrap before filtering to avoid discontinuities, then wrap back.
		y = np.unwrap(y)

	# Pick a valid odd window length.
	n = int(y.size)
	w = int(window_length)
	if w % 2 == 0:
		w += 1
	w = min(w, n if (n % 2 == 1) else max(1, n - 1))
	# Ensure window length is large enough for the polynomial order.
	min_w = int(polyorder) + 2
	if w < min_w:
		# Not enough samples to filter meaningfully.
		df[sg_col] = _wrap_to_pi_np(y) if is_angle else y
		return

	savgol_filter = _require_scipy_savgol_filter()
	y_sg = savgol_filter(y, window_length=w, polyorder=int(polyorder), mode="interp")
	if is_angle:
		y_sg = _wrap_to_pi_np(np.asarray(y_sg, dtype=float))
	df[sg_col] = np.asarray(y_sg, dtype=float)


def load_trajectory_csv(path: Path) -> Tuple[np.ndarray, np.ndarray, float]:
	"""Load a trajectory CSV into (X_raw, U_raw, dt_s).

	Expected columns:
		- time_s (optional but recommended)
		- lateral_offset_m_sg OR lateral_offset_m
		- target_orientation_rad_sg OR target_orientation_rad
		- target_curvature_1pm_sg OR target_curvature_1pm
		- target curvature derivative column: target_curvature_dot_1pm2 OR target_curvature_1pm_dot
		- target curvature second derivative (control): target_curvature_ddot_1pm3 OR target_curvature_1pm_ddot
	"""
	df = pd.read_csv(path)

	# Enforce SG columns: compute them from raw if the planner didn't write them.
	_ensure_sg_column(df, raw_col="lateral_offset_m", sg_col="lateral_offset_m_sg", is_angle=False)
	_ensure_sg_column(df, raw_col="target_orientation_rad", sg_col="target_orientation_rad_sg", is_angle=True)
	_ensure_sg_column(df, raw_col="target_curvature_1pm", sg_col="target_curvature_1pm_sg", is_angle=False)

	lat_col = "lateral_offset_m_sg"
	psi_col = "target_orientation_rad_sg"
	kappa_col = "target_curvature_1pm_sg"

	kappa_dot_col = "target_curvature_dot_1pm2" if "target_curvature_dot_1pm2" in df.columns else "target_curvature_1pm_dot"
	u_col = "target_curvature_ddot_1pm3" if "target_curvature_ddot_1pm3" in df.columns else "target_curvature_1pm_ddot"
	if kappa_dot_col not in df.columns:
		raise ValueError(
			"Trajectory CSV missing curvature derivative column: expected one of "
			"['target_curvature_dot_1pm2', 'target_curvature_1pm_dot']"
		)
	if u_col not in df.columns:
		raise ValueError(
			"Trajectory CSV missing curvature second-derivative control column: expected one of "
			"['target_curvature_ddot_1pm3', 'target_curvature_1pm_ddot']"
		)

	X = np.vstack(
		[
			df[lat_col].to_numpy(dtype=float),
			df[psi_col].to_numpy(dtype=float),
			df[kappa_col].to_numpy(dtype=float),
			df[kappa_dot_col].to_numpy(dtype=float),
		]
	).T
	U = df[u_col].to_numpy(dtype=float).reshape(-1, 1)

	if "time_s" in df.columns and len(df) >= 2:
		t = df["time_s"].to_numpy(dtype=float)
		dt = float(np.median(np.diff(t)))
		if not np.isfinite(dt) or dt <= 0:
			dt = 0.1
	else:
		dt = 0.1

	mask = np.isfinite(X).all(axis=1) & np.isfinite(U).all(axis=1)
	X = X[mask]
	U = U[mask]
	if len(X) < 2:
		raise ValueError("Trajectory CSV has <2 valid samples after filtering.")
	return X, U, dt


def _load_optional_reference_and_speed_from_trajectory_csv(
	path: Path,
	*,
	expected_len: Optional[int] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
	"""Load optional reference columns and speed from a merged-format trajectory CSV.

	Returns (ref_orientation_rad, ref_curvature_1pm, speed_mps), each possibly None.

	NOTE: reference signals are preferred SG-filtered (reference_*_sg) but raw fallbacks are accepted.
	"""
	df = pd.read_csv(Path(path))

	# If raw reference columns exist but SG columns are missing, compute SG versions
	# so downstream always consumes *_sg when possible.
	if "reference_orientation_rad_sg" not in df.columns and "reference_orientation_rad" in df.columns:
		_ensure_sg_column(
			df,
			raw_col="reference_orientation_rad",
			sg_col="reference_orientation_rad_sg",
			is_angle=True,
		)
	if "reference_curvature_1pm_sg" not in df.columns and "reference_curvature_1pm" in df.columns:
		_ensure_sg_column(
			df,
			raw_col="reference_curvature_1pm",
			sg_col="reference_curvature_1pm_sg",
			is_angle=False,
		)

	ref_ori_col = None
	if "reference_orientation_rad_sg" in df.columns:
		ref_ori_col = "reference_orientation_rad_sg"
	elif "reference_orientation_sg" in df.columns:
		ref_ori_col = "reference_orientation_sg"
	elif "reference_orientation_rad" in df.columns:
		ref_ori_col = "reference_orientation_rad"
	elif "reference_orientation" in df.columns:
		ref_ori_col = "reference_orientation"

	ref_kappa_col = None
	if "reference_curvature_1pm_sg" in df.columns:
		ref_kappa_col = "reference_curvature_1pm_sg"
	elif "reference_curvature_sg" in df.columns:
		ref_kappa_col = "reference_curvature_sg"
	elif "reference_curvature_1pm" in df.columns:
		ref_kappa_col = "reference_curvature_1pm"
	elif "reference_curvature" in df.columns:
		ref_kappa_col = "reference_curvature"

	ref_ori = df[ref_ori_col].to_numpy(dtype=float) if ref_ori_col is not None else None
	ref_kappa = df[ref_kappa_col].to_numpy(dtype=float) if ref_kappa_col is not None else None
	speed = df["target_speed_mps"].to_numpy(dtype=float) if "target_speed_mps" in df.columns else None

	if expected_len is not None:
		n = int(expected_len)
		for name, arr in [
			(ref_ori_col or "reference_orientation_*_sg", ref_ori),
			(ref_kappa_col or "reference_curvature_*_sg", ref_kappa),
			("target_speed_mps", speed),
		]:
			if arr is not None and len(arr) != n:
				raise ValueError(f"Trajectory CSV column '{name}' has length {len(arr)} but expected {n}.")

	if ref_ori is not None and (not np.isfinite(ref_ori).all()):
		ref_ori = None
	if ref_kappa is not None and (not np.isfinite(ref_kappa).all()):
		ref_kappa = None
	if speed is not None and (not np.isfinite(speed).all()):
		speed = None

	return ref_ori, ref_kappa, speed


def _load_optional_time_from_trajectory_csv(
	path: Path,
	*,
	expected_len: Optional[int] = None,
) -> Optional[np.ndarray]:
	"""Load optional time vector from a trajectory CSV.

	Returns None when no usable time column is present.
	"""
	df = pd.read_csv(Path(path), usecols=lambda c: c == "time_s")
	if "time_s" not in df.columns:
		return None
	t = df["time_s"].to_numpy(dtype=float).reshape(-1)
	if expected_len is not None and int(expected_len) != int(len(t)):
		return None
	if t.size < 2 or (not np.isfinite(t).all()):
		return None
	dt = np.diff(t)
	if np.any(dt <= 0):
		return None
	return t


def _estimate_indicator_to_boundary_touch_time_s(
	lat: np.ndarray,
	*,
	dt_s: float,
	time_s: Optional[np.ndarray],
	target_lat_offset_m: Optional[float],
	boundary_touch_fraction: float,
	start_window: int = 5,
) -> Optional[float]:
	"""Estimate time from indicator trigger (trajectory start) to lane-boundary touch.

	Surrogate used for optimization:
	- indicator trigger time := first trajectory sample
	- boundary-touch progress := `boundary_touch_fraction` of the start->target lateral shift
	"""
	lat = np.asarray(lat, dtype=float).reshape(-1)
	if lat.size < 2 or (not np.isfinite(lat).all()):
		return None

	win = int(max(1, min(int(start_window), int(lat.size))))
	lat_start = float(np.median(lat[:win]))
	if target_lat_offset_m is None:
		lat_target = float(np.median(lat[-win:]))
	else:
		lat_target = float(target_lat_offset_m)

	delta = float(lat_target - lat_start)
	if (not np.isfinite(delta)) or abs(delta) < 1e-6:
		return None

	frac = float(boundary_touch_fraction)
	if not np.isfinite(frac):
		return None
	frac = float(np.clip(frac, 1e-6, 1.0))
	target_progress = float(abs(delta) * frac)
	sign = 1.0 if delta >= 0.0 else -1.0
	progress = (lat - lat_start) * sign

	hits = np.where(progress >= target_progress)[0]
	if hits.size == 0:
		return None
	i = int(hits[0])

	if time_s is not None:
		t = np.asarray(time_s, dtype=float).reshape(-1)
		if t.size != lat.size or (not np.isfinite(t).all()):
			t = None
		elif np.any(np.diff(t) <= 0):
			t = None
	else:
		t = None
	if t is None:
		dt = float(dt_s)
		if (not np.isfinite(dt)) or dt <= 0:
			dt = 0.1
		t = np.arange(lat.size, dtype=float) * dt

	if i <= 0:
		t_touch = float(t[0])
	else:
		p0 = float(progress[i - 1])
		p1 = float(progress[i])
		t0 = float(t[i - 1])
		t1 = float(t[i])
		if (not np.isfinite(p0)) or (not np.isfinite(p1)) or p1 <= p0:
			touch_alpha = 1.0
		else:
			touch_alpha = float((target_progress - p0) / max(p1 - p0, 1e-12))
			touch_alpha = float(np.clip(touch_alpha, 0.0, 1.0))
		t_touch = t0 + touch_alpha * (t1 - t0)

	t_ind = float(t[0])
	return float(max(0.0, t_touch - t_ind))


def _plot_trajectory_csv_to_png(csv_path: Path, png_path: Path) -> None:
	"""Save a quick diagnostic plot for a merged-format trajectory CSV."""
	try:
		import matplotlib

		matplotlib.use("Agg", force=True)
		import matplotlib.pyplot as plt
	except Exception as e:
		raise RuntimeError(
			"Plotting requires matplotlib. Install it in your environment (e.g. 'pip install matplotlib')."
		) from e

	csv_path = Path(csv_path)
	png_path = Path(png_path)
	df = pd.read_csv(csv_path)
	if len(df) == 0:
		raise RuntimeError(f"Cannot plot empty trajectory CSV: {csv_path}")

	# Use time if present; else fallback to sample index.
	if "time_s" in df.columns:
		t = df["time_s"].to_numpy(dtype=float)
	else:
		t = np.arange(len(df), dtype=float)

	fig, ax = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
	ax = ax.reshape(-1)

	def _plot(col: str, title: str, idx: int) -> None:
		if col not in df.columns:
			ax[idx].set_title(f"{title} (missing: {col})")
			return
		y = df[col].to_numpy(dtype=float)
		ax[idx].plot(t, y)
		ax[idx].set_title(title)
		ax[idx].grid(True)

	def _plot_first(cols: Tuple[str, ...], title: str, idx: int) -> None:
		for c in cols:
			if c in df.columns:
				_plot(c, title, idx)
				return
		ax[idx].set_title(f"{title} (missing: {list(cols)})")

	_plot("lateral_offset_m_sg", "lateral_offset_m_sg", 0)
	_plot("target_orientation_rad_sg", "target_orientation_rad_sg", 1)
	_plot("target_curvature_1pm_sg", "target_curvature_1pm_sg", 2)
	_plot_first(("target_curvature_dot_1pm2", "target_curvature_1pm_dot"), "target_curvature_dot", 3)
	_plot_first(("target_curvature_ddot_1pm3", "target_curvature_1pm_ddot"), "target_curvature_ddot (control)", 4)

	# Hide the last unused subplot.
	ax[5].axis("off")

	ax[4].set_xlabel("time_s" if "time_s" in df.columns else "sample")
	ax[5].set_xlabel("time_s" if "time_s" in df.columns else "sample")

	png_path.parent.mkdir(parents=True, exist_ok=True)
	fig.tight_layout()
	fig.savefig(png_path, dpi=150)
	plt.close(fig)


def _safe_median_dt(t_s: np.ndarray, *, default_dt: float = 0.1) -> float:
	t_s = np.asarray(t_s, dtype=float).reshape(-1)
	if t_s.size < 2:
		return float(default_dt)
	dt = float(np.median(np.diff(t_s)))
	if not np.isfinite(dt) or dt <= 0:
		return float(default_dt)
	return dt


def _as_float_series(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[pd.Series]:
	for c in candidates:
		if c in df.columns:
			return df[c].astype(float)
	return None


def _compute_arc_length_from_xy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
	x = np.asarray(x, dtype=float).reshape(-1)
	y = np.asarray(y, dtype=float).reshape(-1)
	if x.size == 0:
		return np.zeros((0,), dtype=float)
	dx = np.diff(x)
	dy = np.diff(y)
	ds = np.sqrt(dx * dx + dy * dy)
	return np.concatenate([[0.0], np.cumsum(ds)])


def load_trajectory_trace_json(path: Path) -> Tuple[np.ndarray, np.ndarray, float]:
	"""Load a trajectory trace JSON into (X_raw, U_raw, dt_s).

	This supports a few common shapes seen in simulation traces:
	- A dict with an "objects" map (e.g. {"objects": {"0": [{...}, ...]}})
	  in which case object "0" (ego) is used.
	- A top-level list of per-sample dicts.

	If the trace does not explicitly provide curvature signals, we derive them:
	- target_orientation_rad: from (v_y, v_x) if available, else from dy/dx.
	- target_curvature_1pm: d(orientation)/ds.
	- target_curvature_dot_1pm2: d(curvature)/dt.
	- target_curvature_ddot_1pm3: d(curvature_dot)/dt.

	Notes:
	- For traces that only contain (x,y) in a Frenet-like frame (x=longitudinal s, y=lateral d),
	  the derived signals match that frame.
	"""
	data = load_json(Path(path))
	points: List[Dict[str, Any]] = []
	if isinstance(data, dict) and isinstance(data.get("objects"), dict):
		objects = data["objects"]
		ego = objects.get("0")
		if not isinstance(ego, list):
			# Fall back to the first list-like object.
			for v in objects.values():
				if isinstance(v, list):
					ego = v
					break
		if isinstance(ego, list):
			points = [p for p in ego if isinstance(p, dict)]
	elif isinstance(data, list):
		points = [p for p in data if isinstance(p, dict)]
	elif isinstance(data, dict) and isinstance(data.get("data"), list):
		points = [p for p in data["data"] if isinstance(p, dict)]
	else:
		raise ValueError(
			"Unsupported trace JSON shape. Expected a list of samples or a dict with 'objects'."
		)

	if len(points) < 2:
		raise ValueError("Trace JSON has <2 samples.")

	df = pd.DataFrame(points)
	# time
	t = _as_float_series(df, ["time_s", "t_s"])
	if t is None:
		t_ns = _as_float_series(df, ["t_ns", "timestamp_ns", "time_ns"])
		if t_ns is not None:
			t = t_ns * 1e-9
	if t is None:
		# No explicit time; assume uniform.
		t = pd.Series(np.arange(len(df), dtype=float) * 0.1)

	df = df.assign(time_s=np.asarray(t, dtype=float))
	df = df.sort_values("time_s").reset_index(drop=True)

	# lateral offset
	lat = _as_float_series(df, ["lateral_offset_m", "lateral_offset", "d", "y"])
	if lat is None:
		raise ValueError("Trace JSON does not contain a recognizable lateral offset (e.g. 'lateral_offset_m' or 'y').")

	# orientation
	orient = _as_float_series(df, ["target_orientation_rad", "target_orientation", "yaw", "heading", "psi"])
	if orient is None:
		v_x = _as_float_series(df, ["v_x", "vx", "v_long", "v_s"])
		v_y = _as_float_series(df, ["v_y", "vy", "v_lat", "v_d"])
		if v_x is not None and v_y is not None:
			orient = np.arctan2(np.asarray(v_y, dtype=float), np.asarray(v_x, dtype=float))
		else:
			x = _as_float_series(df, ["x", "s", "arc_length_m"])
			y = _as_float_series(df, ["y", "d", "lateral_offset_m"])
			if x is not None and y is not None:
				xa = np.asarray(x, dtype=float)
				ya = np.asarray(y, dtype=float)
				t_s = df["time_s"].to_numpy(dtype=float)
				dx_dt = np.gradient(xa, t_s)
				dy_dt = np.gradient(ya, t_s)
				orient = np.arctan2(dy_dt, dx_dt)
			else:
				orient = np.zeros(len(df), dtype=float)

	# curvature and derivatives
	kappa = _as_float_series(df, ["target_curvature_1pm", "target_curvature", "curvature_1pm", "curvature", "kappa"])
	if kappa is None:
		# Derive curvature as d(orientation)/ds.
		theta = np.unwrap(np.asarray(orient, dtype=float))
		t_s = df["time_s"].to_numpy(dtype=float)
		# Prefer explicit arc-length if available; else use x as longitudinal; else compute from xy.
		s = _as_float_series(df, ["arc_length_m", "s", "x"])
		if s is None:
			x = _as_float_series(df, ["x"])
			y = _as_float_series(df, ["y"])
			if x is not None and y is not None:
				s = _compute_arc_length_from_xy(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
			else:
				# Fall back to time as a proxy (not ideal, but keeps the pipeline runnable).
				s = t_s.copy()
		sa = np.asarray(s, dtype=float)
		# Avoid duplicated s values (gradient would blow up).
		if sa.size >= 2:
			ds = np.diff(sa)
			if np.any(ds == 0):
				sa = sa + 1e-9 * np.arange(sa.size, dtype=float)
		kappa = np.gradient(theta, sa)

	t_s = df["time_s"].to_numpy(dtype=float)
	kappa = np.asarray(kappa, dtype=float)
	kappa_dot = _as_float_series(df, ["target_curvature_dot_1pm2", "target_curvature_dot", "curvature_dot_1pm2", "curvature_dot", "kappa_dot"])
	if kappa_dot is None:
		kappa_dot = np.gradient(kappa, t_s)
	else:
		kappa_dot = np.asarray(kappa_dot, dtype=float)

	kappa_ddot = _as_float_series(
		df,
		[
			"target_curvature_ddot_1pm3",
			"target_curvature_ddot",
			"curvature_ddot_1pm3",
			"curvature_ddot",
			"kappa_ddot",
		],
	)
	if kappa_ddot is None:
		kappa_ddot = np.gradient(np.asarray(kappa_dot, dtype=float), t_s)
	else:
		kappa_ddot = np.asarray(kappa_ddot, dtype=float)

	X = np.vstack(
		[
			np.asarray(lat, dtype=float),
			np.asarray(orient, dtype=float),
			np.asarray(kappa, dtype=float),
			np.asarray(kappa_dot, dtype=float),
		]
	).T
	U = np.asarray(kappa_ddot, dtype=float).reshape(-1, 1)
	mask = np.isfinite(X).all(axis=1) & np.isfinite(U).all(axis=1)
	X = X[mask]
	U = U[mask]
	if len(X) < 2:
		raise ValueError("Trace JSON has <2 valid samples after filtering.")

	dt = _safe_median_dt(t_s, default_dt=0.1)
	return X, U, dt


def load_trajectory_file(path: Path) -> Tuple[np.ndarray, np.ndarray, float]:
	"""Load a trajectory from either CSV (merged format) or JSON trace."""
	path = Path(path)
	sfx = path.suffix.lower()
	if sfx == ".csv":
		return load_trajectory_csv(path)
	if sfx == ".json":
		return load_trajectory_trace_json(path)
	raise ValueError(f"Unsupported trajectory file extension {path.suffix!r}; expected .csv or .json")


def _theta_to_mpc_cost_from_lane_change_theta(theta: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
	"""Map the 7 lane-change weights to MPC state/control costs.

	We map the 7 weights to the 4-state + 1-input linear kinematic model used by
	the optional MPC package:
		state = [d, theta, kappa, kappa_dot], input = kappa_ddot

	Lane-change theta order (see PLANNER_CHANGE_LANE_PARAM_SPECS indices):
		0 accel, 1 jerk, 2 offset_fast, 3 offset_slow, 4 snap, 5 vel_fast, 6 vel_slow

	Mapping:
		d      <- avg(offset_fast, offset_slow)
		theta  <- avg(vel_fast, vel_slow)
		kappa  <- accel
		kappa_dot <- jerk
		kappa_ddot (input) <- snap
	"""
	arr = np.asarray(list(theta), dtype=float).reshape(-1)
	if arr.size != 7:
		raise ValueError(f"Expected theta of length 7 for MPC lane-change planner, got {arr.size}.")
	if not np.isfinite(arr).all():
		raise ValueError("theta must be finite")

	acc = float(arr[0])
	jerk = float(arr[1])
	off_fast = float(arr[2])
	off_slow = float(arr[3])
	snap = float(arr[4])
	vel_fast = float(arr[5])
	vel_slow = float(arr[6])

	state_cost = np.array([
		0.5 * (off_fast + off_slow),
		0.5 * (vel_fast + vel_slow),
		acc,
		jerk,
	], dtype=float)
	control_cost = np.array([snap], dtype=float)

	# Guard against non-positive weights: MPC solver expects cost weights >= 0.
	state_cost = np.maximum(state_cost, 0.0)
	control_cost = np.maximum(control_cost, 0.0)

	# Avoid scale blow-ups: only the *relative* weights should matter for shaping.
	# Keep a consistent mean scale so outer optimization can't "win" by sending all
	# weights to very large values.
	vec = np.concatenate([state_cost, control_cost])
	mean = float(np.mean(vec))
	if np.isfinite(mean) and mean > 0:
		scale = 1.0 / mean
		state_cost = state_cost * scale
		control_cost = control_cost * scale
	else:
		# If everything is zero, keep a tiny positive floor so the MPC problem remains well-posed.
		state_cost = np.full_like(state_cost, 1e-6)
		control_cost = np.full_like(control_cost, 1e-6)
	return state_cost, control_cost


def generate_trajectory_csv_with_mpcsolver(
	*,
	df_ref: "pd.DataFrame",
	out_path: Path,
	theta: Sequence[float],
	opt_timesteps: int = 50,
	dt_s: float = 0.1,
	solver: str = "osqp",
) -> None:
	"""Generate a merged-format trajectory CSV using the optional MPC solver package.

	This version does NOT generate any synthetic lane-change reference.
	Instead, it consumes a preprocessed lane-change reference segment `df_ref` and
	interpolates its SG-filtered reference signals onto the MPC uniform time grid.

	Expected columns in df_ref:
	- time_s
	- lateral_offset_m_sg
	- reference_orientation_sg
	- reference_curvature_sg
	- target_speed_mps

	Desired state per step k:
	  [d_ref[k], theta_ref[k], kappa_ref[k], 0.0]
	"""
	out_path = Path(out_path)
	out_path.parent.mkdir(parents=True, exist_ok=True)

	if df_ref is None:
		raise ValueError("df_ref must be a pandas DataFrame")
	if len(df_ref) < 2:
		raise ValueError(f"df_ref must have at least 2 rows, got {len(df_ref)}")

	opt_timesteps = int(opt_timesteps)
	if opt_timesteps <= 1:
		raise ValueError("opt_timesteps must be > 1")
	dt_s = float(dt_s)
	if not np.isfinite(dt_s) or dt_s <= 0:
		raise ValueError("dt_s must be positive")
	if solver not in {"ipopt", "osqp"}:
		raise ValueError("solver must be 'ipopt' or 'osqp'")

	from kinematics.kinematic_model import (
		INP_DIM,
		STATE_DIM,
		DIST_DIM,
		get_kin_model_dist_mat,
		get_kin_model_inp_mat,
		get_kin_model_sys_mat,
	)
	from kinematics.kinematic_parameters import DEFAULT_MAX_CURVATURE_DDOT, DEFAULT_SATURATION_VELOCITY
	from optimization.formulation_utils import get_cost_factor_matrices, get_saturated_longitudinal_velocity
	from optimization.mpcsolver import MPCSolver, MPCSolverType

	# --- MPC time grid
	N = int(opt_timesteps)
	t_s = np.arange(N + 1, dtype=float) * dt_s

	# --- validate/clean reference dataframe
	df_ref = df_ref.copy()
	required_cols = (
		"time_s",
		"lateral_offset_m_sg",
		"reference_orientation_sg",
		"reference_curvature_sg",
		"target_speed_mps",
	)
	missing = [c for c in required_cols if c not in df_ref.columns]
	if missing:
		raise ValueError(f"df_ref is missing required columns: {missing}")

	df_ref = df_ref.dropna(subset=["time_s"]).copy()
	df_ref["time_s"] = pd.to_numeric(df_ref["time_s"], errors="coerce")
	df_ref = df_ref.dropna(subset=["time_s"]).sort_values("time_s").reset_index(drop=True)

	# Collapse duplicate timestamps for stable interpolation.
	try:
		df_ref = df_ref.groupby("time_s", as_index=False).mean(numeric_only=True)
	except TypeError:
		df_ref = df_ref.groupby("time_s", as_index=False).mean()

	t_raw = df_ref["time_s"].to_numpy(dtype=float).reshape(-1)
	if t_raw.size < 2 or (not np.isfinite(t_raw).all()):
		raise ValueError("df_ref time_s must be finite with at least 2 samples")
	if not np.all(np.diff(t_raw) > 0):
		raise ValueError("df_ref time_s must be strictly increasing after de-duplication")

	# Normalize reference time to start at 0.
	t_ref = (t_raw - float(t_raw[0])).astype(float)

	def _interp_hold_last(y_raw: np.ndarray, *, is_angle: bool) -> Tuple[np.ndarray, np.ndarray]:
		"""Interpolate y(t_ref) onto MPC grid t_s with endpoint hold.

		Returns (wrapped_or_normal, unwrapped_or_same).
		"""
		y_raw = np.asarray(y_raw, dtype=float).reshape(-1)
		y_raw = _fill_nans_linear(y_raw)
		if not np.isfinite(y_raw).all():
			raise ValueError("Reference signal contains non-finite values after NaN fill")
		if is_angle:
			y_unwrapped = np.unwrap(y_raw)
			yi_unwrapped = np.interp(
				t_s,
				t_ref,
				y_unwrapped,
				left=float(y_unwrapped[0]),
				right=float(y_unwrapped[-1]),
			)
			yi = _wrap_to_pi_np(np.asarray(yi_unwrapped, dtype=float))
			return np.asarray(yi, dtype=float), np.asarray(yi_unwrapped, dtype=float)
		yi = np.interp(
			t_s,
			t_ref,
			y_raw,
			left=float(y_raw[0]),
			right=float(y_raw[-1]),
		)
		yi = np.asarray(yi, dtype=float)
		return yi, yi

	# --- interpolate references onto MPC grid
	d_ref, _ = _interp_hold_last(df_ref["lateral_offset_m_sg"].to_numpy(dtype=float), is_angle=False)
	theta_ref, theta_ref_unwrapped = _interp_hold_last(df_ref["reference_orientation_sg"].to_numpy(dtype=float), is_angle=True)
	kappa_ref, _ = _interp_hold_last(df_ref["reference_curvature_sg"].to_numpy(dtype=float), is_angle=False)
	speed_ref, _ = _interp_hold_last(df_ref["target_speed_mps"].to_numpy(dtype=float), is_angle=False)

	# speed sanity
	speed_ref = np.asarray(speed_ref, dtype=float)
	speed_ref = np.where(np.isfinite(speed_ref) & (speed_ref > 0), speed_ref, 1e-3)

	# --- desired state: [d_ref, theta_ref, kappa_ref, 0.0]
	mpc_desired_state = np.stack(
		[d_ref, theta_ref, kappa_ref, np.zeros_like(d_ref, dtype=float)],
		axis=1,
	)[:, :, None]
	assert mpc_desired_state.shape == (N + 1, STATE_DIM, 1)

	# --- build time-varying system matrices from speed profile (step speed = average of endpoints)
	v_steps = 0.5 * (speed_ref[:-1] + speed_ref[1:])
	v_steps = np.maximum(np.asarray(v_steps, dtype=float).reshape(-1), 1e-3)
	assert v_steps.shape == (N,)

	mpc_sys_mat = np.stack([np.asarray(get_kin_model_sys_mat(float(v), dt_s), dtype=float) for v in v_steps], axis=0)
	mpc_inp_mat = np.stack([np.asarray(get_kin_model_inp_mat(float(v), dt_s), dtype=float) for v in v_steps], axis=0)
	mpc_dist_mat = np.stack([np.asarray(get_kin_model_dist_mat(float(v), dt_s), dtype=float) for v in v_steps], axis=0)

	# MPC cost weights from theta.
	state_cost_w, control_cost_w = _theta_to_mpc_cost_from_lane_change_theta(theta)
	mpc_state_cost = np.tile(state_cost_w.reshape(1, -1), (N + 1, 1))
	mpc_control_cost = np.tile(control_cost_w.reshape(1, -1), (N, 1))

	# Velocity-dependent scaling factors used by the optional MPC package.
	long_vel_sat = get_saturated_longitudinal_velocity(v_steps, float(DEFAULT_SATURATION_VELOCITY))
	mpc_state_cost_fac, mpc_contr_cost_fac = get_cost_factor_matrices(long_vel_sat)

	# Disturbance as in prepare_formulation.py: [0, avg(reference_orientation)] per step.
	# Average in unwrapped domain to avoid wrap artifacts near +/-pi.
	psi_avg = 0.5 * (theta_ref_unwrapped[:-1] + theta_ref_unwrapped[1:])
	mpc_disturbance = np.stack([np.zeros((N,), dtype=float), np.asarray(psi_avg, dtype=float)], axis=1)[:, :, None]
	assert mpc_disturbance.shape == (N, DIST_DIM, 1)

	# Initial condition.
	x_init = np.asarray(mpc_desired_state[0], dtype=float).reshape(STATE_DIM, 1)

	# Solve open-loop MPC.
	mpc = MPCSolver.create(MPCSolverType.SEQUENTIAL_IMPLICIT, opt_timesteps=N, solver=str(solver))
	u_min = -float(DEFAULT_MAX_CURVATURE_DDOT)
	u_max = float(DEFAULT_MAX_CURVATURE_DDOT)
	state_3d, input_3d = mpc.solve_with_state_space_solution(
		x_init=x_init,
		mpc_sys_mat=np.asarray(mpc_sys_mat, dtype=float),
		mpc_inp_mat=np.asarray(mpc_inp_mat, dtype=float),
		mpc_dist_mat=np.asarray(mpc_dist_mat, dtype=float),
		mpc_disturbance=np.asarray(mpc_disturbance, dtype=float),
		mpc_desired_state=np.asarray(mpc_desired_state, dtype=float),
		mpc_state_cost_fac=np.asarray(mpc_state_cost_fac, dtype=float),
		mpc_contr_cost_fac=np.asarray(mpc_contr_cost_fac, dtype=float),
		mpc_state_cost=np.asarray(mpc_state_cost, dtype=float),
		mpc_control_cost=np.asarray(mpc_control_cost, dtype=float),
		input_min=float(u_min),
		input_max=float(u_max),
	)
	# Reshape for export.
	state = np.asarray(state_3d, dtype=float).reshape(N + 1, STATE_DIM)
	u = np.asarray(input_3d, dtype=float).reshape(N, INP_DIM)
	u_pad = np.vstack([u, u[-1:]])

	df = pd.DataFrame(
		{
			"time_s": t_s,
			"lateral_offset_m_sg": state[:, 0],
			"target_orientation_rad_sg": state[:, 1],
			"target_curvature_1pm_sg": state[:, 2],
			"target_curvature_dot_1pm2": state[:, 3],
			"target_curvature_ddot_1pm3": u_pad[:, 0],
			"target_speed_mps": speed_ref,
			"reference_orientation_sg": theta_ref,
			"reference_curvature_sg": kappa_ref,
			# Backwards-compatible aliases used by downstream loaders.
			"reference_orientation_rad_sg": theta_ref,
			"reference_curvature_1pm_sg": kappa_ref,
		}
	)
	df.to_csv(out_path, index=False)


def compute_hlo_cost(
	X_raw: np.ndarray,
	U_raw: np.ndarray,
	*,
	omega: Sequence[float],
	dt_s: float,
	reference_orientation_rad: Optional[np.ndarray] = None,
	reference_curvature_1pm: Optional[np.ndarray] = None,
	speed_mps: Optional[np.ndarray] = None,
	target_lat_offset_m: Optional[float] = None,
	softness_m: float = 0.2,
	threshold_m: float = 0.5,
	feature_scales: Optional[Sequence[float]] = None,
	feature_scale_eps: float = 1e-6,
) -> float:
	"""Compute a simple HLO scalar cost from a trajectory.

	This implements the LaTeX-aligned 8/9-feature objective used by DDIOC
	after removing the duration term and adding a velocity-aware lateral-rate feature:
		phi_0: (d - d_ref)^2 (with d_ref defaulting to d_tgt)
		phi_1: v^2 * (d_dot)^2  where d_dot≈(d[k]-d[k-1])/dt
		phi_2: v^4 * (kappa)^2
		phi_3: v^4 * (kappa_dot)^2
		phi_4: v^4 * (kappa - kappa_ref)^2
		phi_5: v^4 * (delta_(kappa-kappa_ref))^2
		phi_6: v^4 * (delta_kappa_dot)^2
		phi_7: v^4 * u^2  (u = kappa_ddot)
		phi_8: v^2 * wrap_to_pi(psi - psi_ref)^2 (only when omega has 9 features)

	No heading term is used.
	"""
	Phi, omega_used = compute_hlo_features_matrix(
		X_raw,
		U_raw,
		omega=omega,
		dt_s=dt_s,
		reference_orientation_rad=reference_orientation_rad,
		reference_curvature_1pm=reference_curvature_1pm,
		speed_mps=speed_mps,
		target_lat_offset_m=target_lat_offset_m,
		softness_m=softness_m,
		threshold_m=threshold_m,
		feature_scales=feature_scales,
		feature_scale_eps=feature_scale_eps,
	)
	return float(np.sum(Phi @ omega_used))


def compute_hlo_features_matrix(
	X_raw: np.ndarray,
	U_raw: np.ndarray,
	*,
	omega: Sequence[float],
	dt_s: float,
	reference_orientation_rad: Optional[np.ndarray] = None,
	reference_curvature_1pm: Optional[np.ndarray] = None,
	speed_mps: Optional[np.ndarray] = None,
	target_lat_offset_m: Optional[float] = None,
	softness_m: float = 0.2,
	threshold_m: float = 0.5,
	feature_scales: Optional[Sequence[float]] = None,
	feature_scale_eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
	"""Return the per-timestep feature matrix Phi (N,m) and the adapted omega (m,).

	This is the shared implementation behind:
	- scalar cost: sum_k Phi[k] @ omega
	- feature sums: sum_k Phi[k]

	Phi is already normalized if feature_scales are provided.
	"""
	X_raw = np.asarray(X_raw, dtype=float)
	U_raw = np.asarray(U_raw, dtype=float)
	omega_arr = np.asarray(list(omega), dtype=float).reshape(-1)
	# Backward compatibility handling (heuristics):
	# - Current objective expects 8 (no psi) or 9 (with psi).
	# - Older objectives may include a leading duration feature and/or may not include the new
	#   lateral-rate feature. We try to adapt by dropping duration (when present) and padding
	#   the missing lateral-rate weight with 0.
	if omega_arr.size == 10:
		# (duration + 9) -> drop duration
		omega_arr = omega_arr[1:]
	if reference_orientation_rad is not None:
		# With psi term available: target length is 9.
		if omega_arr.size == 9:
			pass
		elif omega_arr.size == 8:
			# Legacy (no lat-rate feature): insert 0 after lat_err
			omega_arr = np.insert(omega_arr, 1, 0.0)
		elif omega_arr.size == 10:
			omega_arr = omega_arr[1:]
		else:
			pass
		if omega_arr.size != 9:
			raise ValueError(
				f"omega must be length 9 when reference_orientation_rad is provided, got {omega_arr.size}"
			)
	else:
		# No psi term: target length is 8.
		if omega_arr.size == 8:
			pass
		elif omega_arr.size == 7:
			# Legacy (no lat-rate feature)
			omega_arr = np.insert(omega_arr, 1, 0.0)
		elif omega_arr.size == 9:
			# Likely legacy duration+psi (or current with psi) but no reference was provided.
			# Drop first entry as duration heuristic.
			omega_arr = omega_arr[1:]
		if omega_arr.size != 8:
			raise ValueError(f"omega must be length 8 when reference_orientation_rad is None, got {omega_arr.size}")
	if not np.isfinite(omega_arr).all():
		raise ValueError("omega must be finite")
	if X_raw.ndim != 2 or X_raw.shape[1] != 4:
		raise ValueError(f"X_raw must be (N,4), got {X_raw.shape}")
	if U_raw.ndim != 2 or U_raw.shape[1] < 1:
		raise ValueError(f"U_raw must be (N,1+), got {U_raw.shape}")
	if len(U_raw) != len(X_raw):
		raise ValueError("X_raw and U_raw must have same length")

	dt_s = float(dt_s)
	if not np.isfinite(dt_s) or dt_s <= 0:
		dt_s = 0.1
	softness_m = float(softness_m)
	if not np.isfinite(softness_m) or softness_m <= 0:
		softness_m = 0.2
	threshold_m = float(threshold_m)

	if target_lat_offset_m is None:
		# Use final lateral offset as a reasonable default target.
		target_lat_offset_m = float(X_raw[-1, 0])
	# In the learned objective we used d_ref = d_tgt.
	d_ref = float(target_lat_offset_m)

	# common signals
	lat = np.asarray(X_raw[:, 0], dtype=float)
	psi = np.asarray(X_raw[:, 1], dtype=float)
	kappa = np.asarray(X_raw[:, 2], dtype=float)
	kappa_dot = np.asarray(X_raw[:, 3], dtype=float)
	u = np.asarray(U_raw[:, 0], dtype=float)

	if speed_mps is None:
		v = np.ones_like(lat, dtype=float)
	else:
		v = np.asarray(speed_mps, dtype=float).reshape(-1)
		if len(v) != len(lat):
			raise ValueError("speed_mps must match X_raw length")
		v = np.where(np.isfinite(v) & (v > 0), v, 1.0)
	v_eps = 1e-3
	v_eff = np.maximum(v, v_eps)
	v2 = v_eff * v_eff
	v4 = v2 * v2

	def _wrap_to_pi(a: np.ndarray) -> np.ndarray:
		return np.arctan2(np.sin(a), np.cos(a))

	# curvature reference: optional but recommended when omega was learned with reference features.
	if reference_curvature_1pm is None:
		kappa_ref = np.zeros_like(kappa)
	else:
		kappa_ref = np.asarray(reference_curvature_1pm, dtype=float).reshape(-1)
		if len(kappa_ref) != len(kappa):
			raise ValueError("reference_curvature_1pm must match X_raw length")

	phi0 = (lat - d_ref) ** 2
	dlat = np.diff(lat, prepend=lat[0])
	d_dot = dlat / float(max(dt_s, 1e-9))
	phi1 = v2 * (d_dot * d_dot)
	kappa_err = kappa - kappa_ref
	phi2 = v4 * (kappa * kappa)
	phi3 = v4 * (kappa_dot * kappa_dot)
	phi4 = v4 * (kappa_err * kappa_err)
	dkappa_err = np.diff(kappa_err, prepend=kappa_err[0])
	phi5 = v4 * (dkappa_err * dkappa_err)
	dkdot = np.diff(kappa_dot, prepend=kappa_dot[0])
	phi6 = v4 * (dkdot * dkdot)
	phi7 = v4 * (u * u)
	Phi_list = [phi0, phi1, phi2, phi3, phi4, phi5, phi6, phi7]
	if omega_arr.size == 9:
		if reference_orientation_rad is None:
			raise ValueError("reference_orientation_rad is required when omega has 9 features")
		psi_ref = np.asarray(reference_orientation_rad, dtype=float).reshape(-1)
		if len(psi_ref) != len(psi):
			raise ValueError("reference_orientation_rad must match X_raw length")
		e_psi = _wrap_to_pi(psi - psi_ref)
		phi8 = v2 * (e_psi * e_psi)
		Phi_list.append(phi8)

	Phi = np.vstack(Phi_list).T
	if feature_scales is not None:
		fs = np.asarray(list(feature_scales), dtype=float).reshape(-1)
		if (not np.isfinite(fs).all()):
			raise ValueError(f"feature_scales must be finite floats, got: {fs}")
		# Backward compatibility: if duration was present, drop it.
		if fs.size == 10:
			fs = fs[1:]
		# Backward compatibility: if lateral-rate scale is missing, insert a neutral scale.
		if fs.size == 8 and omega_arr.size == 9:
			fs = np.insert(fs, 1, 1.0)
		elif fs.size == 7 and omega_arr.size == 8:
			fs = np.insert(fs, 1, 1.0)
		if fs.size != omega_arr.size:
			raise ValueError(f"feature_scales must be length {int(omega_arr.size)} floats, got: {fs}")
		fs = np.maximum(fs, 1e-3)
		eps = float(feature_scale_eps)
		if not np.isfinite(eps) or eps < 0:
			raise ValueError(f"feature_scale_eps must be finite and >= 0, got: {eps}")
		Phi = Phi / (fs.reshape(1, -1) + eps)
	return Phi, omega_arr


def compute_hlo_feature_sums(
	X_raw: np.ndarray,
	U_raw: np.ndarray,
	*,
	omega: Sequence[float],
	dt_s: float,
	reference_orientation_rad: Optional[np.ndarray] = None,
	reference_curvature_1pm: Optional[np.ndarray] = None,
	speed_mps: Optional[np.ndarray] = None,
	target_lat_offset_m: Optional[float] = None,
	softness_m: float = 0.2,
	threshold_m: float = 0.5,
	feature_scales: Optional[Sequence[float]] = None,
	feature_scale_eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
	"""Return (feature_sums, omega_used) where feature_sums has shape (m,).

	feature_sums[j] = sum_k Phi[k, j] with the same normalization and omega adaptation
	as used by compute_hlo_cost.
	"""
	Phi, omega_used = compute_hlo_features_matrix(
		X_raw,
		U_raw,
		omega=omega,
		dt_s=dt_s,
		reference_orientation_rad=reference_orientation_rad,
		reference_curvature_1pm=reference_curvature_1pm,
		speed_mps=speed_mps,
		target_lat_offset_m=target_lat_offset_m,
		softness_m=softness_m,
		threshold_m=threshold_m,
		feature_scales=feature_scales,
		feature_scale_eps=feature_scale_eps,
	)
	return np.sum(Phi, axis=0), omega_used


def _resolve_hlo_target_lat_offset_m(
	X_raw: np.ndarray,
	*,
	mode: str,
	fixed_target_lat_offset_m: Optional[float],
	candidates_rel_m: Optional[Sequence[float]],
	window: int,
) -> Optional[float]:
	"""Resolve target lateral offset for HLO cost.

	Modes:
		- auto: fixed if provided else final
		- final: target = final lateral offset (handled by compute_hlo_cost when passing None)
		- fixed: use fixed_target_lat_offset_m
		- nearest_rel: estimate start/end lateral offsets, compute end_rel=end-start,
			then snap to nearest candidate (in meters) and return an absolute target (start + snapped_rel).
	"""
	mode = str(mode).strip().lower()
	if mode not in {"auto", "final", "fixed", "nearest_rel"}:
		raise ValueError(f"Unknown hlo_target_lat_offset_mode: {mode!r}")

	if mode == "auto":
		mode = "fixed" if fixed_target_lat_offset_m is not None else "final"

	if mode == "final":
		return None
	if mode == "fixed":
		if fixed_target_lat_offset_m is None:
			raise ValueError("hlo_target_lat_offset_mode='fixed' requires --hlo_target_lat_offset_m")
		return float(fixed_target_lat_offset_m)

	# nearest_rel
	X_raw = np.asarray(X_raw, dtype=float)
	if X_raw.ndim != 2 or X_raw.shape[1] < 1:
		raise ValueError(f"X_raw must be (N,4+), got {X_raw.shape}")
	lat = np.asarray(X_raw[:, 0], dtype=float).reshape(-1)
	if lat.size < 2:
		return None

	window = int(window)
	if window <= 0:
		window = 1
	window = int(min(window, max(1, lat.size)))
	lat0 = float(np.median(lat[:window]))
	lat_end = float(np.median(lat[-window:]))
	end_rel = float(lat_end - lat0)

	if candidates_rel_m is None:
		candidates = np.asarray([0.0, -3.5], dtype=float)
	else:
		candidates = np.asarray(list(candidates_rel_m), dtype=float).reshape(-1)
		if candidates.size == 0:
			raise ValueError("--hlo_target_lat_offset_candidates_rel must contain at least one float")
	if not np.isfinite(candidates).all():
		raise ValueError("--hlo_target_lat_offset_candidates_rel must be finite")

	idx = int(np.argmin(np.abs(end_rel - candidates)))
	snapped_rel = float(candidates[idx])
	return float(lat0 + snapped_rel)


# -----------------------------------------------------------------------------
# External commands and optimization


_FLOAT_RE = re.compile(r"([+-]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][+-]?\d+)?)")


def _extract_cost_from_text(text: str, cost_regex: Optional[str]) -> float:
	"""Extract a cost float from command output."""
	if cost_regex:
		m = re.search(cost_regex, text)
		if not m:
			raise ValueError(f"Could not parse cost using regex: {cost_regex!r}")
		return float(m.group(1))
	m = _FLOAT_RE.search(text)
	if not m:
		raise ValueError("Could not find any float in output to parse as cost")
	return float(m.group(1))


def _run_cmd(cmd: str, *, cwd: Optional[Path] = None, env_extra: Optional[Dict[str, str]] = None) -> str:
	"""Run a shell-like command (string) safely and return stdout+stderr."""
	args = shlex.split(cmd)
	env = os.environ.copy()
	if env_extra:
		env.update({k: str(v) for k, v in env_extra.items()})
	proc = subprocess.run(
		args,
		cwd=str(cwd) if cwd is not None else None,
		env=env,
		stdout=subprocess.PIPE,
		stderr=subprocess.STDOUT,
		text=True,
		check=False,
	)
	out = proc.stdout or ""
	if proc.returncode != 0:
		raise RuntimeError(f"Command failed (code={proc.returncode}): {cmd}\n--- output ---\n{out}")
	return out


@dataclass
class BilevelResult:
	theta_star: np.ndarray
	cost_star: float
	n_evals: int
	history: List[Tuple[np.ndarray, float]]


def write_result_csv(
	*,
	path: Path,
	result: BilevelResult,
	theta_keys: Sequence[str],
	qp_json: Path,
	hlo_omega: Optional[Sequence[float]],
	method: str,
	maxiter: int,
	planner_overwrite_out: Optional[Path] = None,
) -> None:
	"""Write a single-row CSV summary for a bilevel run.

	This is meant for easy copy/paste into reports and for downstream scripts.
	"""
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)

	row: Dict[str, Any] = {
		"timestamp_utc": datetime.now(timezone.utc).isoformat(),
		"qp_json": str(Path(qp_json)),
		"method": str(method),
		"maxiter": int(maxiter),
		"n_evals": int(result.n_evals),
		"cost_star": float(result.cost_star),
		"planner_overwrite_out": str(planner_overwrite_out) if planner_overwrite_out is not None else "",
	}

	# Record theta keys and values
	for i, k in enumerate(list(theta_keys)):
		row[f"theta_key_{i}"] = str(k)
	for i, v in enumerate(np.asarray(result.theta_star, dtype=float).reshape(-1).tolist()):
		row[f"theta_star_{i}"] = float(v)

	if hlo_omega is not None:
		om = np.asarray(list(hlo_omega), dtype=float).reshape(-1)
		for i in range(int(om.size)):
			row[f"hlo_omega_{i}"] = float(om[i])

	# Always overwrite for deterministic behavior.
	pd.DataFrame([row]).to_csv(path, index=False)


def bilevel_optimize(
	*,
	base_qp_json: Path,
	theta0: Sequence[float],
	theta_keys: Sequence[str],
	baseline_theta: Optional[Sequence[float]] = None,
	eval_only_theta: Optional[Sequence[float]] = None,
	qp_repeat: int = 10,
	planner_cmd: Optional[str],
	planner_mpcsolver: bool = True,
	mpc_opt_timesteps: int = 50,
	mpc_dt_s: float = 0.1,
	mpc_reference_csv: Optional[Path] = None,
	mpc_reference_merged_csv: Optional[Path] = None,
	mpc_lc_ids: Optional[Sequence[int]] = None,
	mpc_batch_size: int = 0,
	mpc_aggregate: str = "mean",
	mpc_solver: str = "osqp",
	traj_csv_path: Optional[Path],
	cost_regex: Optional[str],
	hlo_omega: Optional[Sequence[float]],
	hlo_feature_scales: Optional[Sequence[float]] = None,
	hlo_feature_scale_eps: float = 1e-6,
	hlo_target_lat_offset_m: Optional[float] = None,
	hlo_target_lat_offset_mode: str = "auto",
	hlo_target_lat_offset_candidates_rel: Optional[Sequence[float]] = None,
	hlo_target_lat_offset_window: int = 5,
	hlo_softness_m: float = 0.2,
	hlo_threshold_m: float = 0.5,
	min_indicator_to_boundary_touch_s: float = 2.2,
	boundary_touch_fraction: float = 0.5,
	timing_constraint_penalty_weight: float = 1e4,
	timing_constraint_missing_penalty: float = 1e6,
	require_reference_curvature: bool = False,
	workdir: Optional[Path] = None,
	tie_lane_change_pairs: bool = False,
	optimizer: str = "scipy",
	maxiter: int = 80,
	method: str = "Powell",
	theta_transform: str = "identity",
	theta_min: Optional[Sequence[float]] = None,
	theta_max: Optional[Sequence[float]] = None,
	theta_reg: float = 0.0,
	bayes_calls: int = 60,
	bayes_init_points: int = 12,
	bayes_acq_func: str = "EI",
	nsga2_pop_size: int = 48,
	mo_objectives: str = "group3",
	nsga2_log: str = "gen",
	cache: bool = True,
	verbose: bool = True,
) -> BilevelResult:
	"""Run outer optimization over theta.

	If `hlo_omega` is provided, we compute HLO cost from a trajectory file.

	If `planner_cmd` is provided, we run it each evaluation to (re-)generate the trajectory.
	The command string may use these placeholders:
		{json_path} -> path to the JSON used for this evaluation
		{traj_csv}  -> path to the trajectory file (if provided)

	If `planner_mpcsolver` is true, we generate a trajectory using the MPC solver from
	the optional MPC package and write it to `traj_csv_path` each evaluation. In this
	workflow, the MPC generator consumes a preprocessed reference segment (CSV/DataFrame)
	and does NOT synthesize a lane-change profile.

	If no planner is provided, we assume `traj_csv_path` already exists.
	"""
	base_qp_json = Path(base_qp_json)
	if not base_qp_json.exists():
		raise FileNotFoundError(base_qp_json)

	theta0 = np.asarray(list(theta0), dtype=float).reshape(-1)
	if not np.isfinite(theta0).all():
		raise ValueError("theta0 must be finite")

	theta_keys = list(theta_keys)
	if tie_lane_change_pairs:
		if theta_keys != _LANE_CHANGE_THETA_KEYS:
			raise ValueError(
				"--tie_lane_change_pairs requires theta_keys to match the 7 tpl_change_lane_weight_* parameters "
				"in the default order."
			)
		if theta0.size != 5:
			raise ValueError("With --tie_lane_change_pairs, theta0 must have length 5.")
	else:
		if len(theta_keys) != len(theta0):
			raise ValueError(f"theta_keys length ({len(theta_keys)}) must match theta0 length ({len(theta0)})")
		if planner_mpcsolver and len(theta0) != 7:
			raise ValueError("--planner_mpcsolver currently expects a 7D theta unless --tie_lane_change_pairs is enabled.")
	qp_repeat = int(qp_repeat)
	if qp_repeat <= 0:
		raise ValueError("qp_repeat must be positive")

	use_internal_hlo = hlo_omega is not None
	if hlo_feature_scales is not None:
		fs = np.asarray(list(hlo_feature_scales), dtype=float).reshape(-1)
		if fs.size not in (8, 9, 10) or (not np.isfinite(fs).all()):
			raise ValueError(f"hlo_feature_scales must be 8, 9, or 10 finite floats, got: {fs}")
		if np.any(fs < 0):
			raise ValueError(f"hlo_feature_scales must be >= 0, got: {fs}")
		# Backward compatibility: older learned objectives include a leading duration scale.
		if fs.size == 10:
			fs = fs[1:]
		hlo_feature_scales = [float(x) for x in fs.tolist()]
	hlo_feature_scale_eps = float(hlo_feature_scale_eps)
	if not np.isfinite(hlo_feature_scale_eps) or hlo_feature_scale_eps < 0:
		raise ValueError("hlo_feature_scale_eps must be finite and >= 0")

	min_indicator_to_boundary_touch_s = float(min_indicator_to_boundary_touch_s)
	boundary_touch_fraction = float(boundary_touch_fraction)
	timing_constraint_penalty_weight = float(timing_constraint_penalty_weight)
	timing_constraint_missing_penalty = float(timing_constraint_missing_penalty)
	if not np.isfinite(min_indicator_to_boundary_touch_s):
		raise ValueError("min_indicator_to_boundary_touch_s must be finite")
	if not np.isfinite(boundary_touch_fraction):
		raise ValueError("boundary_touch_fraction must be finite")
	if (not np.isfinite(timing_constraint_penalty_weight)) or timing_constraint_penalty_weight < 0:
		raise ValueError("timing_constraint_penalty_weight must be finite and >= 0")
	if (not np.isfinite(timing_constraint_missing_penalty)) or timing_constraint_missing_penalty < 0:
		raise ValueError("timing_constraint_missing_penalty must be finite and >= 0")

	optimizer = str(optimizer).lower().strip()
	if optimizer not in {"scipy", "bayes", "nsga2"}:
		raise ValueError("optimizer must be one of: scipy, bayes, nsga2")
	if optimizer == "scipy" and minimize is None:
		raise RuntimeError("scipy is required for optimizer='scipy' but is not available")
	if optimizer == "nsga2":
		_import_pymoo()

	workdir = Path(workdir) if workdir is not None else base_qp_json.parent
	workdir.mkdir(parents=True, exist_ok=True)

	planner_cmd_uses_traj = (planner_cmd is not None) and ("{traj_csv}" in planner_cmd)
	needs_traj_csv = use_internal_hlo or planner_cmd_uses_traj 
	if needs_traj_csv and traj_csv_path is None:
		# The user does not need a CSV in advance; we pick a deterministic path under workdir
		# and regenerate it for each evaluation.
		traj_csv_path = workdir / "trajectory_eval.csv"

	if use_internal_hlo and (traj_csv_path is None or hlo_omega is None):
		raise ValueError("Internal HLO requires --hlo_omega and a trajectory file path (auto-assigned if omitted).")

	# Reference data for built-in MPC trajectory generation.
	mpc_reference_df: Optional[pd.DataFrame] = None
	df_all: Optional[pd.DataFrame] = None
	lc_ids_all: List[int] = []
	lc_ids_selected: List[int] = []

	if bool(planner_mpcsolver):
		mpc_aggregate = str(mpc_aggregate).strip().lower()
		if mpc_aggregate not in {"mean", "sum", "max"}:
			raise ValueError("mpc_aggregate must be one of: mean, sum, max")
		mpc_batch_size = int(mpc_batch_size)
		if mpc_batch_size < 0:
			raise ValueError("mpc_batch_size must be >= 0")

		if mpc_reference_merged_csv is not None:
			mpc_reference_merged_csv = Path(mpc_reference_merged_csv)
			if not mpc_reference_merged_csv.exists():
				raise FileNotFoundError(mpc_reference_merged_csv)
			df_all = pd.read_csv(mpc_reference_merged_csv)
			if "lc_id" not in df_all.columns:
				raise ValueError("Merged reference CSV must contain column 'lc_id'.")
			lc_ids_all = sorted([int(x) for x in pd.unique(df_all["lc_id"]) if pd.notna(x)])
			if len(lc_ids_all) == 0:
				raise ValueError("Merged reference CSV contains no lc_id values")
			if mpc_lc_ids is None:
				lc_ids_selected = list(lc_ids_all)
			else:
				req = [int(x) for x in list(mpc_lc_ids)]
				missing = [x for x in req if x not in set(lc_ids_all)]
				if missing:
					raise ValueError(f"Requested lc_id not present in merged reference CSV: {missing}")
				lc_ids_selected = sorted(req)
		else:
			if mpc_reference_csv is None:
				raise ValueError(
					"planner_mpcsolver requires either mpc_reference_merged_csv (with lc_id) or mpc_reference_csv "
					"(single lane-change segment)."
				)
			mpc_reference_csv = Path(mpc_reference_csv)
			if not mpc_reference_csv.exists():
				raise FileNotFoundError(mpc_reference_csv)
			mpc_reference_df = pd.read_csv(mpc_reference_csv)

	base_data = load_json(base_qp_json)

	theta_reg = float(theta_reg)
	if not np.isfinite(theta_reg) or theta_reg < 0:
		raise ValueError("theta_reg must be a finite non-negative float")

	history: List[Tuple[np.ndarray, float]] = []
	cache_cost: Dict[Tuple[float, ...], float] = {}
	cache_feat: Dict[Tuple[float, ...], np.ndarray] = {}
	expected_feat_dim: Optional[int] = None
	nsga2_log = str(nsga2_log).strip().lower()
	if nsga2_log not in {"gen", "eval", "none"}:
		raise ValueError("nsga2_log must be one of: gen, eval, none")

	def _aggregate_costs(costs: Sequence[float], *, mode: str) -> float:
		arr = np.asarray(list(costs), dtype=float).reshape(-1)
		if arr.size == 0:
			raise ValueError("No costs to aggregate")
		mode = str(mode).strip().lower()
		if mode == "sum":
			return float(np.sum(arr))
		if mode == "max":
			return float(np.max(arr))
		return float(np.mean(arr))

	def _traj_base_for_lc(base_path: Path, *, lc_id: int) -> Path:
		base_path = Path(base_path)
		if base_path.suffix.lower() != ".csv":
			base_path = base_path.with_suffix(".csv")
		return base_path.with_name(f"{base_path.stem}_lc{int(lc_id)}{base_path.suffix}")

	def _lc_ids_eval_for_theta(theta_full: np.ndarray) -> List[int]:
		"""Deterministically select lc_id subset for this theta (for minibatching)."""
		if df_all is None:
			return []
		if mpc_batch_size <= 0 or mpc_batch_size >= len(lc_ids_selected):
			return list(lc_ids_selected)
		import hashlib

		tb = np.asarray(theta_full, dtype=np.float64).tobytes()
		digest = hashlib.sha256(tb).digest()
		seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
		rng = np.random.default_rng(seed)
		return [
			int(x)
			for x in rng.choice(
				np.asarray(lc_ids_selected, dtype=int),
				size=int(mpc_batch_size),
				replace=False,
			).tolist()
		]

	def _evaluate_cost_from_trajectory(path: Path) -> float:
		path = Path(path)
		if not path.exists() or path.stat().st_size == 0:
			raise RuntimeError(
				f"Trajectory file missing/empty at {path}. Provide --planner_cmd to generate it per-evaluation, "
				"or point --traj_csv at an existing file. (Supported: .csv or .json)"
			)
		X, U, dt = load_trajectory_file(path)
		ref_ori = None
		ref_kappa = None
		speed_mps = None
		p_sfx = path.suffix.lower()
		if p_sfx == ".csv":
			ref_ori, ref_kappa, speed_mps = _load_optional_reference_and_speed_from_trajectory_csv(path, expected_len=int(len(X)))
		if bool(require_reference_curvature):
			if p_sfx != ".csv":
				raise ValueError(
					"Reference-aware HLO cost requires a CSV trajectory with SG-filtered reference curvature "
					"(reference_curvature_sg or reference_curvature_1pm_sg)."
				)
			if ref_kappa is None:
				raise ValueError(
					"Trajectory CSV is missing SG-filtered reference curvature required for reference-aware HLO cost "
					"(reference_curvature_sg or reference_curvature_1pm_sg)."
				)
		resolved_target = _resolve_hlo_target_lat_offset_m(
			X,
			mode=str(hlo_target_lat_offset_mode),
			fixed_target_lat_offset_m=hlo_target_lat_offset_m,
			candidates_rel_m=hlo_target_lat_offset_candidates_rel,
			window=int(hlo_target_lat_offset_window),
		)
		cost = float(
			compute_hlo_cost(
				X,
				U,
				omega=hlo_omega,
				dt_s=dt,
				reference_orientation_rad=ref_ori,
				reference_curvature_1pm=ref_kappa,
				speed_mps=speed_mps,
				target_lat_offset_m=resolved_target,
				softness_m=float(hlo_softness_m),
				threshold_m=float(hlo_threshold_m),
				feature_scales=hlo_feature_scales,
				feature_scale_eps=hlo_feature_scale_eps,
			)
		)

		if float(min_indicator_to_boundary_touch_s) > 0.0:
			time_s = None
			if p_sfx == ".csv":
				try:
					time_s = _load_optional_time_from_trajectory_csv(path, expected_len=int(len(X)))
				except Exception:
					time_s = None
			t_touch = _estimate_indicator_to_boundary_touch_time_s(
				X[:, 0],
				dt_s=float(dt),
				time_s=time_s,
				target_lat_offset_m=resolved_target,
				boundary_touch_fraction=float(boundary_touch_fraction),
				start_window=int(hlo_target_lat_offset_window),
			)
			if t_touch is None:
				cost = float(cost + timing_constraint_missing_penalty)
			elif t_touch < float(min_indicator_to_boundary_touch_s):
				gap = float(min_indicator_to_boundary_touch_s) - float(t_touch)
				cost = float(cost + timing_constraint_penalty_weight * gap * gap)

		return float(cost)

	def _evaluate_feature_sums_from_trajectory(path: Path) -> Tuple[np.ndarray, np.ndarray]:
		path = Path(path)
		if not path.exists() or path.stat().st_size == 0:
			raise RuntimeError(
				f"Trajectory file missing/empty at {path}. Provide --planner_cmd to generate it per-evaluation, "
				"or point --traj_csv at an existing file. (Supported: .csv or .json)"
			)
		X, U, dt = load_trajectory_file(path)
		ref_ori = None
		ref_kappa = None
		speed_mps = None
		p_sfx = path.suffix.lower()
		if p_sfx == ".csv":
			ref_ori, ref_kappa, speed_mps = _load_optional_reference_and_speed_from_trajectory_csv(
				path, expected_len=int(len(X))
			)
		resolved_target = _resolve_hlo_target_lat_offset_m(
			X,
			mode=str(hlo_target_lat_offset_mode),
			fixed_target_lat_offset_m=hlo_target_lat_offset_m,
			candidates_rel_m=hlo_target_lat_offset_candidates_rel,
			window=int(hlo_target_lat_offset_window),
		)
		feat_sums, omega_used = compute_hlo_feature_sums(
			X,
			U,
			omega=hlo_omega,
			dt_s=dt,
			reference_orientation_rad=ref_ori,
			reference_curvature_1pm=ref_kappa,
			speed_mps=speed_mps,
			target_lat_offset_m=resolved_target,
			softness_m=float(hlo_softness_m),
			threshold_m=float(hlo_threshold_m),
			feature_scales=hlo_feature_scales,
			feature_scale_eps=hlo_feature_scale_eps,
		)
		return np.asarray(feat_sums, dtype=float).reshape(-1), np.asarray(omega_used, dtype=float).reshape(-1)

	def _aggregate_vectors(vectors: Sequence[np.ndarray], *, mode: str) -> np.ndarray:
		arr = np.asarray([np.asarray(v, dtype=float).reshape(-1) for v in list(vectors)], dtype=float)
		if arr.ndim != 2 or arr.shape[0] == 0:
			raise ValueError("No vectors to aggregate")
		mode = str(mode).strip().lower()
		if mode == "sum":
			return np.sum(arr, axis=0)
		if mode == "max":
			return np.max(arr, axis=0)
		return np.mean(arr, axis=0)

	def _run_planner_for_theta(theta_full: np.ndarray) -> Path:
		"""Write eval JSON and (re)generate trajectory files. Returns the eval JSON path."""
		# Prepare evaluation-specific JSON (avoid mutating base_data).
		data = json.loads(json.dumps(base_data))
		apply_theta_to_qp_json(data, theta_keys=theta_keys, theta=theta_full.tolist(), qp_repeat=qp_repeat)
		json_path = workdir / "qp_config_eval.json"
		save_json_atomic(json_path, data)
		if planner_cmd is None and (not planner_mpcsolver):
			return json_path
		if traj_csv_path is None:
			raise RuntimeError("Planner requires a trajectory output path")
		base_path = Path(traj_csv_path)
		# Remove stale trajectories so we don't accidentally evaluate old files.
		to_remove: List[Path] = []
		if df_all is None:
			to_remove.append(base_path)
		else:
			for lc_id in lc_ids_selected:
				to_remove.append(_traj_base_for_lc(base_path, lc_id=int(lc_id)))
		for p in to_remove:
			try:
				p.unlink(missing_ok=True)
			except TypeError:  # py<3.8 compatibility
				if p.exists():
					p.unlink()
		if planner_cmd is not None:
			cmd = planner_cmd.format(json_path=str(json_path), traj_csv=str(base_path))
			if verbose:
				print(f"[planner] {cmd}")
			_run_cmd(cmd, cwd=workdir)
			if not base_path.exists() or base_path.stat().st_size == 0:
				raise RuntimeError(f"Planner did not produce a valid trajectory file at: {base_path}")
			return json_path
		# Built-in MPC solver.
		if df_all is not None:
			lc_ids_eval = _lc_ids_eval_for_theta(theta_full)
			for lc_id in lc_ids_eval:
				df_seg = df_all[df_all["lc_id"] == lc_id].copy()
				if len(df_seg) < 2:
					raise RuntimeError(f"Reference segment for lc_id={lc_id} has <2 rows")
				out_lc = _traj_base_for_lc(base_path, lc_id=int(lc_id))
				if verbose:
					print(f"[planner:mpcsolver] writing {out_lc} (lc_id={int(lc_id)})")
				generate_trajectory_csv_with_mpcsolver(
					df_ref=df_seg,
					out_path=out_lc,
					theta=theta_full.tolist(),
					opt_timesteps=int(mpc_opt_timesteps),
					dt_s=float(mpc_dt_s),
					solver=str(mpc_solver),
				)
				if not out_lc.exists() or out_lc.stat().st_size == 0:
					raise RuntimeError(f"Built-in MPC planner did not produce: {out_lc}")
		else:
			assert mpc_reference_df is not None
			if verbose:
				print(f"[planner:mpcsolver] writing {base_path}")
			generate_trajectory_csv_with_mpcsolver(
				df_ref=mpc_reference_df,
				out_path=base_path,
				theta=theta_full.tolist(),
				opt_timesteps=int(mpc_opt_timesteps),
				dt_s=float(mpc_dt_s),
				solver=str(mpc_solver),
			)
			if not base_path.exists() or base_path.stat().st_size == 0:
				raise RuntimeError(f"Built-in MPC planner did not produce: {base_path}")
		return json_path

	def _evaluate_theta_impl(theta_full: np.ndarray) -> Tuple[float, np.ndarray]:
		"""Evaluate scalar cost for a full (expanded) theta (7D if lane-change)."""
		_run_planner_for_theta(theta_full)
		assert traj_csv_path is not None
		base_path = Path(traj_csv_path)
		if df_all is None:
			cost = float(_evaluate_cost_from_trajectory(base_path))
			return float(cost), theta_full
		lc_ids_eval = _lc_ids_eval_for_theta(theta_full)
		lc_costs = [
			float(_evaluate_cost_from_trajectory(_traj_base_for_lc(base_path, lc_id=int(lc_id))))
			for lc_id in lc_ids_eval
		]
		cost = float(_aggregate_costs(lc_costs, mode=str(mpc_aggregate)))
		return float(cost), theta_full

	def _evaluate_theta_impl_features(theta_full: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
		"""Evaluate feature sums and implied scalar cost for a full theta."""
		_run_planner_for_theta(theta_full)
		assert traj_csv_path is not None
		base_path = Path(traj_csv_path)
		if df_all is None:
			feat, omega_used = _evaluate_feature_sums_from_trajectory(base_path)
			cost = float(np.dot(feat, omega_used))
			return feat, cost, theta_full
		lc_ids_eval = _lc_ids_eval_for_theta(theta_full)
		feats: List[np.ndarray] = []
		omega_used: Optional[np.ndarray] = None
		for lc_id in lc_ids_eval:
			f, om = _evaluate_feature_sums_from_trajectory(_traj_base_for_lc(base_path, lc_id=int(lc_id)))
			feats.append(np.asarray(f, dtype=float).reshape(-1))
			omega_used = np.asarray(om, dtype=float).reshape(-1)
		assert omega_used is not None
		feat_agg = _aggregate_vectors(feats, mode=str(mpc_aggregate))
		cost = float(np.dot(feat_agg, omega_used))
		return np.asarray(feat_agg, dtype=float).reshape(-1), cost, theta_full

	def _expand_theta(theta: np.ndarray) -> np.ndarray:
		theta = np.asarray(theta, dtype=float).reshape(-1)
		if not tie_lane_change_pairs:
			return theta
		return _lane_change_expand_theta5_to_theta7(theta.tolist())

	def _evaluate_theta(theta: Sequence[float], *, record: bool = True, use_cache: bool = True, print_eval: bool = True) -> float:
		theta_param = np.asarray(list(theta), dtype=float).reshape(-1)
		theta_full = _expand_theta(theta_param)
		key = tuple(np.round(theta_full, 12).tolist())
		if cache and use_cache and key in cache_cost:
			return float(cache_cost[key])

		try:
			cost, _ = _evaluate_theta_impl(theta_full)
		except Exception as e:
			# During iterative optimization, a few candidate thetas may cause the planner
			# (or downstream parsing) to fail. Penalize instead of aborting the whole run.
			# For eval-only/baseline paths (record=False), fail fast so the user sees the error.
			if not record:
				raise
			if verbose:
				print(f"[eval] ERROR for theta={theta_full.tolist()}: {type(e).__name__}: {e}")
			cost = 1e12

		# Optional regularization to keep theta in a reasonable range.
		if theta_reg > 0:
			den = float(np.mean(theta0 * theta0) + 1e-12)
			pen = float(np.mean((theta_param - theta0) ** 2) / den)
			cost = float(cost + theta_reg * pen)

		if record:
			history.append((theta_full.copy(), float(cost)))
		if cache and use_cache:
			cache_cost[key] = float(cost)
		if print_eval and verbose:
			print(f"[eval] cost={cost:.6g} theta={theta_full.tolist()}")
		return float(cost)

	def _objectives_from_feature_sums(feat_sums: np.ndarray) -> np.ndarray:
		feat_sums = np.asarray(feat_sums, dtype=float).reshape(-1)
		mode = str(mo_objectives).strip().lower()
		if mode == "features":
			return feat_sums
		if mode != "group3":
			raise ValueError("mo_objectives must be one of: group3, features")
		# Group into 3 interpretable objectives (all are minimization objectives):
		# - tracking: lateral offset + curvature tracking (+ optional heading tracking)
		# - smoothness: lateral-rate + curvature-rate + delta terms
		# - effort: curvature magnitude + control effort
		phi0 = float(feat_sums[0])
		phi1 = float(feat_sums[1])
		phi2 = float(feat_sums[2])
		phi3 = float(feat_sums[3])
		phi4 = float(feat_sums[4])
		phi5 = float(feat_sums[5])
		phi6 = float(feat_sums[6])
		phi7 = float(feat_sums[7])
		phi8 = float(feat_sums[8]) if feat_sums.size >= 9 else 0.0
		obj_track = phi0 + phi4 + phi8
		obj_smooth = phi1 + phi3 + phi5 + phi6
		obj_effort = phi2 + phi7
		return np.asarray([obj_track, obj_smooth, obj_effort], dtype=float)

	def _evaluate_theta_multi(
		theta: Sequence[float],
		*,
		record: bool = True,
		use_cache: bool = True,
		print_eval: bool = True,
	) -> Tuple[np.ndarray, float, np.ndarray]:
		"""Return (objectives, scalar_cost, theta_full) for NSGA-II."""
		nonlocal expected_feat_dim
		theta_param = np.asarray(list(theta), dtype=float).reshape(-1)
		theta_full = _expand_theta(theta_param)
		key = tuple(np.round(theta_full, 12).tolist())
		if cache and use_cache and key in cache_feat and key in cache_cost:
			feat_sums = np.asarray(cache_feat[key], dtype=float).reshape(-1)
			obj = _objectives_from_feature_sums(feat_sums)
			return obj, float(cache_cost[key]), theta_full
		try:
			feat_sums, cost, _ = _evaluate_theta_impl_features(theta_full)
			if expected_feat_dim is None:
				expected_feat_dim = int(np.asarray(feat_sums, dtype=float).reshape(-1).size)
		except Exception as e:
			if not record:
				raise
			if verbose:
				print(f"[eval-mo] ERROR for theta={theta_full.tolist()}: {type(e).__name__}: {e}")
			d = int(expected_feat_dim) if expected_feat_dim is not None else 9
			feat_sums = np.full((d,), 1e12, dtype=float)
			cost = 1e12
		# Optional regularization in decision space (applied to scalar cost only; objectives remain feature-based).
		if theta_reg > 0:
			den = float(np.mean(theta0 * theta0) + 1e-12)
			pen = float(np.mean((theta_param - theta0) ** 2) / den)
			cost = float(cost + theta_reg * pen)
		if record:
			history.append((theta_full.copy(), float(cost)))
		if cache and use_cache:
			cache_feat[key] = np.asarray(feat_sums, dtype=float).reshape(-1)
			cache_cost[key] = float(cost)
		obj = _objectives_from_feature_sums(np.asarray(feat_sums, dtype=float).reshape(-1))
		if print_eval and verbose:
			print(f"[eval-mo] F={obj.tolist()} cost={float(cost):.6g} theta={theta_full.tolist()}")
		return obj, float(cost), theta_full

	# Eval-only mode: evaluate one theta and exit without running an optimizer.
	# Placed before baseline so we don't evaluate twice.
	if eval_only_theta is not None:
		t = np.asarray(list(eval_only_theta), dtype=float).reshape(-1)
		if not np.isfinite(t).all():
			raise ValueError("eval_only_theta must be finite")
		if tie_lane_change_pairs:
			if t.size == 7:
				t_param = _lane_change_reduce_theta7_to_theta5(t.tolist())
			elif t.size == 5:
				t_param = t
			else:
				raise ValueError(f"With tie_lane_change_pairs, eval_only_theta must have length 5 or 7, got {t.size}")
		else:
			if t.size != theta0.size:
				raise ValueError(f"eval_only_theta length ({t.size}) must match theta0 length ({theta0.size})")
			t_param = t
		c = _evaluate_theta(t_param.tolist(), record=False, use_cache=False, print_eval=False)
		print(f"[eval-only] HLO cost = {c:.6g} theta={_expand_theta(t_param).tolist()}")
		# Helpful artifact paths for downstream inspection.
		print(f"[eval-only] workdir:   {workdir}")
		print(f"[eval-only] eval_json: {workdir / 'qp_config_eval.json'}")
		if traj_csv_path is not None:
			base_path = Path(traj_csv_path)
			if df_all is None:
				print(f"[eval-only] traj_csv:  {base_path}")
				png_path = workdir / "default_theta_trajectory.png"
				if base_path.exists() and base_path.stat().st_size > 0:
					_plot_trajectory_csv_to_png(base_path, png_path)
					print(f"[eval-only] plot_png:  {png_path}")
			else:
				ids = _lc_ids_eval_for_theta(_expand_theta(t_param))
				paths = [_traj_base_for_lc(base_path, lc_id=int(lc_id)) for lc_id in ids]
				for p in paths[:5]:
					print(f"[eval-only] traj_csv:  {p}")
				# Plot the first available one.
				for p in paths:
					if p.exists() and p.stat().st_size > 0:
						png_path = workdir / f"default_theta_trajectory_lc{p.stem.split('_lc')[-1]}.png"
						_plot_trajectory_csv_to_png(p, png_path)
						print(f"[eval-only] plot_png:  {png_path}")
						break
		# If an external planner produced MCAPs under workdir, report the newest one.
		try:
			mcaps = sorted(workdir.glob("*.mcap"), key=lambda p: p.stat().st_mtime)
			if mcaps:
				print(f"[eval-only] mcap:      {mcaps[-1]}")
		except Exception:
			pass
		return BilevelResult(theta_star=_expand_theta(t_param), cost_star=float(c), n_evals=len(history), history=history)

	# Baseline evaluation: print HLO cost for the lane-change default theta.
	if baseline_theta is not None:
		base = np.asarray(list(baseline_theta), dtype=float).reshape(-1)
		if not np.isfinite(base).all():
			raise ValueError("baseline_theta must be finite")
		if tie_lane_change_pairs:
			# Optimization runs in 5D; accept either 5D (already tied) or 7D (reduce by averaging).
			if base.size == 7:
				base_param = _lane_change_reduce_theta7_to_theta5(base.tolist())
			elif base.size == 5:
				base_param = base
			else:
				raise ValueError(f"With tie_lane_change_pairs, baseline_theta must have length 5 or 7, got {base.size}")
		else:
			if base.size != theta0.size:
				raise ValueError(f"baseline_theta length ({base.size}) must match theta0 length ({theta0.size})")
			base_param = base
		baseline_cost = _evaluate_theta(base_param.tolist(), record=False, use_cache=False, print_eval=False)
		print(f"[baseline] default_theta HLO cost = {baseline_cost:.6g}")
		# Preserve the baseline artifacts (subsequent evaluations overwrite the eval files).
		try:
			if traj_csv_path is not None:
				base_path = Path(traj_csv_path)
				if df_all is None:
					traj_dst = workdir / "trajectory_default_theta.csv"
					if base_path.exists():
						shutil.copyfile(base_path, traj_dst)
						png_path = workdir / "default_theta_trajectory.png"
						if traj_dst.exists() and traj_dst.stat().st_size > 0:
							_plot_trajectory_csv_to_png(traj_dst, png_path)
							print(f"[baseline] traj_csv:  {traj_dst}")
							print(f"[baseline] plot_png:  {png_path}")
				else:
					ids = _lc_ids_eval_for_theta(_expand_theta(base_param))
					paths = [_traj_base_for_lc(base_path, lc_id=int(lc_id)) for lc_id in ids]
					# Copy the first available one to a stable filename.
					for p in paths:
						if p.exists() and p.stat().st_size > 0:
							traj_dst = workdir / "trajectory_default_theta.csv"
							shutil.copyfile(p, traj_dst)
							png_path = workdir / "default_theta_trajectory.png"
							_plot_trajectory_csv_to_png(traj_dst, png_path)
							print(f"[baseline] traj_csv:  {traj_dst}")
							print(f"[baseline] plot_png:  {png_path}")
							break
			json_src = workdir / "qp_config_eval.json"
			json_dst = workdir / "qp_config_default_theta.json"
			if json_src.exists():
				shutil.copyfile(json_src, json_dst)
		except Exception:
			# Keep baseline printing robust; evaluation already succeeded.
			pass

	if optimizer == "scipy":
		scipy_method = _normalize_scipy_minimize_method(method)
		if bool(verbose) and str(scipy_method) != str(method).strip():
			print(f"[scipy] method '{method}' -> '{scipy_method}'")

		# Optimize in unconstrained space but evaluate with physical theta.
		x0 = _opt_vars_from_theta0(
			theta0,
			transform=str(theta_transform),
			theta_min=theta_min,
			theta_max=theta_max,
		)

		def _evaluate_x(x: np.ndarray) -> float:
			x = np.asarray(x, dtype=float).reshape(-1)
			theta = _theta_from_opt_vars(
				x,
				transform=str(theta_transform),
				theta_min=theta_min,
				theta_max=theta_max,
			).reshape(-1)
			return _evaluate_theta(theta.tolist())

		res = minimize(
			fun=_evaluate_x,
			x0=x0,
			method=str(scipy_method),
			options={"maxiter": int(maxiter), "disp": bool(verbose)},
		)

		theta_star_param = _theta_from_opt_vars(
			np.asarray(res.x, dtype=float),
			transform=str(theta_transform),
			theta_min=theta_min,
			theta_max=theta_max,
		)
		theta_star = _expand_theta(theta_star_param)
		# Ensure workdir artifacts reflect the final solution (theta*), not an arbitrary last eval.
		# This overwrites trajectory_eval.csv (and qp_config_eval.json) deterministically.
		cost_star = _evaluate_theta(theta_star_param.tolist(), record=False, use_cache=False, print_eval=False)
		if traj_csv_path is not None:
			base_path = Path(traj_csv_path)
			plot_src: Optional[Path] = None
			if df_all is None:
				plot_src = base_path
			else:
				ids = _lc_ids_eval_for_theta(theta_star)
				if ids:
					plot_src = _traj_base_for_lc(base_path, lc_id=int(ids[0]))
			if plot_src is not None and plot_src.exists() and plot_src.stat().st_size > 0:
				png_path = workdir / "trajectory_latest.png"
				_plot_trajectory_csv_to_png(plot_src, png_path)
				if verbose:
					print(f"[plot] wrote {png_path}")
		return BilevelResult(theta_star=theta_star, cost_star=cost_star, n_evals=len(history), history=history)

	if optimizer == "nsga2":
		if bool(verbose) and str(method).strip() and str(method).strip() != "Powell":
			print(f"[nsga2] NOTE: ignoring --method={method!r} (only used for --optimizer=scipy)")
		if hlo_omega is None:
			raise ValueError("optimizer='nsga2' requires internal HLO (provide --hlo_omega or --hlo_omega_objective_json)")
		nsga2_pop_size = int(nsga2_pop_size)
		if nsga2_pop_size <= 2:
			raise ValueError("nsga2_pop_size must be > 2")
		# NSGA-II operates directly in bounded decision space.
		lo = _broadcast_param(theta_min, n=theta0.size, default=0.0)
		hi = _broadcast_param(theta_max, n=theta0.size, default=1.0)
		if np.any(hi <= lo):
			raise ValueError("For optimizer='nsga2', theta_max must be > theta_min (elementwise)")
		pymoo = _import_pymoo()
		ElementwiseProblem = pymoo["ElementwiseProblem"]
		NSGA2 = pymoo["NSGA2"]
		SBX = pymoo["SBX"]
		PM = pymoo["PM"]
		FloatRandomSampling = pymoo["FloatRandomSampling"]
		pymoo_minimize = pymoo["pymoo_minimize"]
		get_termination = pymoo["get_termination"]

		# Determine number of objectives by evaluating theta0 once (cached).
		F0, _c0, _th0 = _evaluate_theta_multi(theta0.tolist(), record=True, use_cache=True, print_eval=False)
		n_obj = int(np.asarray(F0, dtype=float).reshape(-1).size)

		class _ThetaProblem(ElementwiseProblem):
			def __init__(self):
				super().__init__(n_var=int(theta0.size), n_obj=int(n_obj), xl=lo, xu=hi)

			def _evaluate(self, x: np.ndarray, out: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
				print_eval = bool(verbose) and (nsga2_log == "eval")
				F, _cost, _theta_full = _evaluate_theta_multi(
					x.tolist(),
					record=True,
					use_cache=True,
					print_eval=print_eval,
				)
				out["F"] = np.asarray(F, dtype=float).reshape(-1)

		def _nsga2_callback(algorithm: Any) -> None:
			if (not verbose) or nsga2_log != "gen":
				return
			try:
				X = np.asarray(algorithm.pop.get("X"), dtype=float)
				F = np.asarray(algorithm.pop.get("F"), dtype=float)
			except Exception:
				return
			if X.ndim != 2 or X.shape[0] == 0:
				return
			best_i = 0
			best_cost = float("inf")
			best_theta_full: Optional[np.ndarray] = None
			best_F: Optional[np.ndarray] = None
			for i in range(int(X.shape[0])):
				try:
					_Fi, ci, thi = _evaluate_theta_multi(X[i].tolist(), record=False, use_cache=True, print_eval=False)
				except Exception:
					continue
				if float(ci) < float(best_cost):
					best_i = int(i)
					best_cost = float(ci)
					best_theta_full = np.asarray(thi, dtype=float).reshape(-1)
					best_F = np.asarray(_Fi, dtype=float).reshape(-1)
			if best_theta_full is None:
				best_theta_full = _expand_theta(np.asarray(X[best_i], dtype=float).reshape(-1))
			if best_F is None:
				best_F = np.asarray(F[best_i], dtype=float).reshape(-1)
			gen = getattr(algorithm, "n_gen", None)
			gen_s = str(int(gen)) if gen is not None else "?"
			print(
				f"[nsga2] gen={gen_s} best_cost={best_cost:.6g} best_F={best_F.tolist()} best_theta={best_theta_full.tolist()}"
			)

		algorithm = NSGA2(
			pop_size=int(nsga2_pop_size),
			sampling=FloatRandomSampling(),
			crossover=SBX(prob=0.9, eta=15),
			mutation=PM(eta=20),
			eliminate_duplicates=True,
		)
		termination = get_termination("n_gen", int(maxiter))
		res = pymoo_minimize(
			_ThetaProblem(),
			algorithm,
			termination,
			seed=0,
			verbose=bool(verbose),
			callback=_nsga2_callback,
		)

		pareto_X = np.asarray(res.X, dtype=float)
		pareto_F = np.asarray(res.F, dtype=float)
		if pareto_X.ndim == 1:
			pareto_X = pareto_X.reshape(1, -1)
		if pareto_F.ndim == 1:
			pareto_F = pareto_F.reshape(1, -1)
		# Choose a single theta* from the Pareto set using the learned scalar objective (omega).
		pareto_costs: List[float] = []
		pareto_theta_full: List[np.ndarray] = []
		for x in pareto_X:
			_F, c, th_full = _evaluate_theta_multi(x.tolist(), record=False, use_cache=True, print_eval=False)
			pareto_costs.append(float(c))
			pareto_theta_full.append(np.asarray(th_full, dtype=float).reshape(-1))
		idx_best = int(np.argmin(np.asarray(pareto_costs, dtype=float)))
		theta_star_param = np.asarray(pareto_X[idx_best], dtype=float).reshape(-1)
		theta_star = _expand_theta(theta_star_param)
		# Ensure workdir artifacts reflect the final chosen solution.
		cost_star = float(_evaluate_theta(theta_star_param.tolist(), record=False, use_cache=False, print_eval=False))
		# Write Pareto front summary.
		try:
			pareto_path = workdir / "pareto_front.csv"
			rows: List[Dict[str, Any]] = []
			for i in range(int(pareto_X.shape[0])):
				row: Dict[str, Any] = {"idx": i, "cost": float(pareto_costs[i])}
				for j, v in enumerate(np.asarray(pareto_theta_full[i], dtype=float).reshape(-1).tolist()):
					row[f"theta_{j}"] = float(v)
				for k, v in enumerate(np.asarray(pareto_F[i], dtype=float).reshape(-1).tolist()):
					row[f"F_{k}"] = float(v)
				rows.append(row)
			pd.DataFrame(rows).to_csv(pareto_path, index=False)
			if verbose:
				print(f"[nsga2] saved Pareto front: {pareto_path}")
		except Exception as e:
			if verbose:
				print(f"[nsga2] Warning: failed to write pareto_front.csv: {e}")

		if traj_csv_path is not None:
			base_path = Path(traj_csv_path)
			plot_src: Optional[Path] = None
			if df_all is None:
				plot_src = base_path
			else:
				ids = _lc_ids_eval_for_theta(theta_star)
				if ids:
					plot_src = _traj_base_for_lc(base_path, lc_id=int(ids[0]))
			if plot_src is not None and plot_src.exists() and plot_src.stat().st_size > 0:
				png_path = workdir / "trajectory_latest.png"
				_plot_trajectory_csv_to_png(plot_src, png_path)
				if verbose:
					print(f"[plot] wrote {png_path}")
		return BilevelResult(theta_star=theta_star, cost_star=cost_star, n_evals=len(history), history=history)

	# For Bayesian optimization, --method is currently not used.
	if bool(verbose) and str(method).strip() and str(method).strip() != "Powell":
		print(f"[bayes] NOTE: ignoring --method={method!r} (only used for --optimizer=scipy)")

	# Bayesian optimization directly over bounded physical theta.
	gp_minimize, Real = _import_skopt()
	lo = _broadcast_param(theta_min, n=theta0.size, default=0.0)
	hi = _broadcast_param(theta_max, n=theta0.size, default=1.0)
	if np.any(hi <= lo):
		raise ValueError("For optimizer='bayes', theta_max must be > theta_min (elementwise)")

	# skopt validates the provided x0 against bounds strictly. For our default MPC
	# settings, theta_min may include a tiny floor (e.g. 1e-6) while default theta0
	# contains exact zeros. Clip the initial point into-bounds to avoid a hard crash.
	theta0_bayes = np.asarray(theta0, dtype=float).reshape(-1)
	out_of_bounds = (theta0_bayes < lo) | (theta0_bayes > hi)
	if np.any(out_of_bounds):
		clipped = np.minimum(np.maximum(theta0_bayes, lo), hi)
		if verbose:
			bad = np.where(out_of_bounds)[0].tolist()
			print(
				"[bayes] theta0 is outside [theta_min, theta_max] for some dimensions; "
				"clipping x0 into bounds before gp_minimize."
			)
			for i in bad:
				print(
					f"  theta[{i}] {float(theta0_bayes[i]):.6g} -> {float(clipped[i]):.6g} "
					f"(bounds [{float(lo[i]):.6g}, {float(hi[i]):.6g}])"
				)
		theta0_bayes = clipped

	space = [Real(float(a), float(b), name=f"theta_{i}") for i, (a, b) in enumerate(zip(lo.tolist(), hi.tolist()))]

	# Note: gp_minimize minimizes the objective. Our objective is already a cost.
	res = gp_minimize(
		func=lambda x: _evaluate_theta(x),
		dimensions=space,
		n_calls=int(bayes_calls),
		n_initial_points=int(bayes_init_points),
		acq_func=str(bayes_acq_func),
		x0=[theta0_bayes.tolist()],
		random_state=0,
		verbose=bool(verbose),
	)

	theta_star = _expand_theta(np.asarray(res.x, dtype=float).reshape(-1))
	# Ensure workdir artifacts reflect the final solution (theta*).
	cost_star = _evaluate_theta(np.asarray(res.x, dtype=float).reshape(-1).tolist(), record=False, use_cache=False, print_eval=False)
	if traj_csv_path is not None:
		base_path = Path(traj_csv_path)
		plot_src: Optional[Path] = None
		if df_all is None:
			plot_src = base_path
		else:
			ids = _lc_ids_eval_for_theta(theta_star)
			if ids:
				plot_src = _traj_base_for_lc(base_path, lc_id=int(ids[0]))
		if plot_src is not None and plot_src.exists() and plot_src.stat().st_size > 0:
			png_path = workdir / "trajectory_latest.png"
			_plot_trajectory_csv_to_png(plot_src, png_path)
			if verbose:
				print(f"[plot] wrote {png_path}")
	return BilevelResult(theta_star=theta_star, cost_star=cost_star, n_evals=len(history), history=history)


# -----------------------------------------------------------------------------
# CLI


def _parse_floats_csv(s: str) -> List[float]:
	parts = [p.strip() for p in s.split(",") if p.strip()]
	if not parts:
		return []
	return [float(p) for p in parts]


def _parse_ints_csv(s: str) -> List[int]:
	parts = [p.strip() for p in s.split(",") if p.strip()]
	if not parts:
		return []
	return [int(p) for p in parts]


def _default_mpc_velocities_mps() -> List[float]:
	"""Default velocity sweep for built-in MPC evaluation.

	The built-in MPC generator (`--planner_mpcsolver`) is typically evaluated across
	workflow we default to a broad set of velocities to avoid overfitting theta to
	a single speed.

	Values are specified in km/h and converted to m/s.
	"""
	vel_kmh = [5.0, 15.0, 30.0, 80.0, 130.0, 180.0]
	return [float(v / 3.6) for v in vel_kmh]


def _find_default_planner_overwrite_json() -> Optional[Path]:
	"""No repository-specific planner overwrite config is assumed by default."""
	return None


def main(argv: Optional[Sequence[str]] = None) -> int:
	parser = argparse.ArgumentParser(description="Bilevel optimization runner (QP JSON -> trajectory -> HLO cost -> optimize theta).")
	_default_planner_overwrite_in = _find_default_planner_overwrite_json()
	parser.add_argument(
		"--qp_json",
		required=False,
		type=Path,
		default=_default_planner_overwrite_in,
		help=(
			"Path to base QP planner JSON config. Can also be a planner overwrite JSON shaped like {\"parameters\": [...]} "
			"(parameter names are used as theta_keys). "
			"Default: none."
		),
	)
	parser.add_argument(
		"--theta0",
		type=str,
		default=None,
		help=(
			"Initial theta as comma-separated floats. If omitted, uses the placeholder values from the original comment."
		),
	)
	parser.add_argument(
		"--theta0_mode",
		type=str,
		choices=["default", "zeros", "random_uniform"],
		default="default",
		help=(
			"How to choose the initial theta when --theta0 is not provided. "
			"'default' starts from the built-in lane-change default weights; "
			"'zeros' starts from all zeros (old behavior); "
			"'random_uniform' samples uniformly within bounds."
		),
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=0,
		help="Random seed used when --theta0_mode=random_uniform (default: 0).",
	)
	parser.add_argument(
		"--theta_keys",
		type=str,
		default=",".join(_LANE_CHANGE_THETA_KEYS),
		help=(
			"Comma-separated keys to set from theta. Two formats are supported: "
			"(1) JSON key-paths (dotted paths, indices like 'a.b[0].c'), or "
			"(2) planner overwrite parameter names (e.g. 'tpl_change_lane_weight_lateral_jerk'). "
			"Default: the 7 tpl_change_lane_weight_* lane-change parameters."
		),
	)
	parser.add_argument(
		"--tie_lane_change_pairs",
		action="store_true",
		help=(
			"Tie the lane-change fast/slow pairs so only 5 parameters are optimized: "
			"offset_fast=offset_slow and velocity_fast=velocity_slow. "
			"This expands 5D -> 7D when writing JSON / running the MPC planner."
		),
	)
	parser.add_argument(
		"--no_tie_lane_change_pairs",
		action="store_true",
		help="Disable automatic tying of lane-change fast/slow pairs.",
	)
	parser.add_argument(
		"--qp_repeat",
		type=int,
		default=10,
		help=(
			"When theta_keys refer to planner overwrite parameter names, set each initValue to this many repeated values "
			"(default: 10)."
		),
	)
	parser.add_argument(
		"--planner_cmd",
		type=str,
		default=None,
		help=(
			"Optional command to run the inner QP planner. Supports placeholders {json_path} and {traj_csv}. "
			"If omitted, assumes the trajectory file already exists."
		),
	)
	parser.add_argument(
		"--planner_mpcsolver",
		action="store_true",
		help=(
			"Generate the trajectory using the optional MPC solver package (no MCAP). "
			"Writes a merged-format CSV to --traj_csv (or the auto default under --workdir)."
		),
	)
	parser.add_argument("--mpc_opt_timesteps", type=int, default=50, help="MPC horizon length when using --planner_mpcsolver.")
	# parser.add_argument("--mpc_dt_s", type=float, default=0.1, help="MPC timestep dt in seconds when using --planner_mpcsolver.")
	parser.add_argument("--mpc_dt_s", type=float, default=0.04, help="MPC timestep dt in seconds when using --planner_mpcsolver.")

	parser.add_argument(
		"--mpc_reference_csv",
		type=str,
		default=None,
		help=(
			"Path to a single preprocessed reference segment CSV to drive the built-in MPC generator. "
			"Must contain time + SG reference columns (see generator docs)."
		),
	)
	parser.add_argument(
		"--mpc_reference_merged_csv",
		type=str,
		default=None,
		help=(
			"Path to a merged preprocessed reference CSV containing multiple segments with a 'lc_id' column. "
			"If set, each theta is evaluated across many lc_id segments (optionally minibatched)."
		),
	)
	parser.add_argument(
		"--mpc_lc_ids",
		type=str,
		default=None,
		help=(
			"Optional comma-separated list of lc_id to evaluate from --mpc_reference_merged_csv. "
			"If omitted, all lc_id values are used."
		),
	)
	parser.add_argument(
		"--mpc_batch_size",
		type=int,
		default=0,
		help=(
			"If >0 and using --mpc_reference_merged_csv, evaluate a deterministic minibatch of this many lc_id per theta. "
			"If 0, evaluates all selected lc_id."
		),
	)
	parser.add_argument(
		"--mpc_aggregate",
		type=str,
		default="mean",
		choices=["mean", "sum", "max"],
		help="How to aggregate costs across lc_id segments when using --mpc_reference_merged_csv (default: mean).",
	)
	parser.add_argument(
		"--mpc_solver",
		type=str,
		default="osqp",
		choices=["ipopt", "osqp"],
		help="Backend used by MPCSolver when using --planner_mpcsolver.",
	)
	parser.add_argument(
		"--theta_transform",
		type=str,
		choices=["identity", "softplus", "sigmoid"],
		default=None,
		help=(
			"How to parameterize theta during outer optimization. "
			"'identity' optimizes theta directly; 'softplus' enforces theta>=0; "
			"'sigmoid' enforces theta in [theta_min, theta_max]. "
			"Default: 'sigmoid' when using --planner_mpcsolver, otherwise 'identity'."
		),
	)
	parser.add_argument(
		"--theta_min",
		type=str,
		default=None,
		help=(
			"Lower bound(s) for theta when --theta_transform=sigmoid. "
			"Provide a single float or a comma-separated list (length = len(theta)). "
			"Default: 0.0 (or broadcast if list)."
		),
	)
	parser.add_argument(
		"--theta_max",
		type=str,
		default=None,
		help=(
			"Upper bound(s) for theta when --theta_transform=sigmoid. "
			"Provide a single float or a comma-separated list (length = len(theta)). "
			"Default: 1.0 for --planner_mpcsolver, otherwise 1.0 if sigmoid is selected."
		),
	)
	parser.add_argument(
		"--theta_reg",
		type=float,
		default=0.0,
		help=(
			"Optional regularization strength to keep theta close to theta0. "
			"Adds: theta_reg * mean((theta-theta0)^2) / mean(theta0^2). Default: 0.0."
		),
	)
	parser.add_argument(
		"--skip_planner",
		action="store_true",
		help=(
			"Do not run --planner_cmd even if provided (useful to avoid expensive MCAP generation). "
			"You must provide an existing --traj_csv (or use an external --hlo_cmd that does not require a trajectory)."
		),
	)
	parser.add_argument(
		"--traj_csv",
		type=Path,
		default=None,
		help=(
			"Trajectory file path produced by planner_cmd, or an existing file if planner_cmd omitted. "
			"Supported: merged-format .csv or JSON trace .json. "
			"Does NOT need to exist in advance. If omitted but required (internal HLO or {traj_csv} placeholders), "
			"defaults to '<workdir>/trajectory_eval.csv'."
		),
	)
	parser.add_argument(
		"--cost_regex",
		type=str,
		default=r"cost\s*[:=]\s*([+-]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][+-]?\d+)?)",
		help="Regex with one capture group for cost when using --hlo_cmd. Default matches 'cost: 1.23'.",
	)
	parser.add_argument(
		"--hlo_omega",
		type=str,
		default=None,
		help=(
			"Compute HLO cost internally using omega as 8 or 9 comma-separated floats. "
			"Order (8): [lat_err_sq, lat_rate_sq_v2, kappa_abs_sq, kappa_dot_abs_sq, kappa_err_sq, delta_kappa_err_sq, delta_kappa_dot_sq, u_sq]. "
			"Order (9): same + [psi_err_sq] where psi_err_sq = v^2*wrap_to_pi(psi-psi_ref)^2 (requires reference_orientation_* columns in trajectory CSV). "
			"Legacy objectives with a duration feature and/or without lat_rate may still load from learned_objective.json when feature_names are present."
		),
	)
	parser.add_argument(
		"--hlo_target_lat_offset_m",
		type=float,
		default=0.0,
		help=(
			"When using internal HLO cost (always used during optimization), set a FIXED target lateral offset in meters. "
			"Default: 0.0."
		),
	)
	parser.add_argument(
		"--hlo_target_lat_offset_mode",
		type=str,
		choices=["auto", "final", "fixed", "nearest_rel"],
		default="fixed",
		help=(
			"How to choose the target lateral offset for the HLO lateral-tracking feature (lat_err_sq). "
			"Default: fixed (i.e. always use --hlo_target_lat_offset_m, which defaults to 0.0). "
			"auto: fixed if --hlo_target_lat_offset_m is provided else final. "
			"final: use the trajectory's final lateral offset (discouraged for lane-change timing). "
			"fixed: always use --hlo_target_lat_offset_m. "
			"nearest_rel: snap the end offset (relative to the start) to the nearest value in --hlo_target_lat_offset_candidates_rel "
			"(default candidates: 0.0,-3.5), then convert back to an absolute target."
		),
	)
	parser.add_argument(
		"--hlo_target_lat_offset_candidates_rel",
		type=str,
		default="0.0,-3.5,3.5",
		help=(
			"Comma-separated candidate relative target offsets (meters) used when --hlo_target_lat_offset_mode=nearest_rel. "
			"Default: '0.0,-3.5,3.5'."
		),
	)
	parser.add_argument(
		"--hlo_target_lat_offset_window",
		type=int,
		default=5,
		help=(
			"Median window size (samples) used to estimate start/end lateral offsets when --hlo_target_lat_offset_mode=nearest_rel "
			"(default: 5)."
		),
	)
	parser.add_argument(
		"--hlo_threshold_m",
		type=float,
		default=0.5,
		help=(
			"Deprecated (duration feature removed): kept for backward compatibility (default: 0.5)."
		),
	)
	parser.add_argument(
		"--hlo_softness_m",
		type=float,
		default=0.2,
		help="Deprecated (duration feature removed): kept for backward compatibility (default: 0.2).",
	)
	parser.add_argument(
		"--min_indicator_to_boundary_touch_s",
		type=float,
		default=2.2,
		help=(
			"Hard timing constraint surrogate used in outer optimization. "
			"Minimum required time (seconds) from indicator trigger (trajectory start) to lane-boundary touch. "
			"Set <=0 to disable. Default: 2.2"
		),
	)
	parser.add_argument(
		"--boundary_touch_fraction",
		type=float,
		default=0.5,
		help=(
			"Fraction of start->target lateral shift used as boundary-touch surrogate. "
			"0.5 approximates touching the lane boundary. Default: 0.5"
		),
	)
	parser.add_argument(
		"--timing_constraint_penalty_weight",
		type=float,
		default=1e4,
		help=(
			"Quadratic penalty weight for timing violations: weight * max(0, min_time - measured_time)^2. "
			"Default: 1e4"
		),
	)
	parser.add_argument(
		"--timing_constraint_missing_penalty",
		type=float,
		default=1e6,
		help=(
			"Penalty added when boundary-touch time cannot be estimated from a trajectory. Default: 1e6"
		),
	)
	parser.add_argument(
		"--hlo_omega_via_pipeline",
		action="store_true",
		help=(
			"Compute omega by running Pipeline/pipeline.py logic (Koopman dynamics + IOC) "
			"as a Python module. Requires --hlo_merged_csv."
		),
	)
	parser.add_argument(
		"--hlo_merged_csv",
		type=Path,
		default=None,
		help="Merged IOC CSV used when computing omega via Pipeline.pipeline.",
	)
	parser.add_argument("--hlo_n_traj", type=int, default=200, help="How many trajectories to load when computing omega via Pipeline.pipeline.")
	parser.add_argument("--hlo_seg_len", type=int, default=31, help="Segment length used when computing omega via Pipeline.pipeline.")
	parser.add_argument(
		"--hlo_lift",
		choices=["poly", "dnn"],
		default="poly",
		help="Koopman lift type when computing omega via Pipeline.pipeline.",
	)
	parser.add_argument("--hlo_degree", type=int, default=2, help="Polynomial lift degree when --hlo_lift=poly.")
	parser.add_argument("--hlo_dnn_psi_dim", type=int, default=20, help="DNN net output dim when --hlo_lift=dnn (total n_psi=n_state+hlo_dnn_psi_dim).")
	parser.add_argument("--hlo_dnn_hidden_dim", type=int, default=64)
	parser.add_argument("--hlo_dnn_hidden_layers", type=int, default=4)
	parser.add_argument("--hlo_dnn_pretrain_steps", type=int, default=0, help="Optional DKR pretraining steps for DNN lift (0=off).")
	parser.add_argument("--hlo_dnn_pretrain_segments", type=int, default=32)
	parser.add_argument("--hlo_dnn_lr", type=float, default=1e-3)
	parser.add_argument("--hlo_dnn_batch_segments", type=int, default=8)
	parser.add_argument("--hlo_dnn_refit_every", type=int, default=50)
	parser.add_argument(
		"--print_hlo_omega_only",
		action="store_true",
		help=(
			"Compute/resolve omega (HLO weights) using the selected source (--hlo_omega / via pipeline / objective_json), "
			"print it as a comma-separated list (8 or 9 floats for the current objective), and exit without running bilevel optimization."
		),
	)
	# Default location where Pipeline/pipeline.py writes omega
	_default_obj_json = (
		_REPO_ROOT / "Datasets" / "Output" / "PipelineRun" / "Learned" / "learned_objective.json"
	)

	parser.add_argument(
		"--hlo_omega_objective_json",
		type=Path,
		default=_default_obj_json if _default_obj_json.exists() else None,
		help=(
			"Load omega from a Pipeline learned_objective.json (expects top-level key 'omega'). "
			"DEFAULT: Datasets/Output/PipelineRun/Learned/learned_objective.json (if it exists)."
		),
	)

	parser.add_argument(
		"--eval_default_theta_only",
		action="store_true",
		help=(
			"Evaluate HLO cost once using the built-in lane-change default theta and exit (no optimization). "
			"Requires a planner to generate a trajectory (e.g. --planner_mpcsolver) or an existing --traj_csv."
		),
	)
	parser.add_argument(
		"--eval_theta",
		type=str,
		default=None,
		help=(
			"Evaluate HLO cost once for an explicit theta (comma-separated floats) and exit (no optimization). "
			"If lane-change fast/slow pairs are tied (default when using the 7 tpl_change_lane_weight_* keys), "
			"you may pass either 5 floats (tied form) or 7 floats (untied; will be reduced by averaging)."
		),
	)
	parser.add_argument(
		"--method",
		type=str,
		default="Powell",
		help=(
			"Local method for scipy.optimize.minimize when --optimizer=scipy. "
			"Examples: Powell, Nelder-Mead. Aliases accepted: 'nelder_mead', 'nelder mead', 'nm'."
		),
	)
	parser.add_argument(
		"--optimizer",
		type=str,
		choices=["scipy", "bayes", "nsga2"],
		default="scipy",
		help=(
			"Outer optimizer. 'scipy' uses scipy.optimize.minimize; 'bayes' uses Bayesian optimization "
			"(Gaussian-process) via scikit-optimize and requires bounds; 'nsga2' uses a population-based "
			"multi-objective evolutionary optimizer (NSGA-II) and writes a Pareto front CSV to <workdir>/pareto_front.csv."
		),
	)
	parser.add_argument(
		"--nsga2_pop_size",
		type=int,
		default=48,
		help="Population size for NSGA-II when --optimizer=nsga2 (default: 48).",
	)
	parser.add_argument(
		"--mo_objectives",
		type=str,
		choices=["group3", "features"],
		default="group3",
		help=(
			"Multi-objective formulation used by NSGA-II. "
			"'group3' groups HLO feature terms into 3 objectives (tracking/smoothness/effort); "
			"'features' uses each HLO feature-sum as its own objective (8 or 9 objectives)."
		),
	)
	parser.add_argument(
		"--nsga2_log",
		type=str,
		choices=["gen", "eval", "none"],
		default="gen",
		help=(
			"Logging for --optimizer=nsga2. "
			"'gen' prints the current best theta once per generation (best by learned scalar HLO cost); "
			"'eval' prints every evaluated candidate; "
			"'none' prints no intermediate theta."
		),
	)
	parser.add_argument("--bayes_calls", type=int, default=60, help="Number of BO evaluations when --optimizer=bayes.")
	parser.add_argument("--bayes_init_points", type=int, default=12, help="Random init points for BO when --optimizer=bayes.")
	parser.add_argument(
		"--bayes_acq_func",
		type=str,
		default="EI",
		choices=["EI", "PI", "LCB"],
		help="Acquisition function for BO when --optimizer=bayes (EI/PI/LCB).",
	)
	parser.add_argument("--maxiter", type=int, default=80, help="Maximum outer iterations (passed to scipy).")
	parser.add_argument("--workdir", type=Path, default=None, help="Working directory for eval JSON and commands.")
	parser.add_argument(
		"--result_csv",
		type=Path,
		default=None,
		help="Optional: write a one-row CSV summary (theta*, cost*, metadata).",
	)
	parser.add_argument("--no_cache", action="store_true", help="Disable caching repeated theta evaluations.")
	parser.add_argument("--quiet", action="store_true", help="Reduce verbosity.")
	parser.add_argument(
		"--print_theta_star_only",
		action="store_true",
		help="Print only the optimized theta* (comma-separated) and exit with code 0.",
	)
	parser.add_argument(
		"--planner_overwrite_in",
		type=Path,
		default=_default_planner_overwrite_in,
		help=(
			"Optional: path to a planner overwrite-parameter-settings JSON (with top-level 'parameters' list). "
			"If provided, the script updates this file in-place using the optimized theta* (atomic write). "
			"Default: none."
		),
	)
	parser.add_argument(
		"--planner_overwrite_out",
		type=Path,
		default=None,
		help=(
			"Optional: output path for the updated planner overwrite JSON. "
			"If provided, the script writes to this path instead of overwriting --planner_overwrite_in."
		),
	)
	parser.add_argument(
		"--planner_repeat",
		type=int,
		default=10,
		help="When writing planner overwrite JSON, repeat each theta value this many times (default: 10).",
	)
	parser.add_argument(
		"--update_qp_json",
		action="store_true",
		help=(
			"After optimization, update --qp_json in-place with the optimized theta* (atomic write). "
			"This modifies the file you pass to --qp_json."
		),
	)
	parser.add_argument(
		"--qp_json_out",
		type=Path,
		default=None,
		help=(
			"Optional: write an updated QP JSON with theta* to this path (atomic write). "
			"Unlike --update_qp_json, this does NOT modify the input file. "
			"Works both for dotted JSON-path theta keys and for planner overwrite JSONs."
		),
	)

	args = parser.parse_args(list(argv) if argv is not None else None)
	if args.qp_json is None:
		raise SystemExit("--qp_json is required (no default config was provided in this repository).")

	# Default lane-change *values* (7D) provided by the user.
	# Used for sensible default bounds (NOT necessarily the starting point theta0).
	lane_change_default_values_7 = [
		0.0,
		0.01524190132707108,
		0.004000080115770993,
		0.004000080115770993,
		0.004000000189989805,
		0.004111199924509229,
		0.004111199924509229,
	]

	# Default theta0 (start point). Starting from a reasonable, non-degenerate
	# set of weights improves optimization stability and avoids "all zeros" solutions.
	default_theta0_7 = list(lane_change_default_values_7)
	zeros_theta0_7 = [0.0] * 7

	def _maybe_parse_bounds(s: Optional[str]) -> Optional[List[float]]:
		if s is None:
			return None
		s = str(s).strip()
		if not s:
			return None
		return _parse_floats_csv(s)

	theta_keys = [k.strip() for k in args.theta_keys.split(",") if k.strip()]

	# By default, auto-tie the lane-change fast/slow pairs when using the 7 standard keys,
	# unless explicitly disabled.
	tie_lane_change_pairs = False
	if not args.no_tie_lane_change_pairs:
		tie_lane_change_pairs = bool(args.tie_lane_change_pairs) or (theta_keys == _LANE_CHANGE_THETA_KEYS)

	# Choose a safe default parameterization for MPC surrogate runs.
	theta_transform = args.theta_transform
	if theta_transform is None:
		theta_transform = "sigmoid" if args.planner_mpcsolver else "identity"

	# Bounds (may be needed for random init and for optimizer='bayes').
	theta_min = _maybe_parse_bounds(args.theta_min)
	theta_max = _maybe_parse_bounds(args.theta_max)
	if tie_lane_change_pairs:
		# Allow users to specify 7 bounds even though optimization runs in 5D.
		if theta_min is not None and len(theta_min) == 7:
			theta_min = _lane_change_reduce_theta7_to_theta5(theta_min).tolist()
		if theta_max is not None and len(theta_max) == 7:
			theta_max = _lane_change_reduce_theta7_to_theta5(theta_max).tolist()
	if str(theta_transform) == "sigmoid" and theta_max is None:
		if args.planner_mpcsolver:
			# Use bounds scaled from the built-in lane-change defaults so the optimizer
			# stays in a realistic regime.
			th0_ref = np.asarray(
				_lane_change_reduce_theta7_to_theta5(lane_change_default_values_7)
				if tie_lane_change_pairs
				else lane_change_default_values_7,
				dtype=float,
			).reshape(-1)
			floor = 1e-4
			theta_max = [float(max(10.0 * v, floor)) for v in th0_ref.tolist()]
		else:
			theta_max = [1.0]
	# For MPC runs with sigmoid parameterization, default theta_min to a fraction of the
	# lane-change defaults (plus a tiny floor). This discourages solutions where most
	# dimensions collapse to ~0 unless the user explicitly sets theta_min=0.
	if str(theta_transform) == "sigmoid" and theta_min is None and args.planner_mpcsolver:
		# If the user explicitly requests a zero initialization, keep 0.0 in-bounds so
		# the start point is truly theta=0 (not clipped to a small positive floor).
		if (args.theta0 is None) and (str(args.theta0_mode) == "zeros"):
			theta_min = [0.0]
		else:
			floor = 1e-6
			frac = 0.1
			th0_ref = np.asarray(
				_lane_change_reduce_theta7_to_theta5(lane_change_default_values_7)
				if tie_lane_change_pairs
				else lane_change_default_values_7,
				dtype=float,
			).reshape(-1)
			theta_min = [float(max(frac * v, floor)) for v in th0_ref.tolist()]

	# Prevent the optimizer from collapsing lane-change comfort weights to exactly 0.
	# This matters because several QP penalties are scaled by v^2 and v^4; a zero weight
	# can effectively remove jerk/snap/velocity regularization at higher speeds.
	if theta_keys == _LANE_CHANGE_THETA_KEYS:
		n = 5 if tie_lane_change_pairs else 7
		lo = _broadcast_param(theta_min, n=n, default=0.0)
		hi: Optional[np.ndarray] = None
		if theta_max is not None:
			hi = _broadcast_param(theta_max, n=n, default=1.0)

		def _scaled_floor(idx: int, *, abs_floor: float, rel_frac_of_max: float) -> float:
			"""Compute a strict-positive lower bound.

			We use an absolute minimum (requested) and, when an upper bound exists,
			a small fraction of that range so the weight stays numerically active.
			"""
			f = float(abs_floor)
			if hi is not None:
				h = float(hi[idx])
				if np.isfinite(h) and h > 0:
					f = max(f, float(rel_frac_of_max) * h)
			return float(f)
		if tie_lane_change_pairs:
			# 5D order: [accel, jerk, offset, snap, velocity]
			lo[0] = max(float(lo[0]), _scaled_floor(0, abs_floor=1e-4, rel_frac_of_max=1e-4))  # accel
			lo[1] = max(float(lo[1]), _scaled_floor(1, abs_floor=1e-5, rel_frac_of_max=1e-5))  # jerk
			lo[3] = max(float(lo[3]), _scaled_floor(3, abs_floor=1e-5, rel_frac_of_max=1e-6))  # snap
			lo[4] = max(float(lo[4]), _scaled_floor(4, abs_floor=1e-4, rel_frac_of_max=1e-4))  # velocity
		else:
			# 7D order: [accel, jerk, offset_fast, offset_slow, snap, vel_fast, vel_slow]
			lo[0] = max(float(lo[0]), _scaled_floor(0, abs_floor=1e-4, rel_frac_of_max=1e-4))  # accel
			lo[1] = max(float(lo[1]), _scaled_floor(1, abs_floor=1e-5, rel_frac_of_max=1e-5))  # jerk
			lo[4] = max(float(lo[4]), _scaled_floor(4, abs_floor=1e-5, rel_frac_of_max=1e-6))  # snap
			lo[5] = max(float(lo[5]), _scaled_floor(5, abs_floor=1e-4, rel_frac_of_max=1e-4))  # vel_fast
			lo[6] = max(float(lo[6]), _scaled_floor(6, abs_floor=1e-4, rel_frac_of_max=1e-4))  # vel_slow
		theta_min = [float(x) for x in lo.tolist()]
		if hi is not None:
			# Ensure max bound is always strictly larger than min bound.
			hi = np.maximum(hi, lo * 10.0)
			eps = 1e-9
			hi = np.maximum(hi, lo + eps)
			theta_max = [float(x) for x in hi.tolist()]

	if args.theta0 is not None:
		theta0 = _parse_floats_csv(args.theta0)
		if tie_lane_change_pairs:
			# Accept either 5D (already tied) or 7D (reduce by averaging).
			if len(theta0) == 7:
				theta0 = _lane_change_reduce_theta7_to_theta5(theta0).tolist()
			elif len(theta0) != 5:
				raise SystemExit("With tied lane-change pairs, --theta0 must have length 5 (or 7 to be reduced).")
	else:
		if args.theta0_mode == "default":
			theta0 = (
				_lane_change_reduce_theta7_to_theta5(default_theta0_7).tolist() if tie_lane_change_pairs else list(default_theta0_7)
			)
		elif args.theta0_mode == "zeros":
			theta0 = (
				_lane_change_reduce_theta7_to_theta5(zeros_theta0_7).tolist() if tie_lane_change_pairs else list(zeros_theta0_7)
			)
		else:
			# Random uniform start within bounds (if available), else fall back to [0,1].
			rng = np.random.default_rng(int(args.seed))
			n = 5 if tie_lane_change_pairs else 7
			lo = _broadcast_param(theta_min, n=n, default=0.0)
			hi = _broadcast_param(theta_max, n=n, default=1.0)
			if np.any(hi <= lo):
				raise SystemExit("Invalid bounds for random init: theta_max must be > theta_min")
			theta0 = (lo + (hi - lo) * rng.random(n)).tolist()

	# Eval-only mode should be fast: if the user passes --eval_theta (or --eval_default_theta_only),
	# we skip iterative omega recomputation via Pipeline.pipeline and instead load omega from a
	# learned_objective.json (or require explicit --hlo_omega).
	eval_only_requested = (
		(args.eval_theta is not None or bool(args.eval_default_theta_only))
		and (not bool(args.print_hlo_omega_only))
	)

	# --- HLO omega selection
	# Precedence:
	#   1) explicit --hlo_omega
	#   2) compute via Pipeline.pipeline (--hlo_omega_via_pipeline)
	#   3) load from learned_objective.json (--hlo_omega_objective_json)
	hlo_feature_scales: Optional[Sequence[float]] = None
	hlo_feature_scale_eps: float = 1e-6
	if args.hlo_omega is not None:
		hlo_omega = _parse_floats_csv(args.hlo_omega)
		if len(hlo_omega) not in (7, 8, 9, 10):
			raise SystemExit(f"--hlo_omega must have 7, 8, 9, or 10 floats, got {len(hlo_omega)}")
	elif bool(args.hlo_omega_via_pipeline) and (not eval_only_requested):
		if args.hlo_merged_csv is None:
			raise SystemExit("--hlo_omega_via_pipeline requires --hlo_merged_csv")
		hlo_omega, hlo_meta = compute_hlo_omega_via_pipeline(
			merged_csv=Path(args.hlo_merged_csv),
			n_traj=int(args.hlo_n_traj),
			seg_len=int(args.hlo_seg_len),
			lift=str(args.hlo_lift),
			degree=int(args.hlo_degree),
			dnn_psi_dim=int(args.hlo_dnn_psi_dim),
			dnn_hidden_dim=int(args.hlo_dnn_hidden_dim),
			dnn_hidden_layers=int(args.hlo_dnn_hidden_layers),
			dnn_pretrain_steps=int(args.hlo_dnn_pretrain_steps),
			dnn_pretrain_segments=int(args.hlo_dnn_pretrain_segments),
			dnn_lr=float(args.hlo_dnn_lr),
			dnn_batch_segments=int(args.hlo_dnn_batch_segments),
			dnn_refit_every=int(args.hlo_dnn_refit_every),
			return_meta=True,
		)
		try:
			hlo_feature_scales = hlo_meta.get("feature_scales", None)
			if hlo_meta.get("feature_scale_eps", None) is not None:
				hlo_feature_scale_eps = float(hlo_meta.get("feature_scale_eps"))
		except Exception:
			hlo_feature_scales = None
		if not args.quiet:
			print(f"[HLO] Computed omega via Pipeline.pipeline (lift={str(args.hlo_lift)})")
	else:
		# Always load omega from learned_objective.json (no recomputation here).
		if bool(args.hlo_omega_via_pipeline) and eval_only_requested and (not args.quiet):
			print(
				"[HLO] Eval-only mode: skipping --hlo_omega_via_pipeline recomputation; "
				"loading omega from learned_objective.json instead."
			)
		obj_path: Optional[Path] = args.hlo_omega_objective_json
		obj_meta: Optional[dict[str, Any]] = None
		if obj_path is None:
			# Try common locations (choose the newest by mtime).
			candidates: List[Path] = []
			try:
				candidates.extend(list((_REPO_ROOT / "Datasets" / "Output").rglob("learned_objective.json")))
			except Exception:
				pass
			try:
				candidates.extend(list((_REPO_ROOT / "Pipeline").rglob("learned_objective.json")))
			except Exception:
				pass
			try:
				candidates.extend(list((_REPO_ROOT / "temp_ddioc_pipeline").rglob("learned_objective.json")))
			except Exception:
				pass
			candidates = [p for p in candidates if p.exists()]
			if candidates:
				obj_path = max(candidates, key=lambda p: p.stat().st_mtime)

		obj_path = Path(obj_path)
		if not obj_path.exists():
			raise SystemExit(f"learned_objective.json not found: {obj_path}")

		try:
			obj_meta = json.loads(obj_path.read_text(encoding="utf-8"))
		except Exception:
			obj_meta = None

		hlo_omega = load_hlo_omega_from_learned_objective_json(obj_path)
		try:
			sc_meta = load_hlo_feature_scales_from_learned_objective_json(obj_path)
			if sc_meta is not None:
				hlo_feature_scales = sc_meta.get("feature_scales", None)
				hlo_feature_scale_eps = float(sc_meta.get("feature_scale_eps", hlo_feature_scale_eps))
				if (not args.quiet) and hlo_feature_scales is not None:
					print(
						f"[HLO] Using feature scales from learned_objective.json "
						f"(method={sc_meta.get('feature_scale_method')}, eps={hlo_feature_scale_eps:g})"
					)
		except Exception as e:
			if not args.quiet:
				print(f"[HLO] Warning: failed to load feature scales from learned_objective.json: {e}")

	if bool(args.print_hlo_omega_only):
		om = np.asarray(list(hlo_omega), dtype=float).reshape(-1)
		if om.size not in (8, 9) or (not np.isfinite(om).all()):
			raise SystemExit(f"Invalid omega: {om}")
		print(",".join([str(float(x)) for x in om.tolist()]))
		return 0

	if args.eval_theta is not None and bool(args.eval_default_theta_only):
		raise SystemExit("Choose only one: --eval_theta or --eval_default_theta_only")

	eval_only_theta: Optional[Sequence[float]] = None
	if args.eval_theta is not None:
		eval_only_theta = _parse_floats_csv(str(args.eval_theta))
	elif bool(args.eval_default_theta_only):
		eval_only_theta = lane_change_default_values_7

	
	planner_cmd = None if args.skip_planner else args.planner_cmd
	planner_mpcsolver = False if args.skip_planner else bool(args.planner_mpcsolver)
	if planner_cmd is not None and planner_mpcsolver:
		raise SystemExit("Choose only one planner: --planner_cmd OR --planner_mpcsolver.")
	if planner_mpcsolver:
		if args.mpc_reference_csv is None and args.mpc_reference_merged_csv is None:
			raise SystemExit("--planner_mpcsolver requires --mpc_reference_csv or --mpc_reference_merged_csv")
		if args.mpc_reference_csv is not None and args.mpc_reference_merged_csv is not None:
			raise SystemExit("Choose only one: --mpc_reference_csv OR --mpc_reference_merged_csv")

	result = bilevel_optimize(
		base_qp_json=args.qp_json,
		theta0=theta0,
		baseline_theta=lane_change_default_values_7,
		eval_only_theta=eval_only_theta,
		theta_keys=theta_keys,
		qp_repeat=int(args.qp_repeat),
		planner_cmd=planner_cmd,
		planner_mpcsolver=planner_mpcsolver,
		mpc_opt_timesteps=int(args.mpc_opt_timesteps),
		mpc_dt_s=float(args.mpc_dt_s),
		mpc_reference_csv=(Path(args.mpc_reference_csv) if args.mpc_reference_csv is not None else None),
		mpc_reference_merged_csv=(Path(args.mpc_reference_merged_csv) if args.mpc_reference_merged_csv is not None else None),
		mpc_lc_ids=(_parse_ints_csv(args.mpc_lc_ids) if args.mpc_lc_ids is not None else None),
		mpc_batch_size=int(args.mpc_batch_size),
		mpc_aggregate=str(args.mpc_aggregate),
		mpc_solver=str(args.mpc_solver),
		traj_csv_path=args.traj_csv,
		cost_regex=args.cost_regex,
		hlo_omega=hlo_omega,
		hlo_feature_scales=hlo_feature_scales,
		hlo_feature_scale_eps=hlo_feature_scale_eps,
		hlo_target_lat_offset_m=args.hlo_target_lat_offset_m,
		hlo_target_lat_offset_mode=str(args.hlo_target_lat_offset_mode),
		hlo_target_lat_offset_candidates_rel=_parse_floats_csv(str(args.hlo_target_lat_offset_candidates_rel)),
		hlo_target_lat_offset_window=int(args.hlo_target_lat_offset_window),
		hlo_softness_m=float(args.hlo_softness_m),
		hlo_threshold_m=float(args.hlo_threshold_m),
		min_indicator_to_boundary_touch_s=float(args.min_indicator_to_boundary_touch_s),
		boundary_touch_fraction=float(args.boundary_touch_fraction),
		timing_constraint_penalty_weight=float(args.timing_constraint_penalty_weight),
		timing_constraint_missing_penalty=float(args.timing_constraint_missing_penalty),
		require_reference_curvature=True,
		workdir=args.workdir,
		tie_lane_change_pairs=bool(tie_lane_change_pairs),
		optimizer=str(args.optimizer),
		maxiter=args.maxiter,
		method=args.method,
		theta_transform=str(theta_transform),
		theta_min=theta_min,
		theta_max=theta_max,
		theta_reg=float(args.theta_reg),
		bayes_calls=int(args.bayes_calls),
		bayes_init_points=int(args.bayes_init_points),
		bayes_acq_func=str(args.bayes_acq_func),
		nsga2_pop_size=int(args.nsga2_pop_size),
		mo_objectives=str(args.mo_objectives),
		nsga2_log=str(args.nsga2_log),
		cache=not args.no_cache,
		verbose=not args.quiet,
	)

	# Eval-only mode: do not perform any optimization/post-processing.
	if bool(args.eval_default_theta_only):
		return 0

	if args.print_theta_star_only:
		print(",".join([str(float(x)) for x in result.theta_star.reshape(-1).tolist()]))
		return 0

	print("\n=== RESULT ===")
	print(f"theta*: {result.theta_star.tolist()}")
	print(f"cost*:  {result.cost_star:.6g}")
	print(f"evals:  {result.n_evals}")

	# Optional: update the passed qp_json in-place with theta*.
	if args.update_qp_json:
		qp_path = Path(args.qp_json)
		qp_data = load_json(qp_path)
		if not isinstance(qp_data, dict):
			raise SystemExit("--update_qp_json requires the JSON to be an object/dict at the top level")
		apply_theta_to_qp_json(
			qp_data,
			theta_keys=theta_keys,
			theta=result.theta_star,
			qp_repeat=int(args.qp_repeat),
		)
		save_json_atomic(qp_path, qp_data)
		print(f"updated: {qp_path}")

	# Optional: write an updated copy of qp_json with theta*.
	if args.qp_json_out is not None:
		qp_in = Path(args.qp_json)
		qp_out = Path(args.qp_json_out)
		qp_data = load_json(qp_in)
		if not isinstance(qp_data, dict):
			raise SystemExit("--qp_json_out requires the JSON to be an object/dict at the top level")
		apply_theta_to_qp_json(
			qp_data,
			theta_keys=theta_keys,
			theta=result.theta_star,
			qp_repeat=int(args.qp_repeat),
		)
		save_json_atomic(qp_out, qp_data)
		print(f"qp_out:  {qp_out}")

	# Optional: write planner overwrite-parameter-settings JSON using theta*
	if args.planner_overwrite_in is not None:
		planner_in = Path(args.planner_overwrite_in)
		planner_out = planner_in if args.planner_overwrite_out is None else Path(args.planner_overwrite_out)
		write_planner_overwrite_parameters_from_theta(
			input_json_path=planner_in,
			output_json_path=planner_out,
			theta=result.theta_star,
			n_repeat=int(args.planner_repeat),
		)
		print(f"planner overwrite: wrote updated parameters to {planner_out}")

	# Optional: write summary CSV
	if args.result_csv is not None:
		# Note: hlo_omega may be None when using external HLO.
		planner_out_path = None
		if args.planner_overwrite_in is not None:
			planner_out_path = planner_out
		write_result_csv(
			path=Path(args.result_csv),
			result=result,
			theta_keys=theta_keys,
			qp_json=Path(args.qp_json),
			hlo_omega=hlo_omega,
			method=str(args.method),
			maxiter=int(args.maxiter),
			planner_overwrite_out=planner_out_path,
		)
		print(f"saved:  {args.result_csv}")

	# Save history for later inspection.
	if args.workdir is not None:
		hist_path = Path(args.workdir) / "bilevel_history.csv"
		rows = []
		for i, (th, c) in enumerate(result.history):
			row = {"iter": i, "cost": float(c)}
			for j, v in enumerate(th.tolist()):
				row[f"theta_{j}"] = float(v)
			rows.append(row)
		pd.DataFrame(rows).to_csv(hist_path, index=False)
		print(f"saved:  {hist_path}")

	return 0


if __name__ == "__main__":
	raise SystemExit(main())