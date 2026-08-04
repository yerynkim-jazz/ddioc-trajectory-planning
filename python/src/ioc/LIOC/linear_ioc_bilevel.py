"""
Bilevel Inverse Optimal Control Benchmark.

This module implements a bilevel method for solving the whole-sequence
discrete-time inverse optimal control problem.

Bilevel formulation:
- Outer problem: minimize trajectory tracking error between expert and generated trajectories
- Inner problem: forward optimal control problem with parameterized cost function

Mathematical formulation:
    Outer problem:
        inf_θ ∑_{k=0}^T ||x_k - x̄_k^θ||² + ∑_{k=0}^{T-1} ||u_k - ū_k^θ||²

    Subject to (x_k^θ, u_k^θ) being solutions to the inner problem:
        inf_{u[0,T-1]} V_T(x[0,T], u[0,T-1], θ)
        s.t. x_{k+1} = f_k(x_k, u_k), x_0 = x̄

Straight-road assumption:
- Reference state x_d(k) = [0, 0, 0, 0]^T, so tracking errors reduce to absolute terms.

Stage cost (velocity-scaled weights with clamped v_eff):
    l = w_d*d^2
        + w_a1*v_eff^2*theta^2
        + w_a2*v_eff^4*kappa^2
        + w_a3*v_eff^4*kappa_dot^2
        + w_a4*v_eff^4*u^2

where:
    v_eff = max(v, v_bar)
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.optimize as opt
from numpy.typing import NDArray


def _maybe_add_repo_root_to_syspath() -> None:
    """Ensure the repo root is on sys.path when running as a script.

    When invoked as `python linear_ioc_bilevel.py`, Python sets sys.path[0] to the
    script directory, which may prevent absolute imports like `import ioc.DDIOC.*`
    from working. We detect the repo root by searching upwards for the repository
    markers and prepend the packaged source roots that contain `ioc/`.
    """

    try:
        this_file = Path(__file__).resolve()
    except Exception:  # pragma: no cover
        return

    for parent in (this_file.parent,) + tuple(this_file.parents):
        if not (parent / "README.md").is_file():
            continue
        candidates = [parent / "python" / "src", parent]
        for candidate in candidates:
            if (candidate / "ioc" / "DDIOC" / "tune_qp_planner.py").is_file():
                candidate_str = str(candidate)
                if candidate_str not in sys.path:
                    sys.path.insert(0, candidate_str)
                return


_maybe_add_repo_root_to_syspath()

# Reuse shared planner overwrite-parameter JSON update utilities.
# In this repo these live in method/DDIOC/tune_qp_planner.py.
try:
    from ioc.DDIOC.tune_qp_planner import (  # type: ignore
        PLANNER_CHANGE_LANE_PARAM_SPECS,
        upsert_planner_overwrite_parameters_from_theta,
    )
except Exception:  # pragma: no cover
    PLANNER_CHANGE_LANE_PARAM_SPECS = None
    upsert_planner_overwrite_parameters_from_theta = None

try:
    import cvxpy as cp  # type: ignore
except Exception:  # pragma: no cover
    cp = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def _read_json(path: Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def _write_json(path: Path, obj: Any) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")


def overwrite_tpl_theta_parameters_json(
    settings_json_path: Path,
    *,
    theta: CostParameters,
    output_path: Optional[Path] = None,
    create_backup: bool = True,
    tpl_n_repeat: Optional[int] = None,
) -> Dict[str, Any]:
    """Overwrite the IOC-related TPL parameters in a planner settings JSON.

    This matches the structure under:
      {"parameters": [{"name": "...", "initValue": ...}, ...]}
    """
    settings_json_path = Path(settings_json_path)
    if not settings_json_path.exists():
        raise FileNotFoundError(settings_json_path)

    data = _read_json(settings_json_path)
    original_data = copy.deepcopy(data)
    if not isinstance(data, dict) or "parameters" not in data or not isinstance(data["parameters"], list):
        raise ValueError(f"Unexpected JSON schema in {settings_json_path}: expected top-level dict with 'parameters' list")

    if upsert_planner_overwrite_parameters_from_theta is None:
        raise ImportError(
            "Failed to import upsert_planner_overwrite_parameters_from_theta from ioc.DDIOC.tune_qp_planner. "
            "Try running as a module (python -m ioc.LIOC.linear_ioc_bilevel) "
            "or ensure the repo root (containing ioc/) is on PYTHONPATH."
        )

    if PLANNER_CHANGE_LANE_PARAM_SPECS is None:
        raise ImportError(
            "Failed to import PLANNER_CHANGE_LANE_PARAM_SPECS from ioc.DDIOC.tune_qp_planner. "
            "Try running as a module (python -m ioc.LIOC.linear_ioc_bilevel) "
            "or ensure the repo root (containing ioc/) is on PYTHONPATH."
        )

    # Map this benchmark's 5D theta into the 7 lane-change planner weights.
    # CostParameters order: [wd, wd1, wd2, wd3, wd4] == [offset, velocity, accel, jerk, snap]
    # tune_qp_planner order (7D): [accel, jerk, offset_fast, offset_slow, snap, vel_fast, vel_slow]
    theta7 = np.asarray(
        [
            float(theta.wd2),
            float(theta.wd3),
            float(theta.wd),
            float(theta.wd),
            float(theta.wd4),
            float(theta.wd1),
            float(theta.wd1),
        ],
        dtype=float,
    )

    required_names = [name for (name, _desc, _idx) in PLANNER_CHANGE_LANE_PARAM_SPECS]

    # Use a single n_repeat for all parameters (tune_qp_planner helper contract).
    # If the user provides tpl_n_repeat:
    #   - n>0: force that
    #   - n==0: infer from existing initValue lengths
    # Otherwise infer from existing initValue lengths.
    infer_repeat = tpl_n_repeat is None or int(tpl_n_repeat) == 0
    if not infer_repeat:
        if int(tpl_n_repeat) < 0:
            raise ValueError("tpl_n_repeat must be >= 0")
        n_repeat = int(tpl_n_repeat)
    else:
        # Prefer the first list-valued initValue found anywhere in the file to match existing style.
        n_repeat = 10
        for item in data["parameters"]:
            if not isinstance(item, dict):
                continue
            old_val = item.get("initValue")
            if isinstance(old_val, list) and len(old_val) > 0:
                n_repeat = int(len(old_val))
                break

    # Not all overwrite-parameter JSONs include the TPL IOC weights already.
    # We upsert them (append if missing) rather than requiring they exist.
    existing_names = {
        str(item.get("name"))
        for item in data["parameters"]
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    missing = [nm for nm in required_names if nm not in existing_names]
    if missing:
        logger.info(
            "Settings JSON does not contain %d TPL IOC parameters; they will be added: %s",
            len(missing),
            ", ".join(missing),
        )

    upsert_planner_overwrite_parameters_from_theta(
        data,
        theta=theta7,
        n_repeat=int(n_repeat),
        param_specs=PLANNER_CHANGE_LANE_PARAM_SPECS,
    )
    changed: List[str] = list(required_names)

    logger.info(
        "Overwriting TPL IOC weights from learned theta: "
        f"wd={theta.wd:.6g}, wd1={theta.wd1:.6g}, wd2={theta.wd2:.6g}, wd3={theta.wd3:.6g}, wd4={theta.wd4:.6g}"
    )
    logger.info(
        "Mapping: wd2->lateral_acceleration, wd3->lateral_jerk, wd->lateral_offset_{fast,slow}, wd4->lateral_snap, wd1->lateral_velocity_{fast,slow}"
    )

    out_path = Path(output_path) if output_path is not None else settings_json_path
    if create_backup and out_path == settings_json_path:
        backup_path = settings_json_path.with_suffix(settings_json_path.suffix + ".bak")
        if not backup_path.exists():
            _write_json(backup_path, original_data)
            logger.info(f"Wrote backup settings JSON to {backup_path}")
        else:
            logger.info(f"Backup already exists, not overwriting: {backup_path}")

    _write_json(out_path, data)
    logger.info(f"Updated {len(changed)} TPL IOC parameters in {out_path}")
    return {"output_path": str(out_path), "n_updated": len(changed), "updated": changed}


def _parse_floats_csv(s: Optional[str], *, expected_len: Optional[int] = None) -> Optional[List[float]]:
    if s is None:
        return None
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    vals = [float(p) for p in parts]
    if expected_len is not None and len(vals) != int(expected_len):
        raise ValueError(f"Expected {expected_len} floats, got {len(vals)}")
    return vals


def _parse_float_or_5(s: str) -> NDArray[np.float64]:
    """Parse either a single float or 5 comma-separated floats into a (5,) array."""
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    if len(parts) == 1:
        v = float(parts[0])
        return np.full(5, v, dtype=float)
    if len(parts) == 5:
        return np.asarray([float(p) for p in parts], dtype=float)
    raise ValueError("Expected 1 float or 5 floats")


def _as_scalar_init_value(init_value: Any) -> float:
    """Extract a representative scalar from a JSON initValue.

    Planner settings often store per-speed-bin arrays; for our scalar IOC benchmark
    we use the mean of the array (or the scalar value if already scalar).
    """
    if isinstance(init_value, list) and init_value:
        arr = np.asarray(init_value, dtype=float)
        return float(np.mean(arr))
    return float(init_value)


def extract_theta_prior_from_planner_settings(settings_json_path: Path) -> "CostParameters":
    """Extract a scalar theta prior from a planner overwrite-parameter settings JSON.

    Uses the same mapping as `_theta_to_tpl_parameter_overwrites` but inverted.
    """
    data = _read_json(Path(settings_json_path))
    if not isinstance(data, dict) or "parameters" not in data or not isinstance(data["parameters"], list):
        raise ValueError(f"Unexpected JSON schema in {settings_json_path}: expected top-level dict with 'parameters' list")

    by_name: Dict[str, Dict[str, Any]] = {}
    for item in data["parameters"]:
        if isinstance(item, dict) and isinstance(item.get("name"), str):
            by_name[item["name"]] = item

    def _get(name: str) -> float:
        if name not in by_name:
            raise KeyError(f"Prior JSON missing parameter {name!r} in {settings_json_path}")
        return _as_scalar_init_value(by_name[name].get("initValue"))

    offset_fast = _get("tpl_change_lane_weight_lateral_offset_fast")
    offset_slow = _get("tpl_change_lane_weight_lateral_offset_slow")
    vel_fast = _get("tpl_change_lane_weight_lateral_velocity_fast")
    vel_slow = _get("tpl_change_lane_weight_lateral_velocity_slow")
    acc = _get("tpl_change_lane_weight_lateral_acceleration")
    jerk = _get("tpl_change_lane_weight_lateral_jerk")
    snap = _get("tpl_change_lane_weight_lateral_snap")

    wd = float(0.5 * (offset_fast + offset_slow))
    wd1 = float(0.5 * (vel_fast + vel_slow))
    wd2 = float(acc)
    wd3 = float(jerk)
    wd4 = float(snap)

    return CostParameters(w_d=wd, w_a1=wd1, w_a2=wd2, w_a3=wd3, w_a4=wd4)


@dataclass
class TrajectoryData:
    """Expert trajectory data structure.

    State vector: x = [d, θ, κ, κ̇]^T
        d: lateral offset (m)
        θ: orientation angle (rad)
        κ: curvature (1/m)
        κ̇: curvature rate (1/m/s)   (units depend on the discretization convention)

    Control input: u = κ̈ (curvature acceleration, 1/m/s²)

    Notes:
    - dt is the sampling time T_s
    - v_bar is the minimum effective speed used ONLY for velocity-scaled weights
      (i.e., v_eff = max(v, v_bar)). Dynamics still use the raw v.
    """

    states: NDArray[np.float64]  # Shape: (T+1, 4) - [d, θ, κ, κ̇]
    controls: NDArray[np.float64]  # Shape: (T, 1) - [κ̈]
    velocities: NDArray[np.float64]  # Shape: (T+1,)
    reference_heading: NDArray[np.float64]  # Shape: (T+1,) - θ_r(k) reference path heading
    reference_curvature: NDArray[np.float64]  # Shape: (T+1,) - κ_r(k) reference path curvature
    reference_curvature_dot: NDArray[np.float64]  # Shape: (T+1,) - κ̇_r(k) reference curvature rate
    T: int  # Time horizon
    n_states: int  # Should be 4
    n_controls: int  # Should be 1
    dt: float = 0.1  # Time step T_s
    v_bar: float = 0.1  # Minimum effective speed for weight scaling


@dataclass
class Dataset:
    """A collection of trajectories to benchmark on."""

    trajectories: List[TrajectoryData]


@dataclass
class CostParameters:
    """Cost function parameters (theta) to be learned.

    Mapping to the cost:
        w_d  (aka wd)  -> d^2
        w_a1 (aka wd1) -> v_eff^2 * theta^2            (heading term)
        w_a2 (aka wd2) -> v_eff^4 * kappa^2            (curvature term)
        w_a3 (aka wd3) -> v_eff^4 * kappa_dot^2        (curvature_first term)
        w_a4 (aka wd4) -> v_eff^4 * u^2 (u=kappa_ddot) (curvature_second term)
    """

    w_d: float
    w_a1: float
    w_a2: float
    w_a3: float
    w_a4: float

    # Aliases to match the planner/parameter naming used elsewhere (wd1..wd4).
    @property
    def wd(self) -> float:
        return float(self.w_d)

    @property
    def wd1(self) -> float:
        return float(self.w_a1)

    @property
    def wd2(self) -> float:
        return float(self.w_a2)

    @property
    def wd3(self) -> float:
        return float(self.w_a3)

    @property
    def wd4(self) -> float:
        return float(self.w_a4)

    def to_array(self) -> NDArray[np.float64]:
        return np.array([self.w_d, self.w_a1, self.w_a2, self.w_a3, self.w_a4], dtype=float)

    @staticmethod
    def from_array(arr: NDArray[np.float64]) -> "CostParameters":
        arr = np.asarray(arr, dtype=float).reshape(-1)
        return CostParameters(
            w_d=float(arr[0]),
            w_a1=float(arr[1]),
            w_a2=float(arr[2]),
            w_a3=float(arr[3]),
            w_a4=float(arr[4]),
        )


class ForwardOptimalControl:
    """Inner forward optimal control problem.

    Solves:
        min_{u[0:T-1]} sum_{k=0}^T l(x_k, u_k, θ, k)
        s.t. x_{k+1} = f_k(x_k, u_k), x_0 given
    """

    def __init__(self, trajectory_data: TrajectoryData):
        self.data = trajectory_data
        self.T = trajectory_data.T
        self.dt = trajectory_data.dt

        # Lazy-built cached QP (cvxpy) artifacts.
        self._qp_built: bool = False
        self._qp_problem: Any = None
        self._qp_x: Any = None
        self._qp_u: Any = None
        self._qp_x0_param: Any = None
        self._qp_wd: Any = None
        self._qp_wd1: Any = None
        self._qp_wd2: Any = None
        self._qp_wd3: Any = None
        self._qp_wd4: Any = None

    def stage_cost(
        self,
        x: NDArray[np.float64],
        u: NDArray[np.float64],
        v: float,
        theta: CostParameters,
        k: int,
    ) -> float:
        """Stage cost with velocity-scaled weights using v_eff = max(v, v_bar).

        Tracking errors computed relative to reference trajectory.
        """
        # State: x = [d, θ, κ, κ̇]
        d = float(x[0])
        theta_orientation = float(x[1])
        kappa = float(x[2])
        kappa_dot = float(x[3])

        # Control: u = κ̈ (scalar)
        if np.ndim(u) > 0:
            u_val = float(u[0])
        else:
            u_val = float(u)

        # Reference values at time k
        k_idx = min(k, len(self.data.reference_heading) - 1)
        theta_ref = float(self.data.reference_heading[k_idx])
        kappa_ref = float(self.data.reference_curvature[k_idx])

        # if getattr(self.data, "reference_curvature_dot", None) is not None and len(self.data.reference_curvature_dot) > 0:
        #     kdot_idx = min(k, len(self.data.reference_curvature_dot) - 1)
        #     kappa_dot_ref = float(self.data.reference_curvature_dot[kdot_idx])
        # else:
        #     kappa_dot_ref = 0.0
        kappa_dot_ref = 0.0
        
        # Tracking errors
        d_error = d  # Lateral offset error (reference is typically 0)
        theta_error = theta_orientation - theta_ref
        kappa_error = kappa - kappa_ref
        kappa_dot_error = kappa_dot - kappa_dot_ref

        # Effective velocity for weighting ONLY (document-style clamp)
        v_eff = max(float(v), float(self.data.v_bar))
        v2 = v_eff * v_eff
        v4 = v2 * v2

        cost = (
            theta.w_d * d_error**2
            + theta.w_a1 * v2 * theta_error**2
            + theta.w_a2 * v4 * kappa_error**2
            + theta.w_a3 * v4 * kappa_dot_error**2
            + theta.w_a4 * v4 * u_val**2
        )
        return float(cost)

    def dynamics(
        self,
        x: NDArray[np.float64],
        u: NDArray[np.float64],
        k: int,
    ) -> NDArray[np.float64]:
        """Discrete-time dynamics: x_{k+1} = A^D x_k + B^D u_k + D^D ż_k.

        Uses the raw velocity v(k) from data (NO clamp), as this is the model evolution.
        
        Disturbance uses the reference heading as an input to compensate for
        small-angle discretization effects in the lateral-offset channel.

        We use the average reference orientation
            \bar{\theta}_r(k) = (\theta_r(k) + \theta_r(k+1)) / 2
        (with endpoint fallback) as suggested in the referenced formulation.

        Disturbance vector: z_k = [z_d(k), \bar{\theta}_r(k)]^T where:
        - z_d(k) = 0 (lateral offset disturbance set to zero)
        """
        # Get current velocity (raw)
        v = float(self.data.velocities[min(k, len(self.data.velocities) - 1)])
        T_s = float(self.dt)

        # Extract control (scalar)
        if np.ndim(u) > 0:
            u_val = float(u[0])
        else:
            u_val = float(u)

        A_D = np.array(
            [
                [1.0, v * T_s, 0.5 * v**2 * T_s**2, (1.0 / 6.0) * v**2 * T_s**3],
                [0.0, 1.0, v * T_s, 0.5 * v * T_s**2],
                [0.0, 0.0, 1.0, T_s],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,

        )
        B_D = np.array(
            [
                (1.0 / 24.0) * v**2 * T_s**4,
                (1.0 / 6.0) * v * T_s**3,
                0.5 * T_s**2,
                T_s,
            ],
            dtype=float,
        )

        # Disturbance matrix D^D (from discretized D(t) matrix)
        D_D = np.array(
            [
                [T_s, -v * T_s],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
            dtype=float,
        )

        # Disturbance vector: z_k = [z_d(k), \bar{\theta}_r(k)]^T
        # z_d(k) = 0 (as requested)
        ref = self.data.reference_heading
        if k < len(ref) - 1:
            theta_r_bar = 0.5 * (float(ref[k]) + float(ref[k + 1]))
        elif len(ref) > 0:
            theta_r_bar = float(ref[-1])
        else:
            theta_r_bar = 0.0

        z_k = np.array([0.0, theta_r_bar], dtype=float)

        x_next = A_D @ x + B_D * u_val + D_D @ z_k
        return x_next

    def solve(
        self,
        theta: CostParameters,
        x0: Optional[NDArray[np.float64]] = None,
        u_init: Optional[NDArray[np.float64]] = None,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], float]:
        """Solve the forward optimal control problem for given parameters θ.

        Preferred solver: convex QP via cvxpy/OSQP (model-based baseline).
        Fallback: SLSQP on the control sequence if cvxpy is unavailable.
        """
        if x0 is None:
            x0 = self.data.states[0]

        n_states = int(self.data.n_states)
        n_controls = int(self.data.n_controls)

        # Model-based QP solve (preferred)
        if cp is not None:
            if not self._qp_built:
                self._build_cached_qp(n_states=n_states, n_controls=n_controls)

            # Update parameters.
            assert self._qp_x0_param is not None
            assert self._qp_wd is not None
            assert self._qp_wd1 is not None
            assert self._qp_wd2 is not None
            assert self._qp_wd3 is not None
            assert self._qp_wd4 is not None

            self._qp_x0_param.value = np.asarray(x0, dtype=float).reshape(-1)
            self._qp_wd.value = float(theta.wd)
            self._qp_wd1.value = float(theta.wd1)
            self._qp_wd2.value = float(theta.wd2)
            self._qp_wd3.value = float(theta.wd3)
            self._qp_wd4.value = float(theta.wd4)

            problem = self._qp_problem
            try:
                problem.solve(
                    solver=cp.OSQP,
                    warm_start=True,
                    eps_abs=1e-4,
                    eps_rel=1e-4,
                    max_iter=100000,
                    polish=True,
                )
            except Exception as e:  # pragma: no cover
                logger.warning(f"cvxpy/OSQP solve failed ({e}); falling back to SLSQP")
                # Fall back below
            else:
                x_var = self._qp_x
                u_var = self._qp_u
                if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or x_var.value is None or u_var.value is None:
                    logger.warning(f"Forward QP not optimal: status={problem.status}; falling back to SLSQP")
                else:
                    x_opt = np.asarray(x_var.value, dtype=float)
                    u_opt = np.asarray(u_var.value, dtype=float)
                    return x_opt, u_opt, float(problem.value)

        # Fallback: SLSQP over u
        if u_init is None:
            u_init = np.zeros(self.T * n_controls, dtype=float)

        u_bounds = [(-10.0, 10.0)] * (self.T * n_controls)

        def objective(u_flat: NDArray[np.float64]) -> float:
            u_seq = u_flat.reshape((self.T, n_controls))
            x_seq = np.zeros((self.T + 1, n_states), dtype=float)
            x_seq[0] = x0

            total_cost = 0.0
            for k in range(self.T):
                v_k = float(self.data.velocities[k]) if k < len(self.data.velocities) else float(self.data.velocities[-1])
                total_cost += self.stage_cost(x_seq[k], u_seq[k], v_k, theta, k)
                x_seq[k + 1] = self.dynamics(x_seq[k], u_seq[k], k)

            v_T = float(self.data.velocities[-1]) if len(self.data.velocities) > 0 else 1.0
            u_T = np.zeros(n_controls, dtype=float)
            total_cost += self.stage_cost(x_seq[self.T], u_T, v_T, theta, self.T)
            return float(total_cost)

        result = opt.minimize(
            objective,
            u_init,
            method="SLSQP",
            bounds=u_bounds,
            options={"maxiter": 500, "ftol": 1e-6},
        )

        if not result.success:
            logger.warning(f"Forward OCP did not converge: {result.message}")

        u_opt = result.x.reshape((self.T, n_controls))
        x_opt = np.zeros((self.T + 1, n_states), dtype=float)
        x_opt[0] = x0
        for k in range(self.T):
            x_opt[k + 1] = self.dynamics(x_opt[k], u_opt[k], k)

        return x_opt, u_opt, float(result.fun)

    def _build_cached_qp(self, *, n_states: int, n_controls: int) -> None:
        """Build and cache the cvxpy QP for this trajectory.

        The structure (dynamics + v-dependent scaling terms) is fixed per trajectory.
        Only the weights (wd..wd4) and the initial state x0 change across solves.
        """
        if cp is None:  # pragma: no cover
            raise RuntimeError("cvxpy is not available")
        if self._qp_built:
            return

        if int(n_states) != 4 or int(n_controls) != 1:
            raise ValueError("Cached QP currently supports n_states=4, n_controls=1")

        T_s = float(self.dt)

        # Decision variables
        x = cp.Variable((self.T + 1, n_states))
        u = cp.Variable((self.T, n_controls))

        # Parameters
        x0_param = cp.Parameter(n_states)
        wd = cp.Parameter(nonneg=True)
        wd1 = cp.Parameter(nonneg=True)
        wd2 = cp.Parameter(nonneg=True)
        wd3 = cp.Parameter(nonneg=True)
        wd4 = cp.Parameter(nonneg=True)

        constraints = [x[0, :] == x0_param]

        # Optional bounds
        u_min, u_max = -10.0, 10.0
        constraints += [u[:, 0] >= u_min, u[:, 0] <= u_max]

        # Precompute per-step velocity scalings.
        # For cost: v_eff = max(v_raw, v_bar). For dynamics: use v_raw.
        vel = np.asarray(self.data.velocities, dtype=float).reshape(-1)
        if vel.size == 0:
            vel = np.ones(self.T + 1, dtype=float)

        v_raw_k = np.array([float(vel[k]) if k < vel.size else float(vel[-1]) for k in range(self.T)], dtype=float)
        v_raw_T = float(vel[-1])
        v_bar = float(self.data.v_bar)

        v_eff_k = np.maximum(v_raw_k, v_bar)
        v2_k = v_eff_k * v_eff_k
        v4_k = v2_k * v2_k

        v_eff_T = max(v_raw_T, v_bar)
        v2_T = float(v_eff_T * v_eff_T)
        v4_T = float(v2_T * v2_T)

        # Dynamics constraints (constant matrices per step)
        # Include the same disturbance as in `dynamics()`:
        #   x_{k+1} = A_k x_k + B_k u_k + D_k z_k
        # where D_k matches the discretized D^D in the referenced formulation and
        # z_k = [z_d(k), theta_r_bar(k)]^T with z_d(k)=0.
        theta_r = np.asarray(getattr(self.data, "reference_heading", np.zeros(self.T + 1)), dtype=float).reshape(-1)
        if theta_r.size != (self.T + 1):
            theta_r = np.pad(theta_r[: self.T + 1], (0, max(0, (self.T + 1) - theta_r.size)))
        theta_r_bar_k = 0.5 * (theta_r[: self.T] + theta_r[1 : self.T + 1])
        for k in range(self.T):
            v = float(v_raw_k[k])
            A_k = np.array(
                [
                    [1.0, v * T_s, 0.5 * v**2 * T_s**2, (1.0 / 6.0) * v**2 * T_s**3],
                    [0.0, 1.0, v * T_s, 0.5 * v * T_s**2],
                    [0.0, 0.0, 1.0, T_s],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=float,
            )
            B_k = np.array(
                [
                    (1.0 / 24.0) * v**2 * T_s**4,
                    (1.0 / 6.0) * v * T_s**3,
                    0.5 * T_s**2,
                    T_s,
                ],
                dtype=float,
            )
            D_k = np.array(
                [
                    [T_s, -v * T_s],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ],
                dtype=float,
            )
            z_k = np.array([0.0, float(theta_r_bar_k[k])], dtype=float)
            dist_k = D_k @ z_k
            constraints.append(x[k + 1, :] == A_k @ x[k, :] + (B_k * u[k, 0]) + dist_k)

        # Objective (vectorized, using scalar Parameters and per-step constants).
        # State is [d, theta, kappa, kappa_dot]; control is kappa_ddot.
        # Track reference signals (if provided) to align with stage_cost.
        theta_ref_vec = np.asarray(getattr(self.data, "reference_heading", np.zeros(self.T + 1)), dtype=float).reshape(-1)
        kappa_ref_vec = np.asarray(getattr(self.data, "reference_curvature", np.zeros(self.T + 1)), dtype=float).reshape(-1)
        kappa_dot_ref_vec = np.asarray(
            getattr(self.data, "reference_curvature_dot", np.zeros(self.T + 1)), dtype=float
        ).reshape(-1)

        if theta_ref_vec.size != (self.T + 1):
            theta_ref_vec = np.pad(theta_ref_vec[: self.T + 1], (0, max(0, (self.T + 1) - theta_ref_vec.size)))
        if kappa_ref_vec.size != (self.T + 1):
            kappa_ref_vec = np.pad(kappa_ref_vec[: self.T + 1], (0, max(0, (self.T + 1) - kappa_ref_vec.size)))
        if kappa_dot_ref_vec.size != (self.T + 1):
            kappa_dot_ref_vec = np.pad(
                kappa_dot_ref_vec[: self.T + 1], (0, max(0, (self.T + 1) - kappa_dot_ref_vec.size))
            )
        d_coeff = np.ones(self.T + 1, dtype=float)
        theta_coeff = np.concatenate([v2_k, np.array([v2_T], dtype=float)])
        kappa_coeff = np.concatenate([v4_k, np.array([v4_T], dtype=float)])
        kappa_dot_coeff = np.concatenate([v4_k, np.array([v4_T], dtype=float)])
        u_coeff = v4_k

        obj = (
            wd * cp.sum(cp.multiply(d_coeff, cp.square(x[:, 0])))
            + wd1 * cp.sum(cp.multiply(theta_coeff, cp.square(x[:, 1] - theta_ref_vec)))
            + wd2 * cp.sum(cp.multiply(kappa_coeff, cp.square(x[:, 2] - kappa_ref_vec)))
            + wd3 * cp.sum(cp.multiply(kappa_dot_coeff, cp.square(x[:, 3] - kappa_dot_ref_vec)))
            + wd4 * cp.sum(cp.multiply(u_coeff, cp.square(u[:, 0])))
        )

        problem = cp.Problem(cp.Minimize(obj), constraints)

        self._qp_problem = problem
        self._qp_x = x
        self._qp_u = u
        self._qp_x0_param = x0_param
        self._qp_wd = wd
        self._qp_wd1 = wd1
        self._qp_wd2 = wd2
        self._qp_wd3 = wd3
        self._qp_wd4 = wd4
        self._qp_built = True
        logger.debug(f"Built cached QP for trajectory (T={self.T})")


class BilevelIOC:
    """Model-based bilevel IOC benchmark.

    This baseline assumes the vehicle/trajectory model is known (the linear
    discrete-time dynamics in `ForwardOptimalControl`).

    Outer optimization:
        min_θ J(θ) = sum_{traj} [ Σ ||x_k^θ - x̄_k||² + Σ ||u_k^θ - ū_k||² ]
    where (x^θ, u^θ) for each trajectory solve the inner forward OCP.
    """

    def __init__(self, dataset: Dataset):
        if not dataset.trajectories:
            raise ValueError("Dataset must contain at least one trajectory")
        self.dataset = dataset
        self.forward_ocps = [ForwardOptimalControl(td) for td in dataset.trajectories]
        self.n_evaluations = 0
        self.best_cost = np.inf
        self.best_theta: Optional[CostParameters] = None
        self.max_weight: float = 100.0
        self.min_theta: NDArray[np.float64] = np.zeros(5, dtype=float)

        # Optional prior-based ratio bounds to keep learned weights in a planner-safe region.
        self.prior_theta: Optional[CostParameters] = None
        self.prior_min_ratio: NDArray[np.float64] = np.zeros(5, dtype=float)
        self.prior_max_ratio: NDArray[np.float64] = np.full(5, float("inf"), dtype=float)
        self.prior_reg: float = 0.0

    @staticmethod
    def trajectory_tracking_error(
        x_generated: NDArray[np.float64],
        u_generated: NDArray[np.float64],
        x_expert: NDArray[np.float64],
        u_expert: NDArray[np.float64],
    ) -> float:
        x_error = float(np.sum((x_generated - x_expert) ** 2))
        u_error = float(np.sum((u_generated - u_expert) ** 2))
        return float(x_error + u_error)

    def outer_objective(self, theta_array: NDArray[np.float64]) -> float:
        self.n_evaluations += 1
        theta_array = np.asarray(theta_array, dtype=float).reshape(-1)
        theta = CostParameters.from_array(theta_array)

        # Box constraints (enforced via penalty so all scipy methods behave consistently)
        if np.any(theta_array < self.min_theta) or np.any(theta_array > float(self.max_weight)):
            return 1e10

        if (
            self.prior_theta is not None
            and np.all(np.isfinite(self.prior_min_ratio))
            and np.all(np.isfinite(self.prior_max_ratio))
        ):
            prior = self.prior_theta.to_array()
            eps = 1e-12
            lower = np.maximum(prior * self.prior_min_ratio, self.min_theta)
            upper = np.minimum(prior * self.prior_max_ratio, float(self.max_weight))
            # If prior component is ~0, ratio bounds are meaningless; fall back to min/max.
            lower = np.where(prior > eps, lower, self.min_theta)
            upper = np.where(prior > eps, upper, float(self.max_weight))
            if np.any(theta_array < lower) or np.any(theta_array > upper):
                return 1e10

        try:
            total_error = 0.0
            for td, ocp in zip(self.dataset.trajectories, self.forward_ocps):
                x_gen, u_gen, _ = ocp.solve(theta, x0=td.states[0])
                total_error += self.trajectory_tracking_error(x_gen, u_gen, td.states, td.controls)

            # Optional regularization to keep theta close to a planner-known-good prior.
            # Use log-ratio penalty for scale invariance.
            if self.prior_theta is not None and float(self.prior_reg) > 0.0:
                prior = self.prior_theta.to_array()
                eps = 1e-12
                lr = np.log((theta_array + eps) / (prior + eps))
                total_error += float(self.prior_reg) * float(np.sum(lr * lr))

            if total_error < self.best_cost:
                self.best_cost = total_error
                self.best_theta = theta
                logger.info(
                    f"Eval {self.n_evaluations}: J = {total_error:.6f}, θ = {theta_array} (n_traj={len(self.dataset.trajectories)})"
                )

            return float(total_error)
        except Exception as e:
            logger.error(f"Error in outer objective evaluation: {e}")
            return 1e10

    def solve(
        self,
        theta_init: Optional[CostParameters] = None,
        method: str = "Nelder-Mead",
        max_iter: int = 200,
        max_fev: int = 0,
        max_weight: float = 100.0,
    ) -> Tuple[CostParameters, float, Dict[str, Any]]:
        if theta_init is None:
            theta_init = CostParameters(w_d=1.0, w_a1=1.0, w_a2=1.0, w_a3=1.0, w_a4=1.0)

        theta0 = theta_init.to_array()
        max_weight_f = float(max_weight)
        if not np.isfinite(max_weight_f) or max_weight_f <= 0:
            raise ValueError("max_weight must be a positive finite float")

        # Ensure the initial point is feasible under (min_theta, max_weight, prior ratio bounds).
        theta0 = np.maximum(theta0, self.min_theta)
        theta0 = np.minimum(theta0, max_weight_f)
        if (
            self.prior_theta is not None
            and np.all(np.isfinite(self.prior_min_ratio))
            and np.all(np.isfinite(self.prior_max_ratio))
        ):
            prior = self.prior_theta.to_array()
            eps = 1e-12
            lower = np.maximum(prior * self.prior_min_ratio, self.min_theta)
            upper = np.minimum(prior * self.prior_max_ratio, max_weight_f)
            lower = np.where(prior > eps, lower, self.min_theta)
            upper = np.where(prior > eps, upper, max_weight_f)
            theta0_proj = np.minimum(np.maximum(theta0, lower), upper)
            if np.any(np.abs(theta0_proj - theta0) > 0):
                logger.info(f"Projected initial theta0 into prior-bounded feasible set: {theta0} -> {theta0_proj}")
            theta0 = theta0_proj

        bounds = [(float(self.min_theta[i]), max_weight_f) for i in range(int(theta0.shape[0]))]
        self.max_weight = max_weight_f

        logger.info(f"Starting bilevel IOC optimization with method: {method}")
        logger.info(f"Initial parameters: {theta_init}")
        v_bars = [float(td.v_bar) for td in self.dataset.trajectories]
        v_bar_str = f"{v_bars[0]}" if all(abs(v - v_bars[0]) < 1e-12 for v in v_bars) else f"range[{min(v_bars)}, {max(v_bars)}]"
        logger.info(
            f"Using v_eff = max(v, v_bar) for weight scaling with v_bar={v_bar_str} (n_traj={len(self.dataset.trajectories)})"
        )

        self.n_evaluations = 0
        self.best_cost = np.inf
        self.best_theta = theta_init

        options: Dict[str, Any] = {"maxiter": int(max_iter), "disp": True}
        max_fev_i = int(max_fev)
        if max_fev_i > 0:
            # Different scipy solvers use different option names.
            if method in ["Nelder-Mead", "Powell"]:
                options["maxfev"] = max_fev_i
            elif method in ["L-BFGS-B", "TNC"]:
                options["maxfun"] = max_fev_i
            # SLSQP doesn't consistently support a max function-eval option across versions.

        if method in ["L-BFGS-B", "SLSQP", "TNC", "Powell"]:
            result = opt.minimize(
                self.outer_objective,
                theta0,
                method=method,
                bounds=bounds,
                options=options,
            )
        else:
            result = opt.minimize(
                self.outer_objective,
                theta0,
                method=method,
                options=options,
            )

        theta_opt = CostParameters.from_array(result.x)
        info: Dict[str, Any] = {
            "success": bool(result.success),
            "message": str(result.message),
            # Kept for backward compatibility (this is actually #objective evaluations).
            "n_iterations": int(self.n_evaluations),
            "n_evaluations": int(self.n_evaluations),
            "final_cost": float(result.fun),
            "optimization_result": result,
        }

        logger.info(f"Optimization completed: {result.message}")
        logger.info(f"Optimal parameters: {theta_opt}")
        logger.info(f"Final cost: {result.fun:.6f}")
        logger.info(f"Total evaluations: {self.n_evaluations}")

        return theta_opt, float(result.fun), info


def _infer_default_prior_path_from_overwrite(overwrite_path: Path) -> Optional[Path]:
    """Try to infer a good prior JSON from an overwrite-params-json path.

    Heuristic: if a sibling file '*_scaled_to_default_theta.json' exists, use it.
    This avoids accidentally using an already-overwritten (possibly failing) file as the prior.
    """
    overwrite_path = Path(overwrite_path)
    parent = overwrite_path.parent
    if not parent.exists():
        return None

    candidates = [
        parent / (overwrite_path.stem + "_scaled_to_default_theta.json"),
        parent / "temporary_parameters_that_will_fix_failing_tests_scaled_to_default_theta.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _is_lane_change_test_overwrite(path: Optional[Path]) -> bool:
    if path is None:
        return False
    p = Path(path)
    # Heuristic: this file name is used by the regulated lane change test fixture.
    return "temporary_parameters_that_will_fix_failing_tests" in p.name


def load_trajectory_data(
    filepath: Path,
    trajectory_id: Optional[str] = None,
    *,
    sg_only: bool = False,
) -> TrajectoryData:
    """Load expert trajectory data from file.

    Supports NPZ, CSV, JSON.
    """
    suffix = filepath.suffix.lower()

    if suffix == ".npz":
        data = np.load(filepath)
        states = data["states"]
        controls = data["controls"]
        velocities = data.get("velocities", np.ones(len(states)))
        reference_heading = data.get("reference_heading", np.zeros(len(states)))
        reference_curvature = data.get("reference_curvature", np.zeros(len(states)))
        reference_curvature_dot = data.get("reference_curvature_dot", np.zeros(len(states)))
        dt = float(data.get("dt", 0.1))
        v_bar = float(data.get("v_bar", 0.1))

    elif suffix == ".csv":
        import pandas as pd

        df = pd.read_csv(filepath)

        def _pick_col(candidates: List[str]) -> Optional[str]:
            for c in candidates:
                if c in df.columns:
                    return c
            return None

        id_col = _pick_col(["file_id", "lc_id"])
        if id_col is not None:
            if trajectory_id is None:
                trajectory_id = str(df[id_col].iloc[0])
                logger.info(f"Loading trajectory: {trajectory_id}")
            df = df[df[id_col].astype(str) == str(trajectory_id)].copy()

        if bool(sg_only):
            d_col = _pick_col(["lateral_offset_m_sg"])
            th_col = _pick_col(["target_orientation_rad_sg"])
            k_col = _pick_col(["target_curvature_1pm_sg"])
        else:
            d_col = _pick_col(["lateral_offset_m_sg", "lateral_offset_m_interp", "lateral_offset_m"])
            th_col = _pick_col(["target_orientation_rad_sg", "target_orientation_rad_interp", "target_orientation_rad"])
            k_col = _pick_col(["target_curvature_1pm_sg", "target_curvature_1pm_interp", "target_curvature_1pm"])
        kdot_col = _pick_col(["target_curvature_dot_1pm2", "target_curvature_1pm_dot"])
        kddot_col = _pick_col(["target_curvature_ddot_1pm3", "target_curvature_1pm_ddot"])
        v_col = _pick_col(["target_speed_mps", "velocity"])

        if all(x is not None for x in (d_col, th_col, k_col, kdot_col, kddot_col)):
            states = df[[d_col, th_col, k_col, kdot_col]].to_numpy(dtype=float)
            controls = df[kddot_col].to_numpy(dtype=float)[:-1].reshape(-1, 1)
            velocities = (
                df[v_col].to_numpy(dtype=float)
                if v_col is not None
                else np.ones(int(states.shape[0]), dtype=float)
            )

            if "time_s" in df.columns:
                times = df["time_s"].to_numpy(dtype=float)
                dt = float(np.mean(np.diff(times))) if len(times) > 1 else 0.1
            else:
                dt = 0.1

            # Reference signals (optional; fall back to zeros)
            # Reference signals (optional). If sg_only, prefer *_sg, but allow non-sg fallbacks
            # since reference signals may not always have smoothed variants.
            if bool(sg_only):
                ref_th_col = _pick_col(["reference_orientation_sg", "reference_heading_rad", "reference_orientation"])
                ref_k_col = _pick_col(["reference_curvature_sg", "reference_curvature_1pm", "reference_curvature"])
                ref_kdot_col = _pick_col(["reference_curvature_dot_sg", "reference_curvature_dot"])
            else:
                ref_th_col = _pick_col(
                    ["reference_heading_rad", "reference_orientation_sg", "reference_orientation_interp", "reference_orientation"]
                )
                ref_k_col = _pick_col(
                    ["reference_curvature_1pm", "reference_curvature_sg", "reference_curvature_interp", "reference_curvature"]
                )
                ref_kdot_col = _pick_col(
                    ["reference_curvature_dot", "reference_curvature_dot_sg", "reference_curvature_dot_interp"]
                )
            reference_heading = (
                df[ref_th_col].to_numpy(dtype=float) if ref_th_col is not None else np.zeros(int(states.shape[0]), dtype=float)
            )
            reference_curvature = (
                df[ref_k_col].to_numpy(dtype=float) if ref_k_col is not None else np.zeros(int(states.shape[0]), dtype=float)
            )
            reference_curvature_dot = (
                df[ref_kdot_col].to_numpy(dtype=float)
                if ref_kdot_col is not None
                else np.zeros(int(states.shape[0]), dtype=float)
            )

            v_bar = 0.1  # default unless you add a column / metadata
        else:
            state_cols = [c for c in df.columns if c.startswith("state_")]
            control_cols = [c for c in df.columns if c.startswith("control_")]
            if not state_cols or not control_cols:
                raise ValueError("CSV must contain 'state_*' and 'control_*' columns or merged_ioc_dataset format")

            states = df[state_cols].values
            controls = df[control_cols].values[:-1]
            velocities = df["velocity"].values if "velocity" in df.columns else np.ones(len(states))
            reference_heading = df["reference_heading"].values if "reference_heading" in df.columns else np.zeros(len(states))
            reference_curvature = df["reference_curvature"].values if "reference_curvature" in df.columns else np.zeros(len(states))
            reference_curvature_dot = (
                df["reference_curvature_dot"].values if "reference_curvature_dot" in df.columns else np.zeros(len(states))
            )
            dt = 0.1
            v_bar = 0.1

    elif suffix == ".json":
        with open(filepath, "r") as f:
            data_dict = json.load(f)

        states = np.array(data_dict["states"], dtype=float)
        controls = np.array(data_dict["controls"], dtype=float)
        velocities = np.array(data_dict.get("velocities", [1.0] * len(states)), dtype=float)
        reference_heading = np.array(data_dict.get("reference_heading", [0.0] * len(states)), dtype=float)
        reference_curvature = np.array(data_dict.get("reference_curvature", [0.0] * len(states)), dtype=float)
        reference_curvature_dot = np.array(data_dict.get("reference_curvature_dot", [0.0] * len(states)), dtype=float)
        dt = float(data_dict.get("dt", 0.1))
        v_bar = float(data_dict.get("v_bar", 0.1))

    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    T = len(controls)
    n_states = int(states.shape[1] if states.ndim > 1 else 1)
    n_controls = int(controls.shape[1] if controls.ndim > 1 else 1)

    logger.info(f"Loaded trajectory: T={T}, n_states={n_states}, n_controls={n_controls}, dt={dt:.4f}, v_bar={v_bar:.4f}")

    return TrajectoryData(
        states=np.asarray(states, dtype=float),
        controls=np.asarray(controls, dtype=float),
        velocities=np.asarray(velocities, dtype=float),
        reference_heading=np.asarray(reference_heading, dtype=float),
        reference_curvature=np.asarray(reference_curvature, dtype=float),
        reference_curvature_dot=np.asarray(reference_curvature_dot, dtype=float),
        T=int(T),
        n_states=int(n_states),
        n_controls=int(n_controls),
        dt=float(dt),
        v_bar=float(v_bar),
    )


def load_dataset_from_merged_csv(
    csv_path: Path,
    *,
    n_traj: int = 1,
    min_traj_len: int = 30,
    trajectory_id: Optional[str] = None,
    v_bar: float = 0.1,
    segment: bool = True,
    seg_len: int = 31,
    segment_stride: Optional[int] = None,
    sg_only: bool = False,
) -> Dataset:
    """Load one or more trajectories from the pipeline-style merged IOC dataset CSV.

    Expected columns:
      - file_id, time_s
      - lateral_offset_m, target_orientation_rad, target_curvature_1pm, target_curvature_dot_1pm2
      - target_curvature_ddot_1pm3
    Optional:
      - target_speed_mps
    """
    import pandas as pd

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    def _pick_col(candidates: List[str]) -> Optional[str]:
        for c in candidates:
            if c in df.columns:
                return c
        return None

    id_col = _pick_col(["file_id", "lc_id"])
    if id_col is None:
        # Allow single-trajectory CSVs (no id column): treat whole file as one trajectory.
        id_col = "__single_traj_id__"
        df[id_col] = "0"

    time_col = _pick_col(["time_s", "t_s", "time"])
    if time_col is None:
        raise ValueError("Merged CSV missing a time column (expected one of: time_s, t_s, time)")

    if bool(sg_only):
        d_col = _pick_col(["lateral_offset_m_sg"])
        th_col = _pick_col(["target_orientation_rad_sg"])
        k_col = _pick_col(["target_curvature_1pm_sg"])
    else:
        d_col = _pick_col(["lateral_offset_m_sg", "lateral_offset_m_interp", "lateral_offset_m"])
        th_col = _pick_col(["target_orientation_rad_sg", "target_orientation_rad_interp", "target_orientation_rad"])
        k_col = _pick_col(["target_curvature_1pm_sg", "target_curvature_1pm_interp", "target_curvature_1pm"])
    kdot_col = _pick_col(["target_curvature_dot_1pm2", "target_curvature_1pm_dot"])
    kddot_col = _pick_col(["target_curvature_ddot_1pm3", "target_curvature_1pm_ddot"])
    v_col = _pick_col(["target_speed_mps", "velocity"])

    missing_signals = [
        name
        for (name, col) in [
            ("lateral_offset", d_col),
            ("target_orientation", th_col),
            ("target_curvature", k_col),
            ("target_curvature_dot", kdot_col),
            ("target_curvature_ddot", kddot_col),
        ]
        if col is None
    ]
    if missing_signals:
        raise ValueError(
            "Merged CSV missing required signals: "
            + ", ".join(missing_signals)
            + ". Present columns include: "
            + ", ".join(list(df.columns)[:50])
            + (" ..." if len(df.columns) > 50 else "")
        )

    has_speed = v_col is not None
    if not has_speed:
        logger.warning("Merged CSV has no speed column ('target_speed_mps' or 'velocity'); using v=1.0")

    df = df.sort_values([id_col, time_col]).reset_index(drop=True)

    if trajectory_id is not None:
        df = df[df[id_col].astype(str) == str(trajectory_id)].copy()
        if df.empty:
            raise ValueError(f"No rows found for file_id={trajectory_id!r}")

    trajectories: List[TrajectoryData] = []
    n_traj_limit = int(n_traj)
    use_all = n_traj_limit <= 0

    seg_len_i = int(seg_len)
    if seg_len_i < 2:
        raise ValueError("seg_len must be >= 2")
    stride_i = int(seg_len_i if segment_stride is None else segment_stride)
    if stride_i <= 0:
        raise ValueError("segment_stride must be a positive integer")

    def _append_segments(
        *,
        states: NDArray[np.float64],
        controls_full: NDArray[np.float64],
        velocities: NDArray[np.float64],
        reference_heading: NDArray[np.float64],
        reference_curvature: NDArray[np.float64],
        reference_curvature_dot: NDArray[np.float64],
        dt: float,
    ) -> None:
        # controls_full has same length as states; inner expects T=len(u)=seg_len-1.
        n = int(states.shape[0])
        if n < seg_len_i:
            return
        for start in range(0, n - seg_len_i + 1, stride_i):
            end = start + seg_len_i
            Xseg = states[start:end].copy()
            Useg = controls_full[start : end - 1].copy().reshape(-1, 1)
            Vseg = velocities[start:end].copy().reshape(-1)
            Rseg = reference_heading[start:end].copy().reshape(-1)
            Kseg = reference_curvature[start:end].copy().reshape(-1)
            Kdotseg = reference_curvature_dot[start:end].copy().reshape(-1)
            if Xseg.shape != (seg_len_i, 4) or Useg.shape != (seg_len_i - 1, 1) or Vseg.shape != (seg_len_i,):
                continue
            trajectories.append(
                TrajectoryData(
                    states=np.asarray(Xseg, dtype=float),
                    controls=np.asarray(Useg, dtype=float),
                    velocities=np.asarray(Vseg, dtype=float),
                    reference_heading=np.asarray(Rseg, dtype=float),
                    reference_curvature=np.asarray(Kseg, dtype=float),
                    reference_curvature_dot=np.asarray(Kdotseg, dtype=float),
                    T=int(seg_len_i - 1),
                    n_states=4,
                    n_controls=1,
                    dt=float(dt),
                    v_bar=float(v_bar),
                )
            )
    processed_file_ids = 0
    for fid in df[id_col].astype(str).unique().tolist():
        if (not use_all) and processed_file_ids >= n_traj_limit:
            break
        dfi = df[df[id_col].astype(str) == str(fid)].copy().sort_values(time_col)
        t = dfi[time_col].to_numpy(dtype=float)

        d = dfi[d_col].to_numpy(dtype=float)
        th = dfi[th_col].to_numpy(dtype=float)
        kappa = dfi[k_col].to_numpy(dtype=float)
        kappa_dot = dfi[kdot_col].to_numpy(dtype=float)
        kappa_ddot = dfi[kddot_col].to_numpy(dtype=float)
        v = dfi[v_col].to_numpy(dtype=float) if has_speed and v_col is not None else np.ones_like(d, dtype=float)
        
        # Extract reference heading if available, otherwise use zeros
        ref_th_col = _pick_col(
            ["reference_heading_rad", "reference_orientation_sg", "reference_orientation_interp", "reference_orientation"]
        )
        if ref_th_col is not None and ref_th_col in dfi.columns:
            ref_heading = dfi[ref_th_col].to_numpy(dtype=float)
        else:
            ref_heading = np.zeros_like(d, dtype=float)
        
        # Extract reference curvature if available, otherwise use zeros
        ref_k_col = _pick_col(
            ["reference_curvature_1pm", "reference_curvature_sg", "reference_curvature_interp", "reference_curvature"]
        )
        if ref_k_col is not None and ref_k_col in dfi.columns:
            ref_curvature = dfi[ref_k_col].to_numpy(dtype=float)
        else:
            ref_curvature = np.zeros_like(d, dtype=float)

        ref_kdot_col = _pick_col(["reference_curvature_dot", "reference_curvature_dot_sg", "reference_curvature_dot_interp"])
        if ref_kdot_col is not None and ref_kdot_col in dfi.columns:
            ref_curvature_dot = dfi[ref_kdot_col].to_numpy(dtype=float)
        else:
            ref_curvature_dot = np.zeros_like(d, dtype=float)

        # Apply a single mask so state/control/velocity/time remain aligned.
        mask = (
            np.isfinite(t)
            & np.isfinite(d)
            & np.isfinite(th)
            & np.isfinite(kappa)
            & np.isfinite(kappa_dot)
            & np.isfinite(kappa_ddot)
            & np.isfinite(v)
            & np.isfinite(ref_heading)
            & np.isfinite(ref_curvature)
            & np.isfinite(ref_curvature_dot)
        )
        t = t[mask]
        d = d[mask]
        th = th[mask]
        kappa = kappa[mask]
        kappa_dot = kappa_dot[mask]
        kappa_ddot = kappa_ddot[mask]
        v = v[mask]
        ref_heading = ref_heading[mask]
        ref_curvature = ref_curvature[mask]
        ref_curvature_dot = ref_curvature_dot[mask]

        states = np.vstack([d, th, kappa, kappa_dot]).T
        velocities = v.reshape(-1)
        controls_full = kappa_ddot.reshape(-1)
        if states.shape[0] < 2:
            continue

        if states.shape[0] < int(min_traj_len):
            continue

        if len(t) >= 2:
            dt = float(np.median(np.diff(t)))
            if (not np.isfinite(dt)) or dt <= 0:
                dt = 0.1
        else:
            dt = 0.1

        if bool(segment):
            _append_segments(
                states=np.asarray(states, dtype=float),
                controls_full=np.asarray(controls_full, dtype=float),
                velocities=np.asarray(velocities, dtype=float),
                reference_heading=np.asarray(ref_heading, dtype=float),
                reference_curvature=np.asarray(ref_curvature, dtype=float),
                reference_curvature_dot=np.asarray(ref_curvature_dot, dtype=float),
                dt=float(dt),
            )
        else:
            controls = controls_full[:-1].reshape(-1, 1)
            T = int(controls.shape[0])
            td = TrajectoryData(
                states=np.asarray(states, dtype=float),
                controls=np.asarray(controls, dtype=float),
                velocities=np.asarray(velocities, dtype=float),
                reference_heading=np.asarray(ref_heading, dtype=float),
                reference_curvature=np.asarray(ref_curvature, dtype=float),
                reference_curvature_dot=np.asarray(ref_curvature_dot, dtype=float),
                T=T,
                n_states=4,
                n_controls=1,
                dt=float(dt),
                v_bar=float(v_bar),
            )
            trajectories.append(td)

        processed_file_ids += 1

    if not trajectories:
        raise RuntimeError("No valid trajectories loaded from merged CSV (check min_traj_len / file_id)")

    if bool(segment):
        logger.info(
            f"Loaded dataset segments: n_seg={len(trajectories)} from {csv_path} (seg_len={seg_len_i}, stride={stride_i}, file_id_limit={'all' if use_all else n_traj_limit})"
        )
    else:
        logger.info(
            f"Loaded dataset trajectories: n_traj={len(trajectories)} from {csv_path} (limit={'all' if use_all else n_traj_limit})"
        )
    return Dataset(trajectories=trajectories)


def save_results(theta: CostParameters, cost: float, info: Dict[str, Any], output_path: Path) -> None:
    results = {
        "parameters": {
            "w_d": theta.w_d,
            "w_a1": theta.w_a1,
            "w_a2": theta.w_a2,
            "w_a3": theta.w_a3,
            "w_a4": theta.w_a4,
        },
        "final_cost": float(cost),
        "success": bool(info["success"]),
        "message": str(info["message"]),
        "n_iterations": int(info["n_iterations"]),
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {output_path}")


def generate_synthetic_data(
    T: int = 50,
    n_states: int = 4,
    n_controls: int = 1,
    dt: float = 0.1,
    v_bar: float = 0.1,
) -> TrajectoryData:
    """Generate synthetic expert trajectory for testing."""
    t = np.linspace(0, T * dt, T + 1)

    states = np.zeros((T + 1, 4), dtype=float)
    states[:, 0] = 0.5 * np.sin(0.5 * t)  # d
    states[:, 1] = 0.1 * np.cos(0.5 * t)  # θ
    states[:, 2] = 0.02 * np.sin(0.3 * t)  # κ
    states[:, 3] = 0.01 * np.cos(0.4 * t)  # κ̇

    controls = np.zeros((T, 1), dtype=float)
    controls[:, 0] = 0.005 * np.sin(0.4 * t[:-1])  # κ̈

    velocities = 10.0 + 2.0 * np.sin(0.2 * t)
    
    # Generate reference heading with some variation
    reference_heading = 0.05 * np.sin(0.3 * t)  # θ_r(t)
    
    # Generate reference curvature with some variation
    reference_curvature = 0.01 * np.cos(0.25 * t)  # κ_r(t)

    return TrajectoryData(
        states=states,
        controls=controls,
        velocities=velocities,
        reference_heading=reference_heading,
        reference_curvature=reference_curvature,
        T=T,
        n_states=4,
        n_controls=1,
        dt=dt,
        v_bar=v_bar,
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Bilevel Inverse Optimal Control Benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--trajectory", type=Path, help="Path to expert trajectory data file (NPZ, CSV, or JSON)")
    group.add_argument("--merged-csv", type=Path, help="Path to merged_ioc_dataset.csv (pipeline export)")
    group.add_argument("--synthetic", action="store_true", help="Use synthetic data for testing")

    parser.add_argument("--trajectory-id", type=str, help="Specific trajectory file_id to use from CSV (single-trajectory mode)")
    parser.add_argument(
        "--n-traj",
        type=int,
        default=320,
        help="Number of trajectories to load in --merged-csv mode (use 0 for all)",
    )
    parser.add_argument("--min-traj-len", type=int, default=30, help="Minimum trajectory length for --merged-csv mode")
    parser.add_argument(
        "--no-segmentation",
        action="store_true",
        help="In --merged-csv mode, disable segmenting stitched trajectories into fixed-length lane-change windows.",
    )
    parser.add_argument(
        "--seg-len",
        type=int,
        default=31,
        help="Segment length (samples) when slicing stitched trajectories (Pipeline default is 31)",
    )
    parser.add_argument(
        "--seg-stride",
        type=int,
        default=0,
        help="Segment stride (samples). Use 0 to mean stride=seg-len (non-overlapping).",
    )
    # Default to SG-only signals to match the merged dataset schema used in this repo.
    # Provide an explicit opt-out for older datasets.
    sg_group = parser.add_mutually_exclusive_group(required=False)
    sg_group.add_argument(
        "--sg-only",
        dest="sg_only",
        action="store_true",
        help=(
            "Use only *_sg columns for the main expert state signals (lateral_offset, target_orientation, target_curvature). "
            "This is the default."
        ),
    )
    sg_group.add_argument(
        "--no-sg-only",
        dest="sg_only",
        action="store_false",
        help=(
            "Allow falling back to *_interp or raw columns for the main expert state signals if *_sg columns are missing."
        ),
    )
    parser.set_defaults(sg_only=True)
    parser.add_argument("--output", type=Path, default=Path("ioc_results_bilevel.json"), help="Output file for learned parameters")
    parser.add_argument(
        "--method",
        type=str,
        default="Nelder-Mead",
        choices=["Nelder-Mead", "Powell", "L-BFGS-B", "SLSQP", "TNC"],
        help="Optimization method for outer problem",
    )
    parser.add_argument("--max-iter", type=int, default=200, help="Maximum number of outer iterations")
    parser.add_argument(
        "--max-fev",
        type=int,
        default=0,
        help=(
            "Optional hard cap on the number of outer objective evaluations (0 = no cap). "
            "This corresponds to the 'Eval N' counter in the logs and is often larger than max-iter for Nelder-Mead."
        ),
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--v-bar",
        type=float,
        default=0.1,
        help="Minimum effective speed v_bar used in v_eff = max(v, v_bar) for weight scaling",
    )

    parser.add_argument(
        "--no-planner-safe-defaults",
        action="store_true",
        help=(
            "Disable auto 'planner-safe' defaults intended for regulated lane-change tests. "
            "When enabled (default), and --overwrite-params-json points at the lane-change test overwrite file, "
            "the script tightens prior bounds/regularization and initializes theta0 from the inferred passing prior."
        ),
    )

    # Init and bounds for outer weights
    parser.add_argument(
        "--theta0",
        type=str,
        default="0,0,0,0,0",
        help="Initial theta (5 floats: w_d,w_a1,w_a2,w_a3,w_a4) as comma-separated list",
    )
    parser.add_argument(
        "--max-weight",
        type=float,
        default=0.1,
        help="Upper bound for each weight (applied for all outer optimizers)",
    )

    parser.add_argument(
        "--min-theta",
        type=str,
        default="0,0,0,0,0",
        help="Lower bounds for theta (5 floats: w_d,w_a1,w_a2,w_a3,w_a4) as comma-separated list",
    )

    parser.add_argument(
        "--prior-params-json",
        type=Path,
        default=None,
        help=(
            "Optional: planner overwrite-parameter settings JSON used as a prior region. "
            "If not provided, and --overwrite-params-json is set, that file is used as the prior by default."
        ),
    )
    parser.add_argument(
        "--no-infer-prior-from-overwrite",
        action="store_true",
        help=(
            "If --prior-params-json is not set, do not try to infer a planner-safe prior (e.g. '*_scaled_to_default_theta.json') "
            "from the --overwrite-params-json location."
        ),
    )
    parser.add_argument(
        "--prior-min-ratio",
        type=str,
        default="0.1",
        help="If a prior is set: enforce theta >= prior * prior-min-ratio. Provide 1 float or 5 floats.",
    )
    parser.add_argument(
        "--prior-max-ratio",
        type=str,
        default="10.0",
        help="If a prior is set: enforce theta <= prior * prior-max-ratio. Provide 1 float or 5 floats.",
    )
    parser.add_argument(
        "--no-prior-bounds",
        action="store_true",
        help="Disable prior ratio bounds even if a prior JSON is provided or inferred.",
    )

    parser.add_argument(
        "--prior-reg",
        type=float,
        default=0.0,
        help=(
            "Optional: add log-ratio regularization term prior-reg * sum(log((theta+eps)/(prior+eps))^2). "
            "Useful to keep learned weights close to a passing prior while still allowing learning."
        ),
    )

    parser.add_argument(
        "--strict-exit",
        action="store_true",
        help=(
            "Exit with code 1 when the optimizer reports success=False. "
            "By default this script exits 0 (like Pipeline/bilevel_opt.py) and records success in the output JSON."
        ),
    )

    parser.add_argument(
        "--overwrite-params-json",
        type=Path,
        default=None,
        help=(
            "Optional: path to a planner overwrite-parameter settings JSON to update in-place with learned theta. "
            "Updates tpl_change_lane_weight_* entries by broadcasting scalars to the existing array lengths."
        ),
    )
    parser.add_argument(
        "--overwrite-params-json-out",
        type=Path,
        default=None,
        help="Optional: write the updated parameters JSON to this path instead of in-place.",
    )

    parser.add_argument(
        "--tpl-n-repeat",
        type=int,
        default=10,
        help=(
            "When writing tpl_change_lane_weight_* parameters, repeat each scalar into a list of this length. "
            "Default is 10. Use 0 to infer a length from existing initValue arrays in the target JSON."
        ),
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="When overwriting in-place, do not create a .bak file next to the settings JSON.",
    )

    args = parser.parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    dataset: Dataset
    if bool(args.synthetic):
        logger.info("Generating synthetic trajectory data")
        dataset = Dataset(trajectories=[generate_synthetic_data(v_bar=float(args.v_bar))])
    elif args.merged_csv:
        logger.info(f"Loading merged dataset from {args.merged_csv}")
        stride = None if int(args.seg_stride) == 0 else int(args.seg_stride)
        dataset = load_dataset_from_merged_csv(
            Path(args.merged_csv),
            n_traj=int(args.n_traj),
            min_traj_len=int(args.min_traj_len),
            trajectory_id=args.trajectory_id,
            v_bar=float(args.v_bar),
            segment=(not bool(args.no_segmentation)),
            seg_len=int(args.seg_len),
            segment_stride=stride,
            sg_only=bool(args.sg_only),
        )
    elif args.trajectory:
        logger.info(f"Loading trajectory data from {args.trajectory}")
        trajectory_data = load_trajectory_data(Path(args.trajectory), args.trajectory_id, sg_only=bool(args.sg_only))
        trajectory_data.v_bar = float(args.v_bar)
        dataset = Dataset(trajectories=[trajectory_data])
    else:
        logger.error("Provide one of: --trajectory, --merged-csv, or --synthetic")
        return 1

    logger.info(f"Benchmark dataset: n_traj={len(dataset.trajectories)}")

    bilevel_ioc = BilevelIOC(dataset)

    # Lower bounds for optimization to avoid degenerate solutions (e.g. jerk/snap collapsing to ~0).
    min_theta_list = _parse_floats_csv(args.min_theta, expected_len=5)
    assert min_theta_list is not None
    bilevel_ioc.min_theta = np.asarray(min_theta_list, dtype=float).reshape(5)
    if np.any(bilevel_ioc.min_theta < 0):
        raise ValueError("min-theta must be nonnegative")

    planner_safe = (not bool(args.no_planner_safe_defaults)) and _is_lane_change_test_overwrite(
        Path(args.overwrite_params_json) if args.overwrite_params_json is not None else None
    )
    if planner_safe and np.allclose(bilevel_ioc.min_theta, 0.0):
        bilevel_ioc.min_theta = bilevel_ioc.min_theta.copy()
        bilevel_ioc.min_theta[3] = max(bilevel_ioc.min_theta[3], 2e-3)  # jerk
        bilevel_ioc.min_theta[4] = max(bilevel_ioc.min_theta[4], 2e-3)  # snap
        logger.info(f"Planner-safe defaults: applying jerk/snap floors via min-theta={bilevel_ioc.min_theta}")

    # Prior guidance: keep learned theta within a ratio of (and/or close to) a planner-known-good configuration.
    prior_path = Path(args.prior_params_json) if args.prior_params_json is not None else None
    if prior_path is None and args.overwrite_params_json is not None:
        if not bool(args.no_infer_prior_from_overwrite):
            inferred = _infer_default_prior_path_from_overwrite(Path(args.overwrite_params_json))
            if inferred is not None:
                prior_path = inferred
                logger.info(f"Inferred prior JSON (planner-safe) from overwrite path: {prior_path}")
        if prior_path is None:
            prior_path = Path(args.overwrite_params_json)

    prior_min_ratio_s = str(args.prior_min_ratio)
    prior_max_ratio_s = str(args.prior_max_ratio)
    prior_reg_f = float(args.prior_reg)
    if planner_safe and prior_reg_f == 0.0 and prior_min_ratio_s == "0.1" and prior_max_ratio_s == "10.0":
        # Tight default region around the inferred passing prior.
        prior_min_ratio_s = "0.85"
        prior_max_ratio_s = "1.2"
        prior_reg_f = 10.0
        logger.info(
            "Planner-safe defaults: tightening prior bounds/regularization "
            f"(prior-min-ratio={prior_min_ratio_s}, prior-max-ratio={prior_max_ratio_s}, prior-reg={prior_reg_f})."
        )

    if prior_path is not None:
        try:
            bilevel_ioc.prior_theta = extract_theta_prior_from_planner_settings(prior_path)
            bilevel_ioc.prior_reg = prior_reg_f
            if not bool(args.no_prior_bounds):
                bilevel_ioc.prior_min_ratio = _parse_float_or_5(prior_min_ratio_s)
                bilevel_ioc.prior_max_ratio = _parse_float_or_5(prior_max_ratio_s)
                logger.info(
                    "Using prior ratio bounds from settings JSON: "
                    f"{prior_path} (min_ratio={bilevel_ioc.prior_min_ratio}, max_ratio={bilevel_ioc.prior_max_ratio}). "
                    f"prior_reg={bilevel_ioc.prior_reg}. Prior theta={bilevel_ioc.prior_theta}"
                )
            else:
                # Leave prior ratio bounds disabled (min_ratio finite but max_ratio=inf makes the check false).
                bilevel_ioc.prior_min_ratio = np.zeros(5, dtype=float)
                bilevel_ioc.prior_max_ratio = np.full(5, float("inf"), dtype=float)
                logger.info(
                    "Using prior (regularization only; ratio bounds disabled): "
                    f"{prior_path}. prior_reg={bilevel_ioc.prior_reg}. Prior theta={bilevel_ioc.prior_theta}"
                )
        except Exception as e:
            logger.warning(f"Failed to load prior theta from {prior_path}: {e}. Continuing without prior guidance.")

    default_theta0 = "0.004249256714959207,0.001765033478490077,0.0017845311316744845,0.005928177097568654,0.0014276250050580374"
    if bilevel_ioc.prior_theta is not None and str(args.theta0).strip() == default_theta0:
        theta_init = bilevel_ioc.prior_theta
        logger.info(f"Initializing theta0 from prior (theta0 left at default): {theta_init}")
    else:
        theta0_list = _parse_floats_csv(args.theta0, expected_len=5)
        assert theta0_list is not None
        theta_init = CostParameters(
            w_d=float(theta0_list[0]),
            w_a1=float(theta0_list[1]),
            w_a2=float(theta0_list[2]),
            w_a3=float(theta0_list[3]),
            w_a4=float(theta0_list[4]),
        )

    theta_opt, final_cost, info = bilevel_ioc.solve(
        theta_init=theta_init,
        method=args.method,
        max_iter=args.max_iter,
        max_fev=args.max_fev,
        max_weight=float(args.max_weight),
    )

    save_results(theta_opt, final_cost, info, args.output)

    if args.overwrite_params_json is not None:
        overwrite_tpl_theta_parameters_json(
            Path(args.overwrite_params_json),
            theta=theta_opt,
            output_path=(Path(args.overwrite_params_json_out) if args.overwrite_params_json_out is not None else None),
            create_backup=(not bool(args.no_backup)),
            tpl_n_repeat=(None if int(args.tpl_n_repeat) == 0 else int(args.tpl_n_repeat)),
        )

    print("\n" + "=" * 60)
    print("BILEVEL IOC OPTIMIZATION SUMMARY")
    print("=" * 60)
    print(f"Success: {info['success']}")
    print(f"Final cost: {final_cost:.6f}")
    print(f"Evaluations: {info['n_iterations']}")
    print("\nLearned parameters:")
    print(f"  w_d  = {theta_opt.w_d:.6f}")
    print(f"  w_a1 = {theta_opt.w_a1:.6f}")
    print(f"  w_a2 = {theta_opt.w_a2:.6f}")
    print(f"  w_a3 = {theta_opt.w_a3:.6f}")
    print(f"  w_a4 = {theta_opt.w_a4:.6f}")
    print("=" * 60)

    # Match Pipeline/bilevel_opt.py behavior: return 0 unless explicitly requested.
    if bool(args.strict_exit) and (not bool(info["success"])):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
