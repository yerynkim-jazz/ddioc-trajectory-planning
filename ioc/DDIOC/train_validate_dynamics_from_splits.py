from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib

# Use a non-interactive backend for headless runs (safe on Windows too)
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# This file lives at: <repo_root>/ioc/DDIOC/<this_file>.py
# so repo root is 2 levels up.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


try:
    # When executed as a module: `python -m ioc.DDIOC.train_validate_dynamics_from_splits`
    from .dynamics_learning import (  # type: ignore
        koopman_regression_full,
        dnn_pretrain_dkr,
        make_dnn_lift,
        make_polynomial_lift,
    )
except Exception:  # pragma: no cover
    # When executed as a script from this folder: `python train_validate_dynamics_from_splits.py`
    from dynamics_learning import (  # noqa: E402
        koopman_regression_full,
        dnn_pretrain_dkr,
        make_dnn_lift,
        make_polynomial_lift,
    )


def _fmt_float(x: Optional[float], *, width: int = 11, prec: int = 4) -> str:
    if x is None:
        return "".rjust(width)
    try:
        xf = float(x)
    except Exception:
        return str(x).rjust(width)
    if not np.isfinite(xf):
        return str(xf).rjust(width)
    # Use scientific notation for small numbers
    return f"{xf:{width}.{prec}e}"


def _print_one_step_table(*, one_step_raw: dict, title: str) -> None:
    state_names = one_step_raw.get("state_names") or []
    train = one_step_raw.get("train") or {}
    test = one_step_raw.get("test") or {}

    train_rmse = train.get("rmse_by_state")
    train_nrmse = train.get("nrmse_by_state")
    test_rmse = test.get("rmse_by_state")
    test_nrmse = test.get("nrmse_by_state")

    state_w = 24
    col_w = 11
    print(f"\n=== One-step (RAW units): RMSE / NRMSE ({title}) ===")
    print(
        "state".ljust(state_w)
        + "tr_RMSE".rjust(col_w)
        + "tr_NRMSE".rjust(col_w)
        + "te_RMSE".rjust(col_w)
        + "te_NRMSE".rjust(col_w)
    )
    print("-" * (state_w + 4 * col_w))

    if not (train_rmse and test_rmse and train_nrmse and test_nrmse):
        print("(missing one_step_raw fields)")
        return

    for j, name in enumerate(state_names):
        tr = float(train_rmse[j])
        tn = float(train_nrmse[j])
        er = float(test_rmse[j])
        en = float(test_nrmse[j])
        name_disp = str(name)
        if len(name_disp) > state_w:
            name_disp = name_disp[: max(0, state_w - 1)] + "…"
        print(
            name_disp.ljust(state_w)
            + _fmt_float(tr, width=col_w)
            + _fmt_float(tn, width=col_w)
            + _fmt_float(er, width=col_w)
            + _fmt_float(en, width=col_w)
        )

    print("-" * (state_w + 4 * col_w))
    print(
        "RMSE_l2".ljust(state_w)
        + _fmt_float(train.get("rmse_l2"), width=col_w)
        + "".rjust(col_w)
        + _fmt_float(test.get("rmse_l2"), width=col_w)
        + "".rjust(col_w)
    )


@dataclass(frozen=True)
class ColumnSpec:
    group_col: str = "lc_source"
    time_col: str = "time_s"
    speed_col: str = "target_speed_mps"
    lat_col: str = ""
    ori_col: str = ""
    kappa_col: str = ""
    kappa_dot_col: str = ""
    u_col: str = ""


def _first_existing(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def infer_columns(df: pd.DataFrame, *, group_col: str, time_col: str) -> ColumnSpec:
    if group_col not in df.columns:
        raise ValueError(f"Missing group_col={group_col!r}")
    if time_col not in df.columns:
        raise ValueError(f"Missing time_col={time_col!r}")

    lat = _first_existing(df, ("lateral_offset_m_sg", "lateral_offset_m_interp", "lateral_offset_m"))
    ori = _first_existing(df, ("target_orientation_rad_sg", "target_orientation_rad_interp", "target_orientation_rad"))
    kappa = _first_existing(df, ("target_curvature_1pm_sg", "target_curvature_1pm_interp", "target_curvature_1pm"))
    kappa_dot = _first_existing(df, ("target_curvature_1pm_dot", "target_curvature_1pm_dot_interp"))
    u = _first_existing(df, ("target_curvature_1pm_ddot", "target_curvature_1pm_ddot_interp"))

    missing = [name for name, val in {
        "lat": lat,
        "ori": ori,
        "kappa": kappa,
        "kappa_dot": kappa_dot,
        "u": u,
    }.items() if val is None]
    if missing:
        raise ValueError(
            "Missing required columns for state/input. "
            f"Could not infer: {missing}. Available columns include: {list(df.columns)[:20]}..."
        )

    return ColumnSpec(
        group_col=group_col,
        time_col=time_col,
        speed_col="target_speed_mps" if "target_speed_mps" in df.columns else "",
        lat_col=str(lat),
        ori_col=str(ori),
        kappa_col=str(kappa),
        kappa_dot_col=str(kappa_dot),
        u_col=str(u),
    )


def load_trajectories(
    csv_path: Path,
    *,
    cols: ColumnSpec,
    min_traj_len: int,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[str], float]:
    df = pd.read_csv(csv_path)
    cols = infer_columns(df, group_col=cols.group_col, time_col=cols.time_col) if not cols.lat_col else cols

    df = df.sort_values([cols.group_col, cols.time_col]).reset_index(drop=True)
    traj_X: List[np.ndarray] = []
    traj_U: List[np.ndarray] = []
    traj_V: List[np.ndarray] = []
    traj_ids: List[str] = []
    dts: List[float] = []

    for gid, gdf in df.groupby(cols.group_col):
        g = gdf.sort_values(cols.time_col)
        t = g[cols.time_col].to_numpy(dtype=float)

        X = np.vstack(
            [
                g[cols.lat_col].to_numpy(dtype=float),
                g[cols.ori_col].to_numpy(dtype=float),
                g[cols.kappa_col].to_numpy(dtype=float),
                g[cols.kappa_dot_col].to_numpy(dtype=float),
            ]
        ).T
        U = g[cols.u_col].to_numpy(dtype=float).reshape(-1, 1)

        if cols.speed_col:
            V = g[cols.speed_col].to_numpy(dtype=float).reshape(-1)
        else:
            V = np.ones(len(g), dtype=float)

        mask = np.isfinite(X).all(axis=1) & np.isfinite(U).all(axis=1) & np.isfinite(V)
        X = X[mask]
        U = U[mask]
        V = V[mask]
        t = t[mask]

        if len(X) < int(min_traj_len):
            continue

        traj_X.append(X)
        traj_U.append(U)
        traj_V.append(V)
        traj_ids.append(str(gid))

        if len(t) >= 2:
            dt = float(np.median(np.diff(t)))
            if not np.isfinite(dt) or dt <= 0:
                dt = 0.0
        else:
            dt = 0.0
        dts.append(dt)

    valid_dt = [d for d in dts if d > 0]
    global_dt_s = float(np.median(valid_dt)) if valid_dt else 0.1

    if not traj_X:
        raise RuntimeError(f"No valid trajectories loaded from {csv_path} (min_traj_len={min_traj_len}).")

    return traj_X, traj_U, traj_V, traj_ids, global_dt_s


def build_segments(
    traj_Xn: List[np.ndarray],
    traj_Un: List[np.ndarray],
    *,
    seg_len: int,
    stride: int,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    segments_X: List[np.ndarray] = []
    segments_U: List[np.ndarray] = []
    seg_len = int(seg_len)
    stride = int(stride)
    if seg_len < 2:
        raise ValueError("seg_len must be >= 2")
    if stride <= 0:
        raise ValueError("stride must be positive")

    for Xn, Un in zip(traj_Xn, traj_Un):
        n = len(Xn)
        if n < seg_len:
            continue
        for start in range(0, n - seg_len + 1, stride):
            end = start + seg_len
            Xseg = Xn[start:end].copy()
            Useg = Un[start : end - 1].copy().reshape(-1, 1)
            if len(Xseg) != seg_len or len(Useg) != (seg_len - 1):
                continue
            segments_X.append(Xseg)
            segments_U.append(Useg)

    if not segments_X:
        raise RuntimeError("No segments built (try smaller seg_len or min_traj_len).")
    return segments_X, segments_U


def koopman_predict_next_x(
    xk: np.ndarray,
    uk: np.ndarray,
    *,
    Kx: np.ndarray,
    Ku: np.ndarray,
    C: np.ndarray,
    lift_np: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    psi = lift_np(xk).reshape(-1, 1)
    psi1 = (Kx @ psi) + (Ku @ np.asarray(uk, dtype=float).reshape(-1, 1))
    x1 = (C @ psi1).reshape(-1)
    return x1


def eval_one_step_mse(
    segments_X: List[np.ndarray],
    segments_U: List[np.ndarray],
    *,
    Kx: np.ndarray,
    Ku: np.ndarray,
    C: np.ndarray,
    lift_np: Callable[[np.ndarray], np.ndarray],
) -> dict:
    se_sum = None
    n = 0
    for Xseg, Useg in zip(segments_X, segments_U):
        for k in range(len(Xseg) - 1):
            x_true = Xseg[k + 1]
            x_hat = koopman_predict_next_x(Xseg[k], Useg[k], Kx=Kx, Ku=Ku, C=C, lift_np=lift_np)
            err = (x_hat - x_true).reshape(-1)
            if se_sum is None:
                se_sum = np.zeros_like(err, dtype=float)
            se_sum += err * err
            n += 1
    if n == 0 or se_sum is None:
        return {"n_transitions": 0, "mse_by_state": None, "mse_mean": None}
    mse = se_sum / float(n)
    return {"n_transitions": int(n), "mse_by_state": mse.tolist(), "mse_mean": float(np.mean(mse))}


def eval_one_step_rmse_nrmse_raw(
    segments_Xn: List[np.ndarray],
    segments_Un: List[np.ndarray],
    *,
    Kx: np.ndarray,
    Ku: np.ndarray,
    C: np.ndarray,
    lift_np: Callable[[np.ndarray], np.ndarray],
    X_mean: np.ndarray,
    X_std: np.ndarray,
) -> dict:
    """One-step RMSE/NRMSE computed in RAW units (denormalized)."""
    X_mean = np.asarray(X_mean, dtype=float).reshape(-1)
    X_std = np.asarray(X_std, dtype=float).reshape(-1)

    se_sum = None
    n = 0
    true_stack = []
    for Xseg, Useg in zip(segments_Xn, segments_Un):
        for k in range(len(Xseg) - 1):
            x_true_n = Xseg[k + 1]
            x_hat_n = koopman_predict_next_x(Xseg[k], Useg[k], Kx=Kx, Ku=Ku, C=C, lift_np=lift_np)
            x_true = x_true_n * X_std + X_mean
            x_hat = x_hat_n * X_std + X_mean
            err = (x_hat - x_true).reshape(-1)
            if se_sum is None:
                se_sum = np.zeros_like(err, dtype=float)
            se_sum += err * err
            n += 1
            true_stack.append(x_true)

    if n == 0 or se_sum is None:
        return {
            "n_transitions": 0,
            "rmse_by_state": None,
            "nrmse_by_state": None,
            "rmse_l2": None,
        }

    true_arr = np.asarray(true_stack, dtype=float)
    sigma = np.std(true_arr, axis=0, ddof=0)
    sigma[sigma == 0] = 1.0

    mse = se_sum / float(n)
    rmse = np.sqrt(mse)
    nrmse = rmse / sigma
    rmse_l2 = float(np.sqrt(np.mean(np.sum((rmse.reshape(1, -1)) ** 2, axis=1))))
    # Better: RMSE_{l2} over all transitions
    # compute directly from mse vector: E||e||^2 = sum_j mse_j
    rmse_l2 = float(np.sqrt(np.sum(mse)))

    return {
        "n_transitions": int(n),
        "rmse_by_state": rmse.tolist(),
        "nrmse_by_state": nrmse.tolist(),
        "rmse_l2": rmse_l2,
        "sigma_by_state": sigma.tolist(),
    }


def eval_rollout_rmse(
    segments_X: List[np.ndarray],
    segments_U: List[np.ndarray],
    *,
    Kx: np.ndarray,
    Ku: np.ndarray,
    C: np.ndarray,
    lift_np: Callable[[np.ndarray], np.ndarray],
    rollout_h: int,
    max_segments: int,
) -> dict:
    rollout_h = int(rollout_h)
    max_segments = int(max_segments)
    se_sum = None
    count = 0
    used = 0

    for Xseg, Useg in zip(segments_X, segments_U):
        if max_segments and used >= max_segments:
            break
        x = Xseg[0].copy()
        H = min(rollout_h, len(Useg), len(Xseg) - 1)
        for k in range(H):
            x = koopman_predict_next_x(x, Useg[k], Kx=Kx, Ku=Ku, C=C, lift_np=lift_np)
            x_true = Xseg[k + 1]
            err = (x - x_true).reshape(-1)
            if se_sum is None:
                se_sum = np.zeros_like(err, dtype=float)
            se_sum += err * err
            count += 1
        used += 1

    if count == 0 or se_sum is None:
        return {"n_steps": 0, "rmse_by_state": None, "rmse_mean": None}
    mse = se_sum / float(count)
    rmse = np.sqrt(mse)
    return {"n_steps": int(count), "rmse_by_state": rmse.tolist(), "rmse_mean": float(np.sqrt(np.mean(mse)))}


def eval_rollout_rmse_vs_horizon_raw(
    segments_Xn: List[np.ndarray],
    segments_Un: List[np.ndarray],
    *,
    Kx: np.ndarray,
    Ku: np.ndarray,
    C: np.ndarray,
    lift_np: Callable[[np.ndarray], np.ndarray],
    X_mean: np.ndarray,
    X_std: np.ndarray,
    horizons: List[int],
    max_segments: int,
) -> dict:
    """Compute RMSE_l2(h) in RAW units for multiple horizons on the same test set."""
    X_mean = np.asarray(X_mean, dtype=float).reshape(-1)
    X_std = np.asarray(X_std, dtype=float).reshape(-1)
    max_segments = int(max_segments)
    if max_segments <= 0:
        max_segments = 10**9

    out = {}
    used = 0
    for h in horizons:
        h = int(h)
        if h < 1:
            continue
        se_sum = 0.0
        count = 0
        used = 0
        for Xseg, Useg in zip(segments_Xn, segments_Un):
            if used >= max_segments:
                break
            H = min(h, len(Useg), len(Xseg) - 1)
            x = Xseg[0].copy()
            for k in range(H):
                x = koopman_predict_next_x(x, Useg[k], Kx=Kx, Ku=Ku, C=C, lift_np=lift_np)
                x_true_n = Xseg[k + 1]
                e = (x * X_std + X_mean) - (x_true_n * X_std + X_mean)
                se_sum += float(np.dot(e, e))
                count += 1
            used += 1
        if count == 0:
            out[str(h)] = None
        else:
            out[str(h)] = float(np.sqrt(se_sum / float(count)))

    return {"rmse_l2_by_h": out}




def _save_rollout_plot(
    *,
    plot_dir: Path,
    horizons: List[int],
    rmse_l2_by_h: dict,
    dt_s: float,
    title: str,
) -> Path:
    plot_dir.mkdir(parents=True, exist_ok=True)
    x = []
    y = []
    for h in horizons:
        val = rmse_l2_by_h.get(str(int(h)))
        if val is None:
            continue
        x.append(int(h))
        y.append(float(val))

    fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=160)
    ax.plot(x, y, marker="o")
    ax.set_xlabel("Horizon (steps)")
    ax.set_ylabel("Rollout RMSE$_{\ell_2}$ (raw units)")
    ax.set_title(title)
    if dt_s and np.isfinite(dt_s) and dt_s > 0:
        ax2 = ax.secondary_xaxis("top", functions=(lambda s: s * dt_s, lambda t: t / dt_s))
        ax2.set_xlabel("Horizon (s)")
    ax.grid(True, alpha=0.3)
    out = plot_dir / "rollout_rmse_vs_horizon.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def _save_overlay_plot(
    *,
    plot_dir: Path,
    t: np.ndarray,
    X_true: np.ndarray,
    X_hat: np.ndarray,
    state_names: List[str],
    max_steps: int,
    title: str,
) -> Path:
    plot_dir.mkdir(parents=True, exist_ok=True)
    max_steps = int(max_steps)
    if max_steps > 0:
        t = t[:max_steps]
        X_true = X_true[:max_steps]
        X_hat = X_hat[:max_steps]

    n_state = X_true.shape[1]
    fig, axes = plt.subplots(n_state, 1, figsize=(7.5, 7.5), dpi=160, sharex=True)
    if n_state == 1:
        axes = [axes]

    for j in range(n_state):
        ax = axes[j]
        ax.plot(t, X_true[:, j], label="true", linewidth=1.5)
        ax.plot(t, X_hat[:, j], label="pred", linewidth=1.2)
        name = state_names[j] if j < len(state_names) else f"x{j}"
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.legend(loc="best")
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(title)
    out = plot_dir / "true_vs_pred_overlay.png"
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--train-csv",
        type=Path,
        default=Path("data_preprocessing") / "splits" / "all_lane_changes_train.csv",
    )
    p.add_argument(
        "--test-csv",
        type=Path,
        default=Path("data_preprocessing") / "splits" / "all_lane_changes_test.csv",
    )
    p.add_argument("--min-traj-len", type=int, default=20)
    p.add_argument("--seg-len", type=int, default=25)
    p.add_argument("--stride", type=int, default=25)
    p.add_argument("--reg", type=float, default=1e-6)

    p.add_argument("--lift", choices=["poly", "dnn"], default="poly")
    p.add_argument("--degree", type=int, default=3)
    p.add_argument("--dnn-psi-dim", type=int, default=20)
    p.add_argument("--dnn-hidden-dim", type=int, default=64)
    p.add_argument("--dnn-hidden-layers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--dnn-pretrain-steps", type=int, default=200, help="Only used when --lift dnn")
    p.add_argument("--dnn-lr", type=float, default=1e-3, help="Only used when --lift dnn")
    p.add_argument("--dnn-batch-segments", type=int, default=8, help="Only used when --lift dnn")
    p.add_argument("--dnn-refit-every", type=int, default=50, help="Only used when --lift dnn")
    p.add_argument("--dnn-print-every", type=int, default=20, help="Only used when --lift dnn")
    p.add_argument(
        "--dnn-fixed-decoder",
        action="store_true",
        help="Use C=[I,0] for DNN lift ψ=[x;net(x)] (usually improves x-space RMSE)",
    )

    p.add_argument("--rollout-h", type=int, default=10)
    p.add_argument("--rollout-max-segments", type=int, default=200)

    p.add_argument(
        "--horizons",
        type=str,
        default="10,20,40,60",
        help="Comma-separated rollout horizons (steps) for the RMSE-vs-horizon plot",
    )

    p.add_argument("--plot-dir", type=Path, default=None)
    p.add_argument("--plot-max-steps", type=int, default=120, help="Overlay plot length in steps")
    p.add_argument(
        "--overlay-id",
        type=str,
        default=None,
        help="Which test trajectory id to overlay (default: first)",
    )

    p.add_argument("--save-npz", type=Path, default=None, help="Optional path to save learned model")

    args = p.parse_args()

    n_state = 4

    cols = ColumnSpec()  # inferred on load
    X_train_raw, U_train_raw, V_train_raw, ids_train, dt_train = load_trajectories(
        args.train_csv, cols=cols, min_traj_len=int(args.min_traj_len)
    )
    X_test_raw, U_test_raw, V_test_raw, ids_test, dt_test = load_trajectories(
        args.test_csv, cols=cols, min_traj_len=int(args.min_traj_len)
    )

    # Normalize using TRAIN only (avoid test leakage)
    X_all = np.vstack(X_train_raw)
    U_all = np.vstack(U_train_raw)
    X_mean = X_all.mean(axis=0)
    X_std = X_all.std(axis=0) + 1e-9
    U_mean = U_all.mean(axis=0)
    U_std = U_all.std(axis=0) + 1e-9

    Xn_train = [(X - X_mean) / X_std for X in X_train_raw]
    Un_train = [(U - U_mean) / U_std for U in U_train_raw]
    Xn_test = [(X - X_mean) / X_std for X in X_test_raw]
    Un_test = [(U - U_mean) / U_std for U in U_test_raw]

    segments_X_train, segments_U_train = build_segments(
        Xn_train, Un_train, seg_len=int(args.seg_len), stride=int(args.stride)
    )
    segments_X_test, segments_U_test = build_segments(
        Xn_test, Un_test, seg_len=int(args.seg_len), stride=int(args.stride)
    )

    lift = str(args.lift).lower().strip()
    if lift == "poly":
        _, lift_np, psi_batch_np, _, _ = make_polynomial_lift(n_state=n_state, degree=int(args.degree))
        lift_obj = None
    else:
        lift_obj = make_dnn_lift(
            n_state=n_state,
            psi_dim=int(args.dnn_psi_dim),
            hidden_dim=int(args.dnn_hidden_dim),
            hidden_layers=int(args.dnn_hidden_layers),
            seed=int(args.seed),
        )
        lift_np = lift_obj.lift_np
        psi_batch_np = lift_obj.psi_batch_np

        def _dnn_identity_decoder(npsi: int) -> np.ndarray:
            C_id = np.zeros((n_state, int(npsi)), dtype=float)
            C_id[:, :n_state] = np.eye(n_state, dtype=float)
            return C_id

        # ---- Pretrain the lift network so it's not just a random feature map.
        steps = int(args.dnn_pretrain_steps)
        if steps > 0:
            # Initialize (Kx, Ku, C) with the current lift before pretraining.
            Kx0, Ku0, C0 = koopman_regression_full(
                segments_X_train,
                segments_U_train,
                psi_batch_np,
                reg=float(args.reg),
            )
            if bool(args.dnn_fixed_decoder):
                C0 = _dnn_identity_decoder(Kx0.shape[0])
            print(
                f"[DNN] Pretraining lift with DKR: steps={steps} lr={float(args.dnn_lr):g} "
                f"batch_segments={int(args.dnn_batch_segments)} refit_every={int(args.dnn_refit_every)}"
            )
            dnn_pretrain_dkr(
                segments_X_train,
                segments_U_train,
                lift=lift_obj,
                Kx=Kx0,
                Ku=Ku0,
                C=C0,
                steps=steps,
                lr=float(args.dnn_lr),
                batch_segments=int(args.dnn_batch_segments),
                seed=int(args.seed),
                refit_every=int(args.dnn_refit_every),
                reg=float(args.reg),
                fix_C_identity=bool(args.dnn_fixed_decoder),
                verbose=True,
                print_every=int(args.dnn_print_every),
            )

    # Final regression after (optional) DNN pretraining
    Kx, Ku, C = koopman_regression_full(
        segments_X_train,
        segments_U_train,
        psi_batch_np,
        reg=float(args.reg),
    )
    if lift == "dnn" and bool(args.dnn_fixed_decoder):
        C = np.zeros_like(C)
        C[:, :n_state] = np.eye(n_state, dtype=float)

    train_one = eval_one_step_mse(
        segments_X_train,
        segments_U_train,
        Kx=Kx,
        Ku=Ku,
        C=C,
        lift_np=lift_np,
    )
    test_one = eval_one_step_mse(
        segments_X_test,
        segments_U_test,
        Kx=Kx,
        Ku=Ku,
        C=C,
        lift_np=lift_np,
    )

    train_roll = eval_rollout_rmse(
        segments_X_train,
        segments_U_train,
        Kx=Kx,
        Ku=Ku,
        C=C,
        lift_np=lift_np,
        rollout_h=int(args.rollout_h),
        max_segments=int(args.rollout_max_segments),
    )
    test_roll = eval_rollout_rmse(
        segments_X_test,
        segments_U_test,
        Kx=Kx,
        Ku=Ku,
        C=C,
        lift_np=lift_np,
        rollout_h=int(args.rollout_h),
        max_segments=int(args.rollout_max_segments),
    )

    # Thesis-style metrics in RAW units
    state_names = [
        "lateral_offset_m",
        "target_orientation_rad",
        "target_curvature_1pm",
        "target_curvature_dot_1pm2",
    ]
    train_one_raw = eval_one_step_rmse_nrmse_raw(
        segments_X_train,
        segments_U_train,
        Kx=Kx,
        Ku=Ku,
        C=C,
        lift_np=lift_np,
        X_mean=X_mean,
        X_std=X_std,
    )
    test_one_raw = eval_one_step_rmse_nrmse_raw(
        segments_X_test,
        segments_U_test,
        Kx=Kx,
        Ku=Ku,
        C=C,
        lift_np=lift_np,
        X_mean=X_mean,
        X_std=X_std,
    )

    horizons = [int(s.strip()) for s in str(args.horizons).split(",") if s.strip()]
    rollout_curve = eval_rollout_rmse_vs_horizon_raw(
        segments_X_test,
        segments_U_test,
        Kx=Kx,
        Ku=Ku,
        C=C,
        lift_np=lift_np,
        X_mean=X_mean,
        X_std=X_std,
        horizons=horizons,
        max_segments=int(args.rollout_max_segments),
    )

    report = {
        "train_csv": str(args.train_csv),
        "test_csv": str(args.test_csv),
        "n_traj_train": len(X_train_raw),
        "n_traj_test": len(X_test_raw),
        "dt_train_s": float(dt_train),
        "dt_test_s": float(dt_test),
        "seg_len": int(args.seg_len),
        "stride": int(args.stride),
        "lift": lift,
        "degree": int(args.degree) if lift == "poly" else None,
        "reg": float(args.reg),
        "one_step": {"train": train_one, "test": test_one},
        "rollout": {"train": train_roll, "test": test_roll},

        "one_step_raw": {"train": train_one_raw, "test": test_one_raw, "state_names": state_names},
        "rollout_rmse_l2_vs_horizon_raw": {"horizons": horizons, **rollout_curve},
    }

    print("\n=== Dynamics Learning (Koopman) ===")
    print(json.dumps(report, indent=2))

    _print_one_step_table(one_step_raw=report["one_step_raw"], title=str(report["lift"]))

    # Optional plots
    if args.plot_dir is not None:
        plot_dir = Path(args.plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)
        rollout_path = _save_rollout_plot(
            plot_dir=plot_dir,
            horizons=horizons,
            rmse_l2_by_h=rollout_curve["rmse_l2_by_h"],
            dt_s=float(dt_test),
            title=f"Rollout RMSE vs Horizon ({lift})",
        )

        # Overlay plot on one representative test trajectory
        # Use the first test trajectory by default.
        overlay_idx = 0
        if args.overlay_id is not None:
            try:
                overlay_idx = ids_test.index(str(args.overlay_id))
            except ValueError:
                overlay_idx = 0

        X_true = np.asarray(X_test_raw[overlay_idx], dtype=float)
        U_true = np.asarray(U_test_raw[overlay_idx], dtype=float)
        # Normalize the chosen trajectory
        Xn = (X_true - X_mean) / X_std
        Un = (U_true - U_mean) / U_std

        # Rollout in normalized space using true inputs, then denormalize
        xn = Xn[0].copy()
        X_hat = [xn * X_std + X_mean]
        for k in range(min(len(Un), len(Xn) - 1)):
            xn = koopman_predict_next_x(xn, Un[k], Kx=Kx, Ku=Ku, C=C, lift_np=lift_np)
            X_hat.append(xn * X_std + X_mean)
        X_hat = np.asarray(X_hat, dtype=float)

        # time axis for overlay
        t = np.asarray(np.arange(X_hat.shape[0]), dtype=float) * float(dt_test)
        overlay_path = _save_overlay_plot(
            plot_dir=plot_dir,
            t=t,
            X_true=X_true[: X_hat.shape[0]],
            X_hat=X_hat,
            state_names=state_names,
            max_steps=int(args.plot_max_steps),
            title=f"True vs Predicted (test traj {ids_test[overlay_idx]}) ({lift})",
        )
        print(f"\n[PLOTS] Wrote {rollout_path}")
        print(f"[PLOTS] Wrote {overlay_path}")

    if args.save_npz is not None:
        save_path = Path(args.save_npz)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            save_path,
            Kx=Kx,
            Ku=Ku,
            C=C,
            X_mean=X_mean,
            X_std=X_std,
            U_mean=U_mean,
            U_std=U_std,
            meta=json.dumps(report),
        )
        print(f"\nSaved model to: {save_path}")


if __name__ == "__main__":
    main()
