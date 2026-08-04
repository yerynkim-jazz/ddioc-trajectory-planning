"""dynamics_learning.py"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, List, Literal, Optional, Tuple

import numpy as np
import casadi as ca

try:
	import torch  # type: ignore
	import torch.nn as nn  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
	torch = None  # type: ignore
	nn = None  # type: ignore


Entities = Literal["ego", "agents", "both"]


@dataclass(frozen=True)
class RootSpec:
	src_root: Path
	tag: str

if torch is not None:
	class ObservableNet(nn.Module):
		def __init__(
			self,
			input_dim: int,
			output_dim: int,
			*,
			hidden_dim: int = 64,
			hidden_layers: int = 4,
		):
			super().__init__()
			input_dim = int(input_dim)
			output_dim = int(output_dim)
			hidden_dim = int(hidden_dim)
			hidden_layers = int(hidden_layers)
			layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
			for _ in range(hidden_layers - 1):
				layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
			layers += [nn.Linear(hidden_dim, output_dim)]
			self.net = nn.Sequential(*layers)

		def forward(self, x):
			if not torch.is_tensor(x):
				x = torch.tensor(x, dtype=torch.float32)
			return self.net(x)


	@dataclass
	class DnnLift:
		n_psi: int
		n_state: int
		net_out: int
		model: Any
		device: Any

		def lift_np(self, x: np.ndarray) -> np.ndarray:
			x = np.asarray(x, dtype=np.float32).reshape(-1)
			with torch.no_grad():
				xt = torch.tensor(x, dtype=torch.float32, device=self.device)
				y = self.model(xt)
				psi = torch.cat([xt, y], dim=0)
				return psi.detach().cpu().numpy().astype(float)

		def psi_batch_np(self, X: np.ndarray) -> np.ndarray:
			X = np.asarray(X, dtype=np.float32)
			with torch.no_grad():
				Xt = torch.tensor(X, dtype=torch.float32, device=self.device)
				Y = self.model(Xt)
				Psi = torch.cat([Xt, Y], dim=1)
				return Psi.detach().cpu().numpy().astype(float)

		def dpsi_dx_np(self, x: np.ndarray) -> np.ndarray:
			x = np.asarray(x, dtype=np.float32).reshape(-1)
			xt = torch.tensor(x, dtype=torch.float32, device=self.device, requires_grad=True)
			y = self.model(xt)
			J_net = []
			for i in range(int(self.net_out)):
				gi = torch.autograd.grad(y[i], xt, retain_graph=True, allow_unused=False)[0]
				J_net.append(gi.detach().cpu().numpy())
			J_net = np.vstack(J_net).astype(float)
			J = np.zeros((int(self.n_psi), int(self.n_state)), dtype=float)
			J[: int(self.n_state), : int(self.n_state)] = np.eye(int(self.n_state))
			J[int(self.n_state) :, :] = J_net
			return J

else:
	@dataclass
	class DnnLift:
		n_psi: int = 0
		n_state: int = 0
		net_out: int = 0
		model: Any = None
		device: Any = None

def make_dnn_lift(
	*,
	n_state: int,
	psi_dim: int = 20,
	hidden_dim: int = 64,
	hidden_layers: int = 4,
	seed: int = 42,
	device: Optional[str] = None,
) -> DnnLift:
	"""Create a DNN Koopman lift.

	Returns a `DnnLift` that exposes:
	- lift_np(x) -> ψ(x)
	- psi_batch_np(X) -> ψ(X)
	- dpsi_dx_np(x) -> dψ/dx

	Notes:
	- ψ_dim is the network output dimension (net_out). Total n_psi = n_state + ψ_dim.
	- `lift_ca` is not provided for DNN lifts (CasADi can't trace PyTorch).
	"""	
	if torch is None:
		raise ModuleNotFoundError("PyTorch is required for --lift dnn. Install 'torch' or use --lift poly.")
	n_state = int(n_state)
	psi_dim = int(psi_dim)
	hidden_dim = int(hidden_dim)
	hidden_layers = int(hidden_layers)
	if n_state <= 0:
		raise ValueError("n_state must be positive")
	if psi_dim <= 0:
		raise ValueError("psi_dim must be positive")

	torch.manual_seed(int(seed))
	np.random.seed(int(seed))
	if device is None:
		dev = torch.device("cpu")
	else:
		dev = torch.device(str(device))

	model = ObservableNet(
		input_dim=n_state,
		output_dim=psi_dim,
		hidden_dim=hidden_dim,
		hidden_layers=hidden_layers,
	)
	model.to(dev)
	model.eval()

	return DnnLift(
		n_psi=int(n_state + psi_dim),
		n_state=n_state,
		net_out=psi_dim,
		model=model,
		device=dev,
	)


def compute_dkr_loss(
	segments_X,
	segments_U,
	*,
	lift: DnnLift,
	Kx: np.ndarray,
	Ku: np.ndarray,
	C: np.ndarray,
) -> Any:
	"""DKR loss (Koopman linearity + reconstruction), aligned with the reference script."""
	if torch is None:
		raise ModuleNotFoundError("PyTorch is required for DKR pretraining.")

	# Convert system matrices once per call (they are constants for backprop)
	Kx_t = torch.tensor(np.asarray(Kx, dtype=np.float32), dtype=torch.float32, device=lift.device)
	Ku_t = torch.tensor(np.asarray(Ku, dtype=np.float32), dtype=torch.float32, device=lift.device)
	C_t = torch.tensor(np.asarray(C, dtype=np.float32), dtype=torch.float32, device=lift.device)

	# Flatten all transitions in the batch to a single big tensor.
	# segments_X: list[(T+1,nx)], segments_U: list[(T,nu) or (T,)]
	Xt_list = []
	Xtp1_list = []
	Ut_list = []
	for x_seg, u_seg in zip(segments_X, segments_U):
		x_seg = np.asarray(x_seg, dtype=np.float32)
		u_seg = np.asarray(u_seg, dtype=np.float32)
		if x_seg.ndim != 2 or x_seg.shape[0] < 2:
			continue
		# u_seg may be (T,1) or (T,)
		if u_seg.ndim == 1:
			u_seg = u_seg.reshape(-1, 1)
		if u_seg.ndim != 2:
			continue
		T = min(x_seg.shape[0] - 1, u_seg.shape[0])
		if T <= 0:
			continue
		Xt_list.append(x_seg[:T])
		Xtp1_list.append(x_seg[1 : T + 1])
		Ut_list.append(u_seg[:T])

	if not Xt_list:
		return torch.tensor(0.0, dtype=torch.float32, device=lift.device)

	Xt_np = np.concatenate(Xt_list, axis=0)
	Xtp1_np = np.concatenate(Xtp1_list, axis=0)
	Ut_np = np.concatenate(Ut_list, axis=0)

	Xt = torch.tensor(Xt_np, dtype=torch.float32, device=lift.device)
	Xtp1 = torch.tensor(Xtp1_np, dtype=torch.float32, device=lift.device)
	Ut = torch.tensor(Ut_np, dtype=torch.float32, device=lift.device)

	Yt = lift.model(Xt)
	Ytp1 = lift.model(Xtp1)
	psi_t = torch.cat([Xt, Yt], dim=1)  # (N, n_psi)
	psi_tp1 = torch.cat([Xtp1, Ytp1], dim=1)

	# Row-form dynamics: psi_{t+1}^T = psi_t^T Kx^T + u_t^T Ku^T
	psi_pred = (psi_t @ Kx_t.T) + (Ut @ Ku_t.T)
	loss_K = torch.mean((psi_tp1 - psi_pred) ** 2)

	# Reconstruction loss: x = C psi  -> row form: x^T = psi^T C^T
	x_pred = psi_t @ C_t.T
	loss_C = torch.mean((Xt - x_pred) ** 2)

	return loss_K + loss_C


def dnn_pretrain_dkr(
	segments_X,
	segments_U,
	*,
	lift: DnnLift,
	Kx: np.ndarray,
	Ku: np.ndarray,
	C: np.ndarray,
	steps: int = 200,
	lr: float = 1e-3,
	batch_segments: int = 8,
	seed: int = 0,
	refit_every: int = 50,
	reg: float = 1e-6,
	fix_C_identity: bool = False,
	verbose: bool = False,
	print_every: int = 20,
) -> None:
	"""Optional pretraining for DNN lift using the DKR loss.

	This updates only θ (the lift network). Optionally refits (Kx, Ku, C) periodically
	using ridge regression on the same segments (keeps training stable).
	"""
	if torch is None:
		raise ModuleNotFoundError("PyTorch is required for DNN pretraining.")

	steps = int(steps)
	if steps <= 0:
		return
	batch_segments = max(1, int(batch_segments))
	refit_every = max(1, int(refit_every))
	print_every = max(1, int(print_every))

	rng = np.random.default_rng(int(seed))
	opt = torch.optim.Adam(lift.model.parameters(), lr=float(lr))

	if bool(fix_C_identity):
		# For ψ(x) = [x; net(x)], the exact decoder is x = [I 0] ψ.
		nx = int(lift.n_state)
		C[:] = 0.0
		C[:, :nx] = np.eye(nx, dtype=float)
	
	lift.model.train()
	for s in range(steps):
		idx = rng.choice(len(segments_X), size=min(batch_segments, len(segments_X)), replace=False)
		Xb = [segments_X[i] for i in idx]
		Ub = [segments_U[i] for i in idx]

		opt.zero_grad(set_to_none=True)
		loss = compute_dkr_loss(Xb, Ub, lift=lift, Kx=Kx, Ku=Ku, C=C)
		loss.backward()
		torch.nn.utils.clip_grad_norm_(lift.model.parameters(), max_norm=10.0)
		opt.step()

		if bool(verbose) and (((s + 1) % print_every) == 0 or (s == 0) or (s + 1) == steps):
			try:
				loss_val = float(loss.detach().cpu().item())
			except Exception:
				loss_val = float("nan")
			print(f"[DNN PRETRAIN] step {s+1:4d}/{steps} loss={loss_val:.6e}")

		if (s + 1) % refit_every == 0:
			Kx[:], Ku[:], C[:] = koopman_regression_full(segments_X, segments_U, lift.psi_batch_np, reg=float(reg))
			if bool(fix_C_identity):
				nx = int(lift.n_state)
				C[:] = 0.0
				C[:, :nx] = np.eye(nx, dtype=float)

	lift.model.eval()


def make_polynomial_lift(n_state: int, degree: int):
	"""Full multivariate polynomial lift (total degree <= degree)."""

	n_state = int(n_state)
	degree = int(degree)
	if n_state <= 0:
		raise ValueError("n_state must be positive")
	if degree < 0:
		raise ValueError("degree must be >= 0")

	# Precompute all exponent vectors e s.t. sum(e)=0..degree.
	# Order: increasing total degree, then lexicographic in e[0], e[1], ...
	# (implemented via recursion on remaining degree).
	exponents: list[tuple[int, ...]] = []

	def _gen_exponents(total_degree: int, idx: int, prefix: list[int]):
		if idx == n_state - 1:
			exponents.append(tuple(prefix + [total_degree]))
			return
		for e in range(total_degree + 1):
			_gen_exponents(total_degree - e, idx + 1, prefix + [e])

	for total_degree in range(degree + 1):
		_gen_exponents(total_degree, 0, [])

	n_psi = len(exponents)

	def lift_np(x: np.ndarray) -> np.ndarray:
		x = np.asarray(x, dtype=float).reshape(-1)
		if x.shape[0] != n_state:
			raise ValueError(f"Expected x with shape ({n_state},), got {x.shape}")

		# ---- SAFETY: avoid overflow in polynomial features
		# Clip state to a reasonable range (in NORMALIZED coordinates).
		# Start with 8..12; tune based on your data scaling.
		x_clip = np.clip(x, -10.0, 10.0)

		psi = np.empty(n_psi, dtype=float)
		for r, exp in enumerate(exponents):
			prod = 1.0
			for i, ei in enumerate(exp):
				if ei:
					# safe power; still can overflow if x_clip too big and ei large
					prod *= float(x_clip[i]) ** int(ei)
			psi[r] = prod

		# If anything still blew up, fall back to finite values
		psi = np.nan_to_num(psi, nan=0.0, posinf=1e6, neginf=-1e6)
		return psi

	def psi_batch_np(X: np.ndarray) -> np.ndarray:
		return np.array([lift_np(x) for x in X], dtype=float)

	def dpsi_dx_np(x: np.ndarray) -> np.ndarray:
		"""Jacobian: (n_psi, n_state). The constant monomial has zero gradient."""

		x = np.asarray(x, dtype=float).reshape(-1)
		if x.shape[0] != n_state:
			raise ValueError(f"Expected x with shape ({n_state},), got {x.shape}")

		J = np.zeros((n_psi, n_state), dtype=float)
		for r, exp in enumerate(exponents):
			# For each dimension k:
			# d/dx_k prod_i x_i^{e_i} = e_k * x_k^{e_k-1} * prod_{i!=k} x_i^{e_i}
			for k in range(n_state):
				ek = exp[k]
				if ek == 0:
					continue

				term = float(ek)
				for i, ei in enumerate(exp):
					if ei == 0:
						continue
					if i == k:
						if ei > 1:
							term *= float(x[i]) ** int(ei - 1)
					else:
						term *= float(x[i]) ** int(ei)
				J[r, k] = term
		return J

	def lift_ca(x: ca.MX) -> ca.MX:
		psi_terms = []
		for exp in exponents:
			term = 1
			for i, ei in enumerate(exp):
				if ei:
					term = term * (x[i] ** int(ei))
			psi_terms.append(term)
		return ca.vertcat(*psi_terms)

	return n_psi, lift_np, psi_batch_np, dpsi_dx_np, lift_ca


def koopman_regression_full(buffer_X, buffer_U, psi_batch_np, reg=1e-6):
	"""Koopman regression (R1 full) on buffer (ridge).

	Learns:
	  psi_{k+1} ≈ Kx psi_k + Ku u_k
	  x_k ≈ C psi_k
	"""

	Xk = np.vstack([X[:-1] for X in buffer_X])  # (N, n_state)
	Xk1 = np.vstack([X[1:] for X in buffer_X])
	Uk = np.vstack([U[:] for U in buffer_U])  # (N, 1)

	Psi_xk = psi_batch_np(Xk).T  # (npsi, N)
	Psi_xk1 = psi_batch_np(Xk1).T  # (npsi, N)

	Z = np.vstack([Psi_xk, Uk.T])  # (npsi+1, N)

	G = Z @ Z.T
	K = Psi_xk1 @ Z.T @ np.linalg.inv(G + reg * np.eye(G.shape[0]))
	Kx = K[:, :-1]
	Ku = K[:, -1:].copy()

	Gpsi = Psi_xk @ Psi_xk.T
	C = Xk.T @ Psi_xk.T @ np.linalg.inv(Gpsi + reg * np.eye(Gpsi.shape[0]))

	return Kx, Ku, C


def _segment_mats_for_woodbury(Xseg, Useg, psi_batch_np):
	"""Build per-segment matrices for Woodbury/RLS updates.

	For one fixed-length segment:
	  Xseg: (seg_len, n_state)
	  Useg: (seg_len-1, 1)

	Returns:
	  Psi_x  : (npsi, N)
	  Psi_x1 : (npsi, N)
	  Z      : (npsi+1, N)  where Z = [Psi_x; u^T]
	  Xk_T   : (n_state, N) where Xk are states at k
	with N = seg_len-1 transitions.
	"""

	Xk = Xseg[:-1]  # (N, n_state)
	Xk1 = Xseg[1:]  # (N, n_state)
	Uk = Useg.reshape(-1, 1)  # (N, 1)

	Psi_x = psi_batch_np(Xk).T  # (npsi, N)
	Psi_x1 = psi_batch_np(Xk1).T  # (npsi, N)

	Z = np.vstack([Psi_x, Uk.T])  # (npsi+1, N)
	Xk_T = Xk.T  # (n_state, N)
	return Psi_x, Psi_x1, Z, Xk_T


def koopman_init_woodbury(Xseg, Useg, psi_batch_np, reg=1e-6):
	"""Initialize Kx/Ku/C and inverse Gram matrices (Pz/Ppsi) from the first segment."""

	Psi_x, Psi_x1, Z, Xk_T = _segment_mats_for_woodbury(Xseg, Useg, psi_batch_np)

	# Pz = (Z Z^T + reg I)^-1
	Gz = Z @ Z.T + reg * np.eye(Z.shape[0])
	Pz = np.linalg.inv(Gz)

	# K = Psi_x1 Z^T Pz
	K = Psi_x1 @ Z.T @ Pz
	Kx = K[:, :-1]
	Ku = K[:, -1:].copy()

	# Ppsi = (Psi_x Psi_x^T + reg I)^-1
	Gpsi = Psi_x @ Psi_x.T + reg * np.eye(Psi_x.shape[0])
	Ppsi = np.linalg.inv(Gpsi)

	# C = Xk Psi_x^T Ppsi   (ridge-consistent)
	C = Xk_T @ Psi_x.T @ Ppsi

	return Kx, Ku, C, Pz, Ppsi


def koopman_update_woodbury(Kx, Ku, C, Pz, Ppsi, Xseg, Useg, psi_batch_np, reg=1e-6):
	"""Woodbury/RLS update using ONE new segment.

	Keeps the same regression targets as `koopman_regression_full`:
	  Psi_x1 ≈ [Kx Ku] [Psi_x; u]
	  x      ≈ C Psi_x
	"""

	Psi_x, Psi_x1, Z, Xk_T = _segment_mats_for_woodbury(Xseg, Useg, psi_batch_np)
	N = Z.shape[1]  # number of transitions in this segment

	# ---- update K (on Z)
	K_full = np.hstack([Kx, Ku])  # (npsi, npsi+1)
	gamma = np.linalg.inv(np.eye(N) + Z.T @ Pz @ Z + reg * np.eye(N))
	K_full_new = K_full + (Psi_x1 - K_full @ Z) @ gamma @ Z.T @ Pz
	Pz_new = Pz - Pz @ Z @ gamma @ Z.T @ Pz

	Kx_new = K_full_new[:, :-1]
	Ku_new = K_full_new[:, -1:].copy()

	# ---- update C (on Psi_x)
	gamma_bar = np.linalg.inv(np.eye(N) + Psi_x.T @ Ppsi @ Psi_x + reg * np.eye(N))
	C_new = C + (Xk_T - C @ Psi_x) @ gamma_bar @ Psi_x.T @ Ppsi
	Ppsi_new = Ppsi - Ppsi @ Psi_x @ gamma_bar @ Psi_x.T @ Ppsi

	return Kx_new, Ku_new, C_new, Pz_new, Ppsi_new


def _compute_rel_opt_timestamps(ts: float, horizon: int) -> np.ndarray:
	if int(horizon) < 2:
		return np.array([0.0], dtype=float)
	return np.linspace(0.0, float(ts) * float(horizon - 1), int(horizon), dtype=float)


def _default_tag_for_root(src_root: Path) -> str:
	name = Path(src_root).name
	return name if name else "dataset"


def _iter_files(root: Path, suffix: str):
	yield from (p for p in Path(root).rglob(f"*{suffix}") if p.is_file())


def _module_candidates(env_var: str, defaults: Tuple[str, ...]) -> List[str]:
	value = str(os.environ.get(env_var, "")).strip()
	if not value:
		return list(defaults)
	return [part.strip() for part in value.split(",") if part.strip()]


def _import_optional_attr(importlib_mod: Any, *, env_var: str, defaults: Tuple[str, ...], attr: str) -> Any:
	errors: List[str] = []
	for module_name in _module_candidates(env_var, defaults):
		try:
			module = importlib_mod.import_module(module_name)
			return getattr(module, attr)
		except Exception as exc:
			errors.append(f"{module_name}: {exc!r}")
	raise ModuleNotFoundError(
		f"Could not import optional helper '{attr}'. Tried modules from {env_var}: {errors}"
	)


def extract_and_merge_from_mcap_roots(
	*,
	src_roots: List[Path],
	tags: Optional[List[str]],
	output_root: Path,
	entities: Entities,
	entities_by_tag: Optional[dict[str, Entities]],
	unzip: bool,
	overwrite_zip: bool,
	max_files_per_root: int,
	verbose: bool,
	ts: float,
	rel_opt_horizon: int,
	max_interpolation_gap_s: float,
	leave_lane_threshold_m: float,
	leave_lane_padding_s: float,
	state_estimation_padding_s: float,
	min_velocity: float,
	write_pkl: bool,
	clean_output: bool,
) -> Path:
	import importlib

	if tags is not None and len(tags) != len(src_roots):
		raise SystemExit("If provided, --tag must be specified once per --src_root.")

	roots: List[RootSpec] = []
	for i, sr in enumerate(src_roots):
		tag = tags[i] if tags is not None else _default_tag_for_root(Path(sr))
		roots.append(RootSpec(src_root=Path(sr), tag=tag))

	output_root = Path(output_root)
	if clean_output and output_root.exists():
		import shutil

		shutil.rmtree(output_root)

	per_demo_root = output_root / "per_demo"
	extracted_csv_dir = output_root / "extracted_csvs"
	merged_csv_path = output_root / "merged_ioc_dataset.csv"

	output_root.mkdir(parents=True, exist_ok=True)
	per_demo_root.mkdir(parents=True, exist_ok=True)

	rel_opt_timestamps = _compute_rel_opt_timestamps(ts=float(ts), horizon=int(rel_opt_horizon))
	log_path = output_root / "batch_log.txt"

	try:
		process_one_mcap = _import_optional_attr(
			importlib,
			env_var="IOC_MCAP_EXTRACT_MODULE",
			defaults=("extract_csv_from_mcap",),
			attr="process_one_mcap",
		)
		safe_extract_zip = _import_optional_attr(
			importlib,
			env_var="IOC_MCAP_EXTRACT_MODULE",
			defaults=("extract_csv_from_mcap",),
			attr="safe_extract_zip",
		)
	except ModuleNotFoundError as e:
		raise SystemExit(
			"MCAP extraction mode depends on optional dataset-extraction helpers that are not bundled with this public repo. "
			"Either provide --merged_csv, or expose compatible helpers via IOC_MCAP_EXTRACT_MODULE."
		) from e

	with open(log_path, "w", encoding="utf-8") as log:
		for root in roots:
			src_root = root.src_root
			tag = root.tag
			entities_for_root: Entities = entities
			if entities_by_tag and tag in entities_by_tag:
				entities_for_root = entities_by_tag[tag]
			out_dir = per_demo_root / tag
			out_dir.mkdir(parents=True, exist_ok=True)

			log.write(f"=== ROOT {tag} ===\n{src_root}\n")

			if unzip:
				zips = sorted(_iter_files(src_root, ".zip"))
				for zp in zips:
					try:
						extract_dir = zp.with_suffix("")
						if extract_dir.exists() and any(extract_dir.iterdir()) and not overwrite_zip:
							continue
						safe_extract_zip(zp, extract_dir, overwrite=overwrite_zip)
					except Exception as e:
						log.write(f"[zip-fail] {zp}\n{e!r}\n")

			mcaps = sorted(_iter_files(src_root, ".mcap"))
			if max_files_per_root and max_files_per_root > 0:
				mcaps = mcaps[:max_files_per_root]

			for mcap_path in mcaps:
				try:
					process_one_mcap(
						mcap_path,
						out_dir,
						entities=entities_for_root,
						ts=float(ts),
						rel_opt_timestamps=rel_opt_timestamps,
						max_interpolation_gap_s=float(max_interpolation_gap_s),
						leave_lane_threshold_m=float(leave_lane_threshold_m),
						leave_lane_padding_s=float(leave_lane_padding_s),
						state_estimation_padding_s=float(state_estimation_padding_s),
						min_velocity=float(min_velocity),
						verbose=bool(verbose),
						write_pkl=bool(write_pkl),
					)
				except Exception as e:
					log.write(f"[mcap-fail] {mcap_path}\n{e!r}\n")

	try:
		extract_csv_files = _import_optional_attr(
			importlib,
			env_var="IOC_CSV_EXTRACT_MODULE",
			defaults=("extract_csv_files",),
			attr="extract_csv_files",
		)
		merge_ioc_csvs = _import_optional_attr(
			importlib,
			env_var="IOC_CSV_MERGE_MODULE",
			defaults=("merge_ioc_csvs",),
			attr="merge_ioc_csvs",
		)
	except ModuleNotFoundError as e:
		raise SystemExit(
			"MCAP extraction mode also depends on optional CSV post-processing helpers that are not bundled with this public repo. "
			"Either provide --merged_csv, or expose compatible helpers via IOC_CSV_EXTRACT_MODULE and IOC_CSV_MERGE_MODULE."
		) from e

	extract_csv_files(
		input_roots=[per_demo_root],
		output_dir=extracted_csv_dir,
		pattern="*.csv",
		clean=True,
		dry_run=False,
	)
	merge_ioc_csvs(
		input_root=str(extracted_csv_dir),
		pattern="*.csv",
		out_csv=str(merged_csv_path),
	)
	return merged_csv_path


def load_trajectories_from_merged_csv(
	csv_path: Path,
	*,
	n_traj: int,
	min_traj_len: int,
	return_ids: bool = True,
	use_reference_columns: bool | None = None,
) -> Tuple[
	List[np.ndarray],
	List[np.ndarray],
	List[np.ndarray],
	List[np.ndarray],
	List[np.ndarray],
	List[str],
	float,
]:
	if use_reference_columns is False:
		raise ValueError("Reference columns are mandatory; remove use_reference_columns=False")

	from typing import Callable

	import pandas as pd

	csv_path = Path(csv_path)
	if not csv_path.exists():
		raise FileNotFoundError(csv_path)

	allowed_columns = {
		"lc_id",
		"lc_id_label",
		"lc_source",
		"time_s",
		"target_speed_mps",
		"lateral_offset_m_sg",
		"target_orientation_rad_sg",
		"target_curvature_1pm_sg",
		"target_curvature_1pm_dot",
		"target_curvature_1pm_ddot",
		"reference_orientation_sg",
		"reference_curvature_sg",
	}

	usecol_fn: Callable[[str], bool] = lambda c: c in allowed_columns
	df = pd.read_csv(csv_path, usecols=usecol_fn)
	if "time_s" not in df.columns:
		raise ValueError("CSV must contain 'time_s' column")

	if "lc_id_label" in df.columns:
		id_col = "lc_id_label"
	elif "lc_id" in df.columns:
		id_col = "lc_id"
	elif "lc_source" in df.columns:
		id_col = "lc_source"
	else:
		raise ValueError("CSV must contain trajectory id column: lc_id_label or lc_id or lc_source")

	if id_col != "file_id":
		df = df.rename(columns={id_col: "file_id"})

	col_lat = "lateral_offset_m_sg"
	col_psi = "target_orientation_rad_sg"
	col_kappa = "target_curvature_1pm_sg"
	col_kappa_dot = "target_curvature_1pm_dot"
	col_u = "target_curvature_1pm_ddot"

	required = [col_lat, col_psi, col_kappa, col_kappa_dot, col_u]
	missing = [c for c in required if c not in df.columns]
	if missing:
		raise ValueError(f"CSV missing required columns: {missing}")

	ref_ori_col = "reference_orientation_sg"
	ref_kappa_col = "reference_curvature_sg"
	if ref_ori_col not in df.columns or ref_kappa_col not in df.columns:
		raise ValueError("CSV missing required reference_orientation_sg / reference_curvature_sg")

	has_speed = "target_speed_mps" in df.columns
	df = df.sort_values(["file_id", "time_s"]).reset_index(drop=True)
	file_ids = df["file_id"].unique()

	traj_X_raw: List[np.ndarray] = []
	traj_U_raw: List[np.ndarray] = []
	traj_V_raw: List[np.ndarray] = []
	traj_ref_orientation_raw: List[np.ndarray] = []
	traj_ref_curvature_raw: List[np.ndarray] = []
	traj_ids: List[str] = []
	traj_dt_s: List[float] = []

	for fid in file_ids:
		dfi = df[df["file_id"] == fid].copy().sort_values("time_s")
		t = dfi["time_s"].to_numpy(dtype=float)

		X = np.vstack(
			[
				dfi[col_lat].to_numpy(),
				dfi[col_psi].to_numpy(),
				dfi[col_kappa].to_numpy(),
				dfi[col_kappa_dot].to_numpy(),
			]
		).T
		U = dfi[col_u].to_numpy().reshape(-1, 1)
		V = dfi["target_speed_mps"].to_numpy().reshape(-1) if has_speed else np.ones(len(dfi), dtype=float)

		ref_ori = dfi[ref_ori_col].to_numpy(dtype=float)
		ref_kappa = dfi[ref_kappa_col].to_numpy(dtype=float)

		mask = np.isfinite(X).all(axis=1) & np.isfinite(U).all(axis=1) & np.isfinite(V)
		mask = mask & np.isfinite(ref_ori) & np.isfinite(ref_kappa)
		X, U, V = X[mask], U[mask], V[mask]
		t = t[mask]
		ref_ori = np.asarray(ref_ori, dtype=float)[mask]
		ref_kappa = np.asarray(ref_kappa, dtype=float)[mask]

		if len(X) < int(min_traj_len):
			continue

		traj_X_raw.append(X.astype(float))
		traj_U_raw.append(U.astype(float))
		traj_V_raw.append(V.astype(float))
		traj_ref_orientation_raw.append(np.asarray(ref_ori, dtype=float))
		traj_ref_curvature_raw.append(np.asarray(ref_kappa, dtype=float))
		traj_ids.append(str(fid))

		if len(t) >= 2:
			dt = float(np.median(np.diff(t)))
			if not np.isfinite(dt) or dt <= 0:
				dt = 0.0
		else:
			dt = 0.0
		traj_dt_s.append(dt)

		if len(traj_X_raw) >= int(n_traj):
			break

	if len(traj_X_raw) == 0:
		raise RuntimeError("No valid trajectories loaded")

	valid_dt = [d for d in traj_dt_s if d > 0]
	global_dt_s = float(np.median(valid_dt)) if valid_dt else 0.1
	return (
		traj_X_raw,
		traj_U_raw,
		traj_V_raw,
		traj_ref_orientation_raw,
		traj_ref_curvature_raw,
		traj_ids if bool(return_ids) else [],
		global_dt_s,
	)


def build_segments_from_trajs(
	traj_Xn: List[np.ndarray],
	traj_Un: List[np.ndarray],
	traj_Vraw: List[np.ndarray],
	traj_Xraw: List[np.ndarray],
	traj_ref_orientation_raw: Optional[List[np.ndarray]] = None,
	traj_ref_curvature_raw: Optional[List[np.ndarray]] = None,
	*,
	seg_len: int,
	segment_stride: int,
	dt_s: float,
) -> tuple[
	List[np.ndarray],
	List[np.ndarray],
	List[np.ndarray],
	Optional[List[np.ndarray]],
	Optional[List[np.ndarray]],
	List[float],
	List[float],
]:
	segments_X: List[np.ndarray] = []
	segments_U: List[np.ndarray] = []
	segments_V: List[np.ndarray] = []
	segments_ref_orientation: List[np.ndarray] = []
	segments_ref_curvature: List[np.ndarray] = []
	segments_target_lat_m: List[float] = []
	segments_dt_s: List[float] = []

	use_ref = (traj_ref_orientation_raw is not None) and (traj_ref_curvature_raw is not None)
	if use_ref:
		if len(traj_ref_orientation_raw) != len(traj_Xn) or len(traj_ref_curvature_raw) != len(traj_Xn):
			raise ValueError("Reference trajectory lists must match traj_Xn length")

	seg_len = int(seg_len)
	segment_stride = int(segment_stride)
	for i, (Xn, Un, Vraw, Xraw) in enumerate(zip(traj_Xn, traj_Un, traj_Vraw, traj_Xraw)):
		n = int(len(Xn))
		if n < seg_len:
			continue
		target_lat_m = float(np.asarray(Xraw)[-1, 0])
		traj_dt = float(dt_s)
		for start in range(0, n - seg_len + 1, segment_stride):
			end = start + seg_len
			Xseg = np.asarray(Xn)[start:end].copy()
			Useg = np.asarray(Un)[start : end - 1].copy().reshape(-1, 1)
			Vseg = np.asarray(Vraw)[start:end].copy()
			if len(Xseg) != seg_len or len(Useg) != (seg_len - 1) or len(Vseg) != seg_len:
				continue
			if use_ref:
				ref_ori_seg = np.asarray(traj_ref_orientation_raw[i], dtype=float)[start:end].copy()
				ref_kappa_seg = np.asarray(traj_ref_curvature_raw[i], dtype=float)[start:end].copy()
				if len(ref_ori_seg) != seg_len or len(ref_kappa_seg) != seg_len:
					continue
				segments_ref_orientation.append(ref_ori_seg)
				segments_ref_curvature.append(ref_kappa_seg)
			segments_X.append(Xseg)
			segments_U.append(Useg)
			segments_V.append(Vseg)
			segments_target_lat_m.append(target_lat_m)
			segments_dt_s.append(traj_dt)

	ref_ori_out = segments_ref_orientation if use_ref else None
	ref_kappa_out = segments_ref_curvature if use_ref else None
	return (
		segments_X,
		segments_U,
		segments_V,
		ref_ori_out,
		ref_kappa_out,
		segments_target_lat_m,
		segments_dt_s,
	)


def prepare_train_test_segments_from_merged_csv(
	*,
	merged_csv: Path,
	n_traj: int,
	min_traj_len: int,
	train_ratio: float,
	split_seed: int,
	seg_len: int,
	segment_stride: Optional[int],
) -> dict:
	(
		traj_X_raw,
		traj_U_raw,
		traj_V_raw,
		traj_ref_orientation_raw,
		traj_ref_curvature_raw,
		traj_ids,
		dt_s,
	) = load_trajectories_from_merged_csv(
		Path(merged_csv),
		n_traj=int(n_traj),
		min_traj_len=int(min_traj_len),
		return_ids=True,
		use_reference_columns=True,
	)

	n_total = len(traj_X_raw)
	perm = np.random.default_rng(int(split_seed)).permutation(n_total)
	n_test = int(np.round((1.0 - float(train_ratio)) * n_total))
	n_test = max(1, n_test)
	n_train = n_total - n_test
	if n_train < 1:
		n_train, n_test = n_total - 1, 1
	train_idx = perm[:n_train]
	test_idx = perm[n_train:]

	X_train_raw = [traj_X_raw[i] for i in train_idx]
	U_train_raw = [traj_U_raw[i] for i in train_idx]
	V_train_raw = [traj_V_raw[i] for i in train_idx]
	ref_ori_train_raw = [traj_ref_orientation_raw[i] for i in train_idx]
	ref_kappa_train_raw = [traj_ref_curvature_raw[i] for i in train_idx]
	ids_train = [traj_ids[i] for i in train_idx]

	X_test_raw = [traj_X_raw[i] for i in test_idx]
	U_test_raw = [traj_U_raw[i] for i in test_idx]
	V_test_raw = [traj_V_raw[i] for i in test_idx]
	ref_ori_test_raw = [traj_ref_orientation_raw[i] for i in test_idx]
	ref_kappa_test_raw = [traj_ref_curvature_raw[i] for i in test_idx]

	X_all = np.vstack(X_train_raw)
	U_all = np.vstack(U_train_raw)
	X_mean, X_std = X_all.mean(axis=0), X_all.std(axis=0) + 1e-9
	U_mean, U_std = U_all.mean(axis=0), U_all.std(axis=0) + 1e-9

	traj_Xn_train = [(X - X_mean) / X_std for X in X_train_raw]
	traj_Un_train = [(U - U_mean) / U_std for U in U_train_raw]
	traj_Xn_test = [(X - X_mean) / X_std for X in X_test_raw]
	traj_Un_test = [(U - U_mean) / U_std for U in U_test_raw]

	if segment_stride is None:
		segment_stride = int(seg_len)
	segment_stride = int(segment_stride)
	if segment_stride <= 0:
		raise ValueError("segment_stride must be positive")

	(
		segments_X,
		segments_U,
		segments_V,
		segments_ref_orientation,
		segments_ref_curvature,
		segments_target_lat_m,
		segments_dt_s,
	) = build_segments_from_trajs(
		traj_Xn_train,
		traj_Un_train,
		V_train_raw,
		X_train_raw,
		ref_ori_train_raw,
		ref_kappa_train_raw,
		seg_len=int(seg_len),
		segment_stride=int(segment_stride),
		dt_s=float(dt_s),
	)
	(
		segments_X_test,
		segments_U_test,
		segments_V_test,
		segments_ref_orientation_test,
		segments_ref_curvature_test,
		segments_target_lat_m_test,
		segments_dt_s_test,
	) = build_segments_from_trajs(
		traj_Xn_test,
		traj_Un_test,
		V_test_raw,
		X_test_raw,
		ref_ori_test_raw,
		ref_kappa_test_raw,
		seg_len=int(seg_len),
		segment_stride=int(segment_stride),
		dt_s=float(dt_s),
	)

	if len(segments_X) == 0:
		raise RuntimeError("No train segments built")
	if segments_ref_orientation is None or segments_ref_curvature is None:
		raise RuntimeError("Missing reference segments")
	if segments_ref_orientation_test is None or segments_ref_curvature_test is None:
		raise RuntimeError("Missing test reference segments")

	return {
		"traj_X_raw": traj_X_raw,
		"traj_ids": traj_ids,
		"ids_train": ids_train,
		"dt_s": float(dt_s),
		"X_mean": np.asarray(X_mean, dtype=float),
		"X_std": np.asarray(X_std, dtype=float),
		"U_mean": np.asarray(U_mean, dtype=float),
		"U_std": np.asarray(U_std, dtype=float),
		"segments_X": segments_X,
		"segments_U": segments_U,
		"segments_V": segments_V,
		"segments_ref_orientation": segments_ref_orientation,
		"segments_ref_curvature": segments_ref_curvature,
		"segments_target_lat_m": segments_target_lat_m,
		"segments_dt_s": segments_dt_s,
		"segments_X_test": segments_X_test,
		"segments_U_test": segments_U_test,
		"segments_V_test": segments_V_test,
		"segments_ref_orientation_test": segments_ref_orientation_test,
		"segments_ref_curvature_test": segments_ref_curvature_test,
		"segments_target_lat_m_test": segments_target_lat_m_test,
		"segments_dt_s_test": segments_dt_s_test,
		"segment_stride": int(segment_stride),
	}


def koopman_predict_next_xn(
	xk_n: np.ndarray,
	uk_n: np.ndarray,
	*,
	Kx: np.ndarray,
	Ku: np.ndarray,
	C: np.ndarray,
	lift_np,
) -> np.ndarray:
	psi_k = lift_np(xk_n)
	psi_k1 = (Kx @ psi_k.reshape(-1, 1)) + (Ku @ uk_n.reshape(-1, 1))
	xk1_hat = (C @ psi_k1).reshape(-1)
	return xk1_hat


def eval_koopman_one_step_mse(
	segments_X: List[np.ndarray],
	segments_U: List[np.ndarray],
	*,
	Kx: np.ndarray,
	Ku: np.ndarray,
	C: np.ndarray,
	lift_np,
	max_segments: int = 0,
) -> dict:
	n_used = 0
	se_sum = None
	for Xseg, Useg in zip(segments_X, segments_U):
		N = len(Xseg) - 1
		for k in range(N):
			xk = Xseg[k]
			uk = Useg[k]
			x_true = Xseg[k + 1]
			x_hat = koopman_predict_next_xn(xk, uk, Kx=Kx, Ku=Ku, C=C, lift_np=lift_np)
			err = (x_hat - x_true).reshape(-1)
			if se_sum is None:
				se_sum = np.zeros_like(err, dtype=float)
			se_sum += err * err
			n_used += 1
		if max_segments and n_used > 0:
			if (n_used // max(1, N)) >= max_segments:
				break
	if n_used == 0 or se_sum is None:
		return {"n_transitions": 0, "mse_by_state": None, "mse_mean": None}
	mse_by_state = (se_sum / float(n_used)).tolist()
	mse_mean = float(np.mean(se_sum / float(n_used)))
	return {"n_transitions": int(n_used), "mse_by_state": mse_by_state, "mse_mean": mse_mean}


def eval_koopman_rollout_rmse(
	segments_X: List[np.ndarray],
	segments_U: List[np.ndarray],
	*,
	Kx: np.ndarray,
	Ku: np.ndarray,
	C: np.ndarray,
	lift_np,
	rollout_h: int = 10,
	max_segments: int = 200,
) -> dict:
	rollout_h = int(rollout_h)
	if rollout_h < 1:
		raise ValueError("rollout_h must be >= 1")
	used = 0
	se_sum = None
	count = 0
	for Xseg, Useg in zip(segments_X, segments_U):
		if used >= int(max_segments):
			break
		x = np.asarray(Xseg[0]).copy()
		H = min(rollout_h, len(Useg), len(Xseg) - 1)
		for k in range(H):
			uk = Useg[k]
			x = koopman_predict_next_xn(x, uk, Kx=Kx, Ku=Ku, C=C, lift_np=lift_np)
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
	rmse_by_state = np.sqrt(mse).tolist()
	rmse_mean = float(np.sqrt(np.mean(mse)))
	return {"n_steps": int(count), "rmse_by_state": rmse_by_state, "rmse_mean": rmse_mean}


def learn_koopman_dynamics(
	*,
	segments_X: List[np.ndarray],
	segments_U: List[np.ndarray],
	lift: str,
	n_state: int,
	degree: int,
	dnn_psi_dim: int,
	dnn_hidden_dim: int,
	dnn_hidden_layers: int,
	dnn_pretrain_steps: int,
	dnn_pretrain_segments: int,
	dnn_lr: float,
	dnn_batch_segments: int,
	dnn_refit_every: int,
	split_seed: int,
	reg: float,
) -> dict:
	lift = str(lift).strip().lower()
	if lift == "poly":
		_, lift_np, psi_batch_np, dpsi_dx_np, _ = make_polynomial_lift(n_state=int(n_state), degree=int(degree))
		lift_obj = None
	elif lift == "dnn":
		lift_obj = make_dnn_lift(
			n_state=int(n_state),
			psi_dim=int(dnn_psi_dim),
			hidden_dim=int(dnn_hidden_dim),
			hidden_layers=int(dnn_hidden_layers),
			seed=int(split_seed),
		)
		lift_np = lift_obj.lift_np
		psi_batch_np = lift_obj.psi_batch_np
		dpsi_dx_np = lift_obj.dpsi_dx_np
	else:
		raise ValueError("lift must be 'poly' or 'dnn'")

	Kx, Ku, C, Pz, Ppsi = koopman_init_woodbury(segments_X[0], segments_U[0], psi_batch_np, reg=float(reg))

	if lift == "dnn" and int(dnn_pretrain_steps) > 0:
		seg_n = min(int(dnn_pretrain_segments), len(segments_X))
		dnn_pretrain_dkr(
			segments_X[:seg_n],
			segments_U[:seg_n],
			lift=lift_obj,
			Kx=Kx,
			Ku=Ku,
			C=C,
			steps=int(dnn_pretrain_steps),
			lr=float(dnn_lr),
			batch_segments=int(dnn_batch_segments),
			seed=int(split_seed),
			refit_every=int(dnn_refit_every),
			reg=float(reg),
		)
		Kx, Ku, C, Pz, Ppsi = koopman_init_woodbury(segments_X[0], segments_U[0], psi_batch_np, reg=float(reg))

	for it in range(1, len(segments_X)):
		Kx, Ku, C, Pz, Ppsi = koopman_update_woodbury(
			Kx,
			Ku,
			C,
			Pz,
			Ppsi,
			segments_X[it],
			segments_U[it],
			psi_batch_np,
			reg=float(reg),
		)

	return {
		"Kx": Kx,
		"Ku": Ku,
		"C": C,
		"lift_np": lift_np,
		"psi_batch_np": psi_batch_np,
		"dpsi_dx_np": dpsi_dx_np,
	}

