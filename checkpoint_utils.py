"""
Checkpoint utilities for saving and restarting G-equation simulations.

This module provides lightweight, portable save/restart helpers that serialize the
minimal solver state needed to resume a simulation later.

Design goals:
- Numpy-based, no pickling of runtime objects or callables
- Save the fields (G, u_x, u_y, S_L), grid/domain, and current time
- Recreate a GEquationSolver2D instance on load with identical dimensions
- Pinned (Dirichlet) regions are preserved if configured on the solver

Limitations:
- velocity_updater functions are not serialized; provide again on restart
- Reinitialization scheduling alignment may change if reinit_interval does not
  divide the remaining steps; behavior is otherwise equivalent
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from g_equation_solver_improved import GEquationSolver2D


@dataclass
class CheckpointMeta:
    nx: int
    ny: int
    Lx: float
    Ly: float
    # Optional run parameters (informational; not enforced during load)
    dt: Optional[float] = None
    time_scheme: Optional[str] = None
    reinit_interval: Optional[int] = None
    reinit_method: Optional[str] = None
    reinit_local: Optional[bool] = None
    save_interval: Optional[int] = None
    notes: Optional[str] = None
    # Extra free-form metadata for exact restarts (e.g., velocity forcing parameters)
    extra: Optional[Dict[str, Any]] = None

    def to_json(self) -> str:
        return json.dumps(self.__dict__)

    @staticmethod
    def from_json(s: str) -> "CheckpointMeta":
        data = json.loads(s)
        return CheckpointMeta(**data)


def save_checkpoint(
    path: str,
    solver: GEquationSolver2D,
    G: np.ndarray,
    t: float,
    meta: Optional[CheckpointMeta] = None,
) -> None:
    """
    Save the current simulation state to a compressed NPZ file.

    Parameters:
    - path: output .npz filepath
    - solver: the solver instance whose fields and grid are used
    - G: current level-set field to save
    - t: current physical time
    - meta: optional metadata about the run (dt, scheme, reinit settings)
    """
    # Ensure arrays
    G_arr = np.asarray(G)
    u_x = np.asarray(solver.u_x)
    u_y = np.asarray(solver.u_y)

    # Normalize S_L to an array for consistency
    if np.isscalar(solver.S_L):
        S_L_arr = np.full((solver.ny, solver.nx), float(solver.S_L))
    else:
        S_L_arr = np.asarray(solver.S_L)

    # Pinned region (optional)
    pinned_mask = getattr(solver, "_pinned_mask", None)
    pinned_values = getattr(solver, "_pinned_values", None)
    if pinned_mask is None:
        pinned_mask = np.zeros_like(G_arr, dtype=bool)
        pinned_values = np.zeros_like(G_arr, dtype=float)

    meta_obj = meta or CheckpointMeta(
        nx=solver.nx, ny=solver.ny, Lx=solver.Lx, Ly=solver.Ly
    )

    np.savez_compressed(
        path,
        G=G_arr,
        t=np.float64(t),
        u_x=u_x,
        u_y=u_y,
        S_L=S_L_arr,
        nx=np.int32(solver.nx),
        ny=np.int32(solver.ny),
        Lx=np.float64(solver.Lx),
        Ly=np.float64(solver.Ly),
        pinned_mask=pinned_mask.astype(bool),
        pinned_values=np.asarray(pinned_values, dtype=float),
        meta_json=meta_obj.to_json(),
    )


def load_checkpoint(path: str) -> Tuple[GEquationSolver2D, np.ndarray, float, CheckpointMeta]:
    """
    Load a checkpoint and reconstruct a solver plus the saved field/time.

    Returns (solver, G, t, meta)
    """
    data = np.load(path, allow_pickle=False)

    nx = int(data["nx"]) if "nx" in data else int(data["G"].shape[1])
    ny = int(data["ny"]) if "ny" in data else int(data["G"].shape[0])
    Lx = float(data["Lx"]) if "Lx" in data else 1.0
    Ly = float(data["Ly"]) if "Ly" in data else 1.0

    G = np.array(data["G"], dtype=float)
    t = float(np.array(data["t"]))
    u_x = np.array(data["u_x"], dtype=float)
    u_y = np.array(data["u_y"], dtype=float)
    S_L_arr = np.array(data["S_L"], dtype=float)

    meta_json = data["meta_json"].item() if "meta_json" in data else json.dumps({})
    meta = CheckpointMeta.from_json(meta_json) if meta_json else CheckpointMeta(nx=nx, ny=ny, Lx=Lx, Ly=Ly)

    # Initialize solver; pass a representative scalar S_L then set array
    solver = GEquationSolver2D(nx=nx, ny=ny, Lx=Lx, Ly=Ly, S_L=float(S_L_arr.mean()), u_x=u_x, u_y=u_y)
    # If S_L was spatially varying, restore the full array
    if not np.allclose(S_L_arr, S_L_arr.mean()):
        solver.S_L = S_L_arr.copy()

    # Restore pinned region if present
    if "pinned_mask" in data and "pinned_values" in data:
        pinned_mask = np.array(data["pinned_mask"], dtype=bool)
        pinned_values = np.array(data["pinned_values"], dtype=float)
        if pinned_mask.any():
            solver.set_pinned_region(pinned_mask, pinned_values)

    return solver, G, t, meta


def restart_solve(
    checkpoint_path: str,
    t_final: float,
    dt: float,
    *,
    save_interval: Optional[int] = None,
    time_scheme: str = "euler",
    reinit_interval: int = 0,
    reinit_method: str = "fast_marching",
    reinit_local: bool = True,
    smooth_ic: bool = False,
    velocity_updater=None,
):
    """
    Convenience wrapper: load a checkpoint and resume the simulation until t_final.

    Returns (G_history, t_history)
    """
    solver, G, t0, _meta = load_checkpoint(checkpoint_path)
    return solver.solve(
        G_initial=G,
        t_final=t_final,
        dt=dt,
        save_interval=save_interval,
        time_scheme=time_scheme,
        reinit_interval=reinit_interval,
        reinit_method=reinit_method,
        reinit_local=reinit_local,
        smooth_ic=smooth_ic,
        velocity_updater=velocity_updater,
        t0=t0,
    )
