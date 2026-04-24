#!/usr/bin/env python3
"""Convert STAR-CCM+ exported CSVs into GeoPT npy triplets.

Input per case:
  - <case>_volume.csv  (must contain coordinates + pressure + velocity)
  - <case>_surface.csv (must contain coordinates + optional normals)

Output per case:
  - x_<id>.npy : (N,7) [x, y, z, sdf_or_0, nx, ny, nz]
  - y_<id>.npy : (N,4) [p, ux, uy, uz]
  - cond_<id>.npy : (C,) condition prompt vector

Usage:
  python tools/starccm_geopt/starccm_csv_to_geopt.py \
    --volume_csv case_001_volume.csv \
    --surface_csv case_001_surface.csv \
    --outdir ./geopt_npys \
    --case_id 1 \
    --cond "Fn=0.26,heel_deg=0.0,yaw_deg=3.0"
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

COMMON_COLUMN_ALIASES = {
    "x": ["Position[0]", "Position[X] (m)", "X (m)"],
    "y": ["Position[1]", "Position[Y] (m)", "Y (m)"],
    "z": ["Position[2]", "Position[Z] (m)", "Z (m)"],
    "ux": ["Velocity[0]", "Velocity[i] (m/s)"],
    "uy": ["Velocity[1]", "Velocity[j] (m/s)"],
    "uz": ["Velocity[2]", "Velocity[k] (m/s)"],
    "p": ["Pressure", "Pressure (Pa)"],
    "nx": ["Normal[0]", "Normal[i]"],
    "ny": ["Normal[1]", "Normal[j]"],
    "nz": ["Normal[2]", "Normal[k]"],
}


def _normalize_colmap(mapping_items: Iterable[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in mapping_items:
        if "=" not in item:
            raise ValueError(f"Invalid --colmap item '{item}', expected alias=column_name")
        alias, col = item.split("=", 1)
        out[alias.strip()] = col.strip()
    return out


def _parse_cond(cond_text: str) -> np.ndarray:
    if not cond_text:
        return np.ones((1,), dtype=np.float32)

    values = []
    for token in cond_text.split(","):
        token = token.strip()
        if not token:
            continue
        if "=" in token:
            _, val = token.split("=", 1)
            values.append(float(val))
        else:
            values.append(float(token))
    if not values:
        return np.ones((1,), dtype=np.float32)
    return np.asarray(values, dtype=np.float32)


def _resolve_column_name(df: pd.DataFrame, alias_key: str, requested_name: str) -> str:
    if requested_name in df.columns:
        return requested_name

    for candidate in COMMON_COLUMN_ALIASES.get(alias_key, []):
        if candidate in df.columns:
            return candidate

    raise KeyError(
        f"CSV missing column for alias '{alias_key}'. "
        f"Tried '{requested_name}' and fallbacks {COMMON_COLUMN_ALIASES.get(alias_key, [])}"
    )


def _read_required_columns(df: pd.DataFrame, colmap: Dict[str, str], keys: Tuple[str, ...]) -> np.ndarray:
    missing = [k for k in keys if k not in colmap]
    if missing:
        raise KeyError(f"Missing aliases in --colmap: {missing}")

    resolved = [_resolve_column_name(df, k, colmap[k]) for k in keys]
    return df[resolved].to_numpy(dtype=np.float32)


def _transform_geometry(surface_xyz: np.ndarray, surface_n: np.ndarray, volume_xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Follow GeoPT DTCHull convention: front -> -X, Y/Z swap.
    s = np.zeros_like(surface_xyz, dtype=np.float32)
    n = np.zeros_like(surface_n, dtype=np.float32)
    v = np.zeros_like(volume_xyz, dtype=np.float32)

    s[:, 0], s[:, 1], s[:, 2] = -surface_xyz[:, 0], surface_xyz[:, 2], surface_xyz[:, 1]
    n[:, 0], n[:, 1], n[:, 2] = -surface_n[:, 0], surface_n[:, 2], surface_n[:, 1]
    v[:, 0], v[:, 1], v[:, 2] = -volume_xyz[:, 0], volume_xyz[:, 2], volume_xyz[:, 1]

    # Shift ground to 0 and center x/z by surface reference.
    ymin = s[:, 1].min()
    s[:, 1] -= ymin
    v[:, 1] -= ymin
    s[:, 0] -= s[:, 0].mean()
    v[:, 0] -= s[:, 0].mean()
    s[:, 2] -= s[:, 2].mean()
    v[:, 2] -= s[:, 2].mean()

    return s, n, v


def _build_signed_distance(volume_xyz: np.ndarray, surface_xyz: np.ndarray, surface_n: np.ndarray) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=1).fit(surface_xyz)
    dists, idx = nn.kneighbors(volume_xyz)
    nearest = surface_xyz[idx[:, 0]]
    nearest_n = surface_n[idx[:, 0]]
    direction = volume_xyz - nearest
    sign = np.sign(np.sum(direction * nearest_n, axis=1, keepdims=True))
    sign[sign == 0] = 1.0
    return (dists * sign).astype(np.float32).reshape(-1, 1)


def convert_one_case(args: argparse.Namespace) -> None:
    os.makedirs(args.outdir, exist_ok=True)

    vol_df = pd.read_csv(args.volume_csv)
    surf_df = pd.read_csv(args.surface_csv)

    colmap = _normalize_colmap(args.colmap)
    vol_xyz = _read_required_columns(vol_df, colmap, ("x", "y", "z"))
    vol_u = _read_required_columns(vol_df, colmap, ("ux", "uy", "uz"))
    vol_p = _read_required_columns(vol_df, colmap, ("p",))

    surf_xyz = _read_required_columns(surf_df, colmap, ("x", "y", "z"))

    if all(k in colmap and colmap[k] in surf_df.columns for k in ("nx", "ny", "nz")):
        surf_n = _read_required_columns(surf_df, colmap, ("nx", "ny", "nz"))
    else:
        # If normals are not exported, use zero normals (SDF sign may be less reliable).
        surf_n = np.zeros_like(surf_xyz, dtype=np.float32)

    s_xyz, s_n, v_xyz = _transform_geometry(surf_xyz, surf_n, vol_xyz)
    sdf = _build_signed_distance(v_xyz, s_xyz, s_n) if args.compute_sdf else np.zeros((v_xyz.shape[0], 1), dtype=np.float32)

    x_volume = np.concatenate([v_xyz, sdf, np.zeros_like(v_xyz)], axis=1)
    x_surface = np.concatenate([s_xyz, np.zeros((s_xyz.shape[0], 1), dtype=np.float32), s_n], axis=1)
    y_volume = np.concatenate([vol_p, vol_u], axis=1).astype(np.float32)
    y_surface = np.zeros((s_xyz.shape[0], 4), dtype=np.float32)

    x = np.concatenate([x_volume, x_surface], axis=0).astype(np.float32)
    y = np.concatenate([y_volume, y_surface], axis=0).astype(np.float32)
    cond = _parse_cond(args.cond)

    np.save(os.path.join(args.outdir, f"x_{args.case_id}.npy"), x)
    np.save(os.path.join(args.outdir, f"y_{args.case_id}.npy"), y)
    np.save(os.path.join(args.outdir, f"cond_{args.case_id}.npy"), cond)

    print(f"[GeoPT] case={args.case_id} done: x={x.shape}, y={y.shape}, cond={cond}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--volume_csv", required=True)
    p.add_argument("--surface_csv", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--case_id", required=True, type=int)
    p.add_argument("--cond", default="", help="Comma-separated prompt values, e.g. 'Fn=0.26,heel_deg=0,yaw_deg=3'")
    p.add_argument("--compute_sdf", action="store_true", help="Compute signed distance for volume points")
    p.add_argument(
        "--colmap",
        nargs="+",
        default=[
            "x=Position[0]", "y=Position[1]", "z=Position[2]",
            "ux=Velocity[0]", "uy=Velocity[1]", "uz=Velocity[2]",
            "p=Pressure",
            "nx=Normal[0]", "ny=Normal[1]", "nz=Normal[2]",
        ],
        help="Column mapping alias=csv_column_name. Required aliases: x,y,z,p,ux,uy,uz (nx,ny,nz optional).",
    )
    return p


if __name__ == "__main__":
    convert_one_case(build_argparser().parse_args())
