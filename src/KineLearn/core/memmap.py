from pathlib import Path
from typing import Any, Dict, Tuple


import numpy as np
import pandas as pd

# Path: src/KineLearn/core/memmap.py


def _n_windows(n_frames: int, window_size: int, stride: int) -> int:
    if n_frames < window_size:
        return 0
    return 1 + (n_frames - window_size) // stride


def make_windowed_memmaps(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    window_size: int,
    stride: int,
    derived_dim: int,
    n_classes: int,
    out_prefix: str,
    debug: bool = False,
) -> Tuple[int, np.memmap, np.memmap, np.ndarray, np.ndarray]:
    """
    Stream frame-level data into windowed np.memmap arrays.
    Returns count, X_mem, Y_mem, vid_indices, start_indices.

    If debug=True, prints/saves per-stem stats and all-zero window counts.
    """
    vids, starts = [], []
    out_prefix = str(out_prefix)
    out_dir = Path(out_prefix).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    missing = []
    for col in ("__stem__", "__frame__"):
        if col not in X.columns:
            missing.append(f"X missing {col}")
        if col not in Y.columns:
            missing.append(f"Y missing {col}")
    if missing:
        raise ValueError(" / ".join(missing))

    # Pre-pass: exact window count
    total_windows = 0
    for stem, grp_feat in X.groupby("__stem__", sort=False):
        n_frames = len(grp_feat)
        total_windows += _n_windows(n_frames, window_size, stride)

    mmX = np.memmap(
        f"{out_prefix}_features.fp32",
        mode="w+",
        dtype="float32",
        shape=(total_windows, window_size, derived_dim),
    )
    mmY = np.memmap(
        f"{out_prefix}_labels.u8",
        mode="w+",
        dtype="uint8",
        shape=(total_windows, window_size, n_classes),
    )
    offset = 0

    # Debug counters
    stem_debug_rows: list[Dict[str, Any]] = []
    zero_window_count = 0

    y_groups = {stem: g for stem, g in Y.groupby("__stem__", sort=False)}
    for stem, grp_feat in X.groupby("__stem__", sort=False):
        try:
            grp_lab = y_groups[stem]
        except KeyError as e:
            raise KeyError(f"Stem '{stem}' present in X but missing in Y") from e

        # Sort both by the per-frame key and drop helper columns
        grp_feat = grp_feat.sort_values("__frame__").reset_index(drop=True)
        grp_lab = grp_lab.sort_values("__frame__").reset_index(drop=True)

        # Sanity check: frame indices match 1:1
        if not np.array_equal(
            grp_feat["__frame__"].to_numpy(), grp_lab["__frame__"].to_numpy()
        ):
            raise ValueError(f"Frame index mismatch within stem {stem}")

        feat_np = grp_feat.drop(columns=["__stem__", "__frame__"]).to_numpy(
            np.float32, copy=False
        )
        lab_np = grp_lab.drop(columns=["__stem__", "__frame__"]).to_numpy(
            np.uint8, copy=False
        )

        n_frames = len(feat_np)

        if debug:
            # per-frame L2 norms (cheap) to detect obviously empty features
            frame_norms = np.linalg.norm(feat_np, axis=1)
            stem_debug_rows.append(
                {
                    "stem": stem,
                    "n_frames": int(n_frames),
                    "min_frame_norm": float(frame_norms.min()) if n_frames else 0.0,
                    "max_frame_norm": float(frame_norms.max()) if n_frames else 0.0,
                    "mean_frame_norm": float(frame_norms.mean()) if n_frames else 0.0,
                }
            )

        for s in range(0, n_frames - window_size + 1, stride):
            winX = feat_np[s : s + window_size]
            winY = lab_np[s : s + window_size]

            # Debug: detect all-zero feature windows
            if debug and not np.any(winX):
                zero_window_count += 1

            mmX[offset] = winX
            mmY[offset] = winY
            vids.append(stem)
            starts.append(int(s))
            offset += 1

    # Resize to true count (truncate the files)
    mmX.flush()
    mmY.flush()
    feat_path = f"{out_prefix}_features.fp32"
    lab_path = f"{out_prefix}_labels.u8"

    # Close the writable memmaps before truncation
    del mmX
    del mmY

    if offset != total_windows:
        raise RuntimeError(
            f"Window count mismatch: prepass={total_windows} wrote={offset}"
        )

    feat_bytes = offset * window_size * derived_dim * np.dtype(np.float32).itemsize
    lab_bytes = offset * window_size * n_classes * np.dtype(np.uint8).itemsize

    with open(feat_path, "r+b") as f:
        f.truncate(feat_bytes)

    with open(lab_path, "r+b") as f:
        f.truncate(lab_bytes)

    # Reopen read-only with the exact final shape

    mmX = np.memmap(
        feat_path, dtype="float32", mode="r", shape=(offset, window_size, derived_dim)
    )

    mmY = np.memmap(
        lab_path, dtype="uint8", mode="r", shape=(offset, window_size, n_classes)
    )

    # Emit debug summary
    if debug:
        dbg_csv = Path(f"{out_prefix}_debug_summary.csv")
        df = pd.DataFrame(stem_debug_rows)
        if not df.empty:
            df = df.sort_values("mean_frame_norm", ascending=True)
        df.to_csv(dbg_csv, index=False)
        print(f"[{out_prefix}] Debug: wrote {dbg_csv}")
        print(
            f"[{out_prefix}] All-zero feature windows: {zero_window_count} / {offset}"
        )

    return offset, mmX, mmY, np.array(vids), np.array(starts)
