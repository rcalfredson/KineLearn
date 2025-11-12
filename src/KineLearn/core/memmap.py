from collections import defaultdict
import os
from pathlib import Path
from typing import Any, Dict, Tuple


import numpy as np
import pandas as pd


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
    keep_tail = window_size - stride
    seen = defaultdict(int)
    remnants: Dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    vids, starts = [], []
    out_prefix = str(out_prefix)
    out_dir = Path(out_prefix).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pre-allocate generously; will reopen with true shape later
    max_windows = len(X)  # conservative upper bound
    mmX = np.memmap(
        f"{out_prefix}_features.fp32",
        mode="w+",
        dtype="float32",
        shape=(max_windows, window_size, derived_dim),
    )
    mmY = np.memmap(
        f"{out_prefix}_labels.u8",
        mode="w+",
        dtype="uint8",
        shape=(max_windows, window_size, n_classes),
    )
    offset = 0

    # Debug counters
    stem_debug_rows: list[Dict[str, Any]] = []
    zero_window_count = 0

    for stem, grp_feat in X.groupby("__stem__"):
        grp_lab = Y.loc[grp_feat.index]
        # Append any leftover tail
        if stem in remnants:
            t_feat, t_lab = remnants.pop(stem)
            grp_feat = pd.concat([t_feat, grp_feat], ignore_index=True)
            grp_lab = pd.concat([t_lab, grp_lab], ignore_index=True)
            seen[stem] -= t_feat.shape[0]

        feat_np = grp_feat.drop(columns=["__stem__"]).to_numpy(np.float32, copy=False)
        lab_np = grp_lab.drop(columns=["__stem__"]).to_numpy(np.uint8, copy=False)
        n_frames = len(feat_np)
        offset_local = seen[stem]

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
            starts.append(offset_local + s)
            offset += 1

        seen[stem] += n_frames
        tail_len = min(keep_tail, n_frames)
        remnants[stem] = (grp_feat.tail(tail_len), grp_lab.tail(tail_len))

    # Resize to true count (truncate the files)
    mmX.flush()
    mmY.flush()
    feat_path = f"{out_prefix}_features.fp32"
    lab_path = f"{out_prefix}_labels.u8"

    # Close the writable memmaps before truncation
    del mmX
    del mmY

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
