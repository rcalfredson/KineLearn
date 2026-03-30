#!/usr/bin/env python3
"""
Run one or more trained KineLearn models on arbitrary feature files.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import yaml

try:
    from KineLearn.scripts.eval import (
        build_bouts_from_mask,
        build_loaded_model,
        load_manifest,
        merge_behavior_frames,
        resolve_weights_path,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from KineLearn.scripts.eval import (
        build_bouts_from_mask,
        build_loaded_model,
        load_manifest,
        merge_behavior_frames,
        resolve_weights_path,
    )


def available_feature_stems(features_dir: Path) -> list[str]:
    prefix = "frame_features_"
    suffix = ".parquet"
    stems = []
    for path in sorted(features_dir.glob(f"{prefix}*{suffix}")):
        name = path.name
        stems.append(name[len(prefix) : -len(suffix)])
    return stems


def resolve_requested_stems(
    requested: list[str], available: list[str], *, where: str
) -> list[str]:
    available_set = set(available)
    resolved: list[str] = []
    for stem in requested:
        if stem in available_set:
            resolved.append(stem)
            continue
        matches = [cand for cand in available if cand.endswith(stem)]
        if len(matches) == 1:
            resolved.append(matches[0])
            continue
        if not matches:
            raise ValueError(
                f"{where}: could not resolve stem '{stem}' against available feature files."
            )
        raise ValueError(
            f"{where}: stem '{stem}' matched multiple feature files: {matches}"
        )
    return resolved


def load_video_stems(video_list_path: Path) -> list[str]:
    with open(video_list_path, "r") as f:
        video_paths = yaml.safe_load(f)
    if not isinstance(video_paths, list) or not all(isinstance(v, str) for v in video_paths):
        raise ValueError(f"{video_list_path} must be a YAML list of video paths.")
    return [Path(v).stem for v in video_paths]


def zero_fill_remaining_nans(
    df: pd.DataFrame, *, df_name: str, helper_columns: tuple[str, ...] = ()
) -> pd.DataFrame:
    value_columns = [c for c in df.columns if c not in helper_columns]
    nan_count = int(df[value_columns].isna().sum().sum())
    if nan_count > 0:
        print(f"⚠️  Final zero-fill parity step on {df_name}: replacing {nan_count} NaNs.")
        df = df.copy()
        df[value_columns] = df[value_columns].fillna(0)
    return df


def default_out_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("results") / "inference" / timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one or more trained KineLearn single-behavior models on arbitrary "
            "frame_features_*.parquet files."
        )
    )
    parser.add_argument(
        "--manifest",
        action="append",
        required=True,
        help="Path to a train_manifest.yml file. Provide once per behavior model.",
    )
    parser.add_argument(
        "--features-dir",
        default="features",
        help="Directory containing frame_features_*.parquet files (default: features).",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--video-list",
        help="Optional YAML list of video paths to select stems for prediction.",
    )
    source.add_argument(
        "--stems",
        nargs="+",
        help="Optional list of feature stems to predict on.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional probability threshold for binary frame predictions and bout output.",
    )
    parser.add_argument(
        "--episode-min-frames",
        type=int,
        default=16,
        help="Minimum positive frames required to keep a predicted bout (default: 16).",
    )
    parser.add_argument(
        "--episode-max-gap",
        type=int,
        default=3,
        help="Maximum internal gap of negative frames allowed within a predicted bout (default: 3).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for prediction windows (default: 256).",
    )
    parser.add_argument(
        "--write-csv",
        action="store_true",
        help="Also export frame-level predictions as CSV in addition to Parquet.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory. Defaults to results/inference/<timestamp>/",
    )
    return parser.parse_args()


def load_features_for_stems(stems: list[str], features_dir: Path) -> pd.DataFrame:
    parts = []
    for stem in stems:
        feat_path = features_dir / f"frame_features_{stem}.parquet"
        if not feat_path.exists():
            raise FileNotFoundError(f"Feature file not found: {feat_path}")
        X = pd.read_parquet(feat_path).copy()
        X["__stem__"] = stem
        X["__frame__"] = np.arange(len(X), dtype=np.int32)
        parts.append(X)
    if not parts:
        raise ValueError("No feature files were loaded.")
    return pd.concat(parts, ignore_index=True)


def align_manifest_features(manifest: dict[str, Any], X: pd.DataFrame) -> pd.DataFrame:
    feature_columns = list(manifest["feature_columns"])
    helper_columns = ("__stem__", "__frame__")

    missing = [c for c in feature_columns if c not in X.columns]
    if missing:
        raise ValueError(
            f"Input features are missing columns required by behavior '{manifest['behavior']}': {missing}"
        )

    aligned = X.loc[:, feature_columns + [c for c in helper_columns if c in X.columns]].copy()
    training_cfg = manifest.get("training", {}) or {}
    if training_cfg.get("final_zero_fill", False):
        aligned = zero_fill_remaining_nans(
            aligned,
            df_name=f"X_infer[{manifest['behavior']}]",
            helper_columns=helper_columns,
        )
    return aligned


def build_window_arrays(
    X: pd.DataFrame,
    *,
    window_size: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    windows = []
    vids = []
    starts = []

    for stem, group in X.groupby("__stem__", sort=False):
        group = group.sort_values("__frame__").reset_index(drop=True)
        feat_np = group.drop(columns=["__stem__", "__frame__"]).to_numpy(np.float32, copy=False)
        n_frames = len(feat_np)
        for start in range(0, n_frames - window_size + 1, stride):
            windows.append(feat_np[start : start + window_size])
            vids.append(stem)
            starts.append(int(start))

    if not windows:
        raise ValueError(
            f"No prediction windows were created. Check window size/stride against the feature files."
        )

    return (
        np.stack(windows, axis=0).astype(np.float32, copy=False),
        np.array(vids, dtype=object),
        np.array(starts, dtype=np.int32),
    )


def prepare_prediction_buffers(X: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    buffers: dict[str, dict[str, np.ndarray]] = {}
    for stem, group in X.groupby("__stem__", sort=False):
        n_frames = len(group)
        buffers[str(stem)] = {
            "prob_sum": np.zeros(n_frames, dtype=np.float64),
            "count": np.zeros(n_frames, dtype=np.int32),
            "frame_index": group.sort_values("__frame__")["__frame__"].to_numpy(dtype=np.int32),
        }
    return buffers


def run_window_predictions(
    model: Any,
    X_windows: np.ndarray,
    vids: np.ndarray,
    starts: np.ndarray,
    *,
    batch_size: int,
    window_size: int,
    base_features: pd.DataFrame,
) -> dict[str, dict[str, np.ndarray]]:
    buffers = prepare_prediction_buffers(base_features)
    n = int(X_windows.shape[0])

    for start_idx in range(0, n, batch_size):
        end_idx = min(start_idx + batch_size, n)
        pred = np.asarray(model.predict_on_batch(X_windows[start_idx:end_idx]), dtype=np.float32)[
            ..., 0
        ]
        for local_idx, global_idx in enumerate(range(start_idx, end_idx)):
            stem = str(vids[global_idx])
            frame_start = int(starts[global_idx])
            frame_end = frame_start + window_size
            buf = buffers[stem]
            buf["prob_sum"][frame_start:frame_end] += pred[local_idx]
            buf["count"][frame_start:frame_end] += 1

    return buffers


def frame_table_from_prediction_buffers(
    buffers: dict[str, dict[str, np.ndarray]],
    *,
    behavior: str,
    threshold: float | None,
) -> pd.DataFrame:
    parts = []
    for stem in sorted(buffers):
        buf = buffers[stem]
        valid = buf["count"] > 0
        if not np.any(valid):
            continue
        frame_idx = buf["frame_index"][valid]
        probs = buf["prob_sum"][valid] / buf["count"][valid]
        payload: dict[str, Any] = {
            "__stem__": stem,
            "__frame__": frame_idx.astype(np.int32),
            f"prob_{behavior}": probs.astype(np.float32),
        }
        if threshold is not None:
            payload[f"pred_{behavior}"] = (probs >= threshold).astype(np.uint8)
        parts.append(pd.DataFrame(payload))

    if not parts:
        raise ValueError(f"No frame predictions were reconstructed for behavior '{behavior}'.")
    return pd.concat(parts, ignore_index=True)


def build_bout_table(
    frame_df: pd.DataFrame,
    *,
    behavior: str,
    min_pred_frames: int,
    max_gap: int,
) -> pd.DataFrame:
    parts = []
    pred_col = f"pred_{behavior}"
    prob_col = f"prob_{behavior}"
    for stem, group in frame_df.groupby("__stem__", sort=True):
        group = group.sort_values("__frame__").reset_index(drop=True)
        pred_mask = group[pred_col].to_numpy(dtype=np.uint8)
        bouts = build_bouts_from_mask(pred_mask, min_length=min_pred_frames, max_gap=max_gap)
        for start, end in bouts:
            sub = group.iloc[start : end + 1]
            parts.append(
                {
                    "__stem__": stem,
                    "behavior": behavior,
                    "start_frame": int(sub["__frame__"].iloc[0]),
                    "end_frame": int(sub["__frame__"].iloc[-1]),
                    "n_frames": int(len(sub)),
                    "mean_probability": float(sub[prob_col].mean()),
                    "max_probability": float(sub[prob_col].max()),
                }
            )
    return pd.DataFrame(parts)


def main() -> None:
    args = parse_args()
    if args.threshold is not None and not (0.0 < args.threshold < 1.0):
        raise ValueError("--threshold must be between 0 and 1 when provided.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.episode_min_frames <= 0:
        raise ValueError("--episode-min-frames must be positive.")
    if args.episode_max_gap < 0:
        raise ValueError("--episode-max-gap must be non-negative.")

    features_dir = Path(args.features_dir)
    known_stems = available_feature_stems(features_dir)
    if not known_stems:
        raise FileNotFoundError(
            f"No frame_features_*.parquet files found in features directory: {features_dir}"
        )

    if args.video_list:
        requested_stems = load_video_stems(Path(args.video_list))
        stems = resolve_requested_stems(requested_stems, known_stems, where="video_list")
    elif args.stems:
        stems = resolve_requested_stems(list(args.stems), known_stems, where="stems")
    else:
        stems = known_stems

    manifests = [load_manifest(Path(p)) for p in args.manifest]
    behaviors = [manifest["behavior"] for manifest in manifests]
    duplicates = sorted({b for b in behaviors if behaviors.count(b) > 1})
    if duplicates:
        raise ValueError(f"Duplicate behaviors in prediction set: {duplicates}")

    out_dir = Path(args.out) if args.out else default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    base_features = load_features_for_stems(stems, features_dir)
    frame_tables = []
    bout_tables = []
    manifest_paths = [Path(p).resolve() for p in args.manifest]

    for manifest, manifest_path in zip(manifests, manifest_paths):
        behavior = manifest["behavior"]
        weights_path = resolve_weights_path(manifest, manifest_path)
        model = build_loaded_model(manifest, weights_path)
        X_behavior = align_manifest_features(manifest, base_features)
        window_cfg = manifest["window"]
        window_size = int(window_cfg["size"])
        stride = int(window_cfg["stride"])

        X_windows, vids, starts = build_window_arrays(
            X_behavior,
            window_size=window_size,
            stride=stride,
        )
        buffers = run_window_predictions(
            model,
            X_windows,
            vids,
            starts,
            batch_size=int(args.batch_size),
            window_size=window_size,
            base_features=X_behavior,
        )
        frame_df = frame_table_from_prediction_buffers(
            buffers,
            behavior=behavior,
            threshold=args.threshold,
        )
        frame_tables.append(frame_df)

        if args.threshold is not None:
            bout_df = build_bout_table(
                frame_df,
                behavior=behavior,
                min_pred_frames=int(args.episode_min_frames),
                max_gap=int(args.episode_max_gap),
            )
            if not bout_df.empty:
                bout_tables.append(bout_df)

        print(
            f"[{behavior}] Predicted {frame_df['__stem__'].nunique()} videos "
            f"across {len(frame_df)} reconstructed frames."
        )

    merged_frames = merge_behavior_frames(frame_tables)

    frame_parquet_path = out_dir / "frame_predictions.parquet"
    merged_frames.to_parquet(frame_parquet_path, index=False)
    print(f"📝 Wrote {frame_parquet_path}")

    if args.write_csv:
        frame_csv_path = out_dir / "frame_predictions.csv"
        merged_frames.to_csv(frame_csv_path, index=False)
        print(f"📝 Wrote {frame_csv_path}")

    if bout_tables:
        merged_bouts = pd.concat(bout_tables, ignore_index=True)
        bouts_csv_path = out_dir / "predicted_bouts.csv"
        merged_bouts.to_csv(bouts_csv_path, index=False)
        print(f"📝 Wrote {bouts_csv_path}")

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "features_dir": str(features_dir.resolve()),
        "out_dir": str(out_dir.resolve()),
        "manifests": [str(path) for path in manifest_paths],
        "behaviors": behaviors,
        "n_videos": int(len(stems)),
        "video_stems": list(stems),
        "threshold": float(args.threshold) if args.threshold is not None else None,
        "episode_settings": (
            {
                "min_pred_frames": int(args.episode_min_frames),
                "max_gap": int(args.episode_max_gap),
            }
            if args.threshold is not None
            else None
        ),
        "artifacts": {
            "frame_predictions_parquet": str(frame_parquet_path.resolve()),
            "frame_predictions_csv": (
                str((out_dir / "frame_predictions.csv").resolve()) if args.write_csv else None
            ),
            "predicted_bouts_csv": (
                str((out_dir / "predicted_bouts.csv").resolve()) if bout_tables else None
            ),
        },
    }
    with open(out_dir / "predict_summary.yml", "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)
    print(f"📝 Wrote {out_dir / 'predict_summary.yml'}")


if __name__ == "__main__":
    main()
