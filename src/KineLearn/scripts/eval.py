#!/usr/bin/env python3
"""
Evaluate one or more single-behavior KineLearn models from training manifests.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from KineLearn.core.models import build_keypoint_bilstm

try:
    import tensorflow as tf
except Exception:
    tf = None

# Path: src/KineLearn/scripts/eval.py


def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def require_keys(d: dict, keys: list[str], where: str) -> None:
    missing = [k for k in keys if k not in d]
    if missing:
        raise ValueError(f"Missing keys {missing} in {where}")


def as_path(value: str | Path) -> Path:
    return value if isinstance(value, Path) else Path(value)


def load_manifest(path: Path) -> dict:
    manifest = load_yaml(path)
    require_keys(
        manifest,
        [
            "behavior",
            "behavior_idx",
            "label_columns",
            "window",
            "artifacts",
            "feature_selection",
            "training_run",
        ],
        f"manifest {path}",
    )
    return manifest


def validate_manifests(manifests: list[dict], subset: str) -> None:
    if not manifests:
        raise ValueError("At least one manifest is required.")

    behaviors = [m["behavior"] for m in manifests]
    dupes = sorted({b for b in behaviors if behaviors.count(b) > 1})
    if dupes:
        raise ValueError(f"Duplicate behaviors in evaluation set: {dupes}")

    base = manifests[0]
    shared_fields = [
        ("kl_config", "KineLearn config"),
        ("split", "split file"),
        ("label_columns", "label columns"),
    ]
    for field, label in shared_fields:
        base_val = base.get(field)
        for manifest in manifests[1:]:
            if manifest.get(field) != base_val:
                raise ValueError(f"All manifests must share the same {label}.")

    base_window = base["window"]
    for manifest in manifests[1:]:
        if manifest["window"] != base_window:
            raise ValueError("All manifests must share the same window size/stride.")

    if subset in {"train", "val"}:
        base_training = base.get("training", {})
        for manifest in manifests[1:]:
            training = manifest.get("training", {})
            for field in ("val_fraction", "seed"):
                if training.get(field) != base_training.get(field):
                    raise ValueError(
                        f"All manifests must share training.{field} when evaluating '{subset}'."
                    )


def resolve_weights_path(manifest: dict, manifest_path: Path) -> Path:
    training_run = manifest.get("training_run", {})
    candidates = [
        training_run.get("evaluation_weights"),
        training_run.get("checkpoint_best_model"),
        training_run.get("checkpoint_interrupted_model"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = as_path(candidate)
        if not path.is_absolute():
            path = (manifest_path.parent / path).resolve()
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No usable weights file found for manifest {manifest_path}."
    )


def open_memmap(artifact: dict, key: str) -> np.memmap:
    path = as_path(artifact[f"{key}_path"])
    dtype = artifact[f"{key}_dtype"]
    shape = tuple(int(x) for x in artifact[f"{key}_shape"])
    return np.memmap(path, mode="r", dtype=dtype, shape=shape)


def load_subset_arrays(manifest: dict, subset: str) -> tuple[np.memmap, np.memmap, np.ndarray, np.ndarray]:
    artifact = manifest["artifacts"][subset]
    mmX = open_memmap(artifact, "X")
    mmY = open_memmap(artifact, "Y")
    vids = np.load(artifact["vids_path"], allow_pickle=True)
    starts = np.load(artifact["starts_path"], allow_pickle=True)
    if len(vids) != int(artifact["count"]) or len(starts) != int(artifact["count"]):
        raise ValueError(f"Index array length mismatch for subset '{subset}'.")
    return mmX, mmY, vids, starts


def build_loaded_model(manifest: dict, weights_path: Path) -> "tf.keras.Model":
    if tf is None:
        raise ImportError("TensorFlow is required for evaluation.")
    window_size = int(manifest["window"]["size"])
    input_dim = int(manifest["feature_selection"]["n_input_features"])
    model = build_keypoint_bilstm(window_size, input_dim)
    model.load_weights(str(weights_path))
    return model


def prepare_frame_buffers(
    vids: np.ndarray, starts: np.ndarray, window_size: int
) -> dict[str, dict[str, np.ndarray]]:
    per_stem_max = defaultdict(int)
    for vid, start in zip(vids, starts):
        stem = str(vid)
        per_stem_max[stem] = max(per_stem_max[stem], int(start) + window_size)

    buffers: dict[str, dict[str, np.ndarray]] = {}
    for stem, n_frames in per_stem_max.items():
        buffers[stem] = {
            "prob_sum": np.zeros(n_frames, dtype=np.float64),
            "count": np.zeros(n_frames, dtype=np.int32),
            "true": np.zeros(n_frames, dtype=np.uint8),
        }
    return buffers


def aggregate_predictions(
    model: "tf.keras.Model",
    mmX: np.memmap,
    mmY: np.memmap,
    vids: np.ndarray,
    starts: np.ndarray,
    *,
    behavior_idx: int,
    window_size: int,
    batch_size: int,
) -> dict[str, dict[str, np.ndarray]]:
    buffers = prepare_frame_buffers(vids, starts, window_size)
    n = int(mmX.shape[0])

    for start_idx in range(0, n, batch_size):
        end_idx = min(start_idx + batch_size, n)
        Xb = np.asarray(mmX[start_idx:end_idx], dtype=np.float32)
        yb = np.asarray(mmY[start_idx:end_idx, :, behavior_idx], dtype=np.uint8)
        pred = np.asarray(model.predict_on_batch(Xb), dtype=np.float32)[..., 0]

        for local_idx, global_idx in enumerate(range(start_idx, end_idx)):
            stem = str(vids[global_idx])
            frame_start = int(starts[global_idx])
            frame_end = frame_start + window_size
            buf = buffers[stem]
            buf["prob_sum"][frame_start:frame_end] += pred[local_idx]
            buf["count"][frame_start:frame_end] += 1
            buf["true"][frame_start:frame_end] = np.maximum(
                buf["true"][frame_start:frame_end], yb[local_idx]
            )

    return buffers


def frame_table_from_buffers(
    buffers: dict[str, dict[str, np.ndarray]],
    *,
    behavior: str,
    threshold: float,
) -> pd.DataFrame:
    parts = []
    for stem in sorted(buffers):
        buf = buffers[stem]
        counts = buf["count"]
        valid = counts > 0
        if not np.any(valid):
            continue
        frame_idx = np.flatnonzero(valid)
        probs = buf["prob_sum"][valid] / counts[valid]
        truth = buf["true"][valid].astype(np.uint8)
        pred = (probs >= threshold).astype(np.uint8)
        parts.append(
            pd.DataFrame(
                {
                    "__stem__": stem,
                    "__frame__": frame_idx.astype(np.int32),
                    f"true_{behavior}": truth,
                    f"prob_{behavior}": probs.astype(np.float32),
                    f"pred_{behavior}": pred,
                }
            )
        )
    if not parts:
        raise ValueError(f"No frame predictions were reconstructed for behavior '{behavior}'.")
    return pd.concat(parts, ignore_index=True)


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    y_true = y_true.astype(np.uint8)
    y_pred = y_pred.astype(np.uint8)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
    }


def build_bouts_from_mask(
    mask: np.ndarray, *, min_length: int = 16, max_gap: int = 3
) -> list[tuple[int, int]]:
    bouts = []
    start = None
    gap = 0
    ones = 0

    for i, value in enumerate(mask.astype(np.uint8)):
        if value:
            if start is None:
                start, ones, gap = i, 1, 0
            else:
                ones += 1
                gap = 0
        elif start is not None:
            gap += 1
            if gap > max_gap:
                end = i - gap
                if ones >= min_length:
                    bouts.append((start, end))
                start = None
                gap = 0
                ones = 0

    if start is not None:
        end = len(mask) - 1
        if ones >= min_length:
            bouts.append((start, end))

    return bouts


def compute_bout_level_metrics(
    pred_bouts: list[tuple[int, int]],
    gt_bouts: list[tuple[int, int]],
    *,
    overlap_threshold: float = 0.2,
) -> dict[str, Any]:
    tp = 0
    matched_gt = 0

    for pred_bout in pred_bouts:
        pred_len = pred_bout[1] - pred_bout[0] + 1
        matched = False
        for gt_bout in gt_bouts:
            overlap = max(
                0, min(pred_bout[1], gt_bout[1]) - max(pred_bout[0], gt_bout[0]) + 1
            )
            if overlap >= overlap_threshold * pred_len:
                matched = True
        if matched:
            tp += 1

    fp = len(pred_bouts) - tp

    for gt_bout in gt_bouts:
        matched = False
        for pred_bout in pred_bouts:
            pred_len = pred_bout[1] - pred_bout[0] + 1
            overlap = max(
                0, min(pred_bout[1], gt_bout[1]) - max(pred_bout[0], gt_bout[0]) + 1
            )
            if overlap >= overlap_threshold * pred_len:
                matched = True
                break
        if matched:
            matched_gt += 1

    fn = len(gt_bouts) - matched_gt
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def identify_bout_errors(
    pred_bouts: list[tuple[int, int]],
    gt_bouts: list[tuple[int, int]],
    *,
    overlap_threshold: float = 0.2,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    matched_pred = set()
    matched_gt = set()

    for i, pred_bout in enumerate(pred_bouts):
        pred_len = pred_bout[1] - pred_bout[0] + 1
        for j, gt_bout in enumerate(gt_bouts):
            overlap = max(
                0, min(pred_bout[1], gt_bout[1]) - max(pred_bout[0], gt_bout[0]) + 1
            )
            if overlap >= overlap_threshold * pred_len:
                matched_pred.add(i)
                matched_gt.add(j)

    fp_bouts = [pred_bouts[i] for i in range(len(pred_bouts)) if i not in matched_pred]
    fn_bouts = [gt_bouts[j] for j in range(len(gt_bouts)) if j not in matched_gt]
    return fp_bouts, fn_bouts


def compute_episode_outputs(
    frame_df: pd.DataFrame,
    *,
    behavior: str,
    min_pred_frames: int,
    max_gap: int,
    overlap_threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metric_rows = []
    error_rows = []

    for stem, group in frame_df.groupby("__stem__", sort=True):
        pred_mask = group[f"pred_{behavior}"].to_numpy(dtype=np.uint8)
        true_mask = group[f"true_{behavior}"].to_numpy(dtype=np.uint8)

        pred_bouts = build_bouts_from_mask(
            pred_mask, min_length=min_pred_frames, max_gap=max_gap
        )
        gt_bouts = build_bouts_from_mask(true_mask, min_length=1, max_gap=0)

        metrics = compute_bout_level_metrics(
            pred_bouts, gt_bouts, overlap_threshold=overlap_threshold
        )
        metric_rows.append(metrics)

        fp_bouts, fn_bouts = identify_bout_errors(
            pred_bouts, gt_bouts, overlap_threshold=overlap_threshold
        )
        for start, end in fp_bouts:
            error_rows.append(
                {
                    "__stem__": stem,
                    "behavior": behavior,
                    "level": "episode",
                    "error_type": "false_positive",
                    "start_frame": int(group["__frame__"].iloc[start]),
                    "end_frame": int(group["__frame__"].iloc[end]),
                }
            )
        for start, end in fn_bouts:
            error_rows.append(
                {
                    "__stem__": stem,
                    "behavior": behavior,
                    "level": "episode",
                    "error_type": "false_negative",
                    "start_frame": int(group["__frame__"].iloc[start]),
                    "end_frame": int(group["__frame__"].iloc[end]),
                }
            )

    total_tp = int(sum(row["tp"] for row in metric_rows))
    total_fp = int(sum(row["fp"] for row in metric_rows))
    total_fn = int(sum(row["fn"] for row in metric_rows))
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    episode_metrics = {
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "n_predicted_episodes": int(total_tp + total_fp),
        "n_true_episodes": int(total_tp + total_fn),
    }
    return episode_metrics, error_rows


def evaluate_manifest(
    manifest: dict,
    manifest_path: Path,
    *,
    subset: str,
    threshold: float,
    batch_size: int | None,
    level: str,
    episode_min_frames: int,
    episode_max_gap: int,
    episode_overlap_threshold: float,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    behavior = manifest["behavior"]
    behavior_idx = int(manifest["behavior_idx"])
    window_size = int(manifest["window"]["size"])
    weights_path = resolve_weights_path(manifest, manifest_path)
    model = build_loaded_model(manifest, weights_path)
    mmX, mmY, vids, starts = load_subset_arrays(manifest, subset)

    eval_batch_size = batch_size or int(manifest.get("training", {}).get("batch_size", 8))
    buffers = aggregate_predictions(
        model,
        mmX,
        mmY,
        vids,
        starts,
        behavior_idx=behavior_idx,
        window_size=window_size,
        batch_size=eval_batch_size,
    )
    frame_df = frame_table_from_buffers(buffers, behavior=behavior, threshold=threshold)
    metric_rows = []
    error_rows: list[dict[str, Any]] = []

    if level in {"frame", "both"}:
        frame_metrics = compute_binary_metrics(
            frame_df[f"true_{behavior}"].to_numpy(),
            frame_df[f"pred_{behavior}"].to_numpy(),
        )
        frame_metrics.update(
            {
                "behavior": behavior,
                "subset": subset,
                "level": "frame",
                "threshold": float(threshold),
                "n_frames": int(len(frame_df)),
                "n_positive_frames": int(frame_df[f"true_{behavior}"].sum()),
                "manifest_path": str(manifest_path.resolve()),
                "weights_path": str(weights_path.resolve()),
            }
        )
        metric_rows.append(frame_metrics)

    if level in {"episode", "both"}:
        episode_metrics, episode_errors = compute_episode_outputs(
            frame_df,
            behavior=behavior,
            min_pred_frames=episode_min_frames,
            max_gap=episode_max_gap,
            overlap_threshold=episode_overlap_threshold,
        )
        episode_metrics.update(
            {
                "behavior": behavior,
                "subset": subset,
                "level": "episode",
                "threshold": float(threshold),
                "episode_min_frames": int(episode_min_frames),
                "episode_max_gap": int(episode_max_gap),
                "episode_overlap_threshold": float(episode_overlap_threshold),
                "manifest_path": str(manifest_path.resolve()),
                "weights_path": str(weights_path.resolve()),
            }
        )
        metric_rows.append(episode_metrics)
        error_rows.extend(episode_errors)

    return frame_df, metric_rows, error_rows


def merge_behavior_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    merged = frames[0]
    for frame_df in frames[1:]:
        merged = merged.merge(frame_df, on=["__stem__", "__frame__"], how="outer")
    merged = merged.sort_values(["__stem__", "__frame__"]).reset_index(drop=True)
    return merged


def build_summary(
    manifests: list[dict],
    manifest_paths: list[Path],
    metrics_rows: list[dict[str, Any]],
    *,
    subset: str,
    level: str,
    threshold: float,
    episode_min_frames: int,
    episode_max_gap: int,
    episode_overlap_threshold: float,
    out_dir: Path,
) -> dict[str, Any]:
    frame_rows = [row for row in metrics_rows if row["level"] == "frame"]
    episode_rows = [row for row in metrics_rows if row["level"] == "episode"]

    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "subset": subset,
        "level": level,
        "threshold": float(threshold),
        "episode_settings": {
            "min_pred_frames": int(episode_min_frames),
            "max_gap": int(episode_max_gap),
            "overlap_threshold": float(episode_overlap_threshold),
        },
        "out_dir": str(out_dir.resolve()),
        "kl_config": manifests[0]["kl_config"],
        "split": manifests[0]["split"],
        "manifests": [str(path.resolve()) for path in manifest_paths],
        "behaviors": sorted({row["behavior"] for row in metrics_rows}),
        "frame_level_metrics": (
            {
                "macro_precision": float(np.mean([row["precision"] for row in frame_rows])),
                "macro_recall": float(np.mean([row["recall"] for row in frame_rows])),
                "macro_f1": float(np.mean([row["f1"] for row in frame_rows])),
                "macro_accuracy": float(np.mean([row["accuracy"] for row in frame_rows])),
            }
            if frame_rows
            else None
        ),
        "episode_level_metrics": (
            {
                "macro_precision": float(np.mean([row["precision"] for row in episode_rows])),
                "macro_recall": float(np.mean([row["recall"] for row in episode_rows])),
                "macro_f1": float(np.mean([row["f1"] for row in episode_rows])),
            }
            if episode_rows
            else None
        ),
        "per_behavior": metrics_rows,
    }


def default_out_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("results") / "evaluations" / timestamp


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate one or more KineLearn single-behavior models."
    )
    parser.add_argument(
        "--manifest",
        action="append",
        required=True,
        help="Path to a train_manifest.yml file. Provide once per behavior model.",
    )
    parser.add_argument(
        "--subset",
        choices=["train", "val", "test"],
        default="test",
        help="Which dataset subset to evaluate (default: test).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for frame-level binary predictions (default: 0.5).",
    )
    parser.add_argument(
        "--level",
        choices=["frame", "episode", "both"],
        default="frame",
        help="Evaluation level to report (default: frame).",
    )
    parser.add_argument(
        "--episode-min-frames",
        type=int,
        default=16,
        help="Minimum positive frames required to keep a predicted episode (default: 16).",
    )
    parser.add_argument(
        "--episode-max-gap",
        type=int,
        default=3,
        help="Maximum internal gap of negative frames allowed within a predicted episode (default: 3).",
    )
    parser.add_argument(
        "--episode-overlap-threshold",
        type=float,
        default=0.2,
        help="Minimum overlap fraction of a predicted episode required to match ground truth (default: 0.2).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional evaluation batch size override. Defaults to each manifest's training batch size.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory for evaluation artifacts. Defaults to results/evaluations/<timestamp>/",
    )
    args = parser.parse_args()

    if not (0.0 < args.threshold < 1.0):
        raise ValueError("--threshold must be between 0 and 1.")
    if not (0.0 < args.episode_overlap_threshold <= 1.0):
        raise ValueError("--episode-overlap-threshold must be in (0, 1].")
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.episode_min_frames <= 0:
        raise ValueError("--episode-min-frames must be positive.")
    if args.episode_max_gap < 0:
        raise ValueError("--episode-max-gap must be non-negative.")

    manifest_paths = [Path(p) for p in args.manifest]
    manifests = [load_manifest(path) for path in manifest_paths]
    validate_manifests(manifests, args.subset)

    out_dir = Path(args.out) if args.out else default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_tables = []
    metrics_rows = []
    error_rows = []
    for manifest, manifest_path in zip(manifests, manifest_paths):
        frame_df, manifest_metrics, manifest_errors = evaluate_manifest(
            manifest,
            manifest_path,
            subset=args.subset,
            threshold=float(args.threshold),
            batch_size=args.batch_size,
            level=args.level,
            episode_min_frames=args.episode_min_frames,
            episode_max_gap=args.episode_max_gap,
            episode_overlap_threshold=float(args.episode_overlap_threshold),
        )
        frame_tables.append(frame_df)
        metrics_rows.extend(manifest_metrics)
        error_rows.extend(manifest_errors)
        for metrics in manifest_metrics:
            print(
                f"[{metrics['behavior']}:{metrics['level']}] "
                f"precision={metrics['precision']:.4f} "
                f"recall={metrics['recall']:.4f} f1={metrics['f1']:.4f}"
            )

    merged_frames = merge_behavior_frames(frame_tables)
    metrics_df = pd.DataFrame(metrics_rows)
    errors_df = pd.DataFrame(error_rows)
    summary = build_summary(
        manifests,
        manifest_paths,
        metrics_rows,
        subset=args.subset,
        level=args.level,
        threshold=float(args.threshold),
        episode_min_frames=args.episode_min_frames,
        episode_max_gap=args.episode_max_gap,
        episode_overlap_threshold=float(args.episode_overlap_threshold),
        out_dir=out_dir,
    )

    summary_path = out_dir / "eval_summary.yml"
    metrics_path = out_dir / "per_behavior_metrics.csv"
    frames_path = out_dir / "frame_predictions.parquet"
    errors_path = out_dir / "episode_errors.csv"

    with open(summary_path, "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)
    metrics_df.to_csv(metrics_path, index=False)
    merged_frames.to_parquet(frames_path, index=False)
    if args.level in {"episode", "both"}:
        errors_df.to_csv(errors_path, index=False)

    print(f"\n📝 Wrote {summary_path}")
    print(f"📝 Wrote {metrics_path}")
    print(f"📝 Wrote {frames_path}")
    if args.level in {"episode", "both"}:
        print(f"📝 Wrote {errors_path}")


if __name__ == "__main__":
    main()
