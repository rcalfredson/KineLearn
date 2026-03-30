#!/usr/bin/env python3
"""
Plot frame-level behavior probability timelines from KineLearn outputs.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def default_out_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("results") / "plots" / "timeline" / timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot per-video behavior probability timelines from a KineLearn "
            "frame_predictions.parquet/CSV file or an output directory that contains one."
        )
    )
    parser.add_argument(
        "source",
        help=(
            "Path to frame_predictions.parquet, frame_predictions.csv, or a directory "
            "containing one of those files."
        ),
    )
    parser.add_argument(
        "--stems",
        nargs="+",
        help="Optional subset of video stems to plot.",
    )
    parser.add_argument(
        "--behaviors",
        nargs="+",
        help="Optional subset of behaviors to plot. Defaults to all probability columns found.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Optional frames-per-second for plotting the x-axis in seconds instead of frames.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional threshold line to draw. If pred_<behavior> columns are absent, this threshold is also used for predicted bout shading.",
    )
    parser.add_argument(
        "--format",
        choices=["png", "pdf", "both"],
        default="png",
        help="Image format to write (default: png).",
    )
    parser.add_argument(
        "--width",
        type=float,
        default=12.0,
        help="Figure width in inches (default: 12).",
    )
    parser.add_argument(
        "--height-per-behavior",
        type=float,
        default=2.4,
        help="Figure height per behavior subplot in inches (default: 2.4).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory. Defaults to results/plots/timeline/<timestamp>/",
    )
    return parser.parse_args()


def resolve_predictions_path(source: Path) -> Path:
    if source.is_file():
        return source
    if not source.is_dir():
        raise FileNotFoundError(f"Source not found: {source}")

    candidates = [
        source / "frame_predictions.parquet",
        source / "frame_predictions.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No frame_predictions.parquet or frame_predictions.csv found in {source}"
    )


def load_predictions_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported prediction table format: {path}")


def infer_behaviors(frame_df: pd.DataFrame) -> list[str]:
    behaviors = []
    for col in frame_df.columns:
        if col.startswith("prob_"):
            behaviors.append(col[len("prob_") :])
    if not behaviors:
        raise ValueError("No prob_<behavior> columns found in prediction table.")
    return sorted(behaviors)


def build_bouts_from_mask(mask: np.ndarray) -> list[tuple[int, int]]:
    bouts = []
    start = None
    mask = mask.astype(np.uint8)
    for i, value in enumerate(mask):
        if value and start is None:
            start = i
        elif not value and start is not None:
            bouts.append((start, i - 1))
            start = None
    if start is not None:
        bouts.append((start, len(mask) - 1))
    return bouts


def predicted_bouts_for_behavior(
    group: pd.DataFrame,
    *,
    behavior: str,
    threshold: float | None,
) -> list[tuple[int, int]]:
    pred_col = f"pred_{behavior}"
    prob_col = f"prob_{behavior}"
    if pred_col in group.columns:
        mask = group[pred_col].to_numpy(dtype=np.uint8)
        return build_bouts_from_mask(mask)
    if threshold is not None:
        mask = (group[prob_col].to_numpy(dtype=np.float32) >= threshold).astype(np.uint8)
        return build_bouts_from_mask(mask)
    return []


def true_bouts_for_behavior(group: pd.DataFrame, *, behavior: str) -> list[tuple[int, int]]:
    true_col = f"true_{behavior}"
    if true_col not in group.columns:
        return []
    mask = group[true_col].to_numpy(dtype=np.uint8)
    return build_bouts_from_mask(mask)


def x_axis_values(group: pd.DataFrame, *, fps: float | None) -> tuple[np.ndarray, str]:
    frames = group["__frame__"].to_numpy(dtype=np.int32)
    if fps is None:
        return frames, "Frame"
    return frames.astype(np.float64) / fps, "Time (s)"


def plot_video_timeline(
    group: pd.DataFrame,
    *,
    stem: str,
    behaviors: list[str],
    threshold: float | None,
    fps: float | None,
    width: float,
    height_per_behavior: float,
    out_dir: Path,
    fmt: str,
) -> list[Path]:
    group = group.sort_values("__frame__").reset_index(drop=True)
    x, xlabel = x_axis_values(group, fps=fps)

    n_behaviors = len(behaviors)
    fig, axes = plt.subplots(
        n_behaviors,
        1,
        figsize=(width, max(1.8, height_per_behavior * n_behaviors)),
        sharex=True,
    )
    if n_behaviors == 1:
        axes = [axes]

    for ax, behavior in zip(axes, behaviors):
        prob_col = f"prob_{behavior}"
        if prob_col not in group.columns:
            raise ValueError(f"Missing required probability column: {prob_col}")
        probs = group[prob_col].to_numpy(dtype=np.float32)

        for start, end in true_bouts_for_behavior(group, behavior=behavior):
            ax.axvspan(x[start], x[end], color="#9bd3ae", alpha=0.25, linewidth=0)

        for start, end in predicted_bouts_for_behavior(
            group, behavior=behavior, threshold=threshold
        ):
            ax.axvspan(x[start], x[end], color="#f3c17b", alpha=0.25, linewidth=0)

        ax.plot(x, probs, color="#2c5d8a", linewidth=1.5)
        if threshold is not None:
            ax.axhline(threshold, color="#9a3b30", linestyle="--", linewidth=1.0)
        ax.set_ylabel(behavior)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, axis="y", alpha=0.25)

    axes[-1].set_xlabel(xlabel)
    fig.suptitle(stem, y=0.995, fontsize=11)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    if fmt in {"png", "both"}:
        png_path = out_dir / "timeline.png"
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        written.append(png_path)
    if fmt in {"pdf", "both"}:
        pdf_path = out_dir / "timeline.pdf"
        fig.savefig(pdf_path, bbox_inches="tight")
        written.append(pdf_path)
    plt.close(fig)
    return written


def main() -> None:
    args = parse_args()
    if args.threshold is not None and not (0.0 < args.threshold < 1.0):
        raise ValueError("--threshold must be between 0 and 1 when provided.")
    if args.fps is not None and args.fps <= 0:
        raise ValueError("--fps must be positive.")
    if args.width <= 0 or args.height_per_behavior <= 0:
        raise ValueError("--width and --height-per-behavior must be positive.")

    source_path = resolve_predictions_path(Path(args.source))
    frame_df = load_predictions_table(source_path)
    required = {"__stem__", "__frame__"}
    missing = sorted(required - set(frame_df.columns))
    if missing:
        raise ValueError(f"Prediction table is missing required columns: {missing}")

    available_behaviors = infer_behaviors(frame_df)
    if args.behaviors:
        missing_behaviors = [b for b in args.behaviors if b not in available_behaviors]
        if missing_behaviors:
            raise ValueError(
                f"Requested behaviors not present in prediction table: {missing_behaviors}"
            )
        behaviors = list(args.behaviors)
    else:
        behaviors = available_behaviors

    if args.stems:
        stems = set(args.stems)
        frame_df = frame_df[frame_df["__stem__"].isin(stems)].copy()
        if frame_df.empty:
            raise ValueError("No rows matched the requested --stems.")

    out_dir = Path(args.out) if args.out else default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for stem, group in frame_df.groupby("__stem__", sort=True):
        stem_dir = out_dir / str(stem)
        written = plot_video_timeline(
            group,
            stem=str(stem),
            behaviors=behaviors,
            threshold=args.threshold,
            fps=args.fps,
            width=float(args.width),
            height_per_behavior=float(args.height_per_behavior),
            out_dir=stem_dir,
            fmt=args.format,
        )
        summary_rows.append(
            {
                "__stem__": str(stem),
                "n_frames": int(len(group)),
                "artifacts": [str(path.resolve()) for path in written],
            }
        )
        print(f"📝 Wrote {len(written)} plot file(s) for {stem}")

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": str(source_path.resolve()),
        "out_dir": str(out_dir.resolve()),
        "behaviors": behaviors,
        "fps": float(args.fps) if args.fps is not None else None,
        "threshold": float(args.threshold) if args.threshold is not None else None,
        "format": args.format,
        "per_video": summary_rows,
    }
    with open(out_dir / "plot_summary.yml", "w") as f:
        import yaml

        yaml.safe_dump(summary, f, sort_keys=False)
    print(f"📝 Wrote {out_dir / 'plot_summary.yml'}")


if __name__ == "__main__":
    main()
