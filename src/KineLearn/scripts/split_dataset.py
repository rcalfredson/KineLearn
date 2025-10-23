#!/usr/bin/env python3
"""
Split a set of processed videos into train/test sets for model training.

This script expects that frame-level feature and label files
(`frame_features_<video_stem>.parquet` / `frame_labels_<video_stem>.parquet`)
already exist for all listed videos.

Usage:
    kinelearn-split -v video_lists/all_videos.yaml --out splits/all_videos_split.yaml --seed 42
"""

import argparse
from datetime import datetime
from pathlib import Path
import random
import yaml
from sklearn.model_selection import train_test_split


def build_default_outpath(video_list_path: str) -> str:
    """Return default split file path with list stem and timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = Path(video_list_path).stem
    return f"data_splits/{stem}_split_{timestamp}.yaml"


def main():
    # First stage: only parse -v so we can build a default output name
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("-v", required=True)
    pre_args, _ = pre_parser.parse_known_args()

    default_out = build_default_outpath(pre_args.v)

    # Full parser
    parser = argparse.ArgumentParser(
        description="Split video list into train/test sets (filename-stem based)."
    )
    parser.add_argument(
        "-v",
        required=True,
        help="Path to a YAML file containing a list of video paths.",
    )
    parser.add_argument(
        "--out",
        default=default_out,
        help=f"Path to save the split info file (default: {default_out})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (optional).",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Fraction of videos to reserve for testing (default: 0.2).",
    )

    args = parser.parse_args()

    # ----------------------------------------------------------------------
    # Load video list
    # ----------------------------------------------------------------------
    with open(args.v, "r") as f:
        video_paths = yaml.safe_load(f)

    if not isinstance(video_paths, list) or not all(
        isinstance(v, str) for v in video_paths
    ):
        raise ValueError(f"{args.v} must be a YAML list of file paths.")

    # Use stems (filename without extension) as identifiers
    video_stems = [Path(vp).stem for vp in video_paths]
    print(f"📹 Loaded {len(video_stems)} videos from {args.v}")

    print(f"💾 Output path: {args.out}")

    # ----------------------------------------------------------------------
    # Apply reproducible random seed
    # ----------------------------------------------------------------------
    if args.seed is not None:
        print(f"🔒 Using random seed {args.seed} for reproducible splitting.")
        random.seed(args.seed)

    # ----------------------------------------------------------------------
    # Split videos into train/test
    # ----------------------------------------------------------------------
    if len(video_stems) < 13:
        test_stems = random.sample(video_stems, 1)
        train_stems = [v for v in video_stems if v not in test_stems]
    else:
        train_stems, test_stems = train_test_split(
            video_stems, test_size=args.test_fraction, random_state=args.seed
        )

    print(f"🧩 Train videos: {len(train_stems)}")
    print(f"🧪 Test videos:  {len(test_stems)}")

    # ----------------------------------------------------------------------
    # Save the split info as YAML
    # ----------------------------------------------------------------------
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    split_info = {
        "seed": args.seed,
        "test_fraction": args.test_fraction,
        "train": train_stems,
        "test": test_stems,
    }

    with open(out_path, "w") as f:
        yaml.safe_dump(split_info, f, sort_keys=False)

    print(f"✅ Video split info saved to {out_path.resolve()}")


if __name__ == "__main__":
    main()
