#!/usr/bin/env python3
"""
Generate and optionally execute split-variability experiments for KineLearn.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import yaml
from sklearn.model_selection import train_test_split


def load_yaml(path: Path) -> Any:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_split_file(path: Path) -> dict[str, Any]:
    if path.suffix.lower() in {".yaml", ".yml"}:
        payload = load_yaml(path)
        if not isinstance(payload, dict):
            raise ValueError(f"Split file {path} must be a mapping.")
        return payload

    with open(path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    sections: dict[str, list[str]] = {}
    current_key = None
    for line in lines:
        if line.endswith(":"):
            current_key = line[:-1].strip().lower()
            sections[current_key] = []
            continue
        if current_key is None:
            raise ValueError(f"Malformed split file {path}: found entries before a section header.")
        sections[current_key].append(line)
    return sections


def save_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure model sensitivity to train/test and train/val split choice by "
            "generating reproducible split files and optionally running training."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--video-list",
        help="YAML list of video paths used to generate outer train/test splits.",
    )
    source.add_argument(
        "--base-split",
        help="Existing train/test split file to hold test set fixed while varying train/val splits.",
    )
    parser.add_argument("--kl-config", required=True, help="KineLearn config YAML.")
    parser.add_argument("--behavior", required=True, help="Behavior to train.")
    parser.add_argument(
        "--features-dir",
        default="features",
        help="Directory containing frame/extracted feature files.",
    )
    parser.add_argument(
        "--outer-seeds",
        nargs="*",
        type=int,
        default=[],
        help="Seeds for outer train/test splits. Required with --video-list.",
    )
    parser.add_argument(
        "--inner-seeds",
        nargs="+",
        type=int,
        required=True,
        help="Seeds for explicit train/val splits within each outer split.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Fraction of videos reserved for test when generating outer splits.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=None,
        help="Fraction of training videos reserved for validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Training seed passed through to kinelearn-train.",
    )
    parser.add_argument(
        "--focal-alpha",
        type=float,
        default=None,
        help="Optional focal alpha override passed through to kinelearn-train.",
    )
    parser.add_argument(
        "--train-command",
        default="kinelearn-train",
        help="Training executable to invoke when --execute is set.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run trainings immediately. Otherwise, only write the experiment plan.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Defaults to results/split_variability/<timestamp>/",
    )
    return parser.parse_args()


def default_out_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("results") / "split_variability" / timestamp


def load_video_stems(video_list_path: Path) -> list[str]:
    video_paths = load_yaml(video_list_path)
    if not isinstance(video_paths, list) or not all(isinstance(v, str) for v in video_paths):
        raise ValueError(f"{video_list_path} must be a YAML list of video paths.")
    return [Path(v).stem for v in video_paths]


def normalize_split_sections(split_info: dict[str, Any], path: Path) -> tuple[list[str], list[str]]:
    lowered = {str(k).strip().lower(): v for k, v in split_info.items()}
    train = lowered.get("train", lowered.get("train videos"))
    test = lowered.get("test", lowered.get("test videos"))
    if train is None or test is None:
        raise ValueError(f"Split file {path} must contain train/test sections.")
    if not isinstance(train, list) or not isinstance(test, list):
        raise ValueError(f"Split file {path} must contain list-valued train/test sections.")
    return list(train), list(test)


def build_outer_splits(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.base_split:
        split_info = load_split_file(Path(args.base_split))
        train_stems, test_stems = normalize_split_sections(split_info, Path(args.base_split))
        return [
            {
                "outer_id": "fixed",
                "outer_seed": None,
                "train_stems": train_stems,
                "test_stems": test_stems,
                "source_split": str(Path(args.base_split).resolve()),
            }
        ]

    if not args.outer_seeds:
        raise ValueError("--outer-seeds is required when using --video-list.")

    stems = load_video_stems(Path(args.video_list))
    outer_splits = []
    for outer_seed in args.outer_seeds:
        train_stems, test_stems = train_test_split(
            stems, test_size=args.test_fraction, random_state=outer_seed
        )
        outer_splits.append(
            {
                "outer_id": f"outer_seed{outer_seed}",
                "outer_seed": int(outer_seed),
                "train_stems": list(train_stems),
                "test_stems": list(test_stems),
                "source_split": None,
            }
        )
    return outer_splits


def manifest_from_stdout(stdout: str) -> str | None:
    matches = re.findall(r"Wrote\s+(.+train_manifest\.yml)", stdout)
    return matches[-1].strip() if matches else None


def build_plan(
    args: argparse.Namespace, out_dir: Path, val_fraction: float
) -> list[dict[str, Any]]:
    runs = []
    split_root = out_dir / "splits"
    outer_splits = build_outer_splits(args)

    for outer in outer_splits:
        outer_dir = split_root / outer["outer_id"]
        if outer["source_split"] is None:
            split_path = outer_dir / "train_test_split.yaml"
            save_yaml(
                split_path,
                {
                    "seed": outer["outer_seed"],
                    "test_fraction": args.test_fraction,
                    "train": outer["train_stems"],
                    "test": outer["test_stems"],
                },
            )
        else:
            split_path = Path(outer["source_split"])

        for inner_seed in args.inner_seeds:
            inner_train, inner_val = train_test_split(
                outer["train_stems"], test_size=val_fraction, random_state=inner_seed
            )
            val_split_path = outer_dir / f"train_val_split_seed{inner_seed}.yaml"
            save_yaml(
                val_split_path,
                {
                    "seed": int(inner_seed),
                    "val_fraction": float(val_fraction),
                    "train": list(inner_train),
                    "val": list(inner_val),
                },
            )

            command = [
                args.train_command,
                "--kl-config",
                args.kl_config,
                "--split",
                str(split_path),
                "--val-split",
                str(val_split_path),
                "--behavior",
                args.behavior,
                "--features-dir",
                args.features_dir,
                "--seed",
                str(args.seed),
            ]
            if args.focal_alpha is not None:
                command.extend(["--focal-alpha", str(args.focal_alpha)])

            runs.append(
                {
                    "outer_id": outer["outer_id"],
                    "outer_seed": outer["outer_seed"],
                    "inner_seed": int(inner_seed),
                    "split_path": str(Path(split_path).resolve()),
                    "val_split_path": str(val_split_path.resolve()),
                    "train_count": len(inner_train),
                    "val_count": len(inner_val),
                    "test_count": len(outer["test_stems"]),
                    "command": command,
                }
            )
    return runs


def write_plan_csv(path: Path, runs: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "outer_id",
        "outer_seed",
        "inner_seed",
        "split_path",
        "val_split_path",
        "train_count",
        "val_count",
        "test_count",
        "command",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            row = dict(run)
            row["command"] = " ".join(row["command"])
            writer.writerow(row)


def aggregate_results(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    kl_config = load_yaml(Path(args.kl_config))
    training_cfg = kl_config.get("training", {})
    val_fraction = (
        float(args.val_fraction)
        if args.val_fraction is not None
        else float(training_cfg.get("val_fraction", 0.2))
    )

    runs = build_plan(args, out_dir, val_fraction)
    write_plan_csv(out_dir / "experiment_plan.csv", runs)
    save_yaml(
        out_dir / "experiment_config.yml",
        {
            "kl_config": str(Path(args.kl_config).resolve()),
            "behavior": args.behavior,
            "features_dir": str(Path(args.features_dir).resolve()),
            "training_seed": int(args.seed),
            "val_fraction": float(val_fraction),
            "test_fraction": float(args.test_fraction),
            "source": {
                "video_list": str(Path(args.video_list).resolve()) if args.video_list else None,
                "base_split": str(Path(args.base_split).resolve()) if args.base_split else None,
            },
            "outer_seeds": list(args.outer_seeds),
            "inner_seeds": list(args.inner_seeds),
            "execute": bool(args.execute),
        },
    )
    print(f"📝 Wrote {out_dir / 'experiment_plan.csv'}")

    if not args.execute:
        print("Dry run only; no trainings launched.")
        return

    summary_rows: list[dict[str, Any]] = []
    for idx, run in enumerate(runs, start=1):
        print(
            f"\n=== Run {idx}/{len(runs)} "
            f"(outer={run['outer_id']}, inner_seed={run['inner_seed']}) ==="
        )
        completed = subprocess.run(
            run["command"],
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.stdout:
            print(completed.stdout, end="")
        if completed.stderr:
            print(completed.stderr, file=sys.stderr, end="")

        row = {
            "outer_id": run["outer_id"],
            "outer_seed": run["outer_seed"],
            "inner_seed": run["inner_seed"],
            "split_path": run["split_path"],
            "val_split_path": run["val_split_path"],
            "returncode": int(completed.returncode),
            "manifest_path": None,
        }

        manifest_path_str = manifest_from_stdout(completed.stdout)
        if manifest_path_str:
            manifest_path = Path(manifest_path_str)
            row["manifest_path"] = str(manifest_path.resolve())
            if manifest_path.exists():
                manifest = load_yaml(manifest_path)
                training_run = manifest.get("training_run", {})
                test_metrics = training_run.get("test_metrics", {})
                row["best_epoch_by_val_loss"] = training_run.get("best_epoch_by_val_loss")
                row["epochs_completed"] = training_run.get("epochs_completed")
                for key, value in test_metrics.items():
                    row[f"test_{key}"] = value

        summary_rows.append(row)
        aggregate_results(out_dir / "results_summary.csv", summary_rows)

    print(f"\n📝 Wrote {out_dir / 'results_summary.csv'}")


if __name__ == "__main__":
    main()
