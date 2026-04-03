#!/usr/bin/env python3
"""
Restore memmap and index artifacts for completed KineLearn training runs.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from KineLearn.core.manifests import load_train_manifest, resolve_recorded_path, save_yaml
from KineLearn.core.memmap import make_windowed_memmaps
from KineLearn.scripts.batch_eval_splits import discover_runs, infer_manifest_path, resolve_source


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
        raise ValueError(f"{where}: stem '{stem}' matched multiple feature files: {matches}")
    return resolved


def load_parquets_for_stems(
    stems: list[str],
    features_dir: Path,
    label_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_parts, y_parts = [], []
    for stem in stems:
        feat_path = features_dir / f"frame_features_{stem}.parquet"
        lab_path = features_dir / f"frame_labels_{stem}.parquet"
        if not feat_path.exists():
            raise FileNotFoundError(f"Feature file not found: {feat_path}")

        X = pd.read_parquet(feat_path).copy()
        if lab_path.exists():
            Y = pd.read_parquet(lab_path).copy()
            for label in label_columns:
                if label not in Y.columns:
                    Y[label] = 0
            extra_cols = [c for c in Y.columns if c not in label_columns]
            if extra_cols:
                Y = Y.drop(columns=extra_cols, errors="ignore")
            Y = Y[label_columns]
        else:
            Y = pd.DataFrame(0, index=range(len(X)), columns=label_columns)

        if len(X) != len(Y):
            raise ValueError(f"Row mismatch for stem '{stem}': features={len(X)} vs labels={len(Y)}")

        frame_idx = np.arange(len(X), dtype=np.int32)
        X["__stem__"] = stem
        X["__frame__"] = frame_idx
        Y["__stem__"] = stem
        Y["__frame__"] = frame_idx
        X_parts.append(X)
        y_parts.append(Y)

    return (
        pd.concat(X_parts, axis=0, ignore_index=True),
        pd.concat(y_parts, axis=0, ignore_index=True),
    )


def align_columns(
    df: pd.DataFrame,
    expected: list[str],
    *,
    df_name: str,
    helper_columns: tuple[str, ...] = (),
    allow_extra: bool = False,
) -> pd.DataFrame:
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing expected columns: {missing}")
    extra = [c for c in df.columns if c not in expected and c not in helper_columns]
    if extra and not allow_extra:
        raise ValueError(f"{df_name} has unexpected columns: {extra}")
    ordered = list(expected) + [c for c in helper_columns if c in df.columns]
    return df.loc[:, ordered]


def zero_fill_remaining_nans(
    df: pd.DataFrame,
    *,
    df_name: str,
    helper_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    value_columns = [c for c in df.columns if c not in helper_columns]
    nan_count = int(df[value_columns].isna().sum().sum())
    if nan_count > 0:
        print(f"⚠️  Final zero-fill parity step on {df_name}: replacing {nan_count} NaNs.")
        df = df.copy()
        df[value_columns] = df[value_columns].fillna(0)
    return df


def default_report_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("results") / "restored_artifacts" / timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild windowed memmap and index artifacts for completed KineLearn "
            "training runs from their saved train manifests."
        )
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help=(
            "Split-variability sweep directory, results_summary.csv, or "
            "experiment_plan.csv to scan for train manifests."
        ),
    )
    parser.add_argument(
        "--manifest",
        action="append",
        default=[],
        help="Direct path to a train_manifest.yml file to restore.",
    )
    parser.add_argument(
        "--features-dir",
        default=None,
        help="Optional features directory override. Defaults to each manifest's features_dir.",
    )
    parser.add_argument(
        "--subset",
        choices=["train", "val", "test", "all"],
        default="all",
        help="Which subset artifacts to restore (default: all).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite already-existing restored artifacts in the target run directory.",
    )
    parser.add_argument(
        "--teardown",
        action="store_true",
        help=(
            "Remove previously restored artifact files instead of rebuilding them. "
            "By default this removes only memmaps and keeps index arrays."
        ),
    )
    parser.add_argument(
        "--remove-indexes",
        action="store_true",
        help=(
            "When used with --teardown, also remove *_vids.npy and *_starts.npy "
            "index arrays in addition to memmaps."
        ),
    )
    parser.add_argument(
        "--report-out",
        default=None,
        help=(
            "Optional output directory for restore_summary.yml. Defaults to "
            "results/restored_artifacts/<timestamp>/."
        ),
    )
    return parser.parse_args()


def collect_manifest_paths(source_paths: list[Path], manifest_paths: list[Path]) -> list[Path]:
    discovered: list[Path] = []
    for source_path in source_paths:
        sweep_dir, table_path = resolve_source(source_path)
        rows = discover_runs(table_path)
        for row in rows:
            manifest_path = infer_manifest_path(row, sweep_dir)
            if manifest_path is not None:
                discovered.append(manifest_path.resolve())
    discovered.extend(path.resolve() for path in manifest_paths)

    unique: dict[Path, None] = {}
    for path in discovered:
        unique.setdefault(path, None)
    return list(unique.keys())


def subsets_to_restore(arg: str) -> list[str]:
    return ["train", "val", "test"] if arg == "all" else [arg]


def subset_targets(manifest_path: Path, artifact: dict[str, Any]) -> dict[str, Path]:
    return {
        "X_path": (manifest_path.parent / Path(artifact["X_path"]).name).resolve(),
        "Y_path": (manifest_path.parent / Path(artifact["Y_path"]).name).resolve(),
        "vids_path": (manifest_path.parent / Path(artifact["vids_path"]).name).resolve(),
        "starts_path": (manifest_path.parent / Path(artifact["starts_path"]).name).resolve(),
    }


def out_prefix_from_targets(targets: dict[str, Path]) -> Path:
    x_path = targets["X_path"]
    suffix = "_features.fp32"
    if not x_path.name.endswith(suffix):
        raise ValueError(f"Unexpected feature memmap filename: {x_path.name}")
    prefix_name = x_path.name[: -len(suffix)]
    return x_path.parent / prefix_name


def subset_is_present(targets: dict[str, Path]) -> bool:
    return all(path.exists() for path in targets.values())


def remove_targets(targets: dict[str, Path], *, include_indexes: bool) -> list[Path]:
    removed: list[Path] = []
    keys = ("X_path", "Y_path", "vids_path", "starts_path") if include_indexes else ("X_path", "Y_path")
    for key in keys:
        path = targets[key]
        if path.exists():
            path.unlink()
            removed.append(path)
    return removed


def validate_restored_subset(
    manifest: dict[str, Any],
    subset: str,
    *,
    count: int,
    mmX: np.memmap,
    mmY: np.memmap,
    vids: np.ndarray,
    starts: np.ndarray,
) -> None:
    artifact = manifest["artifacts"][subset]
    if int(count) != int(artifact["count"]):
        raise ValueError(
            f"Restored {subset} count mismatch for {manifest['run_dir']}: "
            f"expected {artifact['count']} got {count}"
        )
    if tuple(int(x) for x in artifact["X_shape"]) != tuple(int(x) for x in mmX.shape):
        raise ValueError(f"Restored {subset} X_shape mismatch for {manifest['run_dir']}")
    if tuple(int(x) for x in artifact["Y_shape"]) != tuple(int(x) for x in mmY.shape):
        raise ValueError(f"Restored {subset} Y_shape mismatch for {manifest['run_dir']}")
    if len(vids) != int(artifact["count"]) or len(starts) != int(artifact["count"]):
        raise ValueError(f"Restored {subset} index length mismatch for {manifest['run_dir']}")


def restore_subset(
    manifest: dict[str, Any],
    manifest_path: Path,
    *,
    subset: str,
    features_dir_override: Path | None,
    overwrite: bool,
) -> dict[str, Any]:
    artifact = manifest["artifacts"][subset]
    targets = subset_targets(manifest_path, artifact)
    if subset_is_present(targets) and not overwrite:
        return {
            "subset": subset,
            "status": "skipped_existing",
            **{key: str(path) for key, path in targets.items()},
        }

    if overwrite:
        remove_targets(targets, include_indexes=True)

    features_dir_value = (
        features_dir_override
        if features_dir_override is not None
        else resolve_recorded_path(manifest["features_dir"], manifest_path)
    )
    features_dir = Path(features_dir_value)
    known_stems = available_feature_stems(features_dir)
    if not known_stems:
        raise FileNotFoundError(f"No frame_features_*.parquet files found in {features_dir}")

    resolved_stems = manifest.get("resolved_stems") or {}
    requested_stems = list(resolved_stems.get(subset) or [])
    if not requested_stems:
        raise ValueError(f"Manifest {manifest_path} does not contain resolved stems for subset '{subset}'.")
    stems = resolve_requested_stems(
        requested_stems,
        known_stems,
        where=f"{manifest_path.name}:{subset}",
    )

    label_columns = list(manifest["label_columns"])
    feature_columns = list(manifest["feature_columns"])
    X, Y = load_parquets_for_stems(stems, features_dir, label_columns)
    helper_columns = ("__stem__", "__frame__")
    if bool((manifest.get("training") or {}).get("final_zero_fill", False)):
        X = zero_fill_remaining_nans(X, df_name=f"{subset}.X", helper_columns=helper_columns)
        Y = zero_fill_remaining_nans(Y, df_name=f"{subset}.Y", helper_columns=helper_columns)

    X = align_columns(
        X,
        feature_columns,
        df_name=f"{subset}.X",
        helper_columns=helper_columns,
        allow_extra=True,
    )
    Y = align_columns(
        Y,
        label_columns,
        df_name=f"{subset}.Y",
        helper_columns=helper_columns,
    )

    window = manifest["window"]
    out_prefix = out_prefix_from_targets(targets)
    count, mmX, mmY, vids, starts = make_windowed_memmaps(
        X,
        Y,
        int(window["size"]),
        int(window["stride"]),
        int(manifest["feature_selection"]["n_input_features"]),
        int(manifest.get("n_classes", len(label_columns))),
        str(out_prefix),
    )
    np.save(targets["vids_path"], vids)
    np.save(targets["starts_path"], starts)
    validate_restored_subset(
        manifest,
        subset,
        count=count,
        mmX=mmX,
        mmY=mmY,
        vids=vids,
        starts=starts,
    )
    return {
        "subset": subset,
        "status": "restored",
        "count": int(count),
        "features_dir": str(features_dir.resolve()),
        **{key: str(path) for key, path in targets.items()},
    }


def teardown_subset(
    manifest: dict[str, Any],
    manifest_path: Path,
    *,
    subset: str,
    remove_indexes: bool,
) -> dict[str, Any]:
    artifact = manifest["artifacts"][subset]
    targets = subset_targets(manifest_path, artifact)
    removed = remove_targets(targets, include_indexes=remove_indexes)
    status = "removed" if removed else "skipped_missing"
    return {
        "subset": subset,
        "status": status,
        "remove_indexes": bool(remove_indexes),
        "removed_paths": [str(path) for path in removed],
        **{key: str(path) for key, path in targets.items()},
    }


def main() -> None:
    args = parse_args()
    if not args.source and not args.manifest:
        raise ValueError("Provide at least one --source or --manifest.")
    if args.remove_indexes and not args.teardown:
        raise ValueError("--remove-indexes can only be used with --teardown.")
    if args.teardown and args.overwrite:
        raise ValueError("--teardown and --overwrite cannot be used together.")

    manifest_paths = collect_manifest_paths(
        [Path(p) for p in args.source],
        [Path(p) for p in args.manifest],
    )
    if not manifest_paths:
        raise ValueError("No manifests were discovered for restoration.")

    features_dir_override = Path(args.features_dir) if args.features_dir else None
    report_out = Path(args.report_out) if args.report_out else default_report_dir()
    report_out.mkdir(parents=True, exist_ok=True)

    restore_rows: list[dict[str, Any]] = []
    for manifest_path in manifest_paths:
        manifest = load_train_manifest(manifest_path)
        for subset in subsets_to_restore(args.subset):
            if args.teardown:
                row = teardown_subset(
                    manifest,
                    manifest_path,
                    subset=subset,
                    remove_indexes=bool(args.remove_indexes),
                )
            else:
                row = restore_subset(
                    manifest,
                    manifest_path,
                    subset=subset,
                    features_dir_override=features_dir_override,
                    overwrite=bool(args.overwrite),
                )
            row.update(
                {
                    "manifest_path": str(manifest_path.resolve()),
                    "behavior": manifest["behavior"],
                    "run_dir_recorded": str(manifest.get("run_dir")),
                }
            )
            restore_rows.append(row)
            print(
                f"[{manifest['behavior']}:{subset}] {row['status']} "
                f"-> {Path(row['X_path']).parent}"
            )

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_inputs": [str(Path(p).resolve()) for p in args.source],
        "manifest_inputs": [str(Path(p).resolve()) for p in args.manifest],
        "subset": args.subset,
        "overwrite": bool(args.overwrite),
        "teardown": bool(args.teardown),
        "remove_indexes": bool(args.remove_indexes),
        "features_dir_override": str(features_dir_override.resolve()) if features_dir_override else None,
        "restored": restore_rows,
    }
    save_yaml(report_out / "restore_summary.yml", summary)
    print(f"📝 Wrote {report_out / 'restore_summary.yml'}")


if __name__ == "__main__":
    main()
