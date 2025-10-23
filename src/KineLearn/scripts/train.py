#!/usr/bin/env python3
"""
KineLearn training (Step 1): load hyperparameters and dataset.

This script:
- Loads KineLearn config (and optional training hyperparameters)
- Loads a train/test split YAML (with video stems)
- Assembles X_train, y_train, X_test, y_test from per-stem Parquet files

Later steps (model definition, training loop, evaluation) will be added on top.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import yaml


# ----------------------------
# Helpers
# ----------------------------
def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def require_keys(d: dict, keys: List[str], where: str) -> None:
    missing = [k for k in keys if k not in d]
    if missing:
        raise ValueError(f"Missing keys {missing} in {where}")


def load_parquets_for_stems(
    stems: List[str],
    features_dir: Path,
    behaviors: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and concatenate per-stem features/labels Parquet files.

    - Features: features/frame_features_<stem>.parquet
    - Labels:   features/frame_labels_<stem>.parquet
    - If a label file is missing, create a zero-filled frame with the given behaviors.
    - Ensures columns for labels exactly match 'behaviors' (order preserved).
    """
    X_parts, y_parts = [], []
    for stem in stems:
        feat_path = features_dir / f"frame_features_{stem}.parquet"
        lab_path = features_dir / f"frame_labels_{stem}.parquet"

        if not feat_path.exists():
            raise FileNotFoundError(f"Feature file not found: {feat_path}")

        X = pd.read_parquet(feat_path)
        if lab_path.exists():
            Y = pd.read_parquet(lab_path)
            # If label file has extra columns, reduce; if missing, add zeros.
            # Final label columns exactly match 'behaviors', in that order.
            for b in behaviors:
                if b not in Y.columns:
                    Y[b] = 0
            extra_cols = [c for c in Y.columns if c not in behaviors]
            if extra_cols:
                Y = Y.drop(columns=extra_cols, errors="ignore")
            Y = Y[behaviors]
        else:
            Y = pd.DataFrame(0, index=range(len(X)), columns=behaviors)

        if len(X) != len(Y):
            raise ValueError(
                f"Row mismatch for stem '{stem}': features={len(X)} vs labels={len(Y)}"
            )

        # (Optional) keep a stem column for tracing/debug
        X = X.copy()
        Y = Y.copy()
        X["__stem__"] = stem
        Y["__stem__"] = stem

        X_parts.append(X)
        y_parts.append(Y)

    X_all = pd.concat(X_parts, axis=0, ignore_index=True)
    Y_all = pd.concat(y_parts, axis=0, ignore_index=True)
    return X_all, Y_all


def summarize_dataset(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    behaviors: List[str],
) -> None:
    print("\n=== Dataset summary ===")
    print(f"Train: X={X_train.shape}, Y={y_train.shape}")
    print(f" Test: X={X_test.shape}, Y={y_test.shape}")

    # Quick label support summary (counts of positive frames) per behavior
    def label_counts(df: pd.DataFrame) -> pd.Series:
        # df contains behavior columns + __stem__
        cols = [c for c in df.columns if c in behaviors]
        return df[cols].sum().astype(int)

    train_pos = label_counts(y_train)
    test_pos = label_counts(y_test)

    print("\nPositive frame counts (train):")
    for b in behaviors:
        print(f"  {b}: {train_pos.get(b, 0)}")

    print("\nPositive frame counts (test):")
    for b in behaviors:
        print(f"  {b}: {test_pos.get(b, 0)}")

    # Show a peek of feature columns (excluding helper column)
    feature_cols = [c for c in X_train.columns if c != "__stem__"]
    print(f"\nTotal feature columns: {len(feature_cols)} (showing first 10)")
    print(feature_cols[:10])


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="KineLearn training (Step 1): load hyperparameters and dataset."
    )
    parser.add_argument(
        "--kl-config",
        required=True,
        help="Path to KineLearn config YAML.",
    )
    parser.add_argument(
        "--split",
        required=True,
        help="Path to train/test split YAML produced by kinelearn-split.",
    )
    parser.add_argument(
        "--features-dir",
        default="features",
        help="Directory containing frame_features_*.parquet and frame_labels_*.parquet (default: features).",
    )
    # Optional quick overrides (useful even before a full training section exists)
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override epochs from config (optional).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch_size from config (optional).",
    )

    args = parser.parse_args()

    kl_config = load_yaml(Path(args.kl_config))
    require_keys(kl_config, ["behaviors"], "KineLearn config")
    behaviors: List[str] = kl_config["behaviors"] or []
    if not isinstance(behaviors, list) or not all(
        isinstance(b, str) for b in behaviors
    ):
        raise ValueError("`behaviors` in KineLearn config must be a list of strings.")

    # Optional training hyperparameters nested under config["training"]
    training_cfg: Dict = kl_config.get("training") or {}
    # Allow CLI overrides
    if args.epochs is not None:
        training_cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        training_cfg["batch_size"] = args.batch_size

    # Provide some sane defaults (used later when we add training)
    training_cfg.setdefault("epochs", 20)
    training_cfg.setdefault("batch_size", 512)
    training_cfg.setdefault("learning_rate", 1e-3)
    training_cfg.setdefault("loss", "binary_crossentropy")
    training_cfg.setdefault("metrics", ["accuracy"])
    training_cfg.setdefault("val_fraction", 0.1)  # split from training later

    print("Loaded training hyperparameters:")
    for k in [
        "epochs",
        "batch_size",
        "learning_rate",
        "loss",
        "metrics",
        "val_fraction",
    ]:
        print(f"  {k}: {training_cfg[k]}")

    split_info = load_yaml(Path(args.split))
    require_keys(split_info, ["train", "test"], "split file")
    train_stems: List[str] = split_info["train"]
    test_stems: List[str] = split_info["test"]

    features_dir = Path(args.features_dir)

    # Assemble datasets
    X_train, y_train = load_parquets_for_stems(train_stems, features_dir, behaviors)
    X_test, y_test = load_parquets_for_stems(test_stems, features_dir, behaviors)

    # Basic sanity: avoid object dtypes sneaking in
    if any(dt == "object" for dt in X_train.dtypes):
        # Drop helper column from check
        objs = [
            c
            for c in X_train.columns
            if X_train[c].dtype == "object" and c != "__stem__"
        ]
        if objs:
            raise TypeError(f"Found non-numeric feature columns: {objs}")

    summarize_dataset(X_train, y_train, X_test, y_test, behaviors)

    # Placeholder return / handoff for the next stage (modeling)
    # For now, just save a tiny manifest so downstream steps can re-load quickly.
    manifest = {
        "kl_config": str(Path(args.kl_config).resolve()),
        "split": str(Path(args.split).resolve()),
        "features_dir": str(features_dir.resolve()),
        "behaviors": behaviors,
        "training": training_cfg,
        "n_train_rows": int(len(X_train)),
        "n_test_rows": int(len(X_test)),
        "n_features": int(len([c for c in X_train.columns if c != "__stem__"])),
    }
    out = Path("results")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "train_manifest.yaml", "w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)
    print("\n📝 Wrote results/train_manifest.yaml")

    # Later: return or proceed to model construction + training
    # (e.g., build_model(training_cfg), then fit/evaluate)


if __name__ == "__main__":
    main()
