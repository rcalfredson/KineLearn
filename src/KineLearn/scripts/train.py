#!/usr/bin/env python3
"""
KineLearn training (Step 1): load hyperparameters and dataset.
Now also harmonizes training hyperparameters and resolves focal-loss alpha by behavior.

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
from sklearn.model_selection import train_test_split
import yaml

from KineLearn.core.memmap import make_windowed_memmaps

# (Optional for future training step)
try:
    import tensorflow as tf
    from tensorflow.keras import backend as K
except Exception:
    tf = K = None


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


def resolve_focal_params(training_cfg: Dict, behavior: str) -> tuple[float, float]:
    """
    Resolve focal loss alpha and gamma.
    training_cfg may contain:
      training["focal"] = {"alpha": <float or {behavior: float}, "gamma": <float>}
    Falls back to alpha=0.7, gamma=2.0 if unspecified.
    """
    focal = training_cfg.get("focal", {}) or {}
    alpha_cfg = focal.get("alpha", 0.7)
    gamma = float(focal.get("gamma", 2.0))
    if isinstance(alpha_cfg, dict):
        if behavior not in alpha_cfg:
            raise ValueError(
                f"No focal.alpha specified for behavior '{behavior}'. Available: {list(alpha_cfg.keys())}"
            )
        alpha = float(alpha_cfg[behavior])
    else:
        alpha = float(alpha_cfg)
    return alpha, gamma


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
        "--behavior",
        required=True,
        help="Behavior name to train (must be present in the KineLearn config's `behaviors` list).",
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

    behavior = args.behavior
    if behavior not in behaviors:
        raise ValueError(
            f"--behavior '{behavior}' not found in config behaviors: {behaviors}"
        )

    # Optional training hyperparameters nested under config["training"]
    training_cfg: Dict = kl_config.get("training") or {}
    # Allow CLI overrides
    if args.epochs is not None:
        training_cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        training_cfg["batch_size"] = args.batch_size

    # Provide some sane defaults (used later when we add training)
    training_cfg.setdefault("epochs", 10)
    training_cfg.setdefault("batch_size", 8)
    training_cfg.setdefault("learning_rate", 1e-3)
    training_cfg.setdefault(
        "loss", "focal"
    )  # focal is the only supported loss initially
    training_cfg.setdefault("metrics", ["accuracy"])
    training_cfg.setdefault("val_fraction", 0.1)  # split from training later

    # Resolve focal params (alpha can be global or per-behavior)
    alpha, gamma = resolve_focal_params(training_cfg, behavior)

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
    if training_cfg.get("loss", "focal") == "focal":
        print(f"  focal.alpha({behavior}): {alpha}")
        print(f"  focal.gamma: {gamma}")

    split_info = load_yaml(Path(args.split))
    require_keys(split_info, ["train", "test"], "split file")
    train_stems: List[str] = split_info["train"]
    test_stems: List[str] = split_info["test"]

    features_dir = Path(args.features_dir)

    X_test, y_test = load_parquets_for_stems(test_stems, features_dir, behaviors)

    if "window" in training_cfg:
        wsize = training_cfg["window"]["size"]
        stride = training_cfg["window"]["stride"]
    else:
        wsize, stride = 60, 10  # defaults

    n_classes = len(behaviors)
    seed = training_cfg.get("seed", 42)

    # Split training stems into train/val sets
    train_stems, val_stems = train_test_split(
        train_stems, test_size=training_cfg["val_fraction"], random_state=seed
    )
    print(
        f"🧩 Split {len(train_stems) + len(val_stems)} total training videos "
        f"into {len(train_stems)} train and {len(val_stems)} validation."
    )

    X_train, y_train = load_parquets_for_stems(train_stems, features_dir, behaviors)
    X_val, y_val = load_parquets_for_stems(val_stems, features_dir, behaviors)

    # Infer derived_dim from training data
    derived_dim = len([c for c in X_train.columns if c != "__stem__"])

    # Basic sanity check
    if any(dt == "object" for dt in X_train.dtypes):
        objs = [
            c
            for c in X_train.columns
            if X_train[c].dtype == "object" and c != "__stem__"
        ]
        if objs:
            raise TypeError(f"Found non-numeric feature columns: {objs}")

    summarize_dataset(X_train, y_train, X_test, y_test, behaviors)

    train_count, mmX_tr, mmY_tr, tr_vids, tr_starts = make_windowed_memmaps(
        X_train, y_train, wsize, stride, derived_dim, n_classes, "results/train"
    )
    val_count, mmX_va, mmY_va, va_vids, va_starts = make_windowed_memmaps(
        X_val, y_val, wsize, stride, derived_dim, n_classes, "results/val"
    )
    test_count, mmX_te, mmY_te, te_vids, te_starts = make_windowed_memmaps(
        X_test, y_test, wsize, stride, derived_dim, n_classes, "results/test"
    )

    # Write training manifest
    manifest = {
        "kl_config": str(Path(args.kl_config).resolve()),
        "split": str(Path(args.split).resolve()),
        "features_dir": str(features_dir.resolve()),
        "behaviors": behaviors,
        "training": training_cfg,
        "window": {"size": wsize, "stride": stride},
        "counts": {
            "train": train_count,
            "val": val_count,
            "test": test_count,
        },
        "n_features": derived_dim,
        "n_classes": n_classes,
    }

    # Include resolved behavior + focal params in manifest for traceability
    manifest["behavior"] = behavior
    if training_cfg.get("loss", "focal") == "focal":
        manifest["focal"] = {"alpha": alpha, "gamma": gamma}

    out = Path("results")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "train_manifest.yml", "w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)

    print("\n📝 Wrote results/train_manifest.yml")


if __name__ == "__main__":
    main()
