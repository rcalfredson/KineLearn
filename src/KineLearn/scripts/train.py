#!/usr/bin/env python3
"""
Train a single-behavior KineLearn classifier from precomputed frame features.

This script:
- loads the KineLearn config and a saved train/test split
- derives a validation subset from the training stems
- loads per-video feature and label Parquet files
- optionally excludes raw absolute x/y keypoint columns from model input
- windows train/val/test subsets into memmap-backed arrays
- trains a keypoints-only BiLSTM with focal loss
- checkpoints on val_loss, with optional reduce-on-plateau and early stopping
- evaluates the selected checkpoint on the test subset
- writes run artifacts and a train_manifest.yml under results/<behavior>/<timestamp>/

Training is single-behavior per run. Focal-loss alpha can be specified in the config
per behavior or overridden at the CLI for split-specific validation tuning.
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml

from KineLearn.core.generators import KeypointWindowGenerator
from KineLearn.core.losses import focal_loss
from KineLearn.core.memmap import make_windowed_memmaps
from KineLearn.core.models import build_keypoint_bilstm

# (Optional for future training step)
try:
    import tensorflow as tf
    from tensorflow.keras import backend as K
except Exception:
    tf = K = None

# Path: src/KineLearn/scripts/train.py


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
        frame_idx = np.arange(len(X), dtype=np.int32)
        X["__frame__"] = frame_idx
        Y["__frame__"] = frame_idx

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
    feature_cols = [c for c in X_train.columns if c not in ("__stem__", "__frame__")]
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


def align_columns(
    df: pd.DataFrame,
    expected: List[str],
    *,
    df_name: str,
    helper_columns: tuple[str, ...] = (),
    allow_extra: bool = False,
) -> pd.DataFrame:
    """
    Reorder a DataFrame to a known column order and fail loudly on mismatch.
    """
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing expected columns: {missing}")

    extra = [c for c in df.columns if c not in expected and c not in helper_columns]
    if extra and not allow_extra:
        raise ValueError(f"{df_name} has unexpected columns: {extra}")

    ordered = list(expected) + [c for c in helper_columns if c in df.columns]
    return df.loc[:, ordered]


def is_absolute_coordinate_column(col: str) -> bool:
    """
    Return True for raw absolute x/y keypoint columns such as `thorax_x`.
    Derived feature columns are excluded.
    """
    if not (col.endswith("_x") or col.endswith("_y")):
        return False

    derived_markers = ("_coord_", "_velocity_", "_acceleration_")
    if any(marker in col for marker in derived_markers):
        return False

    return True


class HistoryCapture(tf.keras.callbacks.Callback if tf is not None else object):
    """
    Lightweight callback that retains epoch-end logs even if training is interrupted.
    """

    def __init__(self):
        if tf is None:
            raise ImportError("TensorFlow is required for HistoryCapture.")
        super().__init__()
        self.history: dict[str, list[float]] = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for key, value in logs.items():
            self.history.setdefault(key, []).append(float(value))


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
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Override the training seed for this run. "
            "This affects the train/validation split and training-time shuffling."
        ),
    )
    parser.add_argument(
        "--focal-alpha",
        type=float,
        default=None,
        help=(
            "Override focal loss alpha for this training run. "
            "Useful for split-specific tuning in single-behavior training."
        ),
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
    if args.seed is not None:
        training_cfg["seed"] = int(args.seed)
    if args.focal_alpha is not None:
        training_cfg.setdefault("focal", {})
        training_cfg["focal"]["alpha"] = float(args.focal_alpha)

    # Provide some sane defaults (used later when we add training)
    training_cfg.setdefault("epochs", 10)
    training_cfg.setdefault("batch_size", 8)
    training_cfg.setdefault("learning_rate", 1e-3)
    training_cfg.setdefault(
        "loss", "focal"
    )  # focal is the only supported loss initially
    training_cfg.setdefault("metrics", ["accuracy"])
    training_cfg.setdefault("val_fraction", 0.1)  # split from training later
    training_cfg.setdefault("seed", 42)
    training_cfg.setdefault("include_absolute_coordinates", False)
    training_cfg.setdefault("early_stopping", False)
    training_cfg.setdefault("early_stopping_patience", 3)
    training_cfg.setdefault("early_stopping_min_delta", 0.0)
    training_cfg.setdefault("keypoint_noise_std", 0.0)

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
        "seed",
        "include_absolute_coordinates",
        "early_stopping",
        "early_stopping_patience",
        "early_stopping_min_delta",
        "keypoint_noise_std",
    ]:
        print(f"  {k}: {training_cfg[k]}")
    if training_cfg.get("loss", "focal") == "focal":
        print(f"  focal.alpha({behavior}): {alpha}")
        print(f"  focal.gamma: {gamma}")

    split_info = load_yaml(Path(args.split))
    require_keys(split_info, ["train", "test"], "split file")
    train_stems: List[str] = split_info["train"]
    test_stems: List[str] = split_info["test"]

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("results") / behavior / run_timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir

    features_dir = Path(args.features_dir)

    X_test, y_test = load_parquets_for_stems(test_stems, features_dir, behaviors)

    wcfg = kl_config.get("window") or training_cfg.get("window") or {}
    wsize = int(wcfg.get("size", 60))
    stride = int(wcfg.get("stride", 10))

    if wsize <= 0 or stride <= 0:
        raise ValueError(
            f"window.size and window.stride must be positive; got size={wsize}, stride={stride}"
        )
    if stride > wsize:
        print(f"⚠️  stride ({stride}) > window_size ({wsize}); windows will be sparse.")

    # Avoid duplicating window inside training in the manifest
    training_cfg: dict = dict(training_cfg)
    training_cfg.pop("window", None)

    n_classes = len(behaviors)
    seed = training_cfg.get("seed", 42)

    # Split training stems into train/val sets
    train_stems, val_stems = train_test_split(
        train_stems, test_size=training_cfg["val_fraction"], random_state=seed
    )
    if not train_stems or not val_stems:
        raise ValueError(
            "Train/validation split produced an empty partition. "
            "Adjust training.val_fraction or provide more training videos."
        )
    print(
        f"🧩 Split {len(train_stems) + len(val_stems)} total training videos "
        f"into {len(train_stems)} train and {len(val_stems)} validation."
    )

    X_train, y_train = load_parquets_for_stems(train_stems, features_dir, behaviors)
    X_val, y_val = load_parquets_for_stems(val_stems, features_dir, behaviors)

    # Basic sanity check
    if any(dt == "object" for dt in X_train.dtypes):
        objs = [
            c
            for c in X_train.columns
            if X_train[c].dtype == "object" and c not in ("__stem__", "__frame__")
        ]
        if objs:
            raise TypeError(f"Found non-numeric feature columns: {objs}")

    summarize_dataset(X_train, y_train, X_test, y_test, behaviors)

    # Feature column order as written into memmaps (must be stable + recorded)
    all_feature_columns = [
        c for c in X_train.columns if c not in ("__stem__", "__frame__")
    ]
    include_absolute_coordinates = bool(
        training_cfg["include_absolute_coordinates"]
    )
    if include_absolute_coordinates:
        feature_columns = list(all_feature_columns)
    else:
        feature_columns = [
            c for c in all_feature_columns if not is_absolute_coordinate_column(c)
        ]
        dropped_columns = [
            c for c in all_feature_columns if is_absolute_coordinate_column(c)
        ]
        print(
            f"Excluding {len(dropped_columns)} absolute coordinate columns from training input."
        )
    derived_dim = len(feature_columns)
    label_columns = list(behaviors)
    behavior_idx = label_columns.index(behavior)
    helper_columns = ("__stem__", "__frame__")

    X_train = align_columns(
        X_train,
        feature_columns,
        df_name="X_train",
        helper_columns=helper_columns,
        allow_extra=True,
    )
    X_val = align_columns(
        X_val,
        feature_columns,
        df_name="X_val",
        helper_columns=helper_columns,
        allow_extra=True,
    )
    X_test = align_columns(
        X_test,
        feature_columns,
        df_name="X_test",
        helper_columns=helper_columns,
        allow_extra=True,
    )
    y_train = align_columns(
        y_train,
        label_columns,
        df_name="y_train",
        helper_columns=helper_columns,
    )
    y_val = align_columns(
        y_val,
        label_columns,
        df_name="y_val",
        helper_columns=helper_columns,
    )
    y_test = align_columns(
        y_test,
        label_columns,
        df_name="y_test",
        helper_columns=helper_columns,
    )

    split_positive_counts = {
        "train": int(y_train[behavior].sum()),
        "val": int(y_val[behavior].sum()),
        "test": int(y_test[behavior].sum()),
    }
    if split_positive_counts["train"] == 0:
        raise ValueError(
            f"Selected behavior '{behavior}' has zero positive frames in training data."
        )
    if split_positive_counts["val"] == 0:
        print(
            f"⚠️  Selected behavior '{behavior}' has zero positive frames in validation data."
        )
    if split_positive_counts["test"] == 0:
        print(
            f"⚠️  Selected behavior '{behavior}' has zero positive frames in test data."
        )

    train_count, mmX_tr, mmY_tr, tr_vids, tr_starts = make_windowed_memmaps(
        X_train,
        y_train,
        wsize,
        stride,
        derived_dim,
        n_classes,
        str(out / "train"),
    )
    val_count, mmX_va, mmY_va, va_vids, va_starts = make_windowed_memmaps(
        X_val,
        y_val,
        wsize,
        stride,
        derived_dim,
        n_classes,
        str(out / "val"),
    )
    test_count, mmX_te, mmY_te, te_vids, te_starts = make_windowed_memmaps(
        X_test,
        y_test,
        wsize,
        stride,
        derived_dim,
        n_classes,
        str(out / "test"),
    )
    for split_name, count in (
        ("train", train_count),
        ("val", val_count),
        ("test", test_count),
    ):
        if count == 0:
            raise ValueError(
                f"No {split_name} windows were created. "
                "Check window.size/window.stride and per-video frame counts."
            )

    # Persist index arrays (vids + starts) for traceability / later evaluation
    def _save_index(
        name: str, vids: np.ndarray, starts: np.ndarray
    ) -> tuple[Path, Path]:
        vids_path = out / f"{name}_vids.npy"
        starts_path = out / f"{name}_starts.npy"
        np.save(vids_path, vids)
        np.save(starts_path, starts)
        return vids_path, starts_path

    tr_vids_path, tr_starts_path = _save_index("train", tr_vids, tr_starts)
    va_vids_path, va_starts_path = _save_index("val", va_vids, va_starts)
    te_vids_path, te_starts_path = _save_index("test", te_vids, te_starts)

    # Record memmap + index paths explicitly in the manifest
    def _artifact_block(
        name: str, count: int, vids_path: Path, starts_path: Path
    ) -> dict:
        prefix = out / name
        X_path = (prefix.parent / f"{prefix.name}_features.fp32").resolve()
        Y_path = (prefix.parent / f"{prefix.name}_labels.u8").resolve()
        return {
            "count": int(count),
            "X_path": str(X_path),
            "Y_path": str(Y_path),
            "vids_path": str(vids_path.resolve()),
            "starts_path": str(starts_path.resolve()),
            "X_dtype": "float32",
            "Y_dtype": "uint8",
            "X_shape": [int(count), int(wsize), int(derived_dim)],
            "Y_shape": [int(count), int(wsize), int(n_classes)],
        }

    # Write training manifest
    manifest = {
        "kl_config": str(Path(args.kl_config).resolve()),
        "split": str(Path(args.split).resolve()),
        "features_dir": str(features_dir.resolve()),
        "run_dir": str(run_dir.resolve()),
        "behaviors": behaviors,
        "label_columns": label_columns,
        "feature_columns": feature_columns,
        "training": training_cfg,
        "window": {"size": wsize, "stride": stride},
        "counts": {
            "train": train_count,
            "val": val_count,
            "test": test_count,
        },
        "feature_selection": {
            "include_absolute_coordinates": include_absolute_coordinates,
            "n_input_features": len(feature_columns),
        },
        "positive_frames": split_positive_counts,
        "n_features": derived_dim,
        "n_classes": n_classes,
    }

    # Include resolved behavior + focal params in manifest for traceability
    manifest["behavior"] = behavior
    manifest["behavior_idx"] = int(behavior_idx)
    if training_cfg.get("loss", "focal") == "focal":
        manifest["focal"] = {"alpha": alpha, "gamma": gamma}

    manifest["artifacts"] = {
        "train": _artifact_block("train", train_count, tr_vids_path, tr_starts_path),
        "val": _artifact_block("val", val_count, va_vids_path, va_starts_path),
        "test": _artifact_block("test", test_count, te_vids_path, te_starts_path),
    }

    if tf is None:
        raise ImportError(
            "TensorFlow is required for training. "
            "Install tensorflow (or tensorflow-cpu) in this environment."
        )

    # ----------------------------
    # Generators (keypoints-only)
    # ----------------------------
    batch_size = int(training_cfg["batch_size"])
    train_gen = KeypointWindowGenerator(
        mmX_tr,
        mmY_tr,
        behavior_idx=behavior_idx,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        noise_std=float(training_cfg["keypoint_noise_std"]),
    )
    val_gen = KeypointWindowGenerator(
        mmX_va,
        mmY_va,
        behavior_idx=behavior_idx,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        noise_std=0.0,
    )
    test_gen = KeypointWindowGenerator(
        mmX_te,
        mmY_te,
        behavior_idx=behavior_idx,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        noise_std=0.0,
    )

    # ----------------------------
    # Model + compile
    # ----------------------------
    model = build_keypoint_bilstm(wsize, derived_dim)

    lr = float(training_cfg["learning_rate"])
    if training_cfg.get("loss", "focal") != "focal":
        raise ValueError(
            f"Unsupported loss: {training_cfg.get('loss')} (only 'focal' supported)"
        )

    loss_fn = focal_loss(alpha=alpha, gamma=gamma)

    # Metrics: keep lightweight + stable
    metrics = [
        tf.keras.metrics.BinaryAccuracy(name="bin_acc", threshold=0.5),
        tf.keras.metrics.Precision(name="precision", thresholds=0.5),
        tf.keras.metrics.Recall(name="recall", thresholds=0.5),
    ]

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss=loss_fn,
        metrics=metrics,
    )

    # ----------------------------
    # Callbacks
    # ----------------------------
    ckpt_path = out / "best_model.weights.h5"
    interrupted_ckpt_path = out / "interrupted_model.weights.h5"
    history_capture = HistoryCapture()
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_path),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
        ),
        tf.keras.callbacks.CSVLogger(str(out / "train_history.csv")),
        history_capture,
    ]

    if training_cfg.get("reduce_lr", False):
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
            )
        )
    if training_cfg.get("early_stopping", False):
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=int(training_cfg["early_stopping_patience"]),
                min_delta=float(training_cfg["early_stopping_min_delta"]),
                restore_best_weights=True,
                verbose=1,
            )
        )

    # ----------------------------
    # Fit
    # ----------------------------
    epochs = int(training_cfg["epochs"])
    interrupted = False
    interruption_reason = None
    try:
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1,
        )
        history_data = {
            k: [float(x) for x in v] for k, v in (history.history or {}).items()
        }
    except KeyboardInterrupt:
        interrupted = True
        interruption_reason = "keyboard_interrupt"
        print("\n⚠️  Training interrupted; saving partial run artifacts.")
        model.save_weights(str(interrupted_ckpt_path))
        history_data = history_capture.history

    val_loss_history = history_data.get("val_loss", [])
    best_epoch = (
        int(np.argmin(val_loss_history)) + 1 if val_loss_history else None
    )

    evaluation_ckpt_path = None
    if ckpt_path.exists():
        model.load_weights(str(ckpt_path))
        evaluation_ckpt_path = ckpt_path
    elif interrupted_ckpt_path.exists():
        model.load_weights(str(interrupted_ckpt_path))
        evaluation_ckpt_path = interrupted_ckpt_path
    else:
        print(
            "⚠️  No saved checkpoint weights were found; evaluating final in-memory model."
        )

    # ----------------------------
    # Evaluate
    # ----------------------------
    test_metrics = model.evaluate(test_gen, verbose=1)
    test_results = dict(zip(model.metrics_names, [float(x) for x in test_metrics]))
    print("\n=== Test metrics ===")
    for k, v in test_results.items():
        print(f"  {k}: {v}")

    manifest["training_run"] = {
        "interrupted": interrupted,
        "interruption_reason": interruption_reason,
        "checkpoint_best_model": str(ckpt_path.resolve()),
        "checkpoint_interrupted_model": (
            str(interrupted_ckpt_path.resolve())
            if interrupted_ckpt_path.exists()
            else None
        ),
        "evaluation_weights": (
            str(evaluation_ckpt_path.resolve()) if evaluation_ckpt_path else None
        ),
        "history_csv": str((out / "train_history.csv").resolve()),
        "epochs_completed": int(len(history_data.get("loss", []))),
        "best_epoch_by_val_loss": best_epoch,
        "final_metrics": {k: float(v[-1]) for k, v in history_data.items() if len(v) > 0},
        "test_metrics": test_results,
    }

    with open(out / "train_manifest.yml", "w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)

    print(f"\n📝 Wrote {out / 'train_manifest.yml'}")


if __name__ == "__main__":
    main()
