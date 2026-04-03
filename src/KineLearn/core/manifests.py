from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


TRAIN_MANIFEST_REQUIRED_KEYS = [
    "behavior",
    "behavior_idx",
    "label_columns",
    "feature_columns",
    "window",
    "artifacts",
    "feature_selection",
    "training_run",
]

def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r") as f:
        payload = yaml.safe_load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    return payload


def save_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def require_keys(d: dict[str, Any], keys: list[str], where: str) -> None:
    missing = [k for k in keys if k not in d]
    if missing:
        raise ValueError(f"Missing keys {missing} in {where}")


def as_path(value: str | Path) -> Path:
    return value if isinstance(value, Path) else Path(value)


def resolve_recorded_path(value: str | Path, manifest_path: Path) -> Path:
    recorded = as_path(value)
    if recorded.is_absolute():
        if recorded.exists():
            return recorded
        adjacent = manifest_path.parent / recorded.name
        if adjacent.exists():
            return adjacent.resolve()
        return recorded

    relative = (manifest_path.parent / recorded).resolve()
    if relative.exists():
        return relative

    adjacent = (manifest_path.parent / recorded.name).resolve()
    if adjacent.exists():
        return adjacent

    return relative


def load_train_manifest(path: Path) -> dict[str, Any]:
    manifest = load_yaml(path)
    if manifest.get("manifest_type") == "ensemble":
        raise ValueError(
            f"{path} is an ensemble manifest. This command expects a train_manifest.yml file."
        )
    require_keys(manifest, TRAIN_MANIFEST_REQUIRED_KEYS, f"manifest {path}")
    return manifest


def validate_train_manifests(manifests: list[dict[str, Any]], subset: str) -> None:
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


def resolve_weights_path(manifest: dict[str, Any], manifest_path: Path) -> Path:
    training_run = manifest.get("training_run", {})
    candidates = [
        training_run.get("evaluation_weights"),
        training_run.get("checkpoint_best_model"),
        training_run.get("checkpoint_interrupted_model"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = resolve_recorded_path(candidate, manifest_path)
        if path.exists():
            return path
    raise FileNotFoundError(f"No usable weights file found for manifest {manifest_path}.")
