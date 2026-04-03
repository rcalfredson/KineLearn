from __future__ import annotations

import contextlib
import io
import sys
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from KineLearn.core.manifests import (
    build_ensemble_manifest_payload,
    load_prediction_source,
)
from KineLearn.scripts.create_ensemble import main as create_ensemble_main
from KineLearn.scripts.predict import (
    frame_table_from_prediction_buffers,
    run_window_predictions,
)


class FakeModel:
    def __init__(self, value: float):
        self.value = float(value)

    def predict_on_batch(self, X: np.ndarray) -> np.ndarray:
        batch, window = X.shape[0], X.shape[1]
        return np.full((batch, window, 1), self.value, dtype=np.float32)


def write_train_manifest(path: Path, *, behavior: str = "genitalia_extension", feature_columns=None):
    if feature_columns is None:
        feature_columns = ["feat_1", "feat_2"]

    weights_path = path.parent / "best_model.weights.h5"
    weights_path.write_text("weights\n")
    payload = {
        "behavior": behavior,
        "behavior_idx": 0,
        "label_columns": [behavior],
        "feature_columns": feature_columns,
        "window": {"size": 3, "stride": 1},
        "artifacts": {},
        "feature_selection": {
            "include_absolute_coordinates": False,
            "n_input_features": len(feature_columns),
        },
        "training": {"final_zero_fill": False},
        "training_run": {
            "evaluation_weights": str(weights_path.resolve()),
        },
        "run_dir": str(path.parent.resolve()),
    }
    with open(path, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


class EnsembleTests(unittest.TestCase):
    def test_build_ensemble_manifest_rejects_incompatible_members(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first = root / "a" / "train_manifest.yml"
            second = root / "b" / "train_manifest.yml"
            first.parent.mkdir(parents=True, exist_ok=True)
            second.parent.mkdir(parents=True, exist_ok=True)
            write_train_manifest(first, feature_columns=["feat_1", "feat_2"])
            write_train_manifest(second, feature_columns=["feat_1", "feat_3"])

            with self.assertRaises(ValueError):
                build_ensemble_manifest_payload(
                    [first, second],
                    [yaml.safe_load(first.read_text()), yaml.safe_load(second.read_text())],
                    name="bad",
                )

    def test_create_ensemble_cli_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first = root / "a" / "train_manifest.yml"
            second = root / "b" / "train_manifest.yml"
            out_dir = root / "ensemble"
            first.parent.mkdir(parents=True, exist_ok=True)
            second.parent.mkdir(parents=True, exist_ok=True)
            write_train_manifest(first)
            write_train_manifest(second)

            old_argv = sys.argv
            sys.argv = [
                "kinelearn-create-ensemble",
                "--manifest",
                str(first),
                "--manifest",
                str(second),
                "--name",
                "ge_mean",
                "--out-dir",
                str(out_dir),
            ]
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    create_ensemble_main()
            finally:
                sys.argv = old_argv

            ensemble_path = out_dir / "ensemble_manifest.yml"
            self.assertTrue(ensemble_path.exists())
            payload = yaml.safe_load(ensemble_path.read_text())
            self.assertEqual(payload["manifest_type"], "ensemble")
            self.assertEqual(payload["aggregation"]["n_members"], 2)
            self.assertEqual(payload["ensemble_name"], "ge_mean")

    def test_load_prediction_source_reads_ensemble_members(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first = root / "a" / "train_manifest.yml"
            second = root / "b" / "train_manifest.yml"
            first.parent.mkdir(parents=True, exist_ok=True)
            second.parent.mkdir(parents=True, exist_ok=True)
            write_train_manifest(first)
            write_train_manifest(second)

            ensemble_payload = build_ensemble_manifest_payload(
                [first, second],
                [yaml.safe_load(first.read_text()), yaml.safe_load(second.read_text())],
                name="ge_mean",
            )
            ensemble_path = root / "ensemble_manifest.yml"
            with open(ensemble_path, "w") as f:
                yaml.safe_dump(ensemble_payload, f, sort_keys=False)

            source = load_prediction_source(ensemble_path)

            self.assertEqual(source["manifest_kind"], "ensemble")
            self.assertEqual(source["behavior"], "genitalia_extension")
            self.assertEqual(source["aggregation"]["n_members"], 2)
            self.assertEqual(len(source["members"]), 2)

    def test_run_window_predictions_can_average_across_members(self) -> None:
        X_windows = np.ones((2, 3, 2), dtype=np.float32)
        vids = np.array(["stem_a", "stem_a"], dtype=object)
        starts = np.array([0, 1], dtype=np.int32)
        base_features = pd.DataFrame(
            {
                "feat_1": [1.0, 1.0, 1.0, 1.0],
                "feat_2": [2.0, 2.0, 2.0, 2.0],
                "__stem__": ["stem_a"] * 4,
                "__frame__": [0, 1, 2, 3],
            }
        )

        buffers = run_window_predictions(
            FakeModel(0.2),
            X_windows,
            vids,
            starts,
            batch_size=4,
            window_size=3,
            base_features=base_features,
        )
        buffers = run_window_predictions(
            FakeModel(0.8),
            X_windows,
            vids,
            starts,
            batch_size=4,
            window_size=3,
            base_features=base_features,
            buffers=buffers,
        )

        frame_df = frame_table_from_prediction_buffers(
            buffers,
            behavior="genitalia_extension",
            threshold=None,
        )

        self.assertTrue(np.allclose(frame_df["prob_genitalia_extension"], 0.5))


if __name__ == "__main__":
    unittest.main()
