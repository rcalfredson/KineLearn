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

from KineLearn.core.manifests import resolve_recorded_path, resolve_weights_path
from KineLearn.scripts.batch_eval_splits import infer_manifest_path
from KineLearn.scripts.restore_run_artifacts import main as restore_main


def _fake_to_parquet(self, path, index=False, *args, **kwargs):
    df = self if index else self.reset_index(drop=True)
    return df.to_pickle(path)


def _fake_read_parquet(path, *args, **kwargs):
    return pd.read_pickle(path)


def write_manifest(path: Path, features_dir: Path) -> None:
    payload = {
        "behavior": "genitalia_extension",
        "behavior_idx": 0,
        "label_columns": ["genitalia_extension"],
        "feature_columns": ["feat_1", "feat_2"],
        "window": {"size": 2, "stride": 1},
        "artifacts": {
            "train": {
                "count": 2,
                "X_path": "/missing/original/train_features.fp32",
                "Y_path": "/missing/original/train_labels.u8",
                "vids_path": "/missing/original/train_vids.npy",
                "starts_path": "/missing/original/train_starts.npy",
                "X_dtype": "float32",
                "Y_dtype": "uint8",
                "X_shape": [2, 2, 2],
                "Y_shape": [2, 2, 1],
            },
            "val": {
                "count": 2,
                "X_path": "/missing/original/val_features.fp32",
                "Y_path": "/missing/original/val_labels.u8",
                "vids_path": "/missing/original/val_vids.npy",
                "starts_path": "/missing/original/val_starts.npy",
                "X_dtype": "float32",
                "Y_dtype": "uint8",
                "X_shape": [2, 2, 2],
                "Y_shape": [2, 2, 1],
            },
            "test": {
                "count": 2,
                "X_path": "/missing/original/test_features.fp32",
                "Y_path": "/missing/original/test_labels.u8",
                "vids_path": "/missing/original/test_vids.npy",
                "starts_path": "/missing/original/test_starts.npy",
                "X_dtype": "float32",
                "Y_dtype": "uint8",
                "X_shape": [2, 2, 2],
                "Y_shape": [2, 2, 1],
            },
        },
        "feature_selection": {
            "include_absolute_coordinates": False,
            "n_input_features": 2,
        },
        "training": {"final_zero_fill": False},
        "training_run": {
            "evaluation_weights": "/missing/original/best_model.weights.h5",
        },
        "features_dir": str(features_dir.resolve()),
        "run_dir": "/missing/original",
        "resolved_stems": {
            "train": ["stem_a"],
            "val": ["stem_b"],
            "test": ["stem_c"],
        },
        "n_classes": 1,
    }
    with open(path, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


class RestoreRunArtifactsTests(unittest.TestCase):
    def test_resolve_recorded_path_falls_back_to_manifest_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            manifest_path = run_dir / "train_manifest.yml"
            manifest_path.write_text("behavior: x\n")
            local_weights = run_dir / "best_model.weights.h5"
            local_weights.write_text("weights\n")

            resolved = resolve_recorded_path("/missing/original/best_model.weights.h5", manifest_path)

            self.assertEqual(resolved, local_weights.resolve())

    def test_resolve_weights_path_uses_adjacent_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            manifest_path = run_dir / "train_manifest.yml"
            manifest_path.write_text("behavior: x\n")
            local_weights = run_dir / "best_model.weights.h5"
            local_weights.write_text("weights\n")

            resolved = resolve_weights_path(
                {"training_run": {"evaluation_weights": "/missing/original/best_model.weights.h5"}},
                manifest_path,
            )

            self.assertEqual(resolved, local_weights.resolve())

    def test_infer_manifest_path_falls_back_when_summary_manifest_path_is_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sweep_dir = root / "ge_nested"
            manifest_dir = sweep_dir / "03-28-2026" / "20260328_142539"
            manifest_dir.mkdir(parents=True, exist_ok=True)
            manifest_path = manifest_dir / "train_manifest.yml"
            manifest_path.write_text(
                yaml.safe_dump(
                    {
                        "split": str((sweep_dir / "splits" / "outer_seed0" / "train_test_split.yaml").resolve()),
                        "val_split": str((sweep_dir / "splits" / "outer_seed0" / "train_val_split_seed0.yaml").resolve()),
                    },
                    sort_keys=False,
                )
            )
            (sweep_dir / "splits" / "outer_seed0").mkdir(parents=True, exist_ok=True)
            (sweep_dir / "splits" / "outer_seed0" / "train_test_split.yaml").write_text("train: []\ntest: []\n")
            (sweep_dir / "splits" / "outer_seed0" / "train_val_split_seed0.yaml").write_text("train: []\nval: []\n")

            resolved = infer_manifest_path(
                {
                    "manifest_path": "/stale/original/train_manifest.yml",
                    "split_path": str((sweep_dir / "splits" / "outer_seed0" / "train_test_split.yaml").resolve()),
                    "val_split_path": str((sweep_dir / "splits" / "outer_seed0" / "train_val_split_seed0.yaml").resolve()),
                },
                sweep_dir,
            )

            self.assertEqual(resolved, manifest_path.resolve())

    @patch("KineLearn.scripts.restore_run_artifacts.pd.read_parquet", side_effect=_fake_read_parquet)
    @patch("pandas.DataFrame.to_parquet", new=_fake_to_parquet)
    def test_restore_cli_rebuilds_subset_artifacts_next_to_manifest(self, _mock_read_parquet) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            features_dir = root / "features"
            run_dir = root / "archive" / "run_a"
            report_dir = root / "report"
            features_dir.mkdir(parents=True, exist_ok=True)
            run_dir.mkdir(parents=True, exist_ok=True)

            for stem in ("stem_a", "stem_b", "stem_c"):
                pd.DataFrame({"feat_1": [1.0, 2.0, 3.0], "feat_2": [4.0, 5.0, 6.0]}).to_parquet(
                    features_dir / f"frame_features_{stem}.parquet",
                    index=False,
                )
                pd.DataFrame({"genitalia_extension": [0, 1, 0]}).to_parquet(
                    features_dir / f"frame_labels_{stem}.parquet",
                    index=False,
                )

            manifest_path = run_dir / "train_manifest.yml"
            write_manifest(manifest_path, features_dir)
            (run_dir / "best_model.weights.h5").write_text("weights\n")

            old_argv = sys.argv
            sys.argv = [
                "kinelearn-restore-run-artifacts",
                "--manifest",
                str(manifest_path),
                "--report-out",
                str(report_dir),
            ]
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    restore_main()
            finally:
                sys.argv = old_argv

            self.assertTrue((run_dir / "train_features.fp32").exists())
            self.assertTrue((run_dir / "val_features.fp32").exists())
            self.assertTrue((run_dir / "test_features.fp32").exists())
            self.assertTrue((run_dir / "train_vids.npy").exists())
            self.assertTrue((report_dir / "restore_summary.yml").exists())

            summary = yaml.safe_load((report_dir / "restore_summary.yml").read_text())
            restored_statuses = {row["subset"]: row["status"] for row in summary["restored"]}
            self.assertEqual(restored_statuses["train"], "restored")
            self.assertEqual(restored_statuses["val"], "restored")
            self.assertEqual(restored_statuses["test"], "restored")

            vids = np.load(run_dir / "train_vids.npy", allow_pickle=True)
            self.assertEqual(vids.tolist(), ["stem_a", "stem_a"])

    @patch("KineLearn.scripts.restore_run_artifacts.pd.read_parquet", side_effect=_fake_read_parquet)
    @patch("pandas.DataFrame.to_parquet", new=_fake_to_parquet)
    def test_teardown_removes_memmaps_but_keeps_indexes_by_default(self, _mock_read_parquet) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            features_dir = root / "features"
            run_dir = root / "archive" / "run_a"
            report_dir = root / "report"
            features_dir.mkdir(parents=True, exist_ok=True)
            run_dir.mkdir(parents=True, exist_ok=True)

            for stem in ("stem_a", "stem_b", "stem_c"):
                pd.DataFrame({"feat_1": [1.0, 2.0, 3.0], "feat_2": [4.0, 5.0, 6.0]}).to_parquet(
                    features_dir / f"frame_features_{stem}.parquet",
                    index=False,
                )
                pd.DataFrame({"genitalia_extension": [0, 1, 0]}).to_parquet(
                    features_dir / f"frame_labels_{stem}.parquet",
                    index=False,
                )

            manifest_path = run_dir / "train_manifest.yml"
            write_manifest(manifest_path, features_dir)
            (run_dir / "best_model.weights.h5").write_text("weights\n")

            old_argv = sys.argv
            sys.argv = [
                "kinelearn-restore-run-artifacts",
                "--manifest",
                str(manifest_path),
                "--report-out",
                str(report_dir),
            ]
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    restore_main()
            finally:
                sys.argv = old_argv

            old_argv = sys.argv
            sys.argv = [
                "kinelearn-restore-run-artifacts",
                "--manifest",
                str(manifest_path),
                "--teardown",
                "--report-out",
                str(report_dir),
            ]
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    restore_main()
            finally:
                sys.argv = old_argv

            self.assertFalse((run_dir / "train_features.fp32").exists())
            self.assertFalse((run_dir / "train_labels.u8").exists())
            self.assertTrue((run_dir / "train_vids.npy").exists())
            self.assertTrue((run_dir / "train_starts.npy").exists())

    @patch("KineLearn.scripts.restore_run_artifacts.pd.read_parquet", side_effect=_fake_read_parquet)
    @patch("pandas.DataFrame.to_parquet", new=_fake_to_parquet)
    def test_teardown_can_remove_indexes_when_requested(self, _mock_read_parquet) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            features_dir = root / "features"
            run_dir = root / "archive" / "run_a"
            report_dir = root / "report"
            features_dir.mkdir(parents=True, exist_ok=True)
            run_dir.mkdir(parents=True, exist_ok=True)

            for stem in ("stem_a", "stem_b", "stem_c"):
                pd.DataFrame({"feat_1": [1.0, 2.0, 3.0], "feat_2": [4.0, 5.0, 6.0]}).to_parquet(
                    features_dir / f"frame_features_{stem}.parquet",
                    index=False,
                )
                pd.DataFrame({"genitalia_extension": [0, 1, 0]}).to_parquet(
                    features_dir / f"frame_labels_{stem}.parquet",
                    index=False,
                )

            manifest_path = run_dir / "train_manifest.yml"
            write_manifest(manifest_path, features_dir)
            (run_dir / "best_model.weights.h5").write_text("weights\n")

            old_argv = sys.argv
            sys.argv = [
                "kinelearn-restore-run-artifacts",
                "--manifest",
                str(manifest_path),
                "--report-out",
                str(report_dir),
            ]
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    restore_main()
            finally:
                sys.argv = old_argv

            old_argv = sys.argv
            sys.argv = [
                "kinelearn-restore-run-artifacts",
                "--manifest",
                str(manifest_path),
                "--teardown",
                "--remove-indexes",
                "--report-out",
                str(report_dir),
            ]
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    restore_main()
            finally:
                sys.argv = old_argv

            self.assertFalse((run_dir / "train_features.fp32").exists())
            self.assertFalse((run_dir / "train_labels.u8").exists())
            self.assertFalse((run_dir / "train_vids.npy").exists())
            self.assertFalse((run_dir / "train_starts.npy").exists())


if __name__ == "__main__":
    unittest.main()
