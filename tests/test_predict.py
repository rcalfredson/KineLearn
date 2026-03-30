from __future__ import annotations

import sys
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from KineLearn.scripts.predict import (
    align_manifest_features,
    available_feature_stems,
    build_bout_table,
    load_features_for_stems,
    write_per_video_outputs,
)


def _fake_to_parquet(self, path, index=False, *args, **kwargs):
    df = self if index else self.reset_index(drop=True)
    return df.to_pickle(path)


def _fake_read_parquet(path, *args, **kwargs):
    return pd.read_pickle(path)


class PredictHelperTests(unittest.TestCase):
    @patch("KineLearn.scripts.predict.pd.read_parquet", side_effect=_fake_read_parquet)
    @patch("pandas.DataFrame.to_parquet", new=_fake_to_parquet)
    def test_available_feature_stems_discovers_feature_files(self, _mock_read_parquet) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            features_dir = Path(tmpdir)
            pd.DataFrame({"feat_1": [1.0, 2.0]}).to_parquet(
                features_dir / "frame_features_stem_a.parquet", index=False
            )
            pd.DataFrame({"feat_1": [3.0, 4.0]}).to_parquet(
                features_dir / "frame_features_stem_b.parquet", index=False
            )

            stems = available_feature_stems(features_dir)

            self.assertEqual(stems, ["stem_a", "stem_b"])

    @patch("KineLearn.scripts.predict.pd.read_parquet", side_effect=_fake_read_parquet)
    @patch("pandas.DataFrame.to_parquet", new=_fake_to_parquet)
    def test_load_features_for_stems_adds_helper_columns(self, _mock_read_parquet) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            features_dir = Path(tmpdir)
            pd.DataFrame({"feat_1": [1.0, 2.0], "feat_2": [3.0, 4.0]}).to_parquet(
                features_dir / "frame_features_stem_a.parquet", index=False
            )

            loaded = load_features_for_stems(["stem_a"], features_dir)

            self.assertIn("__stem__", loaded.columns)
            self.assertIn("__frame__", loaded.columns)
            self.assertEqual(loaded["__stem__"].tolist(), ["stem_a", "stem_a"])
            self.assertEqual(loaded["__frame__"].tolist(), [0, 1])

    def test_align_manifest_features_raises_on_missing_columns(self) -> None:
        manifest = {
            "behavior": "genitalia_extension",
            "feature_columns": ["feat_1", "feat_missing"],
            "training": {},
        }
        X = pd.DataFrame(
            {
                "feat_1": [1.0, 2.0],
                "__stem__": ["stem_a", "stem_a"],
                "__frame__": [0, 1],
            }
        )

        with self.assertRaises(ValueError):
            align_manifest_features(manifest, X)

    def test_build_bout_table_returns_expected_regions(self) -> None:
        frame_df = pd.DataFrame(
            {
                "__stem__": ["stem_a"] * 6,
                "__frame__": [0, 1, 2, 3, 4, 5],
                "prob_genitalia_extension": [0.1, 0.8, 0.9, 0.2, 0.85, 0.9],
                "pred_genitalia_extension": [0, 1, 1, 0, 1, 1],
            }
        )

        bout_df = build_bout_table(
            frame_df,
            behavior="genitalia_extension",
            min_pred_frames=2,
            max_gap=0,
        )

        self.assertEqual(len(bout_df), 2)
        self.assertEqual(
            bout_df[["start_frame", "end_frame"]].values.tolist(),
            [[1, 2], [4, 5]],
        )

    @patch("pandas.DataFrame.to_parquet", new=_fake_to_parquet)
    def test_write_per_video_outputs_creates_video_subdirs(self) -> None:
        frame_df = pd.DataFrame(
            {
                "__stem__": ["stem_a", "stem_a", "stem_b", "stem_b"],
                "__frame__": [0, 1, 0, 1],
                "prob_genitalia_extension": [0.1, 0.9, 0.2, 0.8],
                "pred_genitalia_extension": [0, 1, 0, 1],
            }
        )
        bout_df = pd.DataFrame(
            {
                "__stem__": ["stem_a", "stem_b"],
                "behavior": ["genitalia_extension", "genitalia_extension"],
                "start_frame": [1, 1],
                "end_frame": [1, 1],
                "n_frames": [1, 1],
                "mean_probability": [0.9, 0.8],
                "max_probability": [0.9, 0.8],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts = write_per_video_outputs(
                frame_df,
                out_dir=Path(tmpdir),
                write_csv=True,
                bout_df=bout_df,
            )

            self.assertTrue((Path(tmpdir) / "videos" / "stem_a" / "frame_predictions.parquet").exists())
            self.assertTrue((Path(tmpdir) / "videos" / "stem_b" / "frame_predictions.parquet").exists())
            self.assertTrue((Path(tmpdir) / "videos" / "stem_a" / "frame_predictions.csv").exists())
            self.assertTrue((Path(tmpdir) / "videos" / "stem_b" / "predicted_bouts.csv").exists())
            self.assertIn("stem_a", artifacts)
            self.assertIn("stem_b", artifacts)


if __name__ == "__main__":
    unittest.main()
