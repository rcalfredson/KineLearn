from __future__ import annotations

import os
os.environ.setdefault("MPLBACKEND", "Agg")

import sys
import tempfile
from pathlib import Path
import unittest

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from KineLearn.scripts.plot_timeline import infer_behaviors, plot_video_timeline


class PlotTimelineTests(unittest.TestCase):
    def test_infer_behaviors_from_probability_columns(self) -> None:
        frame_df = pd.DataFrame(
            {
                "__stem__": ["stem_a"],
                "__frame__": [0],
                "prob_genitalia_extension": [0.2],
                "prob_back_leg_together": [0.3],
            }
        )

        behaviors = infer_behaviors(frame_df)

        self.assertEqual(behaviors, ["back_leg_together", "genitalia_extension"])

    def test_plot_timeline_writes_png_for_each_stem(self) -> None:
        frame_df = pd.DataFrame(
            {
                "__stem__": ["stem_a"] * 6 + ["stem_b"] * 6,
                "__frame__": list(range(6)) + list(range(6)),
                "prob_genitalia_extension": [
                    0.1, 0.7, 0.8, 0.2, 0.6, 0.1,
                    0.2, 0.1, 0.75, 0.8, 0.3, 0.2,
                ],
                "pred_genitalia_extension": [
                    0, 1, 1, 0, 1, 0,
                    0, 0, 1, 1, 0, 0,
                ],
                "true_genitalia_extension": [
                    0, 1, 1, 0, 0, 0,
                    0, 0, 1, 1, 0, 0,
                ],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            out_root = Path(tmpdir)
            for stem, group in frame_df.groupby("__stem__", sort=True):
                written = plot_video_timeline(
                    group,
                    stem=str(stem),
                    behaviors=["genitalia_extension"],
                    threshold=0.6,
                    fps=None,
                    width=8.0,
                    height_per_behavior=2.5,
                    out_dir=out_root / str(stem),
                    fmt="png",
                )
                self.assertEqual(len(written), 1)

            self.assertTrue((out_root / "stem_a" / "timeline.png").exists())
            self.assertTrue((out_root / "stem_b" / "timeline.png").exists())

    def test_plot_timeline_supports_threshold_without_pred_columns(self) -> None:
        frame_df = pd.DataFrame(
            {
                "__stem__": ["stem_a"] * 5,
                "__frame__": list(range(5)),
                "prob_genitalia_extension": [0.1, 0.7, 0.8, 0.2, 0.1],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            written = plot_video_timeline(
                frame_df,
                stem="stem_a",
                behaviors=["genitalia_extension"],
                threshold=0.6,
                fps=60.0,
                width=8.0,
                height_per_behavior=2.5,
                out_dir=Path(tmpdir) / "stem_a",
                fmt="both",
            )

            self.assertEqual(len(written), 2)
            self.assertTrue((Path(tmpdir) / "stem_a" / "timeline.png").exists())
            self.assertTrue((Path(tmpdir) / "stem_a" / "timeline.pdf").exists())


if __name__ == "__main__":
    unittest.main()
