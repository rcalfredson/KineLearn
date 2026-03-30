from __future__ import annotations

import sys
import tempfile
from pathlib import Path
import types
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

sklearn = types.ModuleType("sklearn")
model_selection = types.ModuleType("sklearn.model_selection")
model_selection.train_test_split = lambda *args, **kwargs: (_ for _ in ()).throw(
    NotImplementedError("train_test_split stub should not be called in these tests")
)
sklearn.model_selection = model_selection
sys.modules.setdefault("sklearn", sklearn)
sys.modules.setdefault("sklearn.model_selection", model_selection)

from KineLearn.scripts.split_variability import load_split_file
from KineLearn.scripts.predict import resolve_requested_stems


class SplitParsingTests(unittest.TestCase):
    def test_load_split_file_parses_legacy_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "legacy_split.txt"
            path.write_text(
                "Train videos:\n"
                "stem_a\n"
                "stem_b\n"
                "Test videos:\n"
                "stem_c\n"
            )

            split_info = load_split_file(path)

            self.assertEqual(split_info["train videos"], ["stem_a", "stem_b"])
            self.assertEqual(split_info["test videos"], ["stem_c"])

    def test_load_split_file_parses_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "split.yaml"
            path.write_text(
                "seed: 42\n"
                "train:\n"
                "  - stem_a\n"
                "  - stem_b\n"
                "test:\n"
                "  - stem_c\n"
            )

            split_info = load_split_file(path)

            self.assertEqual(split_info["train"], ["stem_a", "stem_b"])
            self.assertEqual(split_info["test"], ["stem_c"])

    def test_resolve_requested_stems_allows_unique_suffix_match(self) -> None:
        available = [
            "output_video_20250730_181758_cropped_wheel_20250730_181758",
            "output_video_20250708_163744_cropped_wheel_20250708_163744",
        ]

        resolved = resolve_requested_stems(
            ["20250730_181758"],
            available,
            where="test",
        )

        self.assertEqual(
            resolved,
            ["output_video_20250730_181758_cropped_wheel_20250730_181758"],
        )

    def test_resolve_requested_stems_raises_on_ambiguous_suffix(self) -> None:
        available = [
            "prefix_a_20250730_181758",
            "prefix_b_20250730_181758",
        ]

        with self.assertRaises(ValueError):
            resolve_requested_stems(["20250730_181758"], available, where="test")


if __name__ == "__main__":
    unittest.main()
