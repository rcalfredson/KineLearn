from __future__ import annotations

import contextlib
import io
import sys
import tempfile
from pathlib import Path
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from KineLearn.scripts.archive_results import (
    build_archive_plan,
    execute_archive,
    main,
    should_omit,
)


class ArchiveResultsTests(unittest.TestCase):
    def test_should_omit_only_memmap_payloads(self) -> None:
        self.assertTrue(should_omit(Path("train_features.fp32")))
        self.assertTrue(should_omit(Path("test_labels.u8")))
        self.assertFalse(should_omit(Path("train_vids.npy")))
        self.assertFalse(should_omit(Path("train_starts.npy")))
        self.assertFalse(should_omit(Path("train_manifest.yml")))

    def test_build_archive_plan_keeps_metadata_and_prunes_memmaps(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "results" / "genitalia_extension" / "20260402_120000"
            destination = root / "archive" / "genitalia_extension" / "20260402_120000"
            source.mkdir(parents=True, exist_ok=True)

            kept_files = [
                source / "train_manifest.yml",
                source / "best_model.weights.h5",
                source / "train_history.csv",
                source / "train_vids.npy",
                source / "train_starts.npy",
                source / "eval" / "frame_predictions.parquet",
            ]
            omitted_files = [
                source / "train_features.fp32",
                source / "train_labels.u8",
                source / "nested" / "val_features.fp32",
            ]

            for path in kept_files + omitted_files:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"abc")

            plan = build_archive_plan(source, destination)

            moved_sources = {src for src, _, _ in plan.moved_files}
            omitted_sources = {src for src, _ in plan.omitted_files}
            self.assertEqual(moved_sources, set(kept_files))
            self.assertEqual(omitted_sources, set(omitted_files))
            self.assertEqual(
                {dst for _, dst, _ in plan.moved_files},
                {
                    destination / "train_manifest.yml",
                    destination / "best_model.weights.h5",
                    destination / "train_history.csv",
                    destination / "train_vids.npy",
                    destination / "train_starts.npy",
                    destination / "eval" / "frame_predictions.parquet",
                },
            )

    def test_main_dry_run_reports_without_mutating(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "results" / "behavior" / "run_a"
            destination = root / "archive" / "behavior" / "run_a"
            source.mkdir(parents=True, exist_ok=True)
            kept = source / "train_manifest.yml"
            omitted = source / "train_features.fp32"
            kept.write_bytes(b"manifest")
            omitted.write_bytes(b"0123456789")

            stdout = io.StringIO()
            old_argv = sys.argv
            sys.argv = [
                "kinelearn-archive-results",
                str(source),
                str(destination),
                "--dry-run",
            ]
            try:
                with contextlib.redirect_stdout(stdout):
                    main()
            finally:
                sys.argv = old_argv

            output = stdout.getvalue()
            self.assertIn("Would move files: 1", output)
            self.assertIn("Would omit memmaps: 1", output)
            self.assertTrue(kept.exists())
            self.assertTrue(omitted.exists())
            self.assertFalse(destination.exists())

    def test_execute_archive_moves_files_and_removes_omitted_memmaps(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "results" / "behavior" / "run_b"
            destination = root / "archive" / "behavior" / "run_b"
            source.mkdir(parents=True, exist_ok=True)

            kept = source / "nested" / "train_manifest.yml"
            omitted = source / "nested" / "train_features.fp32"
            kept.parent.mkdir(parents=True, exist_ok=True)
            kept.write_text("manifest: true\n")
            omitted.write_bytes(b"123456")

            plan = build_archive_plan(source, destination)
            execute_archive(plan, verbose=False)

            self.assertFalse(source.exists())
            self.assertTrue((destination / "nested" / "train_manifest.yml").exists())
            self.assertFalse((destination / "nested" / "train_features.fp32").exists())


if __name__ == "__main__":
    unittest.main()
