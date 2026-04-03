#!/usr/bin/env python3
"""
Create a lightweight KineLearn ensemble manifest from multiple train manifests.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from KineLearn.core.manifests import (
    build_ensemble_manifest_payload,
    load_train_manifest,
    save_yaml,
)


def default_out_dir(behavior: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("results") / "ensembles" / behavior / timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create an ensemble_manifest.yml from multiple compatible KineLearn "
            "train_manifest.yml files."
        )
    )
    parser.add_argument(
        "--manifest",
        action="append",
        required=True,
        help="Path to a train_manifest.yml file. Provide once per ensemble member.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Optional human-readable ensemble name recorded in the manifest.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help=(
            "Output directory for the ensemble manifest. Defaults to "
            "results/ensembles/<behavior>/<timestamp>/."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    member_paths = [Path(p) for p in args.manifest]
    member_manifests = [load_train_manifest(path) for path in member_paths]
    behavior = member_manifests[0]["behavior"]

    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir(behavior)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = build_ensemble_manifest_payload(
        member_paths,
        member_manifests,
        name=args.name,
    )
    manifest_path = out_dir / "ensemble_manifest.yml"
    save_yaml(manifest_path, payload)

    print(
        f"Created ensemble for behavior '{behavior}' with "
        f"{payload['aggregation']['n_members']} members."
    )
    print(f"📝 Wrote {manifest_path}")


if __name__ == "__main__":
    main()
