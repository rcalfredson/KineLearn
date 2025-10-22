import argparse
from pathlib import Path

import cv2
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
import yaml

from KineLearn.core.features import extract_features
from KineLearn.core.keypoints import convert_h5_to_csv
from KineLearn.core.path import find_unique

training_features_dict = {
    "coordinates": [],
    "velocity": [],
    "acceleration": [],
    "angles": [],
    "distances": [],
}

prefix_map = {
    "coordinates": "coord_",
    "velocity": "velocity_",
    "acceleration": "acceleration_",
    "angles": "angle_",
    "distances": "distance_",
}


def main():
    parser = argparse.ArgumentParser(description="Calculate features from DLC outputs.")
    parser.add_argument(
        "-v",
        required=True,
        help="Path to a YAML file containing a list of videos to process",
    )
    parser.add_argument(
        "--kl-config", required=True, help="Path to KineLearn configuration file"
    )
    parser.add_argument(
        "--create-scalers",
        action="store_true",
        help=(
            "If set, create new StandardScaler objects for features. "
            "If not set, existing scalers must be available to load; "
            "otherwise an error will be raised."
        ),
    )
    parser.add_argument(
        "--out",
        default="features/",
        help="Directory where the computed feature files will be saved (default: features/)",
    )

    args = parser.parse_args()

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    scaler_dir = Path("scalers")
    scaler_dir.mkdir(parents=True, exist_ok=True)

    # Load videos
    with open(args.v, "r") as f:
        video_paths = yaml.safe_load(f)

    # Load KineLearn config
    with open(args.kl_config, "r") as f:
        kl_config = yaml.safe_load(f)

    # Load DLC config
    if not "dlc_config" in kl_config:
        raise ValueError(
            "Missing DeepLabCut config path. Add key 'dlc_config' to your KineLearn config file."
        )
    with open(kl_config["dlc_config"], "r") as f:
        dlc_config = yaml.safe_load(f)

    # Ensure all videos have the same frame rate
    fps_list = []
    for vp in video_paths:
        cap = cv2.VideoCapture(vp)
        if not cap.isOpened():
            raise IOError(f"Cannot open video to read FPS: {vp}")
        fps_list.append(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
    unique_fps = set(fps_list)
    if len(unique_fps) != 1:
        raise ValueError(f"Mixed FPS detected: {sorted(unique_fps)}")
    fps = unique_fps.pop()
    print(f"FPS: {fps}")

    for video_path_str in video_paths:
        # find DLC CSV in the same folder
        video_path = Path(video_path_str)
        video_dir = video_path.parent
        basename = video_path.stem  # filename without extension

        # Build search patterns using DLC config
        task = dlc_config["Task"]
        date = dlc_config["date"]

        csv_pattern = f"{basename}DLC*{task}{date}*.csv"
        h5_pattern = f"{basename}DLC*{task}{date}*.h5"

        dlc_file = find_unique(video_dir, [csv_pattern], must_contain="DLC")

        print(f"Processing {basename}")

        if dlc_file is None:
            h5_file = find_unique(video_dir, [h5_pattern])
            if h5_file:
                print(f" → No CSV found, converting {h5_file} to CSV…")
                dlc_file = convert_h5_to_csv([h5_file], skip_csv=True)[0]
            else:
                raise FileNotFoundError(
                    f"No DLC CSV found for video {basename} (Task={task}, date={date}) in {video_dir}"
                )

        # Get ground truth label file
        tsv_pattern = f"*{video_path.stem}*.tsv"
        # labels_file = find_unique(video_dir, [tsv_pattern], must_contain="ground_truth")
        features_csv = output_dir / f"extracted_features_{video_path.stem}.csv"

        df_combined, df_xy, df_p = extract_features(dlc_file, kl_config)

        df_out = pd.concat([df_combined, df_p], axis=1)
        df_out.to_csv(features_csv, index=False)
        print(f"Features written: {features_csv}")

        # Accumulate for potential scaler fitting
        for ft, prefix in prefix_map.items():
            if prefix == "":
                continue
            training_features_dict[ft].append(df_combined.filter(like=prefix))

    if args.create_scalers:
        print("\nFitting new scalers...")
        scalers = {}
        kl_config.setdefault("scalers", {})

        config_stem = Path(args.kl_config).stem

        for ft, df_list in training_features_dict.items():
            if not df_list:
                continue
            all_data = pd.concat(df_list, ignore_index=True)
            scaler = StandardScaler().fit(all_data)

            scaler_path = scaler_dir / f"scaler_{config_stem}_{ft}.pkl"
            joblib.dump(scaler, scaler_path)
            scalers[ft] = scaler_path
            kl_config["scalers"][ft] = str(scaler_path)

            print(f"  → saved {ft} scaler: {scaler_path}")

        # Update the KineLearn config file with new scaler paths
        with open(args.kl_config, "w") as f:
            yaml.safe_dump(kl_config, f)
        print(f"Updated KineLearn config with scaler paths: {args.kl_config}")
    else:
        # Load existing scalers
        if "scalers" not in kl_config:
            raise ValueError(
                "No scalers defined in config and --create-scalers not set. "
                "Run with --create-scalers first."
            )

        scalers = {}
        for ft, path_str in kl_config["scalers"].items():
            path = Path(path_str)
            if not path.exists():
                raise FileNotFoundError(f"Scaler file not found: {path}")
            scalers[ft] = joblib.load(path)
            print(f"Loaded existing scaler for {ft}: {path}")


if __name__ == "__main__":
    main()
