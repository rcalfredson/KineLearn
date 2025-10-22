import argparse
from pathlib import Path

import cv2
import yaml

from KineLearn.core.keypoints import convert_h5_to_csv
from KineLearn.core.path import find_unique


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

    print("kl config:", kl_config)
    print("dlc config:", dlc_config)

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

        if dlc_file is None:
            h5_file = find_unique(video_dir, [h5_pattern])
            if h5_file:
                print(f" → No CSV found, converting {h5_file} to CSV…")
                dlc_file = convert_h5_to_csv([h5_file], skip_csv=True)[0]
            else:
                raise FileNotFoundError(
                    f"No DLC CSV found for video {basename} (Task={task}, date={date}) in {video_dir}"
                )


if __name__ == "__main__":
    main()
