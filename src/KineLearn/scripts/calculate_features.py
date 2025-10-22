import argparse

import cv2
import yaml


def main():
    parser = argparse.ArgumentParser(description="Calculate features from DLC outputs.")
    parser.add_argument(
        "-v",
        required=True,
        help="Path to a YAML file containing a list of videos to process",
    )
    parser.add_argument(
        "--dlc-config", required=True, help="Path to DeepLabCut configuration file"
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

    with open(args.v, "r") as f:
        video_paths = yaml.safe_load(f)

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


if __name__ == "__main__":
    main()
