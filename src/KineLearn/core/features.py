import numpy as np
import pandas as pd

from KineLearn.core.geometry import compute_angle, compute_distance


def extract_features(dlc_file, kl_config):
    """
    Extract motion and geometry features from a DeepLabCut CSV file using parameters from KineLearn config.

    Parameters
    ----------
    dlc_file : Path or str
        Path to the DeepLabCut CSV file.
    kl_config : dict
        Parsed KineLearn YAML configuration (contains feature definitions).

    Returns
    -------
    df_combined : pd.DataFrame
        All derived features including coordinates, relative, velocity, acceleration, angles, distances.
    df_xy : pd.DataFrame
        Absolute X/Y coordinates.
    df_p : pd.DataFrame
        Likelihood / probability columns.
    """
    df = pd.read_csv(dlc_file)
    df.columns = df.columns.str.strip()

    # --- Basic column parsing ---
    keypoints = sorted(set(col[:-2] for col in df.columns if col.endswith("_x")))
    xy_cols = [f"{kp}_{ax}" for kp in keypoints for ax in ("x", "y")]
    p_cols = [f"{kp}_p" for kp in keypoints]

    df_xy = df[xy_cols].copy().apply(pd.to_numeric, errors="coerce")
    df_p = df[p_cols].copy().apply(pd.to_numeric, errors="coerce")
    df_p.columns = [col[:-2] + "_probability" for col in df_p.columns]

    # --- Load geometric parameters from config ---
    features_cfg = kl_config["features"]
    ref_pt = features_cfg.get("ref_pt", "thorax")
    bl_pts = features_cfg.get("body_length_pts", ["head", "thorax"])

    # --- Compute body length normalization vector ---
    body_length = compute_distance(df_xy, bl_pts[0], bl_pts[1]).to_numpy()

    # --- Compute relative positions (normalized to ref_pt & body_length) ---
    df_relative = df_xy.copy()
    for col in df_xy.columns:
        if "_x" in col:
            df_relative[col] = (df_xy[col] - df_xy[f"{ref_pt}_x"]) / body_length
        elif "_y" in col:
            df_relative[col] = (df_xy[col] - df_xy[f"{ref_pt}_y"]) / body_length

    df_relative.columns = [
        col.replace("_x", "_coord_x").replace("_y", "_coord_y")
        for col in df_relative.columns
    ]

    # --- Compute velocity & acceleration (framewise differences) ---
    df_velocity = df_xy.diff().fillna(0) / body_length[:, np.newaxis]
    df_velocity.columns = [
        col.replace("_x", "_velocity_x").replace("_y", "_velocity_y")
        for col in df_velocity.columns
    ]

    df_acceleration = df_velocity.diff().fillna(0)
    df_acceleration.columns = [
        col.replace("_velocity_x", "_acceleration_x").replace(
            "_velocity_y", "_acceleration_y"
        )
        for col in df_velocity.columns
    ]

    # --- Compute angles dynamically ---
    angle_defs = features_cfg.get("angles", [])
    angle_features = {}
    for pts in angle_defs:
        a, b, c = pts
        name = f"angle_{a}_{b}_{c}"
        angle_features[name] = compute_angle(df_xy, a, b, c)
    df_angles = pd.DataFrame(angle_features)

    # --- Compute distances dynamically ---
    dist_defs = features_cfg.get("distances", [])
    dist_features = {}
    for p1, p2 in dist_defs:
        name = f"distance_{p1}_{p2}"
        dist_features[name] = compute_distance(df_xy, p1, p2)
    df_distances = pd.DataFrame(dist_features)

    # --- Combine all derived features ---
    df_derived = pd.concat(
        [df_relative, df_velocity, df_acceleration, df_angles, df_distances],
        axis=1,
    )
    df_combined = pd.concat([df_xy, df_derived], axis=1)

    return df_combined, df_xy, df_p
