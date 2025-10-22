import numpy as np


def compute_angle(df, p1, p2, p3):
    """
    Compute the angle (in degrees) formed by three keypoints: p1 -> p2 -> p3.
    Assumes that the DataFrame contains columns like 'p1_x', 'p1_y', etc.
    """
    v1_x = df[f"{p1}_x"] - df[f"{p2}_x"]
    v1_y = df[f"{p1}_y"] - df[f"{p2}_y"]
    v2_x = df[f"{p3}_x"] - df[f"{p2}_x"]
    v2_y = df[f"{p3}_y"] - df[f"{p2}_y"]
    dot_product = v1_x * v2_x + v1_y * v2_y
    norm_v1 = np.sqrt(v1_x**2 + v1_y**2)
    norm_v2 = np.sqrt(v2_x**2 + v2_y**2)
    cosine_angle = dot_product / (norm_v1 * norm_v2 + 1e-10)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cosine_angle))


def compute_distance(df, p1, p2):
    """
    Compute the Euclidean distance between two keypoints.
    """
    return np.sqrt(
        (df[f"{p1}_x"] - df[f"{p2}_x"]) ** 2 + (df[f"{p1}_y"] - df[f"{p2}_y"]) ** 2
    )
