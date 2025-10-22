import numpy as np
import pandas as pd


def parse_boris_labels(behaviors, label_file, num_frames):
    """
    Parse BORIS annotation file into a frame-by-behavior label matrix.

    Parameters
    ----------
    behaviors : list of str
        List of behavior names to include as columns in the output matrix.
    label_file : str or Path
        Path to a BORIS annotation file (tab-delimited) containing
        'Behavior', 'Behavior type' (START/STOP), and 'Image index' columns.
    num_frames : int
        Total number of frames in the corresponding video.

    Returns
    -------
    pd.DataFrame
        A DataFrame of shape (num_frames, len(behaviors)), where each column
        corresponds to a behavior and each cell is 1 if that behavior was
        active in that frame, else 0.

    Notes
    -----
    Each behavior is activated between its own START and STOP events,
    independently of other behaviors. Overlaps between different behaviors
    are fully supported. Unmatched START events are ignored unless followed
    by a STOP.
    """
    df_gt = pd.read_csv(label_file, sep="\t")
    label_array = np.zeros((num_frames, len(behaviors)), dtype=int)
    df_gt = df_gt.sort_values(by="Image index")
    active_starts = {}
    for _, row in df_gt.iterrows():
        behavior = row["Behavior"].strip()
        btype = row["Behavior type"].strip().upper()
        frame_idx = int(row["Image index"])
        if behavior not in behaviors:
            continue
        col_idx = behaviors.index(behavior)
        if btype == "START":
            active_starts[behavior] = frame_idx
        elif btype == "STOP" and behavior in active_starts:
            start_frame = active_starts.pop(behavior)
            label_array[start_frame:frame_idx, col_idx] = 1
    return pd.DataFrame(label_array, columns=behaviors)
