from pathlib import Path

import pandas as pd


def convert_h5_to_csv(h5_files, skip_csv=False):
    """
    Convert DeepLabCut H5 files to CSV format for downstream processing.

    Parameters
    ----------
    h5_files : list[Path | str]
        List of paths to DeepLabCut H5 files.
    skip_csv : bool
        If True, skip conversion if the CSV file already exists.

    Returns
    -------
    list[Path]
        Paths to the generated or existing CSV files.
    """
    csv_files = []

    for h5_file in h5_files:
        h5_path = Path(h5_file)
        csv_path = h5_path.with_suffix(".csv")

        # Skip conversion if the file exists and skipping is enabled
        if skip_csv and csv_path.exists():
            print(f"Skipping conversion for {h5_file}. CSV file exists: {csv_path}")
            csv_files.append(csv_path)
            continue

        # Proceed with conversion
        try:
            print(f"Converting H5 file to CSV: {h5_path}")
            # Load H5 file
            data = pd.read_hdf(h5_path)

            # Flatten hierarchical columns and replace "_likelihood" with "_p"
            flat_data = data.copy()
            flat_data.columns = [
                "_".join(col[-2:]).replace("_likelihood", "_p").strip()
                for col in flat_data.columns.values
            ]

            # Add an explicit index column
            flat_data.insert(0, "index", flat_data.index)

            # Save to CSV
            flat_data.to_csv(csv_path, index=False, encoding="utf-8")
            print(f"CSV file saved: {csv_path}")
            csv_files.append(csv_path)
        except Exception as e:
            print(f"Error converting H5 to CSV for {h5_file}: {e}")
    return csv_files
