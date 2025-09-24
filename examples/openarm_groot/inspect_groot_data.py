"""
This script inspects the openarm_groot dataset to verify its structure and content.

Usage:
uv run examples/openarm_groot/inspect_groot_data.py --data-dir /path/to/your/data
"""

from pathlib import Path
import polars as pl
import tyro

def main(data_dir: Path):
    """
    Inspects the openarm_groot dataset.

    Args:
        data_dir: The path to the raw openarm_groot data.
    """
    print(f"Inspecting directory: {data_dir}")

    if not data_dir.exists():
        print(f"Directory not found: {data_dir}")
        return

    data_path = data_dir / "data"
    videos_path = data_dir / "videos"

    if not data_path.exists() or not videos_path.exists():
        print(f"'data' or 'videos' subdirectory not found in {data_dir}")
        return

    # Find the first parquet file to inspect
    parquet_files = sorted(list(data_path.glob("**/*.parquet")))
    if not parquet_files:
        print("No .parquet files found in the data directory.")
        return

    trajectory_file = parquet_files[0]
    print(f"Found {len(parquet_files)} parquet files. Inspecting the first one: {trajectory_file}")

    # Find the corresponding video file
    episode_name = trajectory_file.stem
    video_file = videos_path / trajectory_file.parent.name / "observation.images.ego_view" / f"{episode_name}.mp4"

    print(f"\n--- Checking for corresponding video file ---")
    if not video_file.exists():
        print(f"Video file not found at: {video_file}")
    else:
        print(f"Found video file: {video_file}")

    print(f"\n--- Reading trajectory file: {trajectory_file} ---")
    df = pl.read_parquet(trajectory_file)

    print("\n--- Trajectory Schema ---")
    for name, dtype in df.schema.items():
        print(f"- {name}: {dtype}")

    print("\n--- Trajectory Data (first 5 rows) ---")
    print(df.head())

    # Check the shape of array columns
    print("\n--- Array Column Shapes ---")
    for col in df.columns:
        if isinstance(df[col].dtype, pl.List):
            try:
                # Get the first non-null value to check the shape
                first_valid = df[col].drop_nulls().head(1)[0]
                if first_valid is not None:
                    shape = first_valid.to_numpy().shape
                    print(f"- {col}: {shape}")
                else:
                    print(f"- {col}: All values are null.")
            except Exception as e:
                print(f"- {col}: Could not determine shape. Error: {e}")

if __name__ == "__main__":
    tyro.cli(main)
