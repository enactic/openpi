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
        print("Directory not found.")
        return

    episode_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir()])

    if not episode_dirs:
        print("No episode directories found.")
        return

    print(f"Found {len(episode_dirs)} episode directories.")

    # Inspect the first episode
    first_episode_dir = episode_dirs[0]
    print(f"\n--- Inspecting first episode: {first_episode_dir.name} ---")

    trajectory_file = first_episode_dir / "trajectory.parquet"
    video_file = first_episode_dir / "video.mp4"

    if not trajectory_file.exists():
        print(f"Trajectory file not found at: {trajectory_file}")
        return

    if not video_file.exists():
        print(f"Video file not found at: {video_file}")

    print(f"Reading trajectory file: {trajectory_file}")
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
                shape = df[col][0].to_numpy().shape
                print(f"- {col}: {shape}")
            except:
                print(f"- {col}: Could not determine shape for first element.")

if __name__ == "__main__":
    tyro.cli(main)
