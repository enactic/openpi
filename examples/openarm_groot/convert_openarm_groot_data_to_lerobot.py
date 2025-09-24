"""
This script converts a custom dataset in the openarm_groot format to the LeRobot format.

Usage:
uv run examples/openarm_groot/convert_openarm_groot_data_to_lerobot.py --data-dir /path/to/your/data
"""

import json
import shutil
from pathlib import Path

import numpy as np
import polars as pl
import tyro
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

# TODO: Replace with a local directory name for your dataset.
REPO_NAME = "openarm_groot_lerobot"
# This is the frames per second of the videos. From the README, this is 20.
FPS = 20


def main(data_dir: Path):
    """
    Converts the openarm_groot dataset to the LeRobot format.

    Args:
        data_dir: The path to the raw openarm_groot data, which contains
                  the `data`, `meta`, and `videos` directories.
    """
    # Clean up any existing dataset in the output directory
    output_path = HF_LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        print(f"Removing existing dataset at {output_path}")
        shutil.rmtree(output_path)

    # Create the LeRobot dataset with the correct feature shapes.
    # We use float32 as it's the standard for training.
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        fps=FPS,
        features={
            "observation.images.ego_view": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "action": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["action"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Load the task descriptions from meta/tasks.jsonl
    tasks_file = data_dir / "meta" / "tasks.jsonl"
    if not tasks_file.exists():
        raise FileNotFoundError(f"tasks.jsonl not found at {tasks_file}")

    task_map = {}
    with tasks_file.open("r") as f:
        for line in f:
            task_data = json.loads(line)
            task_map[task_data["task_index"]] = task_data["task"]
    print(f"Loaded {len(task_map)} task descriptions.")

    # Find all trajectory files in the data directory
    parquet_files = sorted(list(data_dir.glob("data/**/*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No .parquet files found in {data_dir / 'data'}")

    print(f"Found {len(parquet_files)} trajectory files to process.")

    for trajectory_file in parquet_files:
        print(f"Processing episode: {trajectory_file.stem}")

        # Determine the corresponding video file path
        episode_name = trajectory_file.stem
        chunk_name = trajectory_file.parent.name
        video_file = (
            data_dir / "videos" / chunk_name / "observation.images.ego_view" / f"{episode_name}.mp4"
        )

        if not video_file.exists():
            print(f"Warning: Video file not found for {episode_name}, skipping episode.")
            continue

        df = pl.read_parquet(trajectory_file)

        for row in df.iter_rows(named=True):
            task_index = row["annotation.human.action.task_description"]
            task_description = task_map.get(task_index, "")

            if not task_description:
                print(f"Warning: No task description found for task_index {task_index}")

            dataset.add_frame(
                {
                    "observation.state": np.array(row["observation.state"], dtype=np.float32),
                    "action": np.array(row["action"], dtype=np.float32),
                    "task": task_description,
                }
            )

        dataset.save_episode(video_path=video_file)

    print("\nDataset conversion complete!")
    print(f"The converted dataset is saved at: {output_path}")


if __name__ == "__main__":
    tyro.cli(main)
