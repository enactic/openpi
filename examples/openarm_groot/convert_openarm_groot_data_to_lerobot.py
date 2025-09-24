"""
This script converts a custom dataset in the openarm_groot format to the LeRobot format.

Usage:
uv run examples/openarm_groot/convert_openarm_groot_data_to_lerobot.py --data-dir /path/to/your/data

To push the dataset to the Hugging Face Hub, use the --push-to-hub flag:
uv run examples/openarm_groot/convert_openarm_groot_data_to_lerobot.py --data-dir /path/to/your/data --push-to-hub
"""

import shutil
from pathlib import Path
import polars as pl
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
import tyro

# TODO: Replace with your Hugging Face username and a dataset name.
REPO_NAME = "your_hf_username/openarm_groot"
# This is the frames per second of the videos. From the README, this is 20.
FPS = 20


def main(data_dir: Path):
    """
    Converts the openarm_groot dataset to the LeRobot format.

    Args:
        data_dir: The path to the raw openarm_groot data.
    """
    # Clean up any existing dataset in the output directory
    output_path = HF_LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        print(f"Removing existing dataset at {output_path}")
        shutil.rmtree(output_path)

    # TODO: Adjust the feature definition based on your dataset's specifics.
    # The shapes for state and actions are placeholders.
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
                "shape": (10,),  # TODO: Change this to your state dimension
                "names": ["state"],
            },
            "action": {
                "dtype": "float32",
                "shape": (7,),  # TODO: Change this to your action dimension
                "names": ["action"],
            },
        },
        # These can be adjusted for performance
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Find all episode directories. This assumes a structure like:
    # data_dir/
    # |- episode_000000/
    # |  |- trajectory.parquet
    # |  |- video.mp4
    # |- episode_000001/
    # |  |- trajectory.parquet
    # |  |- video.mp4
    # ...
    episode_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir()])

    for episode_dir in episode_dirs:
        print(f"Processing episode: {episode_dir.name}")
        trajectory_file = episode_dir / "trajectory.parquet"
        video_file = episode_dir / "video.mp4"

        if not trajectory_file.exists() or not video_file.exists():
            print(f"Skipping {episode_dir.name}: missing trajectory or video file.")
            continue

        # Read the trajectory data
        df = pl.read_parquet(trajectory_file)

        # Get the task description. Assuming it's the same for all steps in an episode.
        # TODO: Change this if the task description varies per step.
        task_description = df["annotation.human.action.task_description"][0]

        for row in df.iter_rows(named=True):
            dataset.add_frame(
                {
                    "observation.state": row["observation.state"],
                    "action": row["action"],
                    "task": task_description,
                }
            )

        dataset.save_episode(video_path=video_file)

    print("Dataset conversion complete.")

    if push_to_hub:
        print("Pushing dataset to Hugging Face Hub...")
        dataset.push_to_hub(
            tags=["openarm_groot", "lerobot"],
            private=False,  # Set to True if you want a private dataset
            push_videos=True,
            license="apache-2.0",
        )
        print("Push to Hub complete.")


if __name__ == "__main__":
    tyro.cli(main)
