#!/usr/bin/env python
"""
A script to convert DVRK (da Vinci Research Kit) robotics data into the LeRobot format (v3.0).

This script processes DVRK surgical robot datasets organized in directory structures
with CSV kinematics data and multiple camera views. It handles both perfect and
recovery demonstrations, extracting dual-arm PSM states, actions, and multi-camera
observations into a LeRobotDataset for the Hugging Face Hub.

Expected DVRK Dataset Structure:
--------------------------------
The script expects a directory structure organized by tissue and subtasks:

/path/to/dataset/
├── tissue_10/                          # Tissue phantom number
│   ├── 1_suture_throw/                 # Subtask directory
│   │   ├── episode_001/                # Individual episode
│   │   │   ├── left_img_dir/           # Left endoscope images
│   │   │   │   └── frame000000_left.jpg
│   │   │   ├── right_img_dir/          # Right endoscope images
│   │   │   │   └── frame000000_right.jpg
│   │   │   ├── endo_psm1/              # PSM1 wrist camera
│   │   │   │   └── frame000000_psm1.jpg
│   │   │   ├── endo_psm2/              # PSM2 wrist camera
│   │   │   │   └── frame000000_psm2.jpg
│   │   │   └── ee_csv.csv              # Kinematics data (16D state + actions)
│   │   └── episode_002/
│   └── 2_needle_pass_recovery/         # Recovery demonstrations
└── tissue_11/

Data Format:
------------
- **Actions**: 16D dual-PSM Cartesian poses + jaw positions (absolute coordinates + quaternions)
- **States**: 16D dual-PSM current poses + jaw positions
- **Images**: 4 camera views (endoscope left/right, PSM1/2 wrist cameras)
- **Metadata**: Tool types, instruction text, recovery/perfect labels

Usage:
------
    python dvrk_zarr_to_lerobot.py --data-path /path/to/dataset --repo-id username/dataset-name

To also push to the Hugging Face Hub:
    python dvrk_zarr_to_lerobot.py --data-path /path/to/dataset --repo-id username/dataset-name --push-to-hub

Dependencies:
-------------
- lerobot v0.6.0
- tyro
- pandas
- PIL
- numpy
"""

import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tyro
from lerobot.datasets.io_utils import write_info
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME
from PIL import Image

states_name = [
    "psm1_pose.position.x",
    "psm1_pose.position.y",
    "psm1_pose.position.z",
    "psm1_pose.orientation.x",
    "psm1_pose.orientation.y",
    "psm1_pose.orientation.z",
    "psm1_pose.orientation.w",
    "psm1_jaw",
    "psm2_pose.position.x",
    "psm2_pose.position.y",
    "psm2_pose.position.z",
    "psm2_pose.orientation.x",
    "psm2_pose.orientation.y",
    "psm2_pose.orientation.z",
    "psm2_pose.orientation.w",
    "psm2_jaw",
]
actions_name = [
    "psm1_sp.position.x",
    "psm1_sp.position.y",
    "psm1_sp.position.z",
    "psm1_sp.orientation.x",
    "psm1_sp.orientation.y",
    "psm1_sp.orientation.z",
    "psm1_sp.orientation.w",
    "psm1_jaw_sp",
    "psm2_sp.position.x",
    "psm2_sp.position.y",
    "psm2_sp.position.z",
    "psm2_sp.orientation.x",
    "psm2_sp.orientation.y",
    "psm2_sp.orientation.z",
    "psm2_sp.orientation.w",
    "psm2_jaw_sp",
]


def read_images(image_dir: str, file_pattern: str) -> np.ndarray:
    """Reads images from a directory into a NumPy array."""
    images = []
    ## count images in the dir
    num_images = len(
        [
            name
            for name in os.listdir(image_dir)
            if os.path.isfile(os.path.join(image_dir, name))
        ]
    )
    for idx in range(num_images):
        filename = os.path.join(image_dir, file_pattern.format(idx))
        if not os.path.exists(filename):
            print(f"Warning: {filename} does not exist.")
            continue
        img = Image.open(filename)
        img_array = np.array(img)[..., :3]  # Ensure 3 channels
        images.append(img_array)
    if images:
        return np.stack(images)
    else:
        return np.empty((0, 0, 0, 3), dtype=np.uint8)


def process_episode(dataset, episode_path, states_name, actions_name, subtask_prompt):
    """Processes a single episode, save the data to lerobot format"""

    # Paths to image directories
    left_dir = os.path.join(episode_path, "left_img_dir")
    right_dir = os.path.join(episode_path, "right_img_dir")
    psm1_dir = os.path.join(episode_path, "endo_psm1")
    psm2_dir = os.path.join(episode_path, "endo_psm2")
    csv_file = os.path.join(episode_path, "ee_csv.csv")

    # Read CSV to determine the number of frames (excluding header)
    df = pd.read_csv(csv_file)

    # Read images from each camera
    left_images = read_images(left_dir, "frame{:06d}_left.jpg")
    right_images = read_images(right_dir, "frame{:06d}_right.jpg")
    psm1_images = read_images(psm1_dir, "frame{:06d}_psm1.jpg")
    psm2_images = read_images(psm2_dir, "frame{:06d}_psm2.jpg")
    # print(left_images.shape, right_images.shape, psm1_images.shape, psm2_images.shape)
    num_frames = min(len(df), left_images.shape[0])

    # Read kinematics data column-wise so each column keeps its own dtype.
    states_data = df[states_name].to_numpy(dtype=np.float32)
    actions_data = df[actions_name].to_numpy(dtype=np.float32)
    host_stamps = df["timestamp"].to_numpy(dtype=np.int64)

    for i in range(num_frames):
        frame = {
            "observation.state": states_data[i],
            "action": actions_data[i],
            "instruction.text": subtask_prompt,
            "observation.meta.tool.psm1": "Large Needle Driver",
            "observation.meta.tool.psm2": "Debakey Forceps",
            # Original acquisition clock, passed through as nanoseconds.
            "observation.meta.host_stamp_ns": np.array(
                [host_stamps[i]], dtype=np.int64
            ),
            "task": subtask_prompt,
        }

        for cam_name, images in [
            ("endoscope.left", left_images),
            ("endoscope.right", right_images),
            ("wrist.left", psm2_images),
            ("wrist.right", psm1_images),
        ]:
            if images.size > 0:
                frame[f"observation.images.{cam_name}"] = images[i]
        dataset.add_frame(frame)

    return dataset


def convert_data_to_lerobot(
    data_path: Path, repo_id: str, *, push_to_hub: bool = False
):
    """
    Converts a single Zarr store with episode boundaries to a LeRobotDataset.

    Args:
        data_path: The path to the Zarr store file/directory.
        repo_id: The repository ID for the dataset on the Hugging Face Hub.
        push_to_hub: Whether to push the dataset to the Hub after conversion.
    """
    final_output_path = HF_LEROBOT_HOME / repo_id
    print(final_output_path)
    if final_output_path.exists():
        print(f"Removing existing dataset at {final_output_path}")
        shutil.rmtree(final_output_path)

    # Initialize a LeRobotDataset with the desired features.
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        use_videos=True,
        robot_type="dvrk",
        fps=30,
        features={
            "observation.images.endoscope.left": {
                "dtype": "video",
                "shape": (540, 960, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.images.endoscope.right": {
                "dtype": "video",
                "shape": (540, 960, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.images.wrist.left": {
                "dtype": "video",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.images.wrist.right": {
                "dtype": "video",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (len(states_name),),
                "names": [states_name],
            },
            "action": {
                "dtype": "float32",
                "shape": (len(actions_name),),
                "names": [actions_name],
            },
            "observation.meta.tool.psm1": {
                "dtype": "string",
                "shape": (1,),
                "names": ["value"],
            },
            "observation.meta.tool.psm2": {
                "dtype": "string",
                "shape": (1,),
                "names": ["value"],
            },
            "observation.meta.host_stamp_ns": {
                "dtype": "int64",
                "shape": (1,),
                "names": ["ns"],
            },
            "instruction.text": {
                "dtype": "string",
                "shape": (1,),
                "description": "Natural language command for the robot",
            },
        },
        image_writer_processes=16,
        image_writer_threads=20,
        tolerance_s=0.1,
    )
    # measure time taken to complete the process
    start_time = time.time()
    perfect_demo_count = 0
    recovery_demo_count = 0
    idx = 10
    tissue_dir = os.path.join(data_path, f"tissue_{idx}")
    if not os.path.exists(tissue_dir):
        print(f"Warning: {tissue_dir} does not exist.")
        exit()
    ## process all demos (perfect and recovery)
    # Sort for deterministic episode order, with perfect subtasks before
    # recovery ones so the count-derived split ranges below stay correct.
    for subtask_name in sorted(
        os.listdir(tissue_dir), key=lambda name: (name.endswith("recovery"), name)
    ):
        try:
            subtask_dir = os.path.join(tissue_dir, subtask_name)
            if not os.path.isdir(subtask_dir):
                continue
            episode_dir = subtask_dir  # fallback for the error message below

            subtask_prompt = " ".join(subtask_name.split("_")[1:])
            is_recovery = subtask_prompt.endswith("recovery")

            if is_recovery:
                subtask_prompt = subtask_prompt[:-9]  # Remove " recovery" suffix

            for episode_name in sorted(os.listdir(subtask_dir)):
                episode_dir = os.path.join(subtask_dir, episode_name)
                if not os.path.isdir(episode_dir):
                    continue
                dataset = process_episode(
                    dataset, episode_dir, states_name, actions_name, subtask_prompt
                )

                dataset.save_episode()
                if is_recovery:
                    recovery_demo_count += 1
                else:
                    perfect_demo_count += 1
        except Exception as e:
            print(f"Error processing episode {episode_dir}: {e}")
            episode_index = dataset.writer.episode_buffer["episode_index"]
            dataset.clear_episode_buffer()
            # Also remove leftover temp video frames so they don't leak into the next episode.
            dataset.writer.cleanup_interrupted_episode(episode_index)
        print(
            f"subtask {subtask_name} processed successful, time taken: {time.time() - start_time}"
        )
    # Finalize the dataset so all buffered metadata and videos are written to disk.
    dataset.finalize()

    print(f"perfect_demo_count: {perfect_demo_count}")

    print(f"recovery_demo_count: {recovery_demo_count}")
    total_episode_count = perfect_demo_count + recovery_demo_count
    print(f"Total episodes processed: {total_episode_count}")
    train_count = int(0.8 * total_episode_count)
    val_count = int(0.1 * total_episode_count)
    # test_count = total_episode_count - train_count - val_count
    ## write split in meta
    dataset.meta.info.splits = {
        "train": "0:{}".format(train_count),
        "val": "{}:{}".format(train_count, train_count + val_count),
        "test": "{}:{}".format(train_count + val_count, total_episode_count),
        "perfect": f"0:{perfect_demo_count}",  # perfect episodes
        "recovery": f"{perfect_demo_count}:{perfect_demo_count + recovery_demo_count}",  # recovery episodes
        # "failure": "140:150",   # failure episodes
    }
    write_info(dataset.meta.info, dataset.root)

    print("Custom split configuration saved!")
    print(f"suturing processed successful, time taken: {time.time() - start_time}")

    if push_to_hub:
        print(f"Pushing dataset to Hugging Face Hub: {repo_id}")
        dataset.push_to_hub()
        print("Push complete.")


def main(
    data_path: Path = Path("/path/to/dataset"),
    repo_id: str = "jchen396/openh_test",
    *,
    push_to_hub: bool = False,
):
    """
    Main entry point for the conversion script.

    Args:
        data_path: The path to the dataset.
        repo_id: The desired Hugging Face Hub repository ID (e.g., 'username/dataset-name').
        push_to_hub: If True, uploads the dataset to the Hub after conversion.
    """
    if not data_path.exists():
        print(f"Error: The provided path does not exist: {data_path}")
        print("Please provide a valid path to your data.")
        return

    if repo_id == "your-username/your-dataset-name":
        print(
            "Warning: Using the default repo_id. Please specify your own with --repo-id."
        )

    convert_data_to_lerobot(data_path, repo_id, push_to_hub=push_to_hub)


if __name__ == "__main__":
    tyro.cli(main)
