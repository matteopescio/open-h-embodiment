#!/usr/bin/env python
"""
A script to convert robotics data from HDF5 files into the LeRobot format (v3.0)
with an efficient MP4 video backend.

This script processes a directory of HDF5 files, where each file represents a
single episode. It extracts observations, actions, and state information, and
packages them into a LeRobotDataset with visual data stored as compressed MP4
videos, then optionally pushes the result to the Hugging Face Hub.

Expected HDF5 File Structure:
------------------------------
The script assumes a directory with zero-indexed HDF5 files (e.g., `data_0.hdf5`).
Each file should contain the following structure:

/data/demo_0/
    ├── action                (Dataset): Actions taken at each step.
    ├── observations/
    │   └── rgb               (Dataset): Room and wrist RGB images with shape
    │                                  (steps, 2, height, width, channels),
    │                                  ordered [room, wrist].
    ├── abs_joint_pos         (Dataset): Absolute joint positions.
    └── timestep              (Dataset): Timestamps for each data point.

Camera intrinsics:
------------------
Including intrinsics is encouraged for any calibrated optical camera (needed for
depth, 3D reconstruction, and stereo). Because intrinsics are static for a
fixed-focal-length camera, they are written once as
'meta/calibration/camera_intrinsics.json' (see CAMERA_INTRINSICS and
write_camera_intrinsics below), not as a per-frame feature.

Usage:
------
    python convert_data_to_lerobot_video.py --data-dir /path/to/your/hdf5/files --repo-id your-username/your-dataset-name

To also push to the Hub:
    python convert_data_to_lerobot_video.py --data-dir /path/to/your/hdf5/files --repo-id your-username/your-dataset-name --push-to-hub
"""

import glob
import json
import os
import shutil
from pathlib import Path

import h5py
import numpy as np
import tqdm
import tyro
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME

# Static per-camera intrinsics for the optical (RGB) camera streams. Including
# intrinsics is encouraged wherever you provide a calibrated camera: they are
# needed for depth, 3D reconstruction, and stereo triangulation.
#
# Intrinsics are static for a fixed-focal-length camera, so they are written
# ONCE as a calibration file (see write_camera_intrinsics below) rather than
# repeated in every frame: static, dataset-level calibration lives as a file
# under meta/calibration/, while per-frame signals stay as observation.meta.*
# features. Key the dict by camera feature name so each optical stream is
# listed, and replace the placeholder values with your own OpenCV pinhole
# calibration.
CAMERA_INTRINSICS = {
    "observation.images.room": {
        "model": "pinhole",
        "width": 224,
        "height": 224,
        "fx": 130.0,
        "fy": 130.0,
        "cx": 112.0,
        "cy": 112.0,
        # OpenCV radtan/plumb_bob coefficients [k1, k2, p1, p2, k3]; use
        # "none" with an empty list if your frames are already undistorted.
        "distortion_model": "opencv_radtan",
        "distortion_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
    },
    "observation.images.wrist": {
        "model": "pinhole",
        "width": 224,
        "height": 224,
        "fx": 130.0,
        "fy": 130.0,
        "cx": 112.0,
        "cy": 112.0,
        "distortion_model": "opencv_radtan",
        "distortion_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
    },
}


def write_camera_intrinsics(dataset_root: Path, intrinsics: dict) -> None:
    """Write per-camera intrinsics to meta/calibration/camera_intrinsics.json.

    Args:
        dataset_root: The LeRobotDataset root (dataset.root); meta/ lives here.
        intrinsics: Mapping of camera feature name -> intrinsics dict.

    Note:
        Call this AFTER dataset.finalize() so the LeRobot writers do not
        clobber the calibration directory.
    """
    calibration_dir = dataset_root / "meta" / "calibration"
    calibration_dir.mkdir(parents=True, exist_ok=True)
    out_path = calibration_dir / "camera_intrinsics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(intrinsics, f, indent=2)
    print(f"Wrote intrinsics for {len(intrinsics)} camera(s) to {out_path}")


def convert_data_to_lerobot(data_dir: Path, repo_id: str, *, push_to_hub: bool = False):
    """
    Converts a directory of HDF5 files to a LeRobotDataset with a video backend.

    Args:
        data_dir: The path to the directory containing the HDF5 files.
        repo_id: The repository ID for the dataset on the Hugging Face Hub.
        push_to_hub: Whether to push the dataset to the Hub after conversion.
    """
    final_output_path = HF_LEROBOT_HOME / repo_id
    if final_output_path.exists():
        print(f"Removing existing dataset at {final_output_path}")
        shutil.rmtree(final_output_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        use_videos=True,
        robot_type="panda",
        fps=30,
        features={
            "observation.images.room": {
                "dtype": "video",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.images.wrist": {
                "dtype": "video",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7"],
            },
            "action": {
                "dtype": "float32",
                "shape": (6,),
                "names": ["x", "y", "z", "roll", "pitch", "yaw"],
            },
            "observation.meta.host_stamp_ns": {
                "dtype": "int64",
                "shape": (1,),
                "names": ["ns"],
            },
        },
        image_writer_processes=16,
        image_writer_threads=20,
        tolerance_s=0.1,
    )

    hdf5_files = sorted(glob.glob(os.path.join(data_dir, "*.hdf5")))

    if not hdf5_files:
        print(f"No HDF5 files found in {data_dir}. Exiting.")
        return

    print(f"Found {len(hdf5_files)} episodes to convert.")

    task_description = "Conduct a liver ultrasound scan"

    for hdf5_path in tqdm.tqdm(hdf5_files, desc="Converting Episodes"):
        try:
            with h5py.File(hdf5_path, "r") as f:
                root_name = "data/demo_0"
                if root_name not in f:
                    print(f"Warning: Skipping {hdf5_path} because '{root_name}' group was not found.")
                    continue

                num_steps = len(f[f"{root_name}/action"])

                # Add each frame from the episode to the internal buffer.
                for step in range(num_steps):
                    frame_data = {
                        "observation.images.room": f[f"{root_name}/observations/rgb"][step, 0],
                        "observation.images.wrist": f[f"{root_name}/observations/rgb"][step, 1],
                        "observation.state": f[f"{root_name}/abs_joint_pos"][step],
                        "action": f[f"{root_name}/action"][step],
                        # Original acquisition clock, passed through as nanoseconds.
                        "observation.meta.host_stamp_ns": np.array(
                            [round(float(f[f"{root_name}/timestep"][step]) * 1e9)], dtype=np.int64
                        ),
                        "task": task_description,
                    }
                    # The canonical timestamp column is computed as frame_index / fps in v3.0.
                    dataset.add_frame(frame_data)

            # After processing all frames for an HDF5 file, save the buffered
            # data as a completed episode. This will trigger the video encoding
            # for the camera streams collected.
            dataset.save_episode()

        except Exception as e:
            print(f"Error processing {hdf5_path}: {e}")
            # It's good practice to clear the buffer on error to prevent
            # a failed episode from contaminating the next one.
            episode_index = dataset.writer.episode_buffer["episode_index"]
            dataset.clear_episode_buffer()
            # Also remove leftover temp video frames so they don't leak into the next episode.
            dataset.writer.cleanup_interrupted_episode(episode_index)

    # Finalize the dataset so all buffered metadata and videos are written to disk.
    dataset.finalize()

    # Write the static per-camera intrinsics as a calibration file. Intrinsics
    # are written after finalize() so the writers do not overwrite the
    # calibration directory.
    write_camera_intrinsics(dataset.root, CAMERA_INTRINSICS)

    print(f"Dataset conversion complete. Saved to {final_output_path}")

    if push_to_hub:
        print(f"Pushing dataset to Hugging Face Hub: {repo_id}")
        dataset.push_to_hub()
        print("Push complete.")


def main(
    data_dir: Path = Path("path/to/your/data"),
    repo_id: str = "your-username/your-dataset-name",
    *,
    push_to_hub: bool = False,
):
    """
    Main entry point for the conversion script.

    Args:
        data_dir: The directory containing HDF5 episode files.
        repo_id: The desired Hugging Face Hub repository ID.
        push_to_hub: If True, uploads the dataset to the Hub.
    """
    if not data_dir.is_dir():
        print(f"Error: The provided data directory does not exist: {data_dir}")
        return

    if repo_id == "your-username/your-dataset-name":
        print("Warning: Using the default repo_id. Please specify your own with --repo-id.")

    convert_data_to_lerobot(data_dir, repo_id, push_to_hub=push_to_hub)


if __name__ == "__main__":
    tyro.cli(main)
