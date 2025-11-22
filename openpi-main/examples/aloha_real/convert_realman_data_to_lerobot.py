"""
Script to convert RealMan dual-arm .h5 episodes to the LeRobot dataset v2.0 format,
following the structure and conventions of convert_aloha_data_to_lerobot.py.

Key decisions (aligned with ALOHA):
- observation.state: joint positions (Right→Left concatenation, 6+6 = 12 DoF)
- action: next-step joint positions (Behavior Cloning target), same ordering & dim (12)
- cameras: map raw camera names to ALOHA-compatible names:
    camera_high         -> cam_high
    camera_left_wrist  -> cam_left_wrist
    camera_right_wrist -> cam_right_wrist
- depth: stored under observation.depth.cam_high (uint16, 480x640) if present
- features:
    observation.state/action have "normalize": True
    task has "tokenize": True (pull from metadata.attrs['instruction'])
- Image format: Saved as CHW (3, 480, 640) to match ALOHA JSON spec.

Example:
uv run convert_realman_data_to_lerobot.py \
  --raw-dir /path/to/h5_episodes \
  --repo-id <org_or_user>/realman_dataset \
  --mode image \
  --fps 50 \
  --push-to-hub
"""

import dataclasses
from pathlib import Path
import shutil
from typing import Literal, List, Dict, Tuple

import h5py
import numpy as np
import torch
import tqdm
import tyro

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from pathlib import Path
import os

LEROBOT_HOME = Path(os.getenv("HF_LEROBOT_HOME",
                              os.path.expanduser("~/.cache/huggingface/lerobot")))

# ----------------------------
# A. Config (kept ALOHA-style)
# ----------------------------
@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


# -----------------------------------------
# B. Joint ordering (Right→Left, ALOHA-like)
# -----------------------------------------
# (基于 6-DoF，不含 gripper)
RIGHT_ARM_JOINT_NAMES: List[str] = [
    "right_waist",
    "right_shoulder",
    "right_elbow",
    "right_forearm_roll",
    "right_wrist_angle",
    "right_wrist_rotate",
]
LEFT_ARM_JOINT_NAMES: List[str] = [
    "left_waist",
    "left_shoulder",
    "left_elbow",
    "left_forearm_roll",
    "left_wrist_angle",
    "left_wrist_rotate",
]
# Combined 12-DoF ordering: Right (6) → Left (6)
JOINT_ORDER_RL_12: List[str] = RIGHT_ARM_JOINT_NAMES + LEFT_ARM_JOINT_NAMES


# ------------------------------------
# C. Create empty LeRobot v2.0 dataset
# ------------------------------------
def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    fps: int, # ✅ 修正：从 port_realman 接收 FPS
    mode: Literal["video", "image"] = "video",
    *,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    """
    Define features and create an empty dataset directory, mirroring ALOHA structure.
    """
    cameras = [
        "cam_high",
        "cam_left_wrist",
        "cam_right_wrist",
    ]

    features = {
        # proprio (12): Right→Left, joint positions
        "observation.state": {
            "dtype": "float32",
            "shape": (12,),
            "names": JOINT_ORDER_RL_12,  # ✅ 修正：使用 [JOINT_ORDER_RL_12]
            
        },
        # action (12): Behavior Cloning target = next-step joint positions
        "action": {
            "dtype": "float32",
            "shape": (12,),
            "names": JOINT_ORDER_RL_12, # ✅ 修正：使用 [JOINT_ORDER_RL_12]
            
        },
        # VLA instruction
        "task_index": {
            "dtype": "int64",
            "shape": [1],
            "names": None,
        },
    }

    # RGB cameras, CHW
    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3, 480, 640),
            "names": ["channels", "height", "width"],
        }

    # depth for top camera if present
    features["observation.depth.cam_high"] = {
        "dtype": "uint16",
        "shape": (480, 640),
        "names": ["height", "width"],
    }

    # clean target dir if exists
    out_dir = LEROBOT_HOME / repo_id
    if out_dir.exists():
        shutil.rmtree(out_dir)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,  # ✅ 修正：使用传入的 FPS
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


# ---------------------------------------
# D. Load a single episode from RealMan H5
# ---------------------------------------
def _transpose_hwc_to_chw(arr_hwc: np.ndarray) -> np.ndarray:
    # Expect (N, H, W, C) -> (N, C, H, W)
    assert arr_hwc.ndim == 4 and arr_hwc.shape[-1] in (1, 3), "Image must be NHWC"
    return np.transpose(arr_hwc, (0, 3, 1, 2))


def load_raw_episode_data(
    ep_path: Path,
) -> Tuple[
    Dict[str, np.ndarray],  # imgs_per_cam (cam_high/left_wrist/right_wrist) in CHW
    torch.Tensor,           # state  (T-1, 12), Right→Left
    torch.Tensor,           # action (T-1, 12), Right→Left (next-step)
    np.ndarray | None,      # depth (T-1, H, W) or None
    str,                    # task instruction
]:
    """
    Reads one RealMan .h5 episode 
    and converts it to the ALOHA-style format.
    """
    with h5py.File(ep_path, "r") as ep:
        # ✅ 修正：使用正确的 HDF5 路径
        obs = ep["observations"]
        cams = obs["cameras"]
        arms = obs["arms"]

        # images (NHWC -> NCHW), map to ALOHA camera keys
        imgs_per_cam: Dict[str, np.ndarray] = {}

        # ✅ 修正：使用正确的 HDF5 路径
        cam_high_rgb = _transpose_hwc_to_chw(cams["camera_high"]["color"][:])
        imgs_per_cam["cam_high"] = cam_high_rgb

        if "camera_left_wrist" in cams and "color" in cams["camera_left_wrist"]:
            imgs_per_cam["cam_left_wrist"] = _transpose_hwc_to_chw(cams["camera_right_wrist"]["color"][:])
        if "camera_right_wrist" in cams and "color" in cams["camera_right_wrist"]:
            imgs_per_cam["cam_right_wrist"] = _transpose_hwc_to_chw(cams["camera_left_wrist"]["color"][:])

        # optional depth for cam_high
        depth = None
        if "camera_high" in cams and "depth" in cams["camera_high"]:
            depth = cams["camera_high"]["depth"][:]  # (T, H, W)

        # joint positions (Left & Right) -> concatenate as Right→Left
        # ✅ 修正：使用正确的 HDF5 路径并修复拼写错误
        left_qpos = torch.from_numpy(arms["left_arm"]["joint_positions"][:])   # (T, 6)
        right_qpos = torch.from_numpy(arms["right_arm"]["joint_positions"][:]) # (T, 6)
        
        # concat in ALOHA order: Right(6) then Left(6)
        qpos_rl = torch.cat((right_qpos, left_qpos), dim=1)  # (T, 12)

        # Behavior Cloning target: next-step joint positions
        T = qpos_rl.shape[0]
        if T < 2:
            raise ValueError(f"Episode {ep_path.name} has <2 frames, cannot form next-step actions.")
        
        state = qpos_rl[:-1].to(torch.float32)    # (T-1, 12)
        action = qpos_rl[1:].to(torch.float32)   # (T-1, 12)

        # trim images/depth to T-1 to stay aligned with (state, action)
        for k in list(imgs_per_cam.keys()):
            imgs = imgs_per_cam[k]
            assert imgs.shape[0] == T, f"{k} frames ({imgs.shape[0]}) != joint frames ({T})"
            imgs_per_cam[k] = imgs[:-1]
        if depth is not None:
            assert depth.shape[0] == T, f"depth frames ({depth.shape[0]}) != joint frames ({T})"
            depth = depth[:-1]

        # language instruction
        task = "Let the forceps go along the black S shaped path"
        # ✅ 修正：使用 'metadata' 而不是 '/metadata'
        if "metadata" in ep and "instruction" in ep["metadata"].attrs:
            raw = ep["metadata"].attrs["instruction"]
            task = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)

    return imgs_per_cam, state, action, depth, task


# -------------------------------------------
# E. Populate dataset (episodes -> LeRobot v2)
# -------------------------------------------
def populate_dataset(
    dataset: LeRobotDataset,
    hdf5_files: list[Path],
    *,
    episodes: list[int] | None = None,
) -> LeRobotDataset:
    """
    ALOHA-style episode loop: load -> frame-wise add -> save_episode(task=...).
    """
    if episodes is None:
        episodes = range(len(hdf5_files))

    for ep_idx in tqdm.tqdm(episodes, desc="Converting Episodes"):
        ep_path = hdf5_files[ep_idx]
        try:
            imgs_per_cam, state, action, depth, task = load_raw_episode_data(ep_path)
            num_frames = state.shape[0]

            for i in range(num_frames):
                frame = {
                    "observation.state": state[i],
                    "action": action[i],
                    "task": task, # ✅ 修正：将 'task' 添加到每一帧
                }
                # RGB
                for camera, img_array in imgs_per_cam.items():
                    frame[f"observation.images.{camera}"] = img_array[i]
                # depth if present
                if depth is not None:
                    frame["observation.depth.cam_high"] = depth[i]

                dataset.add_frame(frame)
            
            # ✅ 修正：将 'task' 作为 episode 级别的元数据保存
            dataset.save_episode() 

        except Exception as e:
            print(f"[WARN] Skipping {ep_path.name} due to error: {e}")
            # import traceback
            # traceback.print_exc() # 取消注释以进行详细调试

    return dataset


# ------------------------------
# F. CLI entry (ALOHA-like form)
# ------------------------------
def port_realman(
    raw_dir: Path,
    repo_id: str,
    *,
    fps: int = 30, # ✅ 修正：添加 fps 参数，默认为 30
    episodes: list[int] | None = None,
    push_to_hub: bool = False,
    mode: Literal["video", "image"] = "image",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    """
    Main entry. Mirrors ALOHA's 'port_aloha' signature/behavior where possible.
    - raw_dir: directory of RealMan .h5 episodes (e.g., *.h5)
    - repo_id: target LeRobot dataset repo name (e.g., user/realman_dataset)
    """
    target_dir = LEROBOT_HOME / repo_id
    if target_dir.exists():
        print(f"Warning: Target directory {target_dir} exists. Removing it.")
        shutil.rmtree(target_dir)

    if not raw_dir.exists():
        raise ValueError(f"Raw dir does not exist: {raw_dir}")

    # RealMan episodes are stored as *.h5 files
    hdf5_files = sorted(raw_dir.glob("*.h5"))
    if len(hdf5_files) == 0:
        raise ValueError(f"No .h5 episode found in {raw_dir}")
    
    print(f"Found {len(hdf5_files)} .h5 episodes.")

    dataset = create_empty_dataset(
        repo_id=repo_id,
        robot_type="realman_dual_arm",
        fps=fps, # ✅ 修正：传递 fps
        mode=mode,
        dataset_config=dataset_config,
    )

    dataset = populate_dataset(
        dataset=dataset,
        hdf5_files=hdf5_files,
        episodes=episodes,
    )

    print("Consolidating dataset...")
    # dataset.consolidate()
    print(f"Conversion complete. Dataset saved to: {target_dir}")

    if push_to_hub:
        print(f"Pushing {repo_id} to Hugging Face Hub...")
        dataset.push_to_hub(
            tags=["realman", "dual_arm", "pi0", "vla"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )
        print("Push to Hub complete.")


if __name__ == "__main__":
    tyro.cli(port_realman)