#!/usr/bin/env python3
"""Offline inference script for Realman π₀ model.

This script loads a trained model checkpoint and performs inference on episodes from
a LeRobot dataset, computing MSE and saving predictions to CSV.

Usage:
uv run /home/ren9/DualArm/openpi/examples/offline_inference.py \
    --config-name pi0_realman \
    --checkpoint checkpoints/pi0_realman/realman_finetune_v1/14999/params \
    --dataset-path /home/ren9/.cache/huggingface/lerobot/realman_test2_dataset \
    --episode 0 \
    --output test2_predictions.csv \
    --batch-size 2
"""
import os
os.environ["JAX_PLATFORM_NAME"] = "gpu"

import argparse
import logging
import pathlib
from typing import Any

import jax
import jax.numpy as jnp
print("JAX default backend:", jax.default_backend())
print("JAX devices:", jax.devices())
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import pandas as pd
from tqdm import tqdm

import openpi.models.model as _model
import openpi.shared.download as download
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.transforms as _transforms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model_and_config(
    config_name: str,
    checkpoint_path: str,
) -> tuple[_model.BaseModel, _config.TrainConfig, _config.DataConfig]:
    """Load model, train config, and data config from checkpoint.
    
    Args:
        config_name: Name of the config (e.g., "pi0_realman")
        checkpoint_path: Path to checkpoint params directory
        
    Returns:
        Tuple of (model, train_config, data_config)
    """
    logger.info(f"Loading config: {config_name}")
    train_config = _config.get_config(config_name)
    
    # Create data config
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    
    # Load checkpoint parameters
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    loaded_params = _model.restore_params(download.maybe_download(checkpoint_path), restore_type=np.ndarray)
    
    # Create model with loaded parameters
    logger.info(f"Creating model: {train_config.model.model_type}")
    model = train_config.model.load(loaded_params)
    
    return model, train_config, data_config


def load_norm_stats(norm_stats_path: pathlib.Path) -> dict[str, _transforms.NormStats]:
    """Load normalization statistics from JSON file.
    
    Args:
        norm_stats_path: Path to norm_stats.json
        
    Returns:
        Dictionary of normalization statistics
    """
    if not norm_stats_path.exists():
        raise FileNotFoundError(f"Normalization stats not found: {norm_stats_path}")
    
    logger.info(f"Loading normalization stats from: {norm_stats_path}")
    return _normalize.load(norm_stats_path)


def create_transforms(
    data_config: _config.DataConfig,
    norm_stats: dict[str, _transforms.NormStats],
) -> tuple[_transforms.DataTransformFn, _transforms.DataTransformFn]:
    """Create input and output transform functions.
    
    Args:
        data_config: Data configuration
        norm_stats: Normalization statistics
        
    Returns:
        Tuple of (input_transform, output_transform)
    """
    # Input transforms: repack → data → normalize → model
    input_transforms = _transforms.compose([
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
        _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs,
    ])
    
    # Output transforms: model → denormalize → data (reversed)
    output_transforms = _transforms.compose([
        *data_config.model_transforms.outputs,
        _transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.data_transforms.outputs,
    ])
    
    return input_transforms, output_transforms


def load_episode_data(
    dataset_path: str,
    episode_index: int,
    action_horizon: int,
) -> list[dict[str, Any]]:
    """Load all frames from a specific episode.
    
    Args:
        dataset_path: Path to LeRobot dataset
        episode_index: Episode index to load
        action_horizon: Action horizon for delta timestamps
        
    Returns:
        List of frame data dictionaries
    """
    logger.info(f"Loading dataset from: {dataset_path}")
    
    # Load dataset metadata
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(dataset_path)
    
    # Create dataset with action sequences
    dataset = lerobot_dataset.LeRobotDataset(
        dataset_path,
        delta_timestamps={
            "action": [t / dataset_meta.fps for t in range(action_horizon)]
        },
    )
    
    # Filter frames from the specified episode
    episode_frames = []
    logger.info(f"Loading episode {episode_index}...")
    
    for idx in tqdm(range(len(dataset)), desc="Loading frames"):
        frame = dataset[idx]
        if frame.get("episode_index") == episode_index:
            episode_frames.append(frame)
    
    if not episode_frames:
        raise ValueError(f"Episode {episode_index} not found in dataset")
    
    logger.info(f"Loaded {len(episode_frames)} frames from episode {episode_index}")
    return episode_frames


def run_inference(
    model: _model.BaseModel,
    frames: list[dict[str, Any]],
    input_transform: _transforms.DataTransformFn,
    output_transform: _transforms.DataTransformFn,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference on episode frames.
    
    Args:
        model: Trained model
        frames: List of frame data
        input_transform: Transform for input data
        output_transform: Transform for output data (denormalization)
        batch_size: Batch size for inference
        
    Returns:
        Tuple of (predictions, ground_truth) arrays with shape (num_frames, 12)
    """
    num_frames = len(frames)
    predictions = []
    ground_truths = []
    
    logger.info(f"Running inference on {num_frames} frames with batch_size={batch_size}")
    
    # Create RNG for sampling
    rng = jax.random.key(42)
    
    # Process in batches
    for start_idx in tqdm(range(0, num_frames, batch_size), desc="Inference"):
        end_idx = min(start_idx + batch_size, num_frames)
        batch_frames = frames[start_idx:end_idx]
        
        # Transform input data
        batch_data = []
        batch_gt = []
        
        for idx, frame in enumerate(batch_frames):
            # 1. DEBUG: Print frame after loading from dataset (only first frame of first batch)
            if start_idx == 0 and idx == 0:
                print("\n=== 1. After Data Loading (from dataset) ===")
                print("Frame keys:", list(frame.keys()))
                # Check image value range
                for img_key in ['observation.images.cam_high', 'observation.images.cam_left_wrist', 'observation.images.cam_right_wrist']:
                    if img_key in frame:
                        img = frame[img_key]
                        if hasattr(img, 'min'):
                            print(f"  [IMAGE VALUES] {img_key}: min={float(img.min()):.4f}, max={float(img.max()):.4f}, mean={float(img.mean()):.4f}")
                print("=" * 50)
                for key, value in frame.items():
                    if isinstance(value, dict):
                        print(f"  {key}: (dict with {len(value)} keys)")
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, dict):
                                print(f"    {sub_key}: (dict with {len(sub_value)} keys)")
                                for sub_sub_key, sub_sub_value in sub_value.items():
                                    if isinstance(sub_sub_value, np.ndarray):
                                        print(f"      {sub_sub_key}: shape={sub_sub_value.shape}, dtype={sub_sub_value.dtype}")
                                    elif hasattr(sub_sub_value, 'shape'):  # torch.Tensor
                                        print(f"      {sub_sub_key}: shape={tuple(sub_sub_value.shape)}, dtype={sub_sub_value.dtype}")
                            elif isinstance(sub_value, np.ndarray):
                                print(f"    {sub_key}: shape={sub_value.shape}, dtype={sub_value.dtype}")
                            elif hasattr(sub_value, 'shape'):  # torch.Tensor
                                print(f"    {sub_key}: shape={tuple(sub_value.shape)}, dtype={sub_value.dtype}")
                    elif isinstance(value, np.ndarray):
                        print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                    elif hasattr(value, 'shape'):  # torch.Tensor
                        print(f"  {key}: shape={tuple(value.shape)}, dtype={value.dtype}")
                    elif isinstance(value, str):
                        print(f"  {key}: type={type(value)}, value={value if len(value) < 50 else value[:50]+'...'}")
                    elif not (key in ["episode_index", "frame_index", "timestamp", "index"]):
                        print(f"  {key}: type={type(value)}")
                print("=" * 50)
            
            # 2. DEBUG: Print before input transform (only first frame of first batch)
            if start_idx == 0 and idx == 0:
                print("\n=== 2. Before Input Transform ===")
                print("Frame keys:", list(frame.keys()))
                for key, value in frame.items():
                    if isinstance(value, dict):
                        print(f"  {key}: (dict with {len(value)} keys)")
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, dict):
                                print(f"    {sub_key}: (dict with {len(sub_value)} keys)")
                                for sub_sub_key, sub_sub_value in sub_value.items():
                                    if isinstance(sub_sub_value, np.ndarray):
                                        print(f"      {sub_sub_key}: shape={sub_sub_value.shape}, dtype={sub_sub_value.dtype}")
                                    elif hasattr(sub_sub_value, 'shape'):  # torch.Tensor
                                        print(f"      {sub_sub_key}: shape={tuple(sub_sub_value.shape)}, dtype={sub_sub_value.dtype}")
                            elif isinstance(sub_value, np.ndarray):
                                print(f"    {sub_key}: shape={sub_value.shape}, dtype={sub_value.dtype}")
                            elif hasattr(sub_value, 'shape'):  # torch.Tensor
                                print(f"    {sub_key}: shape={tuple(sub_value.shape)}, dtype={sub_value.dtype}")
                    elif isinstance(value, np.ndarray):
                        print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                    elif hasattr(value, 'shape'):  # torch.Tensor
                        print(f"  {key}: shape={tuple(value.shape)}, dtype={value.dtype}")
                    elif isinstance(value, str):
                        print(f"  {key}: type={type(value)}, value={value if len(value) < 50 else value[:50]+'...'}")
                    elif not (key in ["episode_index", "frame_index", "timestamp", "index"]):
                        print(f"  {key}: type={type(value)}")
                print("=" * 50)
            
            transformed = input_transform(frame)
            
            # 3. DEBUG: Print after input transform (only first frame of first batch)
            if start_idx == 0 and idx == 0:
                print("\n=== 3. After Input Transform ===")
                for key, value in transformed.items():
                    if isinstance(value, dict):
                        print(f"  {key}: (dict with {len(value)} keys)")
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, np.ndarray):
                                print(f"    {sub_key}: shape={sub_value.shape}, dtype={sub_value.dtype}")
                            else:
                                print(f"    {sub_key}: type={type(sub_value)}")
                    elif isinstance(value, np.ndarray):
                        print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                    else:
                        print(f"  {key}: type={type(value)}")
                print("=" * 50)
            batch_data.append(transformed)
            # Store ground truth action (first step only, since we use scheme A)
            batch_gt.append(frame["action"][0])  # Shape: (12,)
        
        # Stack batch (use numpy stack)
        batch = jax.tree.map(lambda *xs: np.stack(xs, axis=0), *batch_data)
        
        # Convert numpy arrays to JAX arrays (required by the model)
        batch = jax.tree.map(lambda x: jnp.asarray(x), batch)
        
        # Create observation from batch
        observation = _model.Observation.from_dict(batch)
        
        # 4. DEBUG: Print observation after creation (only first batch)
        if start_idx == 0:
            print("\n=== 4. After Observation Creation ===")
            print(f"  images: (dict with {len(observation.images)} keys)")
            for key, value in observation.images.items():
                arr = np.array(value)
                print(f"    {key}: shape={arr.shape}, dtype={arr.dtype}")
            print(f"  image_masks: (dict with {len(observation.image_masks)} keys)")
            for key, value in observation.image_masks.items():
                arr = np.array(value)
                print(f"    {key}: shape={arr.shape}, dtype={arr.dtype}")
            state_arr = np.array(observation.state)
            print(f"  state: shape={state_arr.shape}, dtype={state_arr.dtype}")
            if observation.tokenized_prompt is not None:
                token_arr = np.array(observation.tokenized_prompt)
                print(f"  tokenized_prompt: shape={token_arr.shape}, dtype={token_arr.dtype}")
            if observation.tokenized_prompt_mask is not None:
                mask_arr = np.array(observation.tokenized_prompt_mask)
                print(f"  tokenized_prompt_mask: shape={mask_arr.shape}, dtype={mask_arr.dtype}")
            print("=" * 50)
        
        # Run model inference using sample_actions
        # Split RNG for this batch
        rng, inference_rng = jax.random.split(rng)
        predicted_actions = model.sample_actions(
            inference_rng, 
            observation, 
            num_steps=10  # Number of diffusion steps for Pi0
        )  # Shape: (batch, action_horizon, action_dim)
        
        # Take only the first step (Scheme A)
        predicted_actions_first_step = predicted_actions[:, 0, :]  # Shape: (batch, action_dim)
        
        # Denormalize predictions and convert delta to absolute actions
        # IMPORTANT: AbsoluteActions needs both 'state' and 'actions'
        for i in range(len(batch_frames)):
            # Get the state from the batch (needed for delta->absolute conversion)
            state = batch["state"][i] if "state" in batch else None
            
            # Convert JAX arrays to numpy arrays (required for in-place operations in transforms)
            # Add action_horizon dimension (1, action_dim) since AbsoluteActions expects 2D
            pred_dict = {
                # [np.newaxis, :]是行向量扩展为二维数组，形状为(1, action_dim)
                "actions": np.array(predicted_actions_first_step[i])[np.newaxis, :],  # Shape: (1, action_dim)
                # 如果state为None，则传递None，否则转换为numpy数组
                "state": np.array(state) if state is not None else None,
            }
            
            # Apply output transforms (includes denormalization and delta->absolute conversion)
            #denorm_pred 是一个字典，包含反归一化后的动作和状态
            denorm_pred = output_transform(pred_dict)
            # Remove the action_horizon dimension to get back to (action_dim,)
            predictions.append(np.array(denorm_pred["actions"][0]))
            ground_truths.append(np.array(batch_gt[i]))
    
    return np.array(predictions), np.array(ground_truths)


def compute_mse(predictions: np.ndarray, ground_truth: np.ndarray) -> dict[str, float]:
    """Compute MSE metrics.
    
    Args:
        predictions: Predicted actions, shape (num_frames, 12)
        ground_truth: Ground truth actions, shape (num_frames, 12)
        
    Returns:
        Dictionary with MSE metrics
    """
    mse_per_joint = np.mean((predictions - ground_truth) ** 2, axis=0)
    mse_overall = np.mean(mse_per_joint)
    
    metrics = {
        "mse_overall": float(mse_overall),
    }
    
    # Add per-joint MSE
    joint_names = [
        "right_waist", "right_shoulder", "right_elbow",
        "right_forearm_roll", "right_wrist_angle", "right_wrist_rotate",
        "left_waist", "left_shoulder", "left_elbow",
        "left_forearm_roll", "left_wrist_angle", "left_wrist_rotate",
    ]
    for i, joint_name in enumerate(joint_names):
        metrics[f"mse_{joint_name}"] = float(mse_per_joint[i])
    
    return metrics


def save_predictions_to_csv(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    episode_index: int,
    output_path: pathlib.Path,
):
    """Save predictions to CSV file.
    
    Args:
        predictions: Predicted actions, shape (num_frames, 12)
        ground_truth: Ground truth actions, shape (num_frames, 12)
        episode_index: Episode index
        output_path: Output CSV file path
    """
    num_frames = predictions.shape[0]
    
    # Joint names for columns
    joint_names = [
        "joint_0_right_waist", "joint_1_right_shoulder", "joint_2_right_elbow",
        "joint_3_right_forearm_roll", "joint_4_right_wrist_angle", "joint_5_right_wrist_rotate",
        "joint_6_left_waist", "joint_7_left_shoulder", "joint_8_left_elbow",
        "joint_9_left_forearm_roll", "joint_10_left_wrist_angle", "joint_11_left_wrist_rotate",
    ]
    
    # Create DataFrame
    data = {}
    
    # Add joint predictions (first 12 columns)
    for i, joint_name in enumerate(joint_names):
        data[joint_name] = predictions[:, i]
    
    # Add indices at the end
    data["episode_index"] = [episode_index] * num_frames
    data["frame_index"] = list(range(num_frames))
    
    # Optional: Add ground truth for comparison
    for i, joint_name in enumerate(joint_names):
        data[f"gt_{joint_name}"] = ground_truth[:, i]
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved predictions to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Offline inference for Realman π₀ model")
    parser.add_argument(
        "--config-name",
        type=str,
        default="pi0_realman",
        help="Name of the training config (default: pi0_realman)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint directory (e.g., checkpoints/pi0_realman/exp/14999)",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to LeRobot dataset",
    )
    parser.add_argument(
        "--episode",
        type=int,
        required=True,
        help="Episode index to run inference on",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="predictions.csv",
        help="Output CSV file path (default: predictions.csv)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for inference (default: 2)",
    )
    parser.add_argument(
        "--norm-stats",
        type=str,
        default="assets/pi0_realman/realman_dataset/",
        help="Path to normalization stats (default: assets/pi0_realman/realman_dataset/norm_stats.json)",
    )
    
    args = parser.parse_args()
    
    # Load model and config
    model, train_config, data_config = load_model_and_config(
        args.config_name,
        args.checkpoint,
    )
    
    # Load normalization stats
    norm_stats_path = pathlib.Path(args.norm_stats)
    norm_stats = load_norm_stats(norm_stats_path)
    
    # Create transforms
    input_transform, output_transform = create_transforms(data_config, norm_stats)
    
    # Load episode data
    frames = load_episode_data(
        args.dataset_path,
        args.episode,
        train_config.model.action_horizon,
    )
    
    # Run inference
    predictions, ground_truth = run_inference(
        model,
        frames,
        input_transform,
        output_transform,
        args.batch_size,
    )
    
    # Compute MSE
    metrics = compute_mse(predictions, ground_truth)
    logger.info("=" * 60)
    logger.info("MSE Metrics:")
    logger.info(f"  Overall MSE: {metrics['mse_overall']:.6f}")
    logger.info("-" * 60)
    for key, value in metrics.items():
        if key.startswith("mse_") and key != "mse_overall":
            logger.info(f"  {key}: {value:.6f}")
    logger.info("=" * 60)
    
    # Save predictions
    output_path = pathlib.Path(args.output)
    save_predictions_to_csv(predictions, ground_truth, args.episode, output_path)
    
    logger.info("Inference complete!")


if __name__ == "__main__":
    main()

