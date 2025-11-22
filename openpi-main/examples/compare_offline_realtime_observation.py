#!/usr/bin/env python3
"""对比离线推理和实时推理的observation输入格式和数值。

此脚本加载相同的模型配置和数据，对比两种推理方式生成的observation是否一致，
以确保实时推理的输入处理没有问题。

Usage:
    python examples/compare_offline_realtime_observation.py \
        --config-name pi0_realman \
        --checkpoint checkpoints/pi0_realman/realman_finetune_v1/14999 \
        --dataset-path ~/.cache/huggingface/lerobot/realman_dataset \
        --norm-stats assets/pi0_realman/realman_dataset/ \
        --episode 0 \
        --frame-index 10
"""

import argparse
import logging
import pathlib
import sys
from typing import Any

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np

import openpi.models.model as _model
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.transforms as _transforms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Helper Functions (from realman_inference.py)
# ============================================================================

def preprocess_image_realtime(image_rgb: np.ndarray) -> np.ndarray:
    """Preprocess image using realman_inference.py method (updated to match fix).
    
    仅做最基本的格式转换，resize 和归一化由 transform pipeline 统一处理。
    
    Args:
        image_rgb: RGB image, shape (H, W, 3)
        
    Returns:
        Preprocessed image in HWC format, uint8 [0, 255], shape (H, W, 3)
    """
    # Only keep HWC format and uint8 dtype (no resize, no normalization)
    # The transform pipeline will handle:
    # 1. Resize (via ResizeImages transform)
    # 2. Normalization (via Normalize transform)
    return image_rgb.astype(np.uint8)


# ============================================================================
# Load Model and Config
# ============================================================================

def load_model_and_config(
    config_name: str,
    checkpoint_path: str,
) -> tuple[_model.BaseModel, _config.TrainConfig, _config.DataConfig]:
    """Load model, train config, and data config."""
    logger.info(f"Loading config: {config_name}")
    train_config = _config.get_config(config_name)
    
    # Create data config
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    
    # Load checkpoint parameters
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    loaded_params = _model.restore_params(checkpoint_path, restore_type=np.ndarray)
    
    # Create model with loaded parameters
    logger.info(f"Creating model: {train_config.model.model_type}")
    model = train_config.model.load(loaded_params)
    
    return model, train_config, data_config


def load_norm_stats(norm_stats_path: pathlib.Path) -> dict[str, _transforms.NormStats]:
    """Load normalization statistics."""
    if not norm_stats_path.exists():
        raise FileNotFoundError(f"Normalization stats not found: {norm_stats_path}")
    
    logger.info(f"Loading normalization stats from: {norm_stats_path}")
    return _normalize.load(norm_stats_path)


def create_transforms(
    data_config: _config.DataConfig,
    norm_stats: dict[str, _transforms.NormStats],
) -> tuple[_transforms.DataTransformFn, _transforms.DataTransformFn]:
    """Create input and output transform functions."""
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


# ============================================================================
# Load Offline Data (from LeRobot dataset)
# ============================================================================

def load_offline_frame(
    dataset_path: str,
    episode_index: int,
    frame_index: int,
    action_horizon: int,
) -> dict[str, Any]:
    """Load a single frame from offline dataset.
    
    Args:
        dataset_path: Path to LeRobot dataset
        episode_index: Episode index
        frame_index: Frame index within the episode
        action_horizon: Action horizon for delta timestamps
        
    Returns:
        Frame data dictionary
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
    
    # Find the frame
    logger.info(f"Loading episode {episode_index}, frame {frame_index}...")
    
    frame_count = 0
    for idx in range(len(dataset)):
        frame = dataset[idx]
        if frame.get("episode_index") == episode_index:
            if frame_count == frame_index:
                logger.info(f"Found frame at dataset index {idx}")
                return frame
            frame_count += 1
    
    raise ValueError(f"Frame {frame_index} not found in episode {episode_index}")


# ============================================================================
# Simulate Realtime Data (mimicking robot + camera input)
# ============================================================================

def create_realtime_frame(
    offline_frame: dict[str, Any],
    action_horizon: int,
    prompt: str,
) -> dict[str, Any]:
    """Create a realtime-style frame from offline data.
    
    This simulates what realman_inference.py does: taking raw state and images,
    then preprocessing them.
    
    Args:
        offline_frame: Offline frame from dataset
        action_horizon: Action horizon
        prompt: Task prompt
        
    Returns:
        Realtime-style frame dictionary
    """
    logger.info("Creating realtime-style frame...")
    
    # Extract state (should be in radians)
    # Convert to numpy if it's a Tensor
    state = offline_frame["observation.state"]
    if hasattr(state, 'numpy'):
        state = state.numpy()
    elif hasattr(state, '__array__'):
        state = np.array(state)
    else:
        state = np.asarray(state)
    
    # Extract raw images and convert back to uint8 RGB
    # Offline images are in CHW format, normalized to [-1, 1]
    cam_high_chw = offline_frame["observation.images.cam_high"]
    cam_left_wrist_chw = offline_frame["observation.images.cam_left_wrist"]
    cam_right_wrist_chw = offline_frame["observation.images.cam_right_wrist"]
    
    # Convert CHW -> HWC and denormalize
    def chw_to_rgb_uint8(img_chw):
        # Convert to numpy if it's a Tensor
        if hasattr(img_chw, 'numpy'):
            img_chw = img_chw.numpy()
        elif hasattr(img_chw, '__array__'):
            img_chw = np.array(img_chw)
        else:
            img_chw = np.asarray(img_chw)
        
        img_hwc = np.transpose(img_chw, (1, 2, 0))  # CHW -> HWC
        img_denorm = (img_hwc + 1.0) * 127.5  # [-1, 1] -> [0, 255]
        return np.clip(img_denorm, 0, 255).astype(np.uint8)
    
    cam_high_rgb = chw_to_rgb_uint8(cam_high_chw)
    cam_left_wrist_rgb = chw_to_rgb_uint8(cam_left_wrist_chw)
    cam_right_wrist_rgb = chw_to_rgb_uint8(cam_right_wrist_chw)
    
    # Preprocess images using realtime method
    cam_high_processed = preprocess_image_realtime(cam_high_rgb)
    cam_left_wrist_processed = preprocess_image_realtime(cam_left_wrist_rgb)
    cam_right_wrist_processed = preprocess_image_realtime(cam_right_wrist_rgb)
    
    # Create placeholder action (not used in inference, but needed by transforms)
    placeholder_action = np.zeros((action_horizon, 12), dtype=np.float32)
    
    # Return in realtime format
    return {
        "observation.state": state,
        "action": placeholder_action,
        "observation.images.cam_high": cam_high_processed,
        "observation.images.cam_left_wrist": cam_left_wrist_processed,
        "observation.images.cam_right_wrist": cam_right_wrist_processed,
        "prompt": prompt,
    }


# ============================================================================
# Comparison Functions
# ============================================================================

def compare_arrays(
    name: str,
    array1: np.ndarray,
    array2: np.ndarray,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> bool:
    """Compare two arrays and print statistics.
    
    Args:
        name: Name of the field being compared
        array1: First array (offline)
        array2: Second array (realtime)
        rtol: Relative tolerance
        atol: Absolute tolerance
        
    Returns:
        True if arrays are close, False otherwise
    """
    print(f"\n{'='*70}")
    print(f"Comparing: {name}")
    print(f"{'='*70}")
    
    # Shape comparison
    print(f"  Shape (offline):   {array1.shape}")
    print(f"  Shape (realtime):  {array2.shape}")
    
    if array1.shape != array2.shape:
        print(f"  ❌ SHAPE MISMATCH!")
        return False
    
    print(f"  ✅ Shapes match")
    
    # Dtype comparison
    print(f"\n  Dtype (offline):   {array1.dtype}")
    print(f"  Dtype (realtime):  {array2.dtype}")
    
    # Statistics
    print(f"\n  Statistics (offline):")
    print(f"    Min:    {np.min(array1):.6f}")
    print(f"    Max:    {np.max(array1):.6f}")
    print(f"    Mean:   {np.mean(array1):.6f}")
    print(f"    Std:    {np.std(array1):.6f}")
    
    print(f"\n  Statistics (realtime):")
    print(f"    Min:    {np.min(array2):.6f}")
    print(f"    Max:    {np.max(array2):.6f}")
    print(f"    Mean:   {np.mean(array2):.6f}")
    print(f"    Std:    {np.std(array2):.6f}")
    
    # Difference statistics
    diff = array1 - array2
    abs_diff = np.abs(diff)
    
    print(f"\n  Difference (offline - realtime):")
    print(f"    Min diff:     {np.min(diff):.6e}")
    print(f"    Max diff:     {np.max(diff):.6e}")
    print(f"    Mean abs diff: {np.mean(abs_diff):.6e}")
    print(f"    Max abs diff:  {np.max(abs_diff):.6e}")
    
    # Check if close
    is_close = np.allclose(array1, array2, rtol=rtol, atol=atol)
    
    if is_close:
        print(f"\n  ✅ VALUES MATCH (rtol={rtol}, atol={atol})")
    else:
        print(f"\n  ❌ VALUES DIFFER (rtol={rtol}, atol={atol})")
        
        # Show some examples of mismatches
        mismatch_mask = ~np.isclose(array1, array2, rtol=rtol, atol=atol)
        num_mismatches = np.sum(mismatch_mask)
        print(f"     Number of mismatches: {num_mismatches}/{array1.size}")
        
        if num_mismatches > 0 and array1.ndim <= 2:
            print(f"     First few mismatches:")
            mismatch_indices = np.where(mismatch_mask)
            for i in range(min(5, num_mismatches)):
                idx = tuple(m[i] for m in mismatch_indices)
                print(f"       Index {idx}: {array1[idx]:.6f} vs {array2[idx]:.6f}")
    
    return is_close


def compare_observations(
    obs1: _model.Observation,
    obs2: _model.Observation,
) -> bool:
    """Compare two Observation objects.
    
    Args:
        obs1: First observation (offline)
        obs2: Second observation (realtime)
        
    Returns:
        True if observations are close, False otherwise
    """
    print("\n" + "="*70)
    print("COMPARING OBSERVATIONS")
    print("="*70)
    
    all_match = True
    
    # Compare state
    if hasattr(obs1, 'state') and hasattr(obs2, 'state'):
        match = compare_arrays("state", np.array(obs1.state), np.array(obs2.state))
        all_match = all_match and match
    
    # Compare images
    if hasattr(obs1, 'images') and hasattr(obs2, 'images'):
        for img_key in obs1.images.keys():
            if img_key in obs2.images:
                match = compare_arrays(
                    f"images.{img_key}",
                    np.array(obs1.images[img_key]),
                    np.array(obs2.images[img_key]),
                )
                all_match = all_match and match
            else:
                print(f"  ❌ Image key '{img_key}' not found in realtime observation")
                all_match = False
    
    # Compare prompt
    if hasattr(obs1, 'prompt') and hasattr(obs2, 'prompt'):
        prompt1 = obs1.prompt
        prompt2 = obs2.prompt
        print(f"\n{'='*70}")
        print(f"Comparing: prompt")
        print(f"{'='*70}")
        print(f"  Offline:  {prompt1}")
        print(f"  Realtime: {prompt2}")
        if prompt1 == prompt2:
            print(f"  ✅ Prompts match")
        else:
            print(f"  ⚠️  Prompts differ (this may be expected)")
    
    return all_match


# ============================================================================
# Main Comparison Logic
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="对比离线推理和实时推理的observation输入"
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="pi0_realman",
        help="训练配置名称 (default: pi0_realman)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="模型checkpoint路径",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="LeRobot数据集路径",
    )
    parser.add_argument(
        "--norm-stats",
        type=str,
        default="assets/pi0_realman/realman_dataset/",
        help="归一化统计文件路径 (default: assets/pi0_realman/realman_dataset/)",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=0,
        help="要加载的episode索引 (default: 0)",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=10,
        help="episode中的frame索引 (default: 10)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Let the forceps go along the black S shaped path",
        help="任务提示词 (default: 'Let the forceps go along the black S shaped path')",
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("🔍 离线推理 vs 实时推理 Observation 对比工具")
    print("="*80)
    
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
    
    # Load offline frame from dataset
    print("\n" + "="*80)
    print("📂 加载离线数据 (offline_inference.py方式)")
    print("="*80)
    offline_frame = load_offline_frame(
        args.dataset_path,
        args.episode,
        args.frame_index,
        train_config.model.action_horizon,
    )
    
    # Create realtime-style frame
    print("\n" + "="*80)
    print("🤖 创建实时风格数据 (realman_inference.py方式)")
    print("="*80)
    realtime_frame = create_realtime_frame(
        offline_frame,
        train_config.model.action_horizon,
        args.prompt,
    )
    
    # Transform both frames
    print("\n" + "="*80)
    print("🔄 应用input transforms")
    print("="*80)
    
    print("  Transforming offline frame...")
    offline_transformed = input_transform(offline_frame)
    
    print("  Transforming realtime frame...")
    realtime_transformed = input_transform(realtime_frame)
    
    # Add batch dimension and convert to JAX arrays
    print("\n" + "="*80)
    print("📦 添加batch维度并转换为JAX数组")
    print("="*80)
    
    offline_batch = jax.tree.map(lambda x: np.expand_dims(x, axis=0), offline_transformed)
    offline_batch = jax.tree.map(lambda x: jnp.asarray(x), offline_batch)
    
    realtime_batch = jax.tree.map(lambda x: np.expand_dims(x, axis=0), realtime_transformed)
    realtime_batch = jax.tree.map(lambda x: jnp.asarray(x), realtime_batch)
    
    # Create Observation objects
    print("\n" + "="*80)
    print("🎯 创建Observation对象")
    print("="*80)
    
    print("  Creating offline observation...")
    offline_obs = _model.Observation.from_dict(offline_batch)
    
    print("  Creating realtime observation...")
    realtime_obs = _model.Observation.from_dict(realtime_batch)
    
    # Compare observations
    print("\n" + "="*80)
    print("⚖️  开始对比")
    print("="*80)
    
    all_match = compare_observations(offline_obs, realtime_obs)
    
    # Summary
    print("\n" + "="*80)
    print("📊 对比总结")
    print("="*80)
    
    if all_match:
        print("✅ 所有字段都匹配！离线推理和实时推理的observation输入一致。")
        print("   实时推理的输入处理是正确的。")
    else:
        print("❌ 存在不匹配的字段！")
        print("   请检查上面的详细对比信息，找出差异原因。")
        print("   可能的原因：")
        print("   1. 图像预处理方法不同")
        print("   2. 归一化参数不同")
        print("   3. Transform pipeline顺序不同")
        print("   4. 数据类型转换问题")
    
    print("="*80)
    
    # Optional: Test inference to ensure it runs
    print("\n" + "="*80)
    print("🧪 测试模型推理 (确保observation格式正确)")
    print("="*80)
    
    rng = jax.random.key(42)
    
    try:
        print("  Running inference with offline observation...")
        rng, inference_rng = jax.random.split(rng)
        offline_pred = model.sample_actions(
            inference_rng,
            offline_obs,
            num_steps=10
        )
        print(f"    ✅ Offline inference successful, output shape: {offline_pred.shape}")
        
        print("  Running inference with realtime observation...")
        rng, inference_rng = jax.random.split(rng)
        realtime_pred = model.sample_actions(
            inference_rng,
            realtime_obs,
            num_steps=10
        )
        print(f"    ✅ Realtime inference successful, output shape: {realtime_pred.shape}")
        
        # Compare predictions
        pred_match = compare_arrays(
            "model predictions",
            np.array(offline_pred[0]),
            np.array(realtime_pred[0]),
            rtol=1e-4,
            atol=1e-6,
        )
        
        if pred_match:
            print("\n✅ 模型预测结果也匹配！说明整个pipeline一致。")
        else:
            print("\n⚠️  模型预测结果有微小差异（可能是由于随机性或数值精度）")
        
    except Exception as e:
        print(f"    ❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("✅ 对比完成！")
    print("="*80)


if __name__ == "__main__":
    main()

