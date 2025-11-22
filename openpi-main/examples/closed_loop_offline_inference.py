#!/usr/bin/env python3
"""Closed-loop offline inference script for Realman π₀ model.

Usage:
uv run /home/ren9/DualArm/openpi/examples/closed_loop_offline_inference.py \
    --config-name pi0_realman \
    --checkpoint checkpoints/pi0_realman/realman_finetune_v1/14999/params \
    --dataset-path /home/ren9/.cache/huggingface/lerobot/realman_test2_dataset \
    --episode 0 \
    --output closed_predictions.csv \
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


# ================================
#  加载模型与配置
# ================================
def load_model_and_config(config_name: str, checkpoint_path: str):
    logger.info(f"Loading config: {config_name}")
    train_config = _config.get_config(config_name)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    params = _model.restore_params(download.maybe_download(checkpoint_path), restore_type=np.ndarray)
    model = train_config.model.load(params)
    return model, train_config, data_config


# ================================
#  加载归一化统计
# ================================
def load_norm_stats(norm_stats_path: pathlib.Path):
    if not norm_stats_path.exists():
        raise FileNotFoundError(f"Normalization stats not found: {norm_stats_path}")
    logger.info(f"Loading normalization stats from: {norm_stats_path}")
    return _normalize.load(norm_stats_path)


# ================================
#  创建变换函数
# ================================
def create_transforms(data_config, norm_stats):
    input_transform = _transforms.compose([
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
        _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs,
    ])
    output_transform = _transforms.compose([
        *data_config.model_transforms.outputs,
        _transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.data_transforms.outputs,
    ])
    return input_transform, output_transform


# ================================
#  加载数据集（本地）
# ================================
def load_episode_data(dataset_path: str, episode_index: int, action_horizon: int):
    logger.info(f"Loading dataset (local=True): {dataset_path}")
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(dataset_path, local=True)
    dataset = lerobot_dataset.LeRobotDataset(
        dataset_path,
        delta_timestamps={"action": [t / dataset_meta.fps for t in range(action_horizon)]},
        local=True,
    )
    frames = []
    logger.info(f"Loading episode {episode_index}...")
    for idx in tqdm(range(len(dataset)), desc="Loading frames"):
        frame = dataset[idx]
        if frame.get("episode_index") == episode_index:
            frames.append(frame)
    if not frames:
        raise ValueError(f"Episode {episode_index} not found")
    logger.info(f"Loaded {len(frames)} frames from episode {episode_index}")
    return frames


# ================================
#  闭环推理逻辑
# ================================
def run_closed_loop_inference(model, frames, input_transform, output_transform, batch_size):
    num_frames = len(frames)
    predictions, ground_truths = [], []
    logger.info(f"Running CLOSED-LOOP inference on {num_frames} frames")

    rng = jax.random.key(42)
    current_state = None  # 上一帧预测的 state（绝对动作）

    for t in tqdm(range(num_frames), desc="Inference (closed-loop)"):
        frame = frames[t]
        gt_action = frame["action"][0]  # Ground truth

        # 当前帧输入
        frame_input = dict(frame)
        if current_state is not None:
            frame_input["state"] = current_state

        # 应用输入变换
        transformed = input_transform(frame_input)
        batch = jax.tree.map(lambda x: jnp.asarray(x)[jnp.newaxis, ...], transformed)
        observation = _model.Observation.from_dict(batch)

        # 模型推理
        rng, inf_rng = jax.random.split(rng)
        pred_actions = model.sample_actions(inf_rng, observation, num_steps=10)
        pred_first = pred_actions[0, 0, :]  # (action_dim,)

        # 反归一化 + delta→absolute
        state = batch["state"][0] if "state" in batch else None
        pred_dict = {
            "actions": np.array(pred_first)[np.newaxis, :],
            "state": np.array(state) if state is not None else None,
        }
        denorm_pred = output_transform(pred_dict)
        abs_action = np.array(denorm_pred["actions"][0])

        predictions.append(abs_action)
        ground_truths.append(np.array(gt_action))
        current_state = abs_action  # 更新闭环状态

    return np.array(predictions), np.array(ground_truths)


# ================================
#  计算MSE
# ================================
def compute_mse(pred, gt):
    mse_per_joint = np.mean((pred - gt) ** 2, axis=0)
    mse_total = float(np.mean(mse_per_joint))
    joint_names = [
        "right_waist","right_shoulder","right_elbow","right_forearm_roll",
        "right_wrist_angle","right_wrist_rotate","left_waist","left_shoulder",
        "left_elbow","left_forearm_roll","left_wrist_angle","left_wrist_rotate",
    ]
    metrics = {"mse_overall": mse_total}
    for i, n in enumerate(joint_names):
        metrics[f"mse_{n}"] = float(mse_per_joint[i])
    return metrics


# ================================
#  保存结果
# ================================
def save_predictions(pred, gt, ep_idx, out_path: pathlib.Path):
    joint_names = [
        "joint_0_right_waist","joint_1_right_shoulder","joint_2_right_elbow",
        "joint_3_right_forearm_roll","joint_4_right_wrist_angle","joint_5_right_wrist_rotate",
        "joint_6_left_waist","joint_7_left_shoulder","joint_8_left_elbow",
        "joint_9_left_forearm_roll","joint_10_left_wrist_angle","joint_11_left_wrist_rotate",
    ]
    data = {name: pred[:, i] for i, name in enumerate(joint_names)}
    data["episode_index"] = [ep_idx] * len(pred)
    data["frame_index"] = list(range(len(pred)))
    for i, n in enumerate(joint_names):
        data[f"gt_{n}"] = gt[:, i]
    df = pd.DataFrame(data)
    df.to_csv(out_path, index=False)
    logger.info(f"Saved predictions to: {out_path}")


# ================================
#  主函数
# ================================
def main():
    parser = argparse.ArgumentParser(description="Closed-loop offline inference for Realman π₀")
    parser.add_argument("--config-name", type=str, default="pi0_realman")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--output", type=str, default="closed_predictions.csv")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--norm-stats", type=str,
        default="assets/pi0_realman/realman_dataset/",
    )
    args = parser.parse_args()

    model, train_cfg, data_cfg = load_model_and_config(args.config-name, args.checkpoint)
    norm_stats = load_norm_stats(pathlib.Path(args.norm_stats))
    in_tf, out_tf = create_transforms(data_cfg, norm_stats)
    frames = load_episode_data(args.dataset_path, args.episode, train_cfg.model.action_horizon)

    preds, gts = run_closed_loop_inference(model, frames, in_tf, out_tf, args.batch_size)
    metrics = compute_mse(preds, gts)

    logger.info("="*60)
    logger.info("MSE Metrics:")
    logger.info(f"Overall MSE: {metrics['mse_overall']:.6f}")
    for k, v in metrics.items():
        if k != "mse_overall":
            logger.info(f"  {k}: {v:.6f}")
    logger.info("="*60)

    save_predictions(preds, gts, args.episode, pathlib.Path(args.output))
    logger.info("Closed-loop inference complete!")


if __name__ == "__main__":
    main()
