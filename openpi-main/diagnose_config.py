#!/usr/bin/env python3
"""诊断训练配置是否正确加载"""
import sys
import traceback

print("=" * 80)
print("OpenPI 训练配置诊断脚本")
print("=" * 80)

try:
    print("\n[1/6] 导入基础库...")
    import jax
    print(f"  ✓ JAX version: {jax.__version__}")
    print(f"  ✓ JAX devices: {jax.devices()}")
    
    print("\n[2/6] 导入 OpenPI 配置模块...")
    from openpi.training import config as _config
    print("  ✓ 配置模块加载成功")
    
    print("\n[3/6] 获取 pi0_realman 配置...")
    config = _config.get_config("pi0_realman")
    print(f"  ✓ 配置名称: {config.name}")
    print(f"  ✓ Repo ID: {config.data.repo_id}")
    print(f"  ✓ 批次大小: {config.batch_size}")
    print(f"  ✓ 训练步数: {config.num_train_steps}")
    
    print("\n[4/6] 检查数据集是否存在...")
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    try:
        dataset = LeRobotDataset(config.data.repo_id)
        print(f"  ✓ 数据集加载成功: {len(dataset)} 样本")
    except Exception as e:
        print(f"  ✗ 数据集加载失败: {e}")
        print("  提示: 检查 repo_id 是否正确，数据集是否在正确路径")
    
    print("\n[5/6] 检查 norm_stats 是否存在...")
    import os
    norm_stats_path = f"./assets/{config.name}/{config.data.repo_id}/norm_stats.json"
    if os.path.exists(norm_stats_path):
        print(f"  ✓ Norm stats 存在: {norm_stats_path}")
        import json
        with open(norm_stats_path) as f:
            stats = json.load(f)
            print(f"  ✓ 统计维度: {len(stats)} 个键")
    else:
        print(f"  ✗ Norm stats 不存在: {norm_stats_path}")
        print("  提示: 需要先运行 compute_norm_stats.py")
    
    print("\n[6/6] 测试配置解析（模拟 train.py 的启动）...")
    import sys
    sys.argv = ["diagnose_config.py", "pi0_realman", "--exp-name=test"]
    try:
        parsed_config = _config.cli()
        print(f"  ✓ 配置解析成功")
        print(f"  ✓ 实验名称: {parsed_config.exp_name}")
        print(f"  ✓ Checkpoint 目录: {parsed_config.checkpoint_dir}")
    except Exception as e:
        print(f"  ✗ 配置解析失败: {e}")
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("✅ 诊断完成！如果以上所有检查都通过，训练应该可以正常启动。")
    print("=" * 80)
    
except Exception as e:
    print(f"\n✗ 诊断过程中出现错误:")
    traceback.print_exc()
    sys.exit(1)

