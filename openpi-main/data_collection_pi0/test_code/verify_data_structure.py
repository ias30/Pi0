#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证HDF5文件数据结构是否符合预期
"""

import h5py
import numpy as np
import sys

# 预期的数据结构
EXPECTED_STRUCTURE = {
    'observations': {
        'global_timestamps': ((None,), np.float64),
        'cameras': {
            'camera_high': {
                'color': ((None, 480, 640, 3), np.uint8),
                'depth': ((None, 480, 640), np.uint16),
                'local_timestamps': ((None,), np.float64),
            },
            'camera_left_wrist': {
                'color': ((None, 480, 640, 3), np.uint8),
                'local_timestamps': ((None,), np.float64),
            },
            'camera_right_wrist': {
                'color': ((None, 480, 640, 3), np.uint8),
                'local_timestamps': ((None,), np.float64),
            }
        },
        'arms': {
            'left_arm': {
                'joint_positions': ((None, 6), np.float64),
                'end_effector_poses': ((None, 6), np.float64),
                'local_timestamps': ((None,), np.float64),
            },
            'right_arm': {
                'joint_positions': ((None, 6), np.float64),
                'end_effector_poses': ((None, 6), np.float64),
                'local_timestamps': ((None,), np.float64),
            }
        },
        'motors': {
            'left_motors': {
                'positions': ((None, 4), np.int32),
                'states': ((None, 4), np.int32),
                'local_timestamps': ((None,), np.float64),
            },
            'right_motors': {
                'positions': ((None, 4), np.int32),
                'states': ((None, 4), np.int32),
                'local_timestamps': ((None,), np.float64),
            }
        }
    },
    'metadata': {}
}


def check_dataset(dataset, expected_shape, expected_dtype, path):
    """检查单个数据集"""
    errors = []
    
    # 检查形状
    actual_shape = dataset.shape
    expected_shape_tuple = expected_shape
    
    # None 表示可变长度
    for i, (actual, expected) in enumerate(zip(actual_shape, expected_shape_tuple)):
        if expected is not None and actual != expected:
            errors.append(f"  ❌ {path}: 维度{i}不匹配 (期望{expected}, 实际{actual})")
    
    # 检查维度数量
    if len(actual_shape) != len(expected_shape_tuple):
        errors.append(f"  ❌ {path}: 维度数量不匹配 (期望{len(expected_shape_tuple)}, 实际{len(actual_shape)})")
    
    # 检查数据类型
    if dataset.dtype != expected_dtype:
        errors.append(f"  ❌ {path}: 数据类型不匹配 (期望{expected_dtype}, 实际{dataset.dtype})")
    
    if not errors:
        print(f"  ✅ {path}: shape={actual_shape}, dtype={dataset.dtype}")
    
    return errors


def verify_structure(h5_file, expected, current_path="/", errors_list=None):
    """递归验证HDF5文件结构"""
    if errors_list is None:
        errors_list = []
    
    for key, value in expected.items():
        full_path = f"{current_path}{key}"
        
        # 检查键是否存在
        if key not in h5_file:
            errors_list.append(f"❌ 缺失: {full_path}")
            continue
        
        item = h5_file[key]
        
        if isinstance(value, dict):
            # 递归检查组
            if not isinstance(item, h5py.Group):
                errors_list.append(f"❌ {full_path} 应该是 Group，但实际是 {type(item)}")
            else:
                print(f"📁 {full_path}")
                verify_structure(item, value, f"{full_path}/", errors_list)
        elif isinstance(value, tuple):
            # 检查数据集
            expected_shape, expected_dtype = value
            if not isinstance(item, h5py.Dataset):
                errors_list.append(f"❌ {full_path} 应该是 Dataset，但实际是 {type(item)}")
            else:
                dataset_errors = check_dataset(item, expected_shape, expected_dtype, full_path)
                errors_list.extend(dataset_errors)
    
    return errors_list


def main(h5_filepath):
    """主验证函数"""
    print("=" * 80)
    print(f"验证HDF5文件结构: {h5_filepath}")
    print("=" * 80)
    
    try:
        with h5py.File(h5_filepath, 'r') as f:
            print("\n📋 检查文件结构...\n")
            errors = verify_structure(f, EXPECTED_STRUCTURE)
            
            print("\n" + "=" * 80)
            if errors:
                print(f"❌ 发现 {len(errors)} 个问题:")
                for error in errors:
                    print(error)
                print("=" * 80)
                return False
            else:
                print("✅ 数据结构验证通过！")
                
                # 显示一些额外信息
                obs_count = f['observations']['global_timestamps'].shape[0]
                print(f"\n📊 统计信息:")
                print(f"  - 观测数量: {obs_count}")
                
                if 'metadata' in f:
                    metadata = f['metadata']
                    if 'episode_duration' in metadata.attrs:
                        duration = metadata.attrs['episode_duration']
                        print(f"  - Episode时长: {duration:.2f}秒")
                        if obs_count > 0 and duration > 0:
                            actual_hz = obs_count / duration
                            print(f"  - 实际采集频率: {actual_hz:.2f} Hz")
                
                print("=" * 80)
                return True
                
    except FileNotFoundError:
        print(f"❌ 文件不存在: {h5_filepath}")
        return False
    except Exception as e:
        print(f"❌ 验证过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python verify_data_structure.py <hdf5_file_path>")
        sys.exit(1)
    
    h5_filepath = sys.argv[1]
    success = main(h5_filepath)
    sys.exit(0 if success else 1)

