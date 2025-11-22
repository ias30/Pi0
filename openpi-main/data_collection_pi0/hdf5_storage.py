#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化版HDF5存储管理器
解决0.5Hz → 30Hz性能瓶颈，基于Diffusion Policy优化思路
更新为新的三相机数据结构
"""

import os
import h5py
import numpy as np
import threading
import time
import queue
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from time_sync import get_timestamp, get_timestamp_ms


class OptimizedHDF5Storage:
    """优化版HDF5存储管理器 - 高性能数据采集"""

    def __init__(self, base_dir: str = "data_collection_episodes"):
        """
        初始化优化版HDF5存储管理器

        Args:
            base_dir: 数据存储基础目录
        """
        self.base_dir = base_dir
        self.current_file: Optional[h5py.File] = None
        self.current_filename: Optional[str] = None
        self.current_episode_start_time: Optional[float] = None
        self.write_lock = threading.Lock()
        self.is_recording = False
        
        # ⚡ 性能优化参数
        self.estimated_duration = 180  # 预估最大录制时长(秒)
        self.target_hz = 30 # 目标采集频率
        self.estimated_samples = int(self.estimated_duration * self.target_hz * 1.2) # 20%余量

        # 预分配空间追踪
        self.allocated_size: int = 0
        self.current_index: int = 0
        
        # flush优化
        self.write_count = 0
        self.flush_interval = 100  # 每100次写入flush一次

        # 确保存储目录存在
        os.makedirs(self.base_dir, exist_ok=True)
        print(f"🚀 优化版HDF5存储管理器已初始化，存储目录: {self.base_dir}")

    def start_episode(self, device_mapping: Optional[Dict[str, Any]] = None) -> str:
        """
        开始新的Episode录制

        Args:
            device_mapping: 设备映射配置信息（将记录到metadata）

        Returns:
            新创建的文件路径
        """
        with self.write_lock:
            if self.is_recording:
                print("警告：已在录制中，先停止当前Episode")
                self.stop_episode()

            timestamp = datetime.now()
            filename = timestamp.strftime("%Y%m%d_%H%M%S_episode.h5")
            filepath = os.path.join(self.base_dir, filename)

            try:
                self.current_file = h5py.File(filepath, 'w',
                                            rdcc_nbytes=1024*1024*64,
                                            rdcc_nslots=521)
                self.current_filename = filepath
                self.current_episode_start_time = get_timestamp()
                self.is_recording = True

                self.allocated_size = self.estimated_samples
                self.current_index = 0
                self.write_count = 0

                self._write_metadata(device_mapping)
                self._preallocate_datasets()

                print(f"✅ 开始新Episode录制: {filepath}")
                print(f"📊 预分配空间: {self.allocated_size} observations")
                return filepath

            except Exception as e:
                self.current_file = None
                self.current_filename = None
                self.is_recording = False
                error_msg = f"创建HDF5文件失败: {e}"
                print(error_msg)
                raise RuntimeError(error_msg)

    def _preallocate_datasets(self):
        """⚡ 预分配数据集空间（新三相机结构）"""
        if self.current_file is None:
            return

        print("🔧 预分配数据集空间...")
        obs_group = self.current_file.create_group('observations')
        
        # Global Timestamps
        obs_group.create_dataset('global_timestamps', (self.allocated_size,), dtype='f8', chunks=True)

        # Cameras - 新结构：1个RealSense + 2个Wrist相机
        cam_group = obs_group.create_group('cameras')
        
        # camera_high (RealSense with depth)
        cam_high = cam_group.create_group('camera_high')
        cam_high.create_dataset('color', (self.allocated_size, 480, 640, 3), dtype='uint8', chunks=(1, 480, 640, 3))
        cam_high.create_dataset('depth', (self.allocated_size, 480, 640), dtype='uint16', chunks=(1, 480, 640))
        cam_high.create_dataset('local_timestamps', (self.allocated_size,), dtype='f8', chunks=True)
        
        # camera_left_wrist (RGB only)
        cam_left = cam_group.create_group('camera_left_wrist')
        cam_left.create_dataset('color', (self.allocated_size, 480, 640, 3), dtype='uint8', chunks=(1, 480, 640, 3))
        cam_left.create_dataset('local_timestamps', (self.allocated_size,), dtype='f8', chunks=True)
        
        # camera_right_wrist (RGB only)
        cam_right = cam_group.create_group('camera_right_wrist')
        cam_right.create_dataset('color', (self.allocated_size, 480, 640, 3), dtype='uint8', chunks=(1, 480, 640, 3))
        cam_right.create_dataset('local_timestamps', (self.allocated_size,), dtype='f8', chunks=True)

        # Arms
        arm_group = obs_group.create_group('arms')
        for arm_id in ['left_arm', 'right_arm']:
            sub_group = arm_group.create_group(arm_id)
            sub_group.create_dataset('joint_positions', (self.allocated_size, 6), dtype='f8', chunks=(128, 6))
            sub_group.create_dataset('end_effector_poses', (self.allocated_size, 6), dtype='f8', chunks=(128, 6))
            sub_group.create_dataset('local_timestamps', (self.allocated_size,), dtype='f8', chunks=True)

        # Motors
        motor_group = obs_group.create_group('motors')
        for motor_id in ['left_motors', 'right_motors']:
            sub_group = motor_group.create_group(motor_id)
            sub_group.create_dataset('positions', (self.allocated_size, 4), dtype='i4', chunks=(128, 4))
            sub_group.create_dataset('states', (self.allocated_size, 4), dtype='i4', chunks=(128, 4))
            sub_group.create_dataset('local_timestamps', (self.allocated_size,), dtype='f8', chunks=True)

    def stop_episode(self) -> Optional[str]:
        """
        停止当前Episode录制
        """
        with self.write_lock:
            if not self.is_recording or self.current_file is None:
                print("当前没有进行录制")
                return None
            
            try:
                self._trim_datasets_to_actual_size()
                self._update_final_metadata()
                
                self.current_file.flush()
                self.current_file.close()
                completed_file = self.current_filename
                
                print(f"✅ Episode录制完成: {completed_file}")
                print(f"📊 实际数据统计: {self.current_index} observations")
                
                self.current_file = None
                self.current_filename = None
                self.current_episode_start_time = None
                self.is_recording = False
                
                return completed_file
                
            except Exception as e:
                error_msg = f"停止Episode录制时出错: {e}"
                print(error_msg)
                
                self.current_file = None
                self.current_filename = None
                self.current_episode_start_time = None
                self.is_recording = False
                
                return None

    def _trim_datasets_to_actual_size(self):
        """⚡ 调整数据集大小到实际使用大小"""
        if self.current_file is None:
            return
        
        actual_size = self.current_index
        if actual_size > 0 and actual_size < self.allocated_size:
            obs_group = self.current_file['observations']
            
            # Recursively resize all datasets
            def resize_datasets(group):
                for key, value in group.items():
                    if isinstance(value, h5py.Dataset):
                        value.resize(actual_size, axis=0)
                    elif isinstance(value, h5py.Group):
                        resize_datasets(value)

            resize_datasets(obs_group)

    def write_observation(self, observation: Dict[str, Any]):
        """
        写入一个完整的observation数据包（新三相机结构）
        """
        if not self.is_recording or self.current_file is None:
            return
        
        with self.write_lock:
            try:
                idx = self.current_index
                if idx >= self.allocated_size:
                    print(f"⚠️  Observation 空间不足，自动扩展...")
                    self._expand_datasets()

                obs_group = self.current_file['observations']

                # Global Timestamps
                obs_group['global_timestamps'][idx] = observation['global_timestamp']

                # Cameras - 新结构
                cameras = observation['cameras']
                
                # camera_high (with depth)
                if 'camera_high' in cameras:
                    cam_data = cameras['camera_high']
                    cam_group = obs_group['cameras']['camera_high']
                    cam_group['color'][idx] = cam_data['color']
                    cam_group['depth'][idx] = cam_data['depth']
                    cam_group['local_timestamps'][idx] = cam_data['local_timestamp']
                
                # camera_left_wrist (RGB only)
                if 'camera_left_wrist' in cameras:
                    cam_data = cameras['camera_left_wrist']
                    cam_group = obs_group['cameras']['camera_left_wrist']
                    cam_group['color'][idx] = cam_data['color']
                    cam_group['local_timestamps'][idx] = cam_data['local_timestamp']
                
                # camera_right_wrist (RGB only)
                if 'camera_right_wrist' in cameras:
                    cam_data = cameras['camera_right_wrist']
                    cam_group = obs_group['cameras']['camera_right_wrist']
                    cam_group['color'][idx] = cam_data['color']
                    cam_group['local_timestamps'][idx] = cam_data['local_timestamp']

                # Arms
                for arm_id in ['left_arm', 'right_arm']:
                    arm_data = observation['arms'][arm_id]
                    arm_group = obs_group['arms'][arm_id]
                    arm_group['joint_positions'][idx] = arm_data['joint_positions']
                    arm_group['end_effector_poses'][idx] = arm_data['end_effector_poses']
                    arm_group['local_timestamps'][idx] = arm_data['local_timestamp']

                # Motors
                for motor_id in ['left_motors', 'right_motors']:
                    motor_data = observation['motors'][motor_id]
                    motor_group = obs_group['motors'][motor_id]
                    motor_group['positions'][idx] = motor_data['positions']
                    motor_group['states'][idx] = motor_data['states']
                    motor_group['local_timestamps'][idx] = motor_data['local_timestamp']
                
                self.current_index += 1
                self._conditional_flush()
            
            except Exception as e:
                error_msg = f"写入observation数据时出错: {e}"
                print(error_msg)

    def _expand_datasets(self):
        """扩展所有数据集空间"""
        new_size = int(self.allocated_size * 1.5)
        
        obs_group = self.current_file['observations']
        
        def expand_recursive(group):
            for key, value in group.items():
                if isinstance(value, h5py.Dataset):
                    value.resize(new_size, axis=0)
                elif isinstance(value, h5py.Group):
                    expand_recursive(value)
                    
        expand_recursive(obs_group)
        self.allocated_size = new_size
        print(f"🔧 数据集空间已扩展至 {new_size}")

    def _conditional_flush(self):
        """⚡ 条件性flush - 减少磁盘I/O"""
        self.write_count += 1
        if self.write_count % self.flush_interval == 0:
            self.current_file.flush()

    def delete_last_episode(self) -> bool:
        """
        删除最近的Episode文件
        """
        try:
            episode_files = [f for f in os.listdir(self.base_dir) if f.endswith('_episode.h5')]
            if not episode_files:
                print("没有找到Episode文件可删除")
                return False
            
            episode_files.sort(reverse=True)
            latest_file = os.path.join(self.base_dir, episode_files[0])
            
            os.remove(latest_file)
            print(f"已删除最近的Episode文件: {latest_file}")
            return True
            
        except Exception as e:
            print(f"删除Episode文件时出错: {e}")
            return False

    def log_error(self, error_message: str):
        """记录错误信息（为了兼容性保留，但现在错误日志存储在metadata中）"""
        print(f"错误记录: {error_message}")

    def _write_metadata(self, device_mapping: Optional[Dict[str, Any]] = None):
        """写入初始元数据
        
        Args:
            device_mapping: 设备映射配置信息
        """
        if self.current_file is None: return
        metadata = self.current_file.create_group('metadata')
        metadata.attrs['episode_start_time'] = self.current_episode_start_time
        metadata.attrs['creation_timestamp'] = get_timestamp_ms()
        metadata.attrs['version'] = '4.0_triple_camera'
        
        # 记录设备映射配置
        if device_mapping:
            metadata.attrs['swap_arms'] = device_mapping.get('swap_arms', False)
            metadata.attrs['device_mapping_description'] = device_mapping.get('description', '')
            # 记录物理设备IP（用于追溯）
            if 'left_device_ip' in device_mapping:
                metadata.attrs['physical_left_device_ip'] = device_mapping['left_device_ip']
            if 'right_device_ip' in device_mapping:
                metadata.attrs['physical_right_device_ip'] = device_mapping['right_device_ip']

    def _update_final_metadata(self):
        """更新最终元数据"""
        if self.current_file is None: return
        metadata = self.current_file['metadata']
        end_time = get_timestamp()
        metadata.attrs['episode_end_time'] = end_time
        metadata.attrs['episode_duration'] = end_time - self.current_episode_start_time
        metadata.attrs['observation_count'] = self.current_index

    def get_status(self) -> Dict[str, Any]:
        """获取当前状态"""
        return {
            'is_recording': self.is_recording,
            'current_file': self.current_filename,
            'observation_count': self.current_index,
            'allocated_size': self.allocated_size,
        }

    def cleanup(self):
        """清理资源"""
        if self.is_recording:
            self.stop_episode()
        print("✅ 优化版HDF5存储管理器已清理")

HDF5Storage = OptimizedHDF5Storage
