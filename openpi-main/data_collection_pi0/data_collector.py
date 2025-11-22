#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主数据采集控制器
协调所有数据源的采集和存储，管理整个数据采集流程
"""

import numpy as np
import threading
import time
from typing import Dict, Any, Optional, List
from enum import Enum

from time_sync import TimeSync, get_global_time_sync
from hdf5_storage import HDF5Storage
from camera_collector import TripleCameraCollector
from arm_collector import DualArmCollector
from motor_collector import DualMotorCollector


class CollectionState(Enum):
    """数据采集状态枚举"""
    IDLE = "idle"
    STARTING = "starting"
    RECORDING = "recording"
    STOPPING = "stopping"
    ERROR = "error"


class DataCollector:
    """主数据采集控制器"""
    
    def __init__(self, storage_base_dir: str = "data_collection_episodes"):
        print("🚀 初始化数据采集系统...")
        
        self.time_sync = get_global_time_sync()
        self.storage = HDF5Storage(storage_base_dir)
        
        self.camera_collector: Optional[TripleCameraCollector] = None
        self.arm_collector: Optional[DualArmCollector] = None
        self.motor_collector: Optional[DualMotorCollector] = None
        
        self.state = CollectionState.IDLE
        self.state_lock = threading.Lock()
        
        # 采集控制
        self.collection_thread: Optional[threading.Thread] = None
        self.collect_stop_flag = threading.Event()
        self.collection_rate = 30 # Hz

        self.episode_count = 0
        self.total_errors = 0
        self.last_episode_path: Optional[str] = None
        
        print("✅ 数据采集系统初始化完成")
    
    def initialize_hardware(self, left_robot_arm=None, left_arm_handle=None,
                          right_robot_arm=None, right_arm_handle=None,
                          left_position_controller=None, right_position_controller=None):
        print("🔧 初始化硬件接口...")
        
        try:
            self.camera_collector = TripleCameraCollector(resolution=(640, 480), fps=30)
            self.camera_collector.set_error_callback(self._on_hardware_error)
            print("📷 相机采集器初始化完成")
        except Exception as e:
            print(f"❌ 相机采集器初始化失败: {e}")
            self.camera_collector = None
        
        if all([left_robot_arm, left_arm_handle, right_robot_arm, right_arm_handle]):
            try:
                self.arm_collector = DualArmCollector(
                    left_robot_arm, left_arm_handle,
                    right_robot_arm, right_arm_handle,
                    target_hz=60 # 机械臂以更高频率采集以保证数据新鲜度
                )
                self.arm_collector.set_error_callback(self._on_hardware_error)
                print("🦾 机械臂采集器初始化完成")
            except Exception as e:
                print(f"❌ 机械臂采集器初始化失败: {e}")
                self.arm_collector = None
        else:
            print("⚠️  机械臂参数不完整，跳过初始化")
        
        if left_position_controller and right_position_controller:
            try:
                self.motor_collector = DualMotorCollector(
                    left_position_controller, right_position_controller,
                    target_hz=30
                )
                self.motor_collector.set_error_callback(self._on_hardware_error)
                print("⚙️  电机采集器初始化完成")
            except Exception as e:
                print(f"❌ 电机采集器初始化失败: {e}")
                self.motor_collector = None
        else:
            print("⚠️  电机控制器参数不完整，跳过初始化")
        
        print("✅ 硬件接口初始化完成")
    
    def connect_all_hardware(self) -> bool:
        print("🔌 连接所有硬件...")
        success_flags = []
        
        if self.camera_collector:
            success_flags.append(self.camera_collector.connect_all())
        if self.arm_collector:
            success_flags.append(self.arm_collector.check_all_connections())
        if self.motor_collector:
            success_flags.append(self.motor_collector.check_all_connections())
        
        overall_success = all(success_flags) if success_flags else False
        print(f"🔌 硬件连接完成: {'✅ 全部成功' if overall_success else '❌ 部分失败'}")
        return overall_success

    def _start_all_hardware_streams(self) -> bool:
        """启动所有硬件的数据流"""
        print("🌊 启动所有硬件数据流...")
        success_flags = []
        if self.camera_collector:
            success_flags.append(self.camera_collector.start_all())
        if self.arm_collector:
            success_flags.append(self.arm_collector.start_all_collection())
        if self.motor_collector:
            success_flags.append(self.motor_collector.start_all_collection())
        return all(success_flags)

    def _stop_all_hardware_streams(self):
        """停止所有硬件的数据流"""
        print("🌊 停止所有硬件数据流...")
        if self.camera_collector: self.camera_collector.stop_all()
        if self.arm_collector: self.arm_collector.stop_all_collection()
        if self.motor_collector: self.motor_collector.stop_all_collection()

    def start_episode(self) -> bool:
        with self.state_lock:
            if self.state != CollectionState.IDLE:
                print(f"❌ 无法开始录制，当前状态: {self.state.value}")
                return False
            self.state = CollectionState.STARTING
        
        try:
            print("🎬 开始新Episode录制...")
            
            # 1. 启动硬件数据流
            if not self._start_all_hardware_streams():
                print("❌ 启动部分硬件数据流失败，中止录制")
                self._stop_all_hardware_streams() # 清理已启动的
                with self.state_lock:
                    self.state = CollectionState.ERROR
                return False
            
            # 等待数据流稳定，确保首次采集有数据
            time.sleep(0.5)

            # 2. 开始HDF5存储
            episode_path = self.storage.start_episode()
            self.last_episode_path = episode_path
            
            # 3. 启动采集线程
            self.collect_stop_flag.clear()
            self.collection_thread = threading.Thread(target=self._collection_worker, daemon=True)
            self.collection_thread.start()

            with self.state_lock:
                self.state = CollectionState.RECORDING
                self.episode_count += 1
            print(f"✅ Episode #{self.episode_count} 录制开始 @ {self.collection_rate}Hz: {episode_path}")
            return True
            
        except Exception as e:
            with self.state_lock:
                self.state = CollectionState.ERROR
            error_msg = f"开始Episode时发生异常: {e}"
            print(f"❌ {error_msg}")
            self.storage.log_error(error_msg)
            self.total_errors += 1
            # 确保资源被清理
            self._emergency_stop_all()
            return False
    
    def _collection_worker(self):
        """核心采集线程，以固定频率打包并写入observation"""
        interval = 1.0 / self.collection_rate
        
        while not self.collect_stop_flag.is_set():
            loop_start_time = time.time()
            try:
                # 1. 获取统一的全局时间戳
                global_obs_timestamp = get_global_time_sync().get_timestamp()
                
                # 2. 打包Observation
                observation = self._package_observation(global_obs_timestamp)
                
                # 3. 写入存储
                if observation:
                    self.storage.write_observation(observation)

            except Exception as e:
                print(f"❌ 采集线程错误: {e}")
                self.total_errors += 1

            # 维持固定频率
            elapsed = time.time() - loop_start_time
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        print("✅ 采集线程已退出")

    def _package_observation(self, global_obs_timestamp: float) -> Optional[Dict[str, Any]]:
        """从各个采集器获取最新数据并打包, 包含全局和本地时间戳（新三相机结构）"""
        obs = {'global_timestamp': global_obs_timestamp, 'cameras': {}, 'arms': {}, 'motors': {}}

        # Camera Data - 新三相机结构
        if self.camera_collector:
            # camera_high (RealSense with depth)
            cam_high_ts, cam_high_color, cam_high_depth = self.camera_collector.camera_high.get_latest_data()
            obs['cameras']['camera_high'] = {
                'color': cam_high_color, 
                'depth': cam_high_depth, 
                'local_timestamp': cam_high_ts
            }
            
            # camera_left_wrist (RGB only)
            cam_left_ts, cam_left_color = self.camera_collector.camera_left_wrist.get_latest_data()
            obs['cameras']['camera_left_wrist'] = {
                'color': cam_left_color,
                'local_timestamp': cam_left_ts
            }
            
            # camera_right_wrist (RGB only)
            cam_right_ts, cam_right_color = self.camera_collector.camera_right_wrist.get_latest_data()
            obs['cameras']['camera_right_wrist'] = {
                'color': cam_right_color,
                'local_timestamp': cam_right_ts
            }
        
        # Arm Data
        if self.arm_collector:
            l_arm_ts, l_j_p, l_ee_p = self.arm_collector.left_arm.get_latest_data()
            r_arm_ts, r_j_p, r_ee_p = self.arm_collector.right_arm.get_latest_data()
            obs['arms']['left_arm'] = {
                'joint_positions': l_j_p, 'end_effector_poses': l_ee_p,
                'local_timestamp': l_arm_ts
            }
            obs['arms']['right_arm'] = {
                'joint_positions': r_j_p, 'end_effector_poses': r_ee_p,
                'local_timestamp': r_arm_ts
            }
        else: # 提供默认值以保证结构完整
            default_arm_data = {
                'joint_positions': [0]*6, 'end_effector_poses': [0]*6,
                'local_timestamp': global_obs_timestamp
            }
            obs['arms']['left_arm'] = default_arm_data
            obs['arms']['right_arm'] = default_arm_data

        # Motor Data
        if self.motor_collector:
            l_motor_ts, l_m_p, l_m_s = self.motor_collector.left_motors.get_latest_data()
            r_motor_ts, r_m_p, r_m_s = self.motor_collector.right_motors.get_latest_data()
            obs['motors']['left_motors'] = {'positions': l_m_p, 'states': l_m_s, 'local_timestamp': l_motor_ts}
            obs['motors']['right_motors'] = {'positions': r_m_p, 'states': r_m_s, 'local_timestamp': r_motor_ts}
        else: # 提供默认值
            default_motor_data = {'positions': [0]*4, 'states': [0]*4, 'local_timestamp': global_obs_timestamp}
            obs['motors']['left_motors'] = default_motor_data
            obs['motors']['right_motors'] = default_motor_data
        
        return obs

    def stop_episode(self) -> Optional[str]:
        with self.state_lock:
            if self.state != CollectionState.RECORDING:
                print(f"❌ 无法停止录制，当前状态: {self.state.value}")
                return None
            self.state = CollectionState.STOPPING
        
        try:
            print("🛑 停止Episode录制...")
            
            # 1. 停止采集线程
            self.collect_stop_flag.set()
            if self.collection_thread and self.collection_thread.is_alive():
                self.collection_thread.join(timeout=2.0)
            
            # 2. 停止硬件数据流
            self._stop_all_hardware_streams()
            
            # 3. 停止存储
            completed_path = self.storage.stop_episode()
            
            with self.state_lock:
                self.state = CollectionState.IDLE
            
            if completed_path:
                print(f"✅ Episode 录制完成: {completed_path}")
            else:
                print("❌ Episode 停止时出现问题")
            
            return completed_path
            
        except Exception as e:
            with self.state_lock:
                self.state = CollectionState.ERROR
            error_msg = f"停止Episode时发生异常: {e}"
            print(f"❌ {error_msg}")
            self.storage.log_error(error_msg)
            self.total_errors += 1
            return None
    
    def delete_last_episode(self) -> bool:
        if self.state != CollectionState.IDLE:
            print(f"❌ 无法删除Episode，当前状态: {self.state.value}")
            return False
        
        success = self.storage.delete_last_episode()
        if success:
            print("🗑️  最近的Episode已删除")
        return success
    
    def _on_hardware_error(self, component_id: str, error_message: str):
        error_msg = f"硬件错误 {component_id}: {error_message}"
        print(f"❌ {error_msg}")
        self.storage.log_error(error_msg)
        self.total_errors += 1
        
        if self.state == CollectionState.RECORDING:
            print("⚠️  录制中发生硬件错误，考虑手动停止录制...")

    def _emergency_stop_all(self):
        """紧急停止所有活动"""
        print("🚨 紧急停止所有采集活动...")
        self.collect_stop_flag.set()
        self._stop_all_hardware_streams()
        if self.storage.is_recording:
            self.storage.stop_episode()
        with self.state_lock:
            self.state = CollectionState.IDLE
    
    def get_status(self) -> Dict[str, Any]:
        status = {
            'state': self.state.value,
            'episode_count': self.episode_count,
            'total_errors': self.total_errors,
            'last_episode_path': self.last_episode_path,
            'is_time_synced': self.time_sync.is_synced(),
            'storage_status': self.storage.get_status()
        }
        return status
    
    def print_status(self):
        status = self.get_status()
        print("\n" + "=" * 60)
        print("📊 数据采集系统状态")
        print("=" * 60)
        print(f"  - 系统状态: {status['state']}")
        print(f"  - Episode 计数: {status['episode_count']}")
        print(f"  - 总错误数: {status['total_errors']}")
        print(f"  - 时间同步: {'✅' if status['is_time_synced'] else '❌'}")
        
        storage_stat = status['storage_status']
        print(f"  - 存储状态: {'录制中' if storage_stat['is_recording'] else '空闲'}")
        if storage_stat['is_recording']:
            print(f"    - 文件: {os.path.basename(storage_stat['current_file'])}")
            print(f"    - 已采集Observations: {storage_stat['observation_count']}")
        
        print("=" * 60)
    
    def is_recording(self) -> bool:
        return self.state == CollectionState.RECORDING
    
    def is_idle(self) -> bool:
        return self.state == CollectionState.IDLE
    
    def cleanup(self):
        print("🧹 清理数据采集系统...")
        if self.is_recording():
            self.stop_episode()
        
        self._stop_all_hardware_streams()
        self.storage.cleanup()
        self.time_sync.stop_sync()
        
        with self.state_lock:
            self.state = CollectionState.IDLE
        
        print("✅ 数据采集系统清理完成")


if __name__ == "__main__":
    collector = DataCollector("test_episodes")
    collector.initialize_hardware()
    
    if collector.start_episode():
        print("模拟录制5秒...")
        time.sleep(5)
        collector.stop_episode()
    
    collector.print_status()
    collector.cleanup()