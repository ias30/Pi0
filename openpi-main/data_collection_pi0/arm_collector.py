#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机械臂数据采集模块
负责采集双臂机械臂的关节角度和末端姿态数据
"""

import threading
import time
from typing import Optional, Callable, Dict, Any, List, Tuple
from time_sync import get_timestamp


class ArmCollector:
    """单个机械臂数据采集器"""
    
    def __init__(self, arm_id: str, robot_arm, arm_handle, target_hz: int = 60):
        self.arm_id = arm_id
        self.robot_arm = robot_arm
        self.arm_handle = arm_handle
        self.target_hz = target_hz
        self.target_interval = 1.0 / target_hz
        
        self._stop_flag = False
        self._capture_thread: Optional[threading.Thread] = None
        self._is_connected = False
        
        self.data_callback: Optional[Callable] = None
        self.error_callback: Optional[Callable] = None
        
        self.sample_count = 0
        self.error_count = 0
        self.last_sample_time = 0.0
        
        # 数据缓存 (已修正)
        self.data_lock = threading.Lock()
        self.last_timestamp: float = 0.0
        self.last_joint_positions: List[float] = [0.0] * 6
        self.last_end_effector_pose: List[float] = [0.0] * 6
        
        print(f"机械臂采集器初始化: {arm_id} @ {target_hz}Hz")
    
    def set_data_callback(self, callback: Callable[[str, List[float], List[float], float], None]):
        self.data_callback = callback
    
    def set_error_callback(self, callback: Callable[[str, str], None]):
        self.error_callback = callback
    
    def get_latest_data(self) -> Tuple[float, List[float], List[float]]:
        """获取最新的数据 (已修正)"""
        with self.data_lock:
            return self.last_timestamp, self.last_joint_positions, self.last_end_effector_pose

    def check_connection(self) -> bool:
        try:
            result, _ = self.robot_arm.rm_get_current_arm_state()
            self._is_connected = (result == 0)
            return self._is_connected
        except Exception as e:
            self._is_connected = False
            error_msg = f"检查连接状态失败: {e}"
            print(f"{self.arm_id}: {error_msg}")
            if self.error_callback:
                self.error_callback(self.arm_id, error_msg)
            return False
    
    def start_collection(self) -> bool:
        if not self.check_connection():
            print(f"{self.arm_id}: 机械臂未连接，无法开始采集")
            return False
        
        if self._capture_thread and self._capture_thread.is_alive():
            print(f"{self.arm_id}: 采集线程已在运行")
            return True
        
        self._stop_flag = False
        self._capture_thread = threading.Thread(target=self._collection_worker, daemon=True)
        self._capture_thread.start()
        
        print(f"{self.arm_id}: 开始数据采集")
        return True
    
    def stop_collection(self):
        self._stop_flag = True
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)
        print(f"{self.arm_id}: 数据采集已停止")
    
    def _collection_worker(self):
        print(f"{self.arm_id}: 采集线程已启动")
        last_collection_time = time.time()
        
        while not self._stop_flag:
            try:
                loop_start_time = time.time()
                elapsed = loop_start_time - last_collection_time
                if elapsed < self.target_interval:
                    time.sleep(self.target_interval - elapsed)
                
                timestamp = get_timestamp()
                data = self._get_arm_data()
                
                if data:
                    joint_positions, end_effector_pose = data
                    with self.data_lock:
                        self.last_timestamp = timestamp
                        self.last_joint_positions = joint_positions
                        self.last_end_effector_pose = end_effector_pose

                    if self.data_callback:
                        self.data_callback(self.arm_id, joint_positions, end_effector_pose, timestamp)
                    
                    self.sample_count += 1
                    self.last_sample_time = timestamp
                else:
                    self.error_count += 1
                    if self.error_callback:
                        self.error_callback(self.arm_id, "获取机械臂数据失败")
                
                last_collection_time = time.time()
                
            except Exception as e:
                self.error_count += 1
                error_msg = f"采集线程异常: {e}"
                print(f"{self.arm_id}: {error_msg}")
                if self.error_callback:
                    self.error_callback(self.arm_id, error_msg)
                time.sleep(0.1)
        
        print(f"{self.arm_id}: 采集线程已退出")
    
    def _get_arm_data(self):
        """获取机械臂当前数据 (已修正)"""
        try:
            # 获取关节状态
            joint_result, joint_data = self.robot_arm.rm_get_current_arm_state()
            if joint_result != 0:
                return None
            
            # 提取关节角度
            joint_positions = list(joint_data.get('joint', [0.0] * 6))[:6]
            
            # 提取末端姿态
            pose_data = joint_data.get('pose', [0.0] * 6)
            end_effector_pose = list(pose_data)[:6]
            
            return joint_positions, end_effector_pose
            
        except Exception as e:
            print(f"{self.arm_id}: 获取机械臂数据时出错: {e}")
            return None
    
    def is_connected(self) -> bool:
        return self._is_connected
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            'arm_id': self.arm_id,
            'is_connected': self._is_connected,
            'sample_count': self.sample_count,
            'error_count': self.error_count,
            'target_hz': self.target_hz,
            'last_sample_time': self.last_sample_time
        }
    
    def cleanup(self):
        self.stop_collection()


class DualArmCollector:
    """双臂机械臂数据采集管理器"""
    
    def __init__(self, left_robot_arm, left_arm_handle, 
                 right_robot_arm, right_arm_handle, target_hz: int = 30):
        self.left_arm = ArmCollector("left_arm", left_robot_arm, left_arm_handle, target_hz)
        self.right_arm = ArmCollector("right_arm", right_robot_arm, right_arm_handle, target_hz)
        
        self.data_callback: Optional[Callable] = None
        self.error_callback: Optional[Callable] = None
        
        self.left_arm.set_data_callback(self._on_arm_data)
        self.left_arm.set_error_callback(self._on_arm_error)
        self.right_arm.set_data_callback(self._on_arm_data)
        self.right_arm.set_error_callback(self._on_arm_error)
    
    def set_data_callback(self, callback: Callable[[str, List[float], List[float], float], None]):
        self.data_callback = callback
    
    def set_error_callback(self, callback: Callable[[str, str], None]):
        self.error_callback = callback
    
    def _on_arm_data(self, arm_id: str, joint_positions: List[float], 
                    end_effector_pose: List[float], timestamp: float):
        if self.data_callback:
            self.data_callback(arm_id, joint_positions, end_effector_pose, timestamp)
    
    def _on_arm_error(self, arm_id: str, error_message: str):
        if self.error_callback:
            self.error_callback(arm_id, error_message)
    
    def check_all_connections(self) -> bool:
        left_ok = self.left_arm.check_connection()
        right_ok = self.right_arm.check_connection()
        success = left_ok and right_ok
        print(f"机械臂连接状态: left_arm={left_ok}, right_arm={right_ok}")
        return success
    
    def start_all_collection(self) -> bool:
        left_ok = self.left_arm.start_collection()
        right_ok = self.right_arm.start_collection()
        success = left_ok and right_ok
        print(f"机械臂采集状态: left_arm={left_ok}, right_arm={right_ok}")
        return success
    
    def stop_all_collection(self):
        self.left_arm.stop_collection()
        self.right_arm.stop_collection()
        print("所有机械臂数据采集已停止")
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            'left_arm': self.left_arm.get_stats(),
            'right_arm': self.right_arm.get_stats()
        }
    
    def cleanup(self):
        self.stop_all_collection()

class ArmInterfaceAdapter:
    @staticmethod
    def create_arm_collector(arm_id: str, robot_arm, arm_handle, target_hz: int = 60):
        return ArmCollector(arm_id, robot_arm, arm_handle, target_hz)
    
    @staticmethod
    def create_dual_arm_collector(left_robot_arm, left_arm_handle,
                                right_robot_arm, right_arm_handle, 
                                target_hz: int = 60):
        return DualArmCollector(left_robot_arm, left_arm_handle,
                              right_robot_arm, right_arm_handle, target_hz)