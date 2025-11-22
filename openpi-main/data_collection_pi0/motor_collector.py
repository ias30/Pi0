#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
电机数据采集模块
负责采集两组电机的位置和状态数据
"""

import threading
import time
from typing import Optional, Callable, Dict, Any, List, Tuple
from time_sync import get_timestamp


class MotorCollector:
    """单组电机数据采集器"""
    
    def __init__(self, motor_group_id: str, position_controller, target_hz: int = 30):
        self.motor_group_id = motor_group_id
        self.position_controller = position_controller
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
        
        # 数据缓存
        self.data_lock = threading.Lock()
        self.last_timestamp: float = 0.0
        self.last_motor_positions: List[int] = [0] * 4
        self.last_motor_states: List[int] = [0] * 4
        
        print(f"电机采集器初始化: {motor_group_id} @ {target_hz}Hz")
    
    def set_data_callback(self, callback: Callable[[str, List[int], List[int], float], None]):
        self.data_callback = callback
    
    def set_error_callback(self, callback: Callable[[str, str], None]):
        self.error_callback = callback

    def get_latest_data(self) -> Tuple[float, List[int], List[int]]:
        """获取最新的数据"""
        with self.data_lock:
            return self.last_timestamp, self.last_motor_positions, self.last_motor_states

    def check_connection(self) -> bool:
        try:
            if hasattr(self.position_controller, 'data_ser'):
                self._is_connected = (self.position_controller.data_ser is not None and 
                                    self.position_controller.data_ser.is_open)
            else:
                self._is_connected = False
            return self._is_connected
        except Exception as e:
            self._is_connected = False
            error_msg = f"检查连接状态失败: {e}"
            print(f"{self.motor_group_id}: {error_msg}")
            if self.error_callback:
                self.error_callback(self.motor_group_id, error_msg)
            return False
    
    def start_collection(self) -> bool:
        if not self.check_connection():
            print(f"{self.motor_group_id}: 电机未连接，无法开始采集")
            return False
        
        if self._capture_thread and self._capture_thread.is_alive():
            print(f"{self.motor_group_id}: 采集线程已在运行")
            return True
        
        self._stop_flag = False
        self._capture_thread = threading.Thread(target=self._collection_worker, daemon=True)
        self._capture_thread.start()
        
        print(f"{self.motor_group_id}: 开始数据采集")
        return True
    
    def stop_collection(self):
        self._stop_flag = True
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)
        print(f"{self.motor_group_id}: 数据采集已停止")
    
    def _collection_worker(self):
        print(f"{self.motor_group_id}: 采集线程已启动")
        last_collection_time = time.time()
        
        while not self._stop_flag:
            try:
                loop_start_time = time.time()
                elapsed = loop_start_time - last_collection_time
                if elapsed < self.target_interval:
                    time.sleep(self.target_interval - elapsed)
                
                timestamp = get_timestamp()
                motor_positions, motor_states = self._get_motor_data()
                
                if motor_positions is not None and motor_states is not None:
                    with self.data_lock:
                        self.last_timestamp = timestamp
                        self.last_motor_positions = motor_positions
                        self.last_motor_states = motor_states

                    if self.data_callback:
                        self.data_callback(self.motor_group_id, motor_positions, motor_states, timestamp)
                    
                    self.sample_count += 1
                    self.last_sample_time = timestamp
                else:
                    self.error_count += 1
                    if self.error_callback:
                        self.error_callback(self.motor_group_id, "获取电机数据失败")
                
                last_collection_time = time.time()
                
            except Exception as e:
                self.error_count += 1
                error_msg = f"采集线程异常: {e}"
                print(f"{self.motor_group_id}: {error_msg}")
                if self.error_callback:
                    self.error_callback(self.motor_group_id, error_msg)
                time.sleep(0.1)
        
        print(f"{self.motor_group_id}: 采集线程已退出")
    
    def _get_motor_data(self):
        try:
            # 获取所有电机的实际位置
            motor_positions = self.position_controller.read_position()
            
            # 只取电机0和1的位置数据
            positions = [motor_positions[0], motor_positions[1], 0, 0]
            
            # 电机2和3的状态从pos_ctrl对象中获取
            # motor_23_current_state为0表示在零点位置，为1表示在极限位置
            current_state = self.position_controller.motor_23_current_state
            states = [0, 0, current_state, current_state]
            
            return positions, states
            
        except Exception as e:
            print(f"{self.motor_group_id}: 获取电机数据时出错: {e}")
            return None, None
    
    def is_connected(self) -> bool:
        return self._is_connected
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            'motor_group_id': self.motor_group_id,
            'is_connected': self._is_connected,
            'sample_count': self.sample_count,
            'error_count': self.error_count,
            'target_hz': self.target_hz,
            'last_sample_time': self.last_sample_time
        }
    
    def cleanup(self):
        self.stop_collection()


class DualMotorCollector:
    """双组电机数据采集管理器"""
    
    def __init__(self, left_position_controller, right_position_controller, target_hz: int = 30):
        self.left_motors = MotorCollector("left_motors", left_position_controller, target_hz)
        self.right_motors = MotorCollector("right_motors", right_position_controller, target_hz)
        
        self.data_callback: Optional[Callable] = None
        self.error_callback: Optional[Callable] = None
        
        self.left_motors.set_data_callback(self._on_motor_data)
        self.left_motors.set_error_callback(self._on_motor_error)
        self.right_motors.set_data_callback(self._on_motor_data)
        self.right_motors.set_error_callback(self._on_motor_error)
    
    def set_data_callback(self, callback: Callable[[str, List[int], List[int], float], None]):
        self.data_callback = callback
    
    def set_error_callback(self, callback: Callable[[str, str], None]):
        self.error_callback = callback
    
    def _on_motor_data(self, motor_group_id: str, motor_positions: List[int], 
                      motor_states: List[int], timestamp: float):
        if self.data_callback:
            self.data_callback(motor_group_id, motor_positions, motor_states, timestamp)
    
    def _on_motor_error(self, motor_group_id: str, error_message: str):
        if self.error_callback:
            self.error_callback(motor_group_id, error_message)
    
    def check_all_connections(self) -> bool:
        left_ok = self.left_motors.check_connection()
        right_ok = self.right_motors.check_connection()
        success = left_ok and right_ok
        print(f"电机连接状态: left_motors={left_ok}, right_motors={right_ok}")
        return success
    
    def start_all_collection(self) -> bool:
        left_ok = self.left_motors.start_collection()
        right_ok = self.right_motors.start_collection()
        success = left_ok and right_ok
        print(f"电机采集状态: left_motors={left_ok}, right_motors={right_ok}")
        return success
    
    def stop_all_collection(self):
        self.left_motors.stop_collection()
        self.right_motors.stop_collection()
        print("所有电机数据采集已停止")
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            'left_motors': self.left_motors.get_stats(),
            'right_motors': self.right_motors.get_stats()
        }
    
    def cleanup(self):
        self.stop_all_collection()

# 电机接口适配器 (保持不变)
class MotorInterfaceAdapter:
    @staticmethod
    def create_motor_collector(motor_group_id: str, position_controller, target_hz: int = 30):
        return MotorCollector(motor_group_id, position_controller, target_hz)
    
    @staticmethod
    def create_dual_motor_collector(left_position_controller, right_position_controller, 
                                  target_hz: int = 30):
        return DualMotorCollector(left_position_controller, right_position_controller, target_hz)