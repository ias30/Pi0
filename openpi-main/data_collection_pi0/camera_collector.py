#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
相机数据采集模块
负责一个Realsense相机(camera_high)和两个手腕相机(wrist cameras)的图像采集
"""

import cv2
import numpy as np
import threading
import time
import subprocess
from typing import Optional, Callable, Dict, Any, Tuple
from time_sync import get_timestamp


class CameraCollector:
    """Realsense相机数据采集器"""
    
    def __init__(self, camera_id: str, device_index: int = 0, 
                 resolution: tuple = (640, 480), fps: int = 30, serial_number: str = None):
        """
        初始化相机采集器
        
        Args:
            camera_id: 相机标识符 (如 "camera_high")
            device_index: 设备索引 (当不使用serial_number时)
            resolution: 分辨率 (width, height)
            fps: 帧率
            serial_number: RealSense设备序列号 (优先使用)
        """
        self.camera_id = camera_id
        self.device_index = device_index
        self.resolution = resolution
        self.fps = fps
        self.serial_number = serial_number
        self.target_interval = 1.0 / fps
        
        self._stop_flag = False
        self._capture_thread: Optional[threading.Thread] = None
        self._is_connected = False
        self._use_realsense = False
        
        self.data_callback: Optional[Callable] = None
        self.error_callback: Optional[Callable] = None
        
        self._color_cap: Optional[cv2.VideoCapture] = None
        self._depth_cap: Optional[cv2.VideoCapture] = None
        
        self.frame_count = 0
        self.error_count = 0
        self.last_frame_time = 0.0

        # 数据缓存
        self.last_color_img: Optional[np.ndarray] = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)
        self.last_depth_img: Optional[np.ndarray] = np.zeros((resolution[1], resolution[0]), dtype=np.uint16)
        self.last_timestamp: float = 0.0
        self.data_lock = threading.Lock()
        
        if serial_number:
            print(f"相机采集器初始化: {camera_id} (RealSense序列号: {serial_number})")
        else:
            print(f"相机采集器初始化: {camera_id} (设备{device_index})")
    
    def set_data_callback(self, callback: Callable[[str, np.ndarray, np.ndarray, float], None]):
        self.data_callback = callback
    
    def set_error_callback(self, callback: Callable[[str, str], None]):
        self.error_callback = callback
    
    def get_latest_data(self) -> Tuple[float, np.ndarray, np.ndarray]:
        """获取最新的数据"""
        with self.data_lock:
            return self.last_timestamp, self.last_color_img, self.last_depth_img

    def connect(self) -> bool:
        try:
            import pyrealsense2 as rs
            return self._connect_realsense()
        except ImportError:
            print(f"{self.camera_id}: Realsense SDK未安装，使用OpenCV连接")
            return self._connect_opencv()
        except Exception as e:
            print(f"{self.camera_id}: Realsense连接失败 ({e})，尝试OpenCV")
            return self._connect_opencv()
    
    def _connect_realsense(self) -> bool:
        try:
            import pyrealsense2 as rs
            
            self.pipeline = rs.pipeline()
            config = rs.config()
            
            if self.serial_number:
                config.enable_device(self.serial_number)
                print(f"{self.camera_id}: 连接RealSense设备 {self.serial_number}")
            
            config.enable_stream(rs.stream.color, self.resolution[0], self.resolution[1], rs.format.bgr8, self.fps)
            config.enable_stream(rs.stream.depth, self.resolution[0], self.resolution[1], rs.format.z16, self.fps)
            
            self.pipeline.start(config)
            self._is_connected = True
            self._use_realsense = True
            
            print(f"{self.camera_id}: Realsense连接成功")
            return True
        except Exception as e:
            print(f"{self.camera_id}: Realsense连接失败: {e}")
            return False
    
    def _connect_opencv(self) -> bool:
        try:
            self._color_cap = cv2.VideoCapture(self.device_index)
            if not self._color_cap.isOpened():
                print(f"{self.camera_id}: 无法打开设备 {self.device_index}")
                return False
            
            self._color_cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self._color_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self._color_cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            actual_width = int(self._color_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self._color_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self._color_cap.get(cv2.CAP_PROP_FPS)
            
            print(f"{self.camera_id}: OpenCV连接成功 {actual_width}x{actual_height}@{actual_fps}fps")
            
            self._is_connected = True
            self._use_realsense = False
            return True
        except Exception as e:
            print(f"{self.camera_id}: OpenCV连接失败: {e}")
            return False
    
    def start_capture(self) -> bool:
        if not self._is_connected:
            if not self.connect():
                return False
        
        if self._capture_thread and self._capture_thread.is_alive():
            print(f"{self.camera_id}: 采集线程已在运行")
            return True
        
        self._stop_flag = False
        self._capture_thread = threading.Thread(target=self._capture_worker, daemon=True)
        self._capture_thread.start()
        
        print(f"{self.camera_id}: 开始图像采集")
        return True
    
    def stop_capture(self):
        self._stop_flag = True
        
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)
            if self._capture_thread.is_alive():
                print(f"{self.camera_id}: 警告 - 采集线程未能及时停止")
        
        print(f"{self.camera_id}: 图像采集已停止")
    
    def _capture_worker(self):
        print(f"{self.camera_id}: 采集线程已启动")
        
        last_capture_time = time.time()
        
        while not self._stop_flag:
            try:
                loop_start_time = time.time()
                
                elapsed = loop_start_time - last_capture_time
                if elapsed < self.target_interval:
                    time.sleep(self.target_interval - elapsed)
                
                timestamp = get_timestamp()
                color_img, depth_img = self._get_frames()
                
                if color_img is not None and depth_img is not None:
                    with self.data_lock:
                        self.last_timestamp = timestamp
                        self.last_color_img = color_img
                        self.last_depth_img = depth_img

                    if self.data_callback:
                        self.data_callback(self.camera_id, color_img, depth_img, timestamp)
                    
                    self.frame_count += 1
                    self.last_frame_time = timestamp
                else:
                    self.error_count += 1
                    if self.error_callback:
                        self.error_callback(self.camera_id, "获取图像失败")
                
                last_capture_time = time.time()
                
            except Exception as e:
                self.error_count += 1
                error_msg = f"采集线程异常: {e}"
                print(f"{self.camera_id}: {error_msg}")
                if self.error_callback:
                    self.error_callback(self.camera_id, error_msg)
                time.sleep(0.1)
        
        print(f"{self.camera_id}: 采集线程已退出")
    
    def _get_frames(self):
        try:
            if self._use_realsense:
                return self._get_realsense_frames()
            else:
                return self._get_opencv_frames()
        except Exception as e:
            print(f"{self.camera_id}: 获取帧时出错: {e}")
            return None, None
    
    def _get_realsense_frames(self):
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                return None, None
            
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            return color_image, depth_image
            
        except Exception as e:
            print(f"{self.camera_id}: Realsense获取帧失败: {e}")
            return None, None
    
    def _get_opencv_frames(self):
        try:
            ret, color_image = self._color_cap.read()
            if not ret:
                return None, None
            
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            depth_image = ((255 - gray) * 4).astype(np.uint16)
            
            return color_image, depth_image
        except Exception as e:
            print(f"{self.camera_id}: OpenCV获取帧失败: {e}")
            return None, None
    
    def disconnect(self):
        self.stop_capture()
        
        try:
            if hasattr(self, '_use_realsense') and self._use_realsense and hasattr(self, 'pipeline'):
                self.pipeline.stop()
            
            if self._color_cap: self._color_cap.release()
            if self._depth_cap: self._depth_cap.release()
            
            self._is_connected = False
            print(f"{self.camera_id}: 相机已断开连接")
        except Exception as e:
            print(f"{self.camera_id}: 断开连接时出错: {e}")
    
    def is_connected(self) -> bool:
        return self._is_connected
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            'camera_id': self.camera_id,
            'is_connected': self._is_connected,
            'frame_count': self.frame_count,
            'error_count': self.error_count,
            'target_fps': self.fps,
            'last_frame_time': self.last_frame_time
        }
    
    def cleanup(self):
        self.disconnect()


class FFmpegWristCamera:
    """基于FFmpeg的手腕相机采集器（仅RGB，30FPS）"""
    
    def __init__(self, camera_id: str, device_index: int = 0, 
                 resolution: tuple = (640, 480), fps: int = 30):
        """
        初始化FFmpeg手腕相机采集器
        
        Args:
            camera_id: 相机标识符 (如 "camera_left_wrist", "camera_right_wrist")
            device_index: 设备索引 /dev/video{device_index}
            resolution: 分辨率 (width, height)
            fps: 帧率
        """
        self.camera_id = camera_id
        self.device_index = device_index
        self.resolution = resolution
        self.fps = fps
        self.target_interval = 1.0 / fps
        
        self._stop_flag = False
        self._capture_thread: Optional[threading.Thread] = None
        self._is_connected = False
        self._ffmpeg_process: Optional[subprocess.Popen] = None
        
        self.data_callback: Optional[Callable] = None
        self.error_callback: Optional[Callable] = None
        
        self.frame_count = 0
        self.error_count = 0
        self.last_frame_time = 0.0
        
        # 数据缓存
        self.last_color_img: Optional[np.ndarray] = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)
        self.last_timestamp: float = 0.0
        self.data_lock = threading.Lock()
        
        print(f"手腕相机采集器初始化: {camera_id} (设备/dev/video{device_index})")
    
    def set_data_callback(self, callback: Callable[[str, np.ndarray, float], None]):
        self.data_callback = callback
    
    def set_error_callback(self, callback: Callable[[str, str], None]):
        self.error_callback = callback
    
    def get_latest_data(self) -> Tuple[float, np.ndarray]:
        """获取最新的数据（仅RGB）"""
        with self.data_lock:
            return self.last_timestamp, self.last_color_img
    
    def connect(self) -> bool:
        """启动FFmpeg进程"""
        try:
            command = [
                'ffmpeg',
                '-f', 'v4l2',
                '-input_format', 'mjpeg',
                '-framerate', str(self.fps),
                '-video_size', f'{self.resolution[0]}x{self.resolution[1]}',
                '-i', f'/dev/video{self.device_index}',
                '-f', 'rawvideo',
                '-pix_fmt', 'bgr24',
                'pipe:1'
            ]
            
            self._ffmpeg_process = subprocess.Popen(
                command, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.DEVNULL
            )
            
            self._is_connected = True
            print(f"{self.camera_id}: FFmpeg连接成功")
            return True
            
        except Exception as e:
            print(f"{self.camera_id}: FFmpeg连接失败: {e}")
            self._is_connected = False
            return False
    
    def start_capture(self) -> bool:
        if not self._is_connected:
            if not self.connect():
                return False
        
        if self._capture_thread and self._capture_thread.is_alive():
            print(f"{self.camera_id}: 采集线程已在运行")
            return True
        
        self._stop_flag = False
        self._capture_thread = threading.Thread(target=self._capture_worker, daemon=True)
        self._capture_thread.start()
        
        print(f"{self.camera_id}: 开始图像采集")
        return True
    
    def stop_capture(self):
        self._stop_flag = True
        
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)
            if self._capture_thread.is_alive():
                print(f"{self.camera_id}: 警告 - 采集线程未能及时停止")
        
        print(f"{self.camera_id}: 图像采集已停止")
    
    def _capture_worker(self):
        print(f"{self.camera_id}: 采集线程已启动")
        
        frame_size = self.resolution[0] * self.resolution[1] * 3
        last_capture_time = time.time()
        
        while not self._stop_flag:
            try:
                loop_start_time = time.time()
                
                elapsed = loop_start_time - last_capture_time
                if elapsed < self.target_interval:
                    time.sleep(self.target_interval - elapsed)
                
                # 从FFmpeg进程读取一帧
                raw_frame = self._ffmpeg_process.stdout.read(frame_size)
                
                if len(raw_frame) != frame_size:
                    self.error_count += 1
                    if self.error_callback:
                        self.error_callback(self.camera_id, "未读取到完整帧")
                    continue
                
                timestamp = get_timestamp()
                color_img = np.frombuffer(raw_frame, np.uint8).reshape((self.resolution[1], self.resolution[0], 3))
                
                with self.data_lock:
                    self.last_timestamp = timestamp
                    self.last_color_img = color_img.copy()
                
                if self.data_callback:
                    self.data_callback(self.camera_id, color_img, timestamp)
                
                self.frame_count += 1
                self.last_frame_time = timestamp
                
                last_capture_time = time.time()
                
            except Exception as e:
                self.error_count += 1
                error_msg = f"采集线程异常: {e}"
                print(f"{self.camera_id}: {error_msg}")
                if self.error_callback:
                    self.error_callback(self.camera_id, error_msg)
                time.sleep(0.1)
        
        print(f"{self.camera_id}: 采集线程已退出")
    
    def disconnect(self):
        self.stop_capture()
        
        try:
            if self._ffmpeg_process:
                self._ffmpeg_process.kill()
                self._ffmpeg_process.wait(timeout=2.0)
                self._ffmpeg_process = None
            
            self._is_connected = False
            print(f"{self.camera_id}: 相机已断开连接")
        except Exception as e:
            print(f"{self.camera_id}: 断开连接时出错: {e}")
    
    def is_connected(self) -> bool:
        return self._is_connected
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            'camera_id': self.camera_id,
            'is_connected': self._is_connected,
            'frame_count': self.frame_count,
            'error_count': self.error_count,
            'target_fps': self.fps,
            'last_frame_time': self.last_frame_time
        }
    
    def cleanup(self):
        self.disconnect()


class TripleCameraCollector:
    """三相机采集管理器：1个RealSense(camera_high) + 2个手腕相机(wrist cameras)"""
    
    def __init__(self, resolution: tuple = (640, 480), fps: int = 30):
        # RealSense相机 (保留原camera0作为camera_high)
        self.camera_high = CameraCollector("camera_high", device_index=0, 
                                           resolution=resolution, fps=fps,
                                           serial_number="031522071209")
        
        # 两个手腕相机（使用FFmpeg）
        self.camera_left_wrist = FFmpegWristCamera("camera_left_wrist", device_index=6,
                                                     resolution=resolution, fps=fps)
        self.camera_right_wrist = FFmpegWristCamera("camera_right_wrist", device_index=8,
                                                      resolution=resolution, fps=fps)
        
        self.data_callback: Optional[Callable] = None
        self.error_callback: Optional[Callable] = None
        
        # 设置回调
        self.camera_high.set_data_callback(self._on_realsense_data)
        self.camera_high.set_error_callback(self._on_camera_error)
        self.camera_left_wrist.set_data_callback(self._on_wrist_data)
        self.camera_left_wrist.set_error_callback(self._on_camera_error)
        self.camera_right_wrist.set_data_callback(self._on_wrist_data)
        self.camera_right_wrist.set_error_callback(self._on_camera_error)
    
    def set_data_callback(self, callback: Callable):
        self.data_callback = callback
    
    def set_error_callback(self, callback: Callable[[str, str], None]):
        self.error_callback = callback
    
    def _on_realsense_data(self, camera_id: str, color_img: np.ndarray, 
                          depth_img: np.ndarray, timestamp: float):
        if self.data_callback:
            self.data_callback(camera_id, color_img, depth_img, timestamp)
    
    def _on_wrist_data(self, camera_id: str, color_img: np.ndarray, timestamp: float):
        if self.data_callback:
            # 手腕相机没有depth，传递None
            self.data_callback(camera_id, color_img, None, timestamp)
    
    def _on_camera_error(self, camera_id: str, error_message: str):
        if self.error_callback:
            self.error_callback(camera_id, error_message)
    
    def connect_all(self) -> bool:
        camera_high_ok = self.camera_high.connect()
        camera_left_ok = self.camera_left_wrist.connect()
        camera_right_ok = self.camera_right_wrist.connect()
        success = camera_high_ok and camera_left_ok and camera_right_ok
        print(f"相机连接状态: camera_high={camera_high_ok}, camera_left_wrist={camera_left_ok}, camera_right_wrist={camera_right_ok}")
        return success
    
    def start_all(self) -> bool:
        camera_high_ok = self.camera_high.start_capture()
        camera_left_ok = self.camera_left_wrist.start_capture()
        camera_right_ok = self.camera_right_wrist.start_capture()
        success = camera_high_ok and camera_left_ok and camera_right_ok
        print(f"相机采集状态: camera_high={camera_high_ok}, camera_left_wrist={camera_left_ok}, camera_right_wrist={camera_right_ok}")
        return success
    
    def stop_all(self):
        self.camera_high.stop_capture()
        self.camera_left_wrist.stop_capture()
        self.camera_right_wrist.stop_capture()
        print("所有相机采集已停止")
    
    def disconnect_all(self):
        self.camera_high.disconnect()
        self.camera_left_wrist.disconnect()
        self.camera_right_wrist.disconnect()
        print("所有相机已断开连接")
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            'camera_high': self.camera_high.get_stats(),
            'camera_left_wrist': self.camera_left_wrist.get_stats(),
            'camera_right_wrist': self.camera_right_wrist.get_stats()
        }
    
    def cleanup(self):
        self.disconnect_all()


# 兼容性别名
DualCameraCollector = TripleCameraCollector


if __name__ == "__main__":
    """测试三个相机的数据采集"""
    print("=" * 60)
    print("开始测试三相机数据采集系统")
    print("=" * 60)
    
    # 创建采集器
    collector = TripleCameraCollector(resolution=(640, 480), fps=30)
    
    # 统计数据
    stats = {
        'camera_high': {'color_count': 0, 'depth_count': 0},
        'camera_left_wrist': {'color_count': 0},
        'camera_right_wrist': {'color_count': 0}
    }
    
    def data_callback(camera_id, color_img, depth_img, timestamp):
        """数据回调函数"""
        if camera_id == 'camera_high':
            stats['camera_high']['color_count'] += 1
            stats['camera_high']['depth_count'] += 1
            print(f"[{camera_id}] 收到数据 - Color: {color_img.shape}, Depth: {depth_img.shape}, TS: {timestamp:.3f}")
        else:
            stats[camera_id]['color_count'] += 1
            print(f"[{camera_id}] 收到数据 - Color: {color_img.shape}, TS: {timestamp:.3f}")
    
    def error_callback(camera_id, error_msg):
        """错误回调函数"""
        print(f"[错误 {camera_id}] {error_msg}")
    
    # 设置回调
    collector.set_data_callback(data_callback)
    collector.set_error_callback(error_callback)
    
    try:
        # 连接所有相机
        print("\n正在连接所有相机...")
        if not collector.connect_all():
            print("警告: 部分相机连接失败，但继续测试")
        
        # 开始采集
        print("\n开始采集数据...")
        if not collector.start_all():
            print("警告: 部分相机启动失败，但继续测试")
        
        # 运行10秒
        print("\n采集中... (持续10秒)")
        test_duration = 10
        start_time = time.time()
        
        while time.time() - start_time < test_duration:
            elapsed = time.time() - start_time
            remaining = test_duration - elapsed
            print(f"\r剩余时间: {remaining:.1f}秒", end='', flush=True)
            time.sleep(1)
        
        print("\n\n停止采集...")
        collector.stop_all()
        
        # 打印统计信息
        print("\n" + "=" * 60)
        print("采集统计结果")
        print("=" * 60)
        
        all_stats = collector.get_stats()
        for cam_name, cam_stat in all_stats.items():
            print(f"\n{cam_name}:")
            print(f"  - 帧数: {cam_stat['frame_count']}")
            print(f"  - 错误数: {cam_stat['error_count']}")
            print(f"  - 目标FPS: {cam_stat['target_fps']}")
            actual_fps = cam_stat['frame_count'] / test_duration
            print(f"  - 实际FPS: {actual_fps:.2f}")
        
        print("\n回调统计:")
        print(f"  camera_high - Color帧: {stats['camera_high']['color_count']}, Depth帧: {stats['camera_high']['depth_count']}")
        print(f"  camera_left_wrist - Color帧: {stats['camera_left_wrist']['color_count']}")
        print(f"  camera_right_wrist - Color帧: {stats['camera_right_wrist']['color_count']}")
        
    except KeyboardInterrupt:
        print("\n\n用户中断测试")
    except Exception as e:
        print(f"\n\n测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n清理资源...")
        collector.cleanup()
        print("测试完成!")
