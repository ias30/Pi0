#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
时间同步模块
提供NTP时间同步功能，确保数据采集的时间戳统一
"""

import time
import threading
import ntplib
from typing import Optional


class TimeSync:
    """时间同步管理器"""
    
    def __init__(self, ntp_server: str = "pool.ntp.org", sync_interval: float = 3600.0):
        """
        初始化时间同步器
        
        Args:
            ntp_server: NTP服务器地址
            sync_interval: 同步间隔（秒）
        """
        self.ntp_server = ntp_server
        self.sync_interval = sync_interval
        self._offset = 0.0  # 时间偏移量
        self._last_sync_time = 0.0
        self._sync_lock = threading.Lock()
        self._stop_sync = False
        self._sync_thread: Optional[threading.Thread] = None
        self._is_synced = False
        
        # 启动同步线程
        self.start_sync()
    
    def start_sync(self):
        """启动时间同步线程"""
        if self._sync_thread is None or not self._sync_thread.is_alive():
            self._stop_sync = False
            self._sync_thread = threading.Thread(target=self._sync_worker, daemon=True)
            self._sync_thread.start()
            print(f"时间同步线程已启动，NTP服务器: {self.ntp_server}")
    
    def stop_sync(self):
        """停止时间同步线程"""
        self._stop_sync = True
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=2.0)
        print("时间同步线程已停止")
    
    def _sync_worker(self):
        """时间同步工作线程"""
        # 首次同步
        self._perform_sync()
        
        while not self._stop_sync:
            time.sleep(min(self.sync_interval, 60.0))  # 每分钟检查一次
            
            if time.time() - self._last_sync_time >= self.sync_interval:
                self._perform_sync()
    
    def _perform_sync(self):
        """执行时间同步"""
        try:
            ntp_client = ntplib.NTPClient()
            response = ntp_client.request(self.ntp_server, timeout=5)
            
            with self._sync_lock:
                # 计算时间偏移量
                self._offset = response.offset
                self._last_sync_time = time.time()
                self._is_synced = True
                
            print(f"时间同步成功，偏移量: {self._offset:.6f}秒")
            
        except Exception as e:
            print(f"时间同步失败: {e}")
            if not self._is_synced:
                # 首次同步失败，使用本地时间
                with self._sync_lock:
                    self._offset = 0.0
                    self._last_sync_time = time.time()
                    self._is_synced = True
                print("使用本地时间作为基准")
    
    def get_timestamp(self) -> float:
        """
        获取同步后的时间戳
        
        Returns:
            同步后的Unix时间戳（秒）
        """
        with self._sync_lock:
            return time.time() + self._offset
    
    def get_timestamp_ms(self) -> int:
        """
        获取同步后的时间戳（毫秒）
        
        Returns:
            同步后的Unix时间戳（毫秒）
        """
        return int(self.get_timestamp() * 1000)
    
    def is_synced(self) -> bool:
        """检查是否已完成时间同步"""
        return self._is_synced
    
    def get_offset(self) -> float:
        """获取当前时间偏移量"""
        with self._sync_lock:
            return self._offset
    
    def format_timestamp(self, timestamp: Optional[float] = None) -> str:
        """
        格式化时间戳为可读字符串
        
        Args:
            timestamp: 时间戳，如果为None则使用当前时间
            
        Returns:
            格式化的时间字符串
        """
        if timestamp is None:
            timestamp = self.get_timestamp()
        
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp)) + \
               f".{int((timestamp % 1) * 1000):03d}"


# 全局时间同步实例
_global_time_sync: Optional[TimeSync] = None


def get_global_time_sync() -> TimeSync:
    """获取全局时间同步实例"""
    global _global_time_sync
    if _global_time_sync is None:
        _global_time_sync = TimeSync()
    return _global_time_sync


def get_timestamp() -> float:
    """获取同步后的时间戳（便捷函数）"""
    return get_global_time_sync().get_timestamp()


def get_timestamp_ms() -> int:
    """获取同步后的时间戳毫秒（便捷函数）"""
    return get_global_time_sync().get_timestamp_ms()


if __name__ == "__main__":
    # 测试时间同步功能
    sync = TimeSync()
    time.sleep(2)  # 等待首次同步完成
    
    print(f"时间是否已同步: {sync.is_synced()}")
    print(f"时间偏移量: {sync.get_offset():.6f}秒")
    print(f"当前同步时间戳: {sync.get_timestamp():.6f}")
    print(f"当前同步时间: {sync.format_timestamp()}")
    
    sync.stop_sync()