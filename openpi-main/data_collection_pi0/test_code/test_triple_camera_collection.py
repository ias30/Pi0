#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试三相机数据采集系统（不需要真实的机械臂和电机）
"""

import time
import sys
from data_collector import DataCollector

def main():
    print("=" * 80)
    print("🧪 测试三相机数据采集系统")
    print("=" * 80)
    
    # 创建数据采集器
    collector = DataCollector("test_episodes")
    
    # 仅初始化相机（不需要机械臂和电机）
    print("\n🔧 初始化硬件（仅相机）...")
    collector.initialize_hardware()
    
    # 连接硬件
    print("\n🔌 连接硬件...")
    hardware_ok = collector.connect_all_hardware()
    
    if not hardware_ok:
        print("\n⚠️  硬件连接失败，但继续测试...")
    else:
        print("\n✅ 硬件连接成功")
    
    try:
        # 开始录制
        print("\n🎬 开始录制...")
        success = collector.start_episode()
        
        if not success:
            print("❌ 无法开始录制")
            return
        
        # 录制10秒
        duration = 10
        print(f"\n📹 录制中... (持续{duration}秒)")
        
        for i in range(duration):
            remaining = duration - i
            print(f"  剩余时间: {remaining}秒", end='\r', flush=True)
            time.sleep(1)
        
        print("\n\n🛑 停止录制...")
        episode_path = collector.stop_episode()
        
        if episode_path:
            print(f"\n✅ Episode录制完成: {episode_path}")
            print("\n📊 显示系统状态:")
            collector.print_status()
            
            # 验证数据结构
            print("\n🔍 验证数据结构...")
            import subprocess
            result = subprocess.run(
                ['python', 'verify_data_structure.py', episode_path],
                capture_output=True,
                text=True
            )
            print(result.stdout)
            if result.returncode != 0:
                print(result.stderr)
        else:
            print("\n❌ Episode停止失败")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
    except Exception as e:
        print(f"\n❌ 测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n🧹 清理资源...")
        collector.cleanup()
        print("✅ 测试完成")


if __name__ == "__main__":
    main()

