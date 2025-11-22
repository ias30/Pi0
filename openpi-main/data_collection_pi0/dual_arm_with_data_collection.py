#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
集成版双臂机械臂控制与数据采集系统
在原有遥操作功能基础上增加高质量数据采集能力
"""

import sys
import os
# 添加父目录到路径，以便导入原始控制模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/opt/ros/noetic/lib/python3/dist-packages')
# Example user-specific paths, ensure these are correct for your setup
sys.path.append('/home/ren9/touch_ws/devel/lib/python3/dist-packages')
sys.path.append('/home/ren9/realman_ws/devel/lib/python3/dist-packages')
import threading
import pygame
import rospy
import numpy as np # 导入numpy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from omni_msgs.msg import OmniButtonEvent

# 导入原始控制系统
from dual_arm_motor_control import (
    ArmMotorController, DEFAULT_MOTOR_LIMITS, LEFT_DEVICE_MOTOR_LIMITS
)
from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e
import time
# 导入数据采集系统
from data_collector import DataCollector
from time_sync import get_global_time_sync


class IntegratedDualArmSystem:
    """集成版双臂系统：遥操作 + 数据采集"""

    def __init__(self):
        rospy.init_node('integrated_dual_arm_system', anonymous=True)

        print("=" * 80)
        print("🚀 初始化集成版双臂机械臂控制与数据采集系统")
        print("=" * 80)

        # =================================================================
        # 🔧 设备配置（与原系统相同）
        # =================================================================
        device_config = {
            'left_device': {
                'serial_port': '/dev/serial/by-path/pci-0000:00:14.0-usb-0:2.3:1.0-port0',
                'robot_ip': '169.254.128.18',
            },
            'right_device': {
                'serial_port': '/dev/serial/by-path/pci-0000:00:14.0-usb-0:2.1:1.0-port0',
                'robot_ip': '169.254.128.19',
            }
        }

        fallback_config = {
            'left_device': {'serial_port': '/dev/ttyUSB0', 'robot_ip': '169.254.128.18'},
            'right_device': {'serial_port': '/dev/ttyUSB1', 'robot_ip': '169.254.128.19'}
        }

        # 检查配置可用性
        import os
        use_by_path = all(os.path.exists(device_config[dev]['serial_port']) for dev in device_config)
        final_config = device_config if use_by_path else fallback_config

        # =================================================================
        # 📦 Bounding Box 参数 (在此处设定)
        # =================================================================
        # 参数格式: {'min': [x_min, y_min, z_min], 'max': [x_max, y_max, z_max]}
        # 单位：米
        LEFT_ARM_BOUNDING_BOX = {
            'min': np.array([-0.398, -0.538, 0.013]),
            'max': np.array([-0.198, -0.438, 0.213])
        }
        RIGHT_ARM_BOUNDING_BOX = {
            'min': np.array([0.198, -0.638, 0.013]),
            'max': np.array([0.398, -0.338, 0.213])
        }


        # =================================================================
        # 🤖 初始化原始控制系统
        # =================================================================
        print("🤖 初始化双臂控制系统...")

        self.left_controller = ArmMotorController(
            "Left Device",
            final_config['left_device']['robot_ip'], 8080,
            final_config['left_device']['serial_port'],
            scale_factor_arm=5.0, max_delta_arm=0.1, motor_step=100,
            thread_mode=rm_thread_mode_e.RM_TRIPLE_MODE_E,
            motor_limits=LEFT_DEVICE_MOTOR_LIMITS, is_left_device=True,
            bounding_box=None # 传递左臂Bounding Box
        )

        self.right_controller = ArmMotorController(
            "Right Device",
            final_config['right_device']['robot_ip'], 8080,
            final_config['right_device']['serial_port'],
            scale_factor_arm=5.0, max_delta_arm=0.1, motor_step=100,
            thread_mode=None, motor_limits=None, is_left_device=False,
            bounding_box=None # 禁用右臂Bounding Box
        )

        # =================================================================
        # 🦾 移动机械臂到初始位置 (修改为串行移动)
        # =================================================================
        print("🦾 正在按顺序移动机械臂到初始关节角度...")
        # 将局部变量改为实例属性（self.xxx），以便在其他方法中调用
        self.initial_joint_left_angles = [-79.5739974975586, -44.62900161743164, -71.25800323486328, 70.08100128173828, 14.241999626159668, 42.44200134277344]
        self.initial_joint_right_angles = [80.212 ,40.867 ,66.459 ,-45.413 ,-17.694 ,0.480]
        
        # 1. 移动左臂并等待完成
        print("⏳ 正在移动左臂...")
        self.left_controller.robot_arm.rm_movej(self.initial_joint_left_angles, 1, 0, 0, True) # block=True会等待移动完成
        print("✅ 左臂已到达初始位置。")

        # 2. 移动右臂并等待完成
        print("⏳ 正在移动右臂...")
        self.right_controller.robot_arm.rm_movej(self.initial_joint_right_angles, 5, 0, 0, True) # block=True会等待移动完成
        print("✅ 右臂已到达初始位置。")
        
        print("✅ 双臂均已到达初始位置。")
        time.sleep(3) # 等待3秒以确保状态稳定

        # =================================================================
        # 📊 初始化数据采集系统
        # =================================================================
        print("📊 初始化数据采集系统...")

        self.data_collector = DataCollector("data_collection_episodes")

        # 初始化数据采集硬件
        self.data_collector.initialize_hardware(
            left_robot_arm=self.left_controller.robot_arm,
            left_arm_handle=self.left_controller.arm_handle,
            right_robot_arm=self.right_controller.robot_arm,
            right_arm_handle=self.right_controller.arm_handle,
            left_position_controller=self.left_controller.pos_ctrl,
            right_position_controller=self.right_controller.pos_ctrl
        )

        # 连接硬件
        hardware_ok = self.data_collector.connect_all_hardware()
        print(f"🔌 硬件连接: {'✅ 成功' if hardware_ok else '⚠️  部分失败'}")

        # =================================================================
        # 🎮 系统控制
        # =================================================================
        self.step = 8192
        self.stop = False

        # ROS订阅
        self._setup_ros_subscriptions()

        # Pygame初始化
        pygame.init()
        pygame.display.set_mode((600, 400))
        pygame.display.set_caption("Integrated Dual-Arm Control & Data Collection")

        self.ros_thread = threading.Thread(target=rospy.spin, daemon=True)
        self.ros_thread.start()

        print("=" * 80)
        print("✅ 系统初始化完成")
        print("=" * 80)
        self._print_usage_instructions()

        # 启动主循环
        self.run()

    def _setup_ros_subscriptions(self):
        """设置ROS订阅"""
        # 左设备订阅
        rospy.Subscriber('/left_device/phantom/pose', PoseStamped,
                        self.left_controller.phantom_pose_callback)
        rospy.Subscriber('/left_device/phantom/joint_states', JointState,
                        self.left_controller.phantom_joint_state_callback)
        rospy.Subscriber('/left_device/phantom/button', OmniButtonEvent,
                        self.left_controller.phantom_button_callback)

        # 右设备订阅
        rospy.Subscriber('/right_device/phantom/pose', PoseStamped,
                        self.right_controller.phantom_pose_callback)
        rospy.Subscriber('/right_device/phantom/joint_states', JointState,
                        self.right_controller.phantom_joint_state_callback)
        rospy.Subscriber('/right_device/phantom/button', OmniButtonEvent,
                        self.right_controller.phantom_button_callback)

    def _print_usage_instructions(self):
        """打印使用说明"""
        print("🎮 系统控制说明:")
        print("=" * 60)
        print("【遥操作控制】")
        print("  🤏 机械臂: 按下灰色按钮 + 移动Phantom设备 (已添加Bounding Box限制)")
        print("  ⚙️  电机: roll/pitch控制电机0&1，白色按钮切换电机2&3")
        print("  ⌨️  键盘: 数字键1-4(左设备), 5-8(右设备)")
        print()
        print("【数据采集控制】")
        print("  🎬 按 R 键: 开始Episode录制")
        print("  🛑 按 C 键: 停止Episode录制")
        print("  🗑️  按 Backspace: 删除最近的Episode")
        print("  📊 按 S 键: 显示系统状态")
        print()
        print("【系统控制】")
        print("  ARM RESET: 按 H 键(左臂), 按 K 键(右臂)")
        print("  🔄 按 0 键: 重置所有电机到零点")
        print("  🚪 按 Q 键: 安全退出系统")
        print("=" * 60)

    def run(self):
        """主运行循环"""
        try:
            main_loop_rate = rospy.Rate(50)  # 200Hz控制频率

            while not self.stop and not rospy.is_shutdown():
                # 处理pygame事件
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.stop = True
                    elif event.type == pygame.KEYDOWN:
                        self._handle_key_event(event)

                if self.stop:
                    break

                # 更新控制器
                self.left_controller.update()
                self.right_controller.update()

                main_loop_rate.sleep()

        except KeyboardInterrupt:
            print("\n🛑 检测到 Ctrl-C，正在安全关闭...")
            self.stop = True
        finally:
            self.cleanup()

    def _handle_key_event(self, event):
        """处理键盘事件"""
        key_name = pygame.key.name(event.key)

        # 数据采集控制
        if key_name == 'r':
            self._start_recording()
        elif key_name == 'c':
            self._stop_recording()
        elif key_name == 'backspace':
            self._delete_last_episode()
        elif key_name == 's':
            self._show_system_status()

        # 电机控制 (原有逻辑)
        elif key_name in ['1', '2', '3', '4']:
            self._handle_left_motor_control(key_name)
        elif key_name in ['5', '6', '7', '8']:
            self._handle_right_motor_control(key_name)
        
        # 新增的复位逻辑
        elif key_name == 'h':
            print("🔄 按下 'h'，正在复位左臂到初始位置...")
            self.left_controller.robot_arm.rm_movej(self.initial_joint_left_angles, 10, 0, 0, True)
            print("✅ 左臂已复位。")
        elif key_name == 'k':
            print("🔄 按下 'k'，正在复位右臂到初始位置...")
            self.right_controller.robot_arm.rm_movej(self.initial_joint_right_angles, 10, 0, 0, True)
            print("✅ 右臂已复位。")

        # 系统控制
        elif key_name == '0':
            self._reset_all_motors()
        elif key_name == 'q':
            self._safe_exit()

    def _start_recording(self):
        """开始录制"""
        if self.data_collector.is_recording():
            print("⚠️  已在录制中")
            return

        success = self.data_collector.start_episode()
        if success:
            print("🎬 Episode录制已开始")
        else:
            print("❌ Episode录制开始失败")

    def _stop_recording(self):
        """停止录制"""
        if not self.data_collector.is_recording():
            print("⚠️  当前未在录制")
            return

        completed_path = self.data_collector.stop_episode()
        if completed_path:
            print(f"✅ Episode录制完成: {completed_path}")
        else:
            print("❌ Episode录制停止失败")

    def _delete_last_episode(self):
        """删除最近的Episode"""
        success = self.data_collector.delete_last_episode()
        if success:
            print("🗑️  最近的Episode已删除")
        else:
            print("❌ 删除Episode失败")

    def _show_system_status(self):
        """显示系统状态"""
        self.data_collector.print_status()

    def _handle_left_motor_control(self, key_name):
        """处理左侧电机控制"""
        if key_name == '1':
            self.left_controller.motor_target_positions[2] += self.step
            self.left_controller.motor_target_positions[3] += self.step
        elif key_name == '2':
            self.left_controller.motor_target_positions[2] -= self.step
            self.left_controller.motor_target_positions[3] -= self.step
        elif key_name == '3':
            self.left_controller.motor_target_positions[2] -= self.step
            self.left_controller.motor_target_positions[3] += self.step
        elif key_name == '4':
            self.left_controller.motor_target_positions[2] += self.step
            self.left_controller.motor_target_positions[3] -= self.step

        print(f"左设备电机目标位置: {self.left_controller.motor_target_positions}")

    def _handle_right_motor_control(self, key_name):
        """处理右侧电机控制"""
        if key_name == '5':
            self.right_controller.motor_target_positions[2] += self.step
            self.right_controller.motor_target_positions[3] += self.step
        elif key_name == '6':
            self.right_controller.motor_target_positions[2] -= self.step
            self.right_controller.motor_target_positions[3] -= self.step
        elif key_name == '7':
            self.right_controller.motor_target_positions[2] -= self.step
            self.right_controller.motor_target_positions[3] += self.step
        elif key_name == '8':
            self.right_controller.motor_target_positions[2] += self.step
            self.right_controller.motor_target_positions[3] -= self.step

        print(f"右设备电机目标位置: {self.right_controller.motor_target_positions}")

    def _reset_all_motors(self):
        """重置所有电机"""
        print("🔄 重置所有电机到零点...")
        self.left_controller.reset_motors_to_zero()
        self.right_controller.reset_motors_to_zero()
        print("✅ 所有电机已重置")

    def _safe_exit(self):
        """安全退出"""
        print("🚪 准备安全退出...")

        # 如果正在录制，先停止
        if self.data_collector.is_recording():
            print("🛑 正在停止录制...")
            self.data_collector.stop_episode()

        # 重置电机
        self._reset_all_motors()

        self.stop = True

    def cleanup(self):
        """清理系统资源"""
        print("🧹 清理系统资源...")

        # 清理数据采集系统
        self.data_collector.cleanup()

        # 清理控制器
        self.left_controller.cleanup()
        self.right_controller.cleanup()

        # 断开机械臂连接
        try:
            RoboticArm.rm_destroy()
            print("🦾 机械臂连接已断开")
        except Exception as e:
            print(f"❌ 断开机械臂时出错: {e}")

        # 关闭ROS
        if not rospy.is_shutdown():
            rospy.signal_shutdown("集成系统正常退出")

        # 清理pygame
        pygame.quit()

        print("✅ 系统清理完成")


if __name__ == "__main__":
    try:
        system = IntegratedDualArmSystem()
    except Exception as e:
        print(f"❌ 系统启动失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("🏁 集成双臂系统程序结束")