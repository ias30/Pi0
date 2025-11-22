#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
同时控制两个机械臂和两组电机
Left Device: 控制第一个机械臂(169.254.128.18) + 第一组电机(/dev/ttyUSB0)
Right Device: 控制第二个机械臂(169.254.128.19) + 第二组电机(/dev/ttyUSB1)
'''
# Add ROS package paths - customize these to your environment
import sys
sys.path.append('/opt/ros/noetic/lib/python3/dist-packages')
# Example user-specific paths, ensure these are correct for your setup
sys.path.append('/home/ren9/touch_ws/devel/lib/python3/dist-packages')
sys.path.append('/home/ren9/realman_ws/devel/lib/python3/dist-packages')

# Essential Imports
import rospy
import threading
import numpy as np
import time
import serial
import pygame

# ROS Message Imports
from omni_msgs.msg import OmniButtonEvent
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState

# Robotic Arm Interface (ensure this path and import are correct)
from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e

# ===================== 电机限位参数配置 =====================
# 默认限位参数（适用于Right Device - USB端口1.3）
DEFAULT_MOTOR_LIMITS = {
    0: {'max': 8192*4.5, 'min': -8192*4.5},   # 电机0: ±4.5圈
    1: {'max': 8192*4.5, 'min': -8192*4.5},   # 电机1: ±3.5圈
    2: {'max': 8192*3.0, 'min': -8192*3.0},   # 电机2: ±3.0圈
    3: {'max': 8192*3.0, 'min': -8192*3.0}    # 电机3: ±9.0圈
}

# Left Device自定义限位参数（USB端口1.1）c
LEFT_DEVICE_MOTOR_LIMITS = {
    0: {'max': 8192*4.5, 'min': -8192*4.5},   # 电机0: ±1.5圈
    1: {'max': 8192*4.0, 'min': -8192*4.0},   # 电机1: ±3.0圈
    2: {'max': 8192*2.0, 'min': -8192*2.0},   # 电机2: ±1.0圈
    3: {'max': 8192*2.0, 'min': -8192*2.0}    # 电机3: ±1.0圈
}
# =========================================================

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0

    def compute(self, setpoint, measured_value, dt):
        error = setpoint - measured_value
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        output = int(self.kp * error + self.ki * self.integral + self.kd * derivative)
        output = max(min(output, 1000), -1000) # 限制输出范围
        self.prev_error = error
        return output


class PositionController:
    def __init__(self, device_name, serial_port, kp=0.015, ki=0.000, kd=0.0001,
                 motor_limits=None):
        self.device_name = device_name
        self.serial_port = serial_port
        self.current_pos = [0, 0, 0, 0]
        self.read_pos = [0, 0, 0, 0]
        self.cmds = [0, 0, 0, 0]
        self.stop = False
        self.pid_ctrl = PIDController(kp, ki, kd)
        self.motor_23_current_state = 0  # 0=位置0（零点），1=位置1（极限位置）

        # 电机限位参数配置
        self.motor_limits = motor_limits if motor_limits is not None else DEFAULT_MOTOR_LIMITS

        self.data_ser = None
        self.t1_ref = None   # position_ctrl 线程的引用
        self.t2_ref = None   # get_data 线程的引用

        self.t1_ref = threading.Thread(target=self.position_ctrl)
        self.t1_ref.daemon = True
        self.t1_ref.start()

    def set_position(self, cmds):
        self.cmds = cmds

    def read_position(self):
        return self.read_pos

    def get_data(self):
        # 第一个循环：尝试同步，找到起始字符 'a'
        while not self.stop:
            try:
                if not (self.data_ser and self.data_ser.is_open):
                    time.sleep(0.1) # 等待串口打开
                    continue

                recv_byte = self.data_ser.read(1)
                if not recv_byte:
                    continue

                recv = recv_byte.decode("ascii")
                if recv == "a":
                    if not (self.data_ser and self.data_ser.is_open):
                        break
                    recv_remainder = self.data_ser.read(79)
                    if len(recv_remainder) == 79:
                        break
                    else:
                        self.data_ser.flushInput()
                        continue
            except serial.SerialException as e:
                print(f"{self.device_name}: get_data串口在同步'a'时发生错误: {e}")
                break
            except UnicodeDecodeError:
                self.data_ser.flushInput()
                continue
            except Exception as e:
                print(f"{self.device_name}: get_data在同步'a'时发生未知错误: {e}")
                break

        # 第二个循环：读取有效数据帧
        while not self.stop:
            try:
                if not (self.data_ser and self.data_ser.is_open):
                    time.sleep(0.1)
                    continue

                recv_bytes = self.data_ser.read(80)
                if not recv_bytes or len(recv_bytes) < 80:
                    if recv_bytes and len(recv_bytes) < 80:
                        self.data_ser.flushInput()
                    continue

                recv = recv_bytes.decode("ascii")
                recv_parts = recv.split(',')
                if len(recv_parts) > 1:
                    recv_parts[-1] = recv_parts[-1].split('\n')[0]
                    recv_parts = recv_parts[1:]
                    if len(recv_parts) == 4:
                        self.current_pos = list(map(int, recv_parts))
                        self.read_pos = list(map(int, recv_parts))

            except serial.SerialException as e:
                print(f"{self.device_name}: get_data串口在读取数据时发生错误: {e}")
                break
            except UnicodeDecodeError:
                if self.data_ser and self.data_ser.is_open:
                    self.data_ser.flushInput()
                continue
            except ValueError:
                if self.data_ser and self.data_ser.is_open:
                    self.data_ser.flushInput()
                continue
            except Exception as e:
                print(f"{self.device_name}: get_data在读取数据时发生未知错误: {e}")
                break
        print(f"{self.device_name}: get_data 线程退出。")

    def spd_calculate(self, motor_id, target_pos, t):
        spd = self.pid_ctrl.compute(target_pos, self.current_pos[motor_id], t)

        # 使用配置的电机角度限制逻辑
        max_limit = self.motor_limits[motor_id]['max']
        min_limit = self.motor_limits[motor_id]['min']

        if motor_id == 0:
            if self.current_pos[0] > max_limit and spd > 0: spd = -1
            elif self.current_pos[0] < min_limit and spd < 0: spd = -1
        elif motor_id == 1:
            if self.current_pos[1] > max_limit and spd > 0: spd = -1
            elif self.current_pos[1] < min_limit and spd < 0: spd = -1
        elif motor_id == 2:
            if self.current_pos[2] > max_limit and spd > 0:
                print(f"{self.device_name}: 电机2 (ID {motor_id}) 超过正向限制范围 ({self.current_pos[2]} > {max_limit})，目标速度 {spd}。仅打印。")
            elif self.current_pos[2] < min_limit and spd < 0:
                print(f"{self.device_name}: 电机2 (ID {motor_id}) 超过负向限制范围 ({self.current_pos[2]} < {min_limit})，目标速度 {spd}。仅打印。")
        elif motor_id == 3:
            if self.current_pos[3] > max_limit and spd > 0: spd = -1
            elif self.current_pos[3] < min_limit and spd < 0: spd = -1

        spd_str = str(spd).rjust(6)
        if len(spd_str) > 6:
            spd_str = "-00001"
        return spd_str

    def isReached(self, motor_id):
        return abs(self.current_pos[motor_id] - self.cmds[motor_id]) < 1000

    def position_ctrl(self):
        try:
            self.data_ser = serial.Serial(self.serial_port, 115200, timeout=1)
            self.data_ser.flushInput()
            print(f"{self.device_name}: 串口 {self.serial_port} 已打开。")

            self.t2_ref = threading.Thread(target=self.get_data)
            self.t2_ref.daemon = True
            self.t2_ref.start()

            while not self.stop:
                spds_command = ""
                for i in range(4):
                    spds_command += self.spd_calculate(i, int(self.cmds[i]), 0.03)

                if len(spds_command) > 24:
                    spds_command = "-00001-00001-00001-00001"

                if self.data_ser and self.data_ser.is_open:
                    try:
                        self.data_ser.write(spds_command.encode("ascii"))
                    except serial.SerialException as e:
                        print(f"{self.device_name}: 写入串口时发生错误: {e}。正在停止...")
                        self.stop = True
                        break
                else:
                    print(f"{self.device_name}: 串口在循环中未打开。正在停止...")
                    self.stop = True
                    break
                time.sleep(0.03)

        except serial.SerialException as e:
            print(f"{self.device_name}: 无法打开串口或串口通信出错: {e}")
            self.data_ser = None
        except Exception as e:
            print(f"{self.device_name}: position_ctrl 线程发生意外错误: {e}")
        finally:
            print(f"{self.device_name}: 进入 position_ctrl 的 finally 清理块。")

            if hasattr(self, 't2_ref') and self.t2_ref and self.t2_ref.is_alive():
                print(f"{self.device_name}: 等待 get_data 线程 (t2) 结束...")
                self.t2_ref.join(timeout=2.0)
                if self.t2_ref.is_alive():
                    print(f"{self.device_name}: get_data 线程 (t2) 未能在2秒内结束。")

            if self.data_ser and self.data_ser.is_open:
                stop_cmd_bytes = b"-00001-00001-00001-00001"
                print(f"{self.device_name}: 发送最终停止指令 '{stop_cmd_bytes.decode()}' 5次。")
                for i in range(5):
                    try:
                        self.data_ser.write(stop_cmd_bytes)
                        self.data_ser.flush()
                        print(f"{self.device_name}: 第 {i+1}/5 次停止指令已发送并刷新。")
                        time.sleep(0.03)
                    except serial.SerialException as se_final:
                        print(f"{self.device_name}: 发送第 {i+1}/5 次停止指令时串口错误: {se_final}。终止发送。")
                        break
                    except Exception as e_final:
                        print(f"{self.device_name}: 发送第 {i+1}/5 次停止指令时发生错误: {e_final}。终止发送。")
                        break

                try:
                    self.data_ser.flush()
                    print(f"{self.device_name}: 最终串口数据已刷新。")
                except Exception as e_flush_final:
                    print(f"{self.device_name}: 最终刷新串口数据时发生错误: {e_flush_final}")

                try:
                    self.data_ser.close()
                    print(f"{self.device_name}: 串口已关闭。")
                except Exception as e_close:
                    print(f"{self.device_name}: 关闭串口时发生错误: {e_close}")
            else:
                print(f"{self.device_name}: 串口未打开或不可用，无法发送停止指令或关闭。")

            self.data_ser = None
            print(f"{self.device_name}: position_ctrl 线程已结束。")


class ArmMotorController:
    def __init__(self, device_name, robot_ip, robot_port, serial_port,
                 scale_factor_arm=10.0, max_delta_arm=0.1, motor_step=100,
                 thread_mode=rm_thread_mode_e.RM_SINGLE_MODE_E, motor_limits=None,
                 is_left_device=False, bounding_box=None):
        self.device_name = device_name
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.is_left_device = is_left_device  # 新增：设备类型标识
        self.bounding_box = bounding_box # 新增：边界框

        # --- 机械臂控制属性 ---
        self.robot_arm = RoboticArm(thread_mode)
        self.arm_handle = self.robot_arm.rm_create_robot_arm(robot_ip, robot_port)
        print(f"{self.device_name}: 机械臂连接ID: {self.arm_handle.id}")

        # 获取并打印当前机械臂状态
        arm_state_result = self.robot_arm.rm_get_current_arm_state()
        print(f"{self.device_name}: 机械臂当前状态: {arm_state_result}")

        self.scale_factor_arm = scale_factor_arm
        self.max_delta_arm = max_delta_arm
        self.last_vr_pos_arm = None
        self.is_arm_moving = False
        self.current_phantom_pose_msg = None
        self.new_phantom_pose_data = False

        # --- 电机控制属性 ---
        # 存储限位参数，用于process_motor_control函数
        self.motor_limits = motor_limits if motor_limits is not None else DEFAULT_MOTOR_LIMITS

        self.pos_ctrl = PositionController(device_name, serial_port, motor_limits=motor_limits)
        time.sleep(1) # 等待电机控制器初始化
        self.motor_target_positions = [0, 0, 0, 0]
        self.motor_step = motor_step

        # --- Phantom数据属性 ---
        self.current_phantom_roll = 0.0
        self.current_phantom_pitch = 0.0
        self.grey_button_state = 0
        self.white_button_state = 0
        self.prev_grey_button_state = 0  # 上一次灰色按钮状态
        self.prev_white_button_state = 0 # 上一次白色按钮状态

        # 电机2&3状态跟踪（0状态或1状态）
        self.motor_23_current_state = 0  # 0=位置0（零点），1=位置1（极限位置）

        self.control_data_lock = threading.Lock()

        # 设置初始电机位置
        self.pos_ctrl.set_position(self.motor_target_positions)
        print(f"{self.device_name}: 电机已发送复位到零点指令")
        print(f"{self.device_name}: 机械臂线程模式: {thread_mode}")
        print(f"{self.device_name}: 设备类型: {'左设备' if is_left_device else '右设备'}")
        if self.bounding_box:
            print(f"{self.device_name}: Bounding Box 已启用: {self.bounding_box}")

    def phantom_pose_callback(self, msg: PoseStamped):
        with self.control_data_lock:
            self.current_phantom_pose_msg = msg
            self.new_phantom_pose_data = True

    def phantom_button_callback(self, msg: OmniButtonEvent):
        with self.control_data_lock:
            self.grey_button_state = msg.grey_button
            self.white_button_state = msg.white_button

    def phantom_joint_state_callback(self, msg: JointState):
        joint_positions = {}
        for name, position in zip(msg.name, msg.position):
            joint_positions[name] = position

        with self.control_data_lock:
            self.current_phantom_roll = joint_positions.get('roll', 0.0)
            self.current_phantom_pitch = joint_positions.get('waist', 0.0)

    def stop_robot_arm_movement(self):
        if self.is_arm_moving:
            try:
                print(f"{self.device_name}: 机械臂停止运动条件满足。发送停止指令。")
                _, state = self.robot_arm.rm_get_current_arm_state()
                current_pose_arm = [float(x) for x in state['pose']]
                self.robot_arm.rm_movep_canfd(current_pose_arm, follow=False)
                print(f"{self.device_name}: 机械臂停止指令已发送。")
                self.is_arm_moving = False
            except Exception as e:
                print(f"{self.device_name}: 停止机械臂运动时出错: {e}")

    def process_robot_arm_movement(self):
        local_current_phantom_pose_msg = None
        local_new_phantom_pose_data = False
        local_grey_button = 0
        local_white_button = 0

        with self.control_data_lock:
            if self.new_phantom_pose_data and self.current_phantom_pose_msg is not None:
                local_current_phantom_pose_msg = self.current_phantom_pose_msg
                local_new_phantom_pose_data = True
                self.new_phantom_pose_data = False
            local_grey_button = self.grey_button_state
            local_white_button = self.white_button_state

        if not local_new_phantom_pose_data:
            return

        vr_pos_arm = np.array([
            local_current_phantom_pose_msg.pose.position.x,
            local_current_phantom_pose_msg.pose.position.y,
            local_current_phantom_pose_msg.pose.position.z
        ])

        # 机械臂启用逻辑：只需灰色按钮按下
        arm_should_be_moving_now = (local_grey_button == 1)

        if not arm_should_be_moving_now:
            if self.is_arm_moving:
                self.stop_robot_arm_movement()
            self.last_vr_pos_arm = vr_pos_arm
            return

        # 机械臂控制已启用（灰色按钮按下）
        if self.last_vr_pos_arm is None:
            self.last_vr_pos_arm = vr_pos_arm
            print(f"{self.device_name}: 机械臂控制已启用。初始化VR基准位置。")
            return

        delta_arm = vr_pos_arm - self.last_vr_pos_arm
        delta_arm = np.clip(delta_arm, -self.max_delta_arm, self.max_delta_arm)

        _, current_arm_state = self.robot_arm.rm_get_current_arm_state()
        current_arm_pose_list = current_arm_state['pose']

        target_arm_pos = [
            float(current_arm_pose_list[0] + delta_arm[0] * self.scale_factor_arm),
            float(current_arm_pose_list[1] + delta_arm[1] * self.scale_factor_arm),
            float(current_arm_pose_list[2] + delta_arm[2] * self.scale_factor_arm),
        ]
        print(f"{self.device_name}: 目标机械臂位置（未限制前）: {target_arm_pos}")
        # *** Bounding Box限制 ***
        if self.bounding_box:
            min_bound = self.bounding_box['min']
            max_bound = self.bounding_box['max']
            clipped_pos = np.clip(target_arm_pos, min_bound, max_bound)
            # 检查是否有clip发生
            if not np.array_equal(target_arm_pos, clipped_pos):
                 print(f"box限制: {target_arm_pos} -> {clipped_pos}")
            target_arm_pos = clipped_pos.tolist()


        target_arm_ori = [float(x) for x in current_arm_pose_list[3:6]]
        target_arm_full_pose = target_arm_pos + target_arm_ori

        # 再次检查按钮状态以确保安全
        with self.control_data_lock:
            final_check_grey = self.grey_button_state

        if final_check_grey == 1:
            result = self.robot_arm.rm_movep_canfd(target_arm_full_pose, follow=False)
            self.is_arm_moving = True
        else:
            print(f"{self.device_name}: 在计算过程中机械臂按钮被释放；运动被中止。")
            if self.is_arm_moving:
                self.stop_robot_arm_movement()

        self.last_vr_pos_arm = vr_pos_arm

    def process_motor_control(self):
        local_grey_btn = 0
        local_white_btn = 0
        local_phantom_roll = 0.0
        local_phantom_pitch = 0.0

        # 获取当前和之前的按钮状态（用于边缘检测）
        with self.control_data_lock:
            local_grey_btn = self.grey_button_state
            local_white_btn = self.white_button_state
            local_phantom_roll = self.current_phantom_roll
            local_phantom_pitch = self.current_phantom_pitch

        # --- 电机0和1：由Phantom的roll和pitch控制（保持不变）---
        conversion_factor_roll = 7 * 8192 / np.pi
        conversion_factor_pitch = 15 * 8192 / np.pi

        self.motor_target_positions[0] = -int(local_phantom_roll * conversion_factor_roll)
        self.motor_target_positions[1] = -int(local_phantom_pitch * conversion_factor_pitch)

        # 使用配置的电机0和1的限制
        self.motor_target_positions[0] = max(min(self.motor_target_positions[0],
                                                self.motor_limits[0]['max']),
                                            self.motor_limits[0]['min'])
        self.motor_target_positions[1] = max(min(self.motor_target_positions[1],
                                                self.motor_limits[1]['max']),
                                            self.motor_limits[1]['min'])

        # --- 电机2和3：白色按钮状态切换控制 ---
        # 使用配置的限位参数
        limit_motor2_abs = self.motor_limits[2]['max']
        limit_motor3_abs = self.motor_limits[3]['max']

        # 只有在机械臂控制未激活时才激活电机控制（即灰色按钮未按下）
        if local_grey_btn == 0:
            # 检测白色按钮的边缘触发（从0到1的变化）
            white_button_pressed = (local_white_btn == 1 and self.prev_white_button_state == 0)

            if white_button_pressed:
                # 状态切换：0↔1 在零点和极限位置间切换
                if self.motor_23_current_state == 0:
                    # 从状态0切换到状态1：根据设备类型应用不同的运动方向
                    if self.is_left_device:
                        # 左设备：电机2正极限，电机3负极限（与右设备相反）
                        self.motor_target_positions[2] = -limit_motor2_abs    # 电机2到正极限
                        self.motor_target_positions[3] = limit_motor3_abs   # 电机3到负极限
                        print(f"{self.device_name}: ⚪ 白色按钮切换 → 状态1: 电机2={limit_motor2_abs}, 电机3={-limit_motor3_abs} [左设备相反运动]")
                    else:
                        # 右设备：电机2负极限，电机3正极限（原逻辑）
                        self.motor_target_positions[2] = -limit_motor2_abs   # 电机2到负极限
                        self.motor_target_positions[3] = limit_motor3_abs    # 电机3到正极限
                        print(f"{self.device_name}: ⚪ 白色按钮切换 → 状态1: 电机2={-limit_motor2_abs}, 电机3={limit_motor3_abs} [右设备标准运动]")
                    self.motor_23_current_state = 1
                    self.pos_ctrl.motor_23_current_state = 1  # 更新PositionController中的状态
                else:
                    # 从状态1切换到状态0：电机2&3回到负值位置（根据设备类型不同）
                    if self.is_left_device:
                        # 左设备：状态0的负值位置
                        self.motor_target_positions[2] = 4250 # 电机2到-50%极限位置
                        self.motor_target_positions[3] = -4250  # 电机3到-50%极限位置
                        print(f"{self.device_name}: ⚪ 白色按钮切换 → 状态0: 电机2=1000, 电机3=-1000 [左设备负值位置]")
                    else:
                        # 右设备：状态0的负值位置
                        self.motor_target_positions[2] = 2500 # 电机2到-30%极限位置
                        self.motor_target_positions[3] = -2500# 电机3到-30%极限位置
                        print(f"{self.device_name}: ⚪ 白色按钮切换 → 状态0: 电机2=1000, 电机3=-1000 [右设备负值位置]")
                    self.motor_23_current_state = 0
                    self.pos_ctrl.motor_23_current_state = 0  # 更新PositionController中的状态
        else:
            # 机械臂控制激活时，不响应白色按钮
            if local_white_btn == 1 and self.prev_white_button_state == 0:
                print(f"{self.device_name}: 🛡️ 白色按钮触发但灰色按钮正在控制机械臂，跳过电机控制")

        # 更新按钮状态（用于下次边缘检测）
        self.prev_grey_button_state = local_grey_btn
        self.prev_white_button_state = local_white_btn

        self.pos_ctrl.set_position(self.motor_target_positions)

    def update(self):
        """每个控制周期调用的更新函数"""
        self.process_robot_arm_movement()
        self.process_motor_control()

    def reset_motors_to_zero(self):
        """重置电机到零点"""
        # 所有电机复位到零点
        self.motor_target_positions = [0, 0, 0, 0]
        self.motor_23_current_state = 0  # 重置电机2&3状态为0
        self.pos_ctrl.set_position(self.motor_target_positions)

        print(f"{self.device_name}: 等待电机到达零点 (最多10秒)...")
        wait_start_time = time.time()
        all_reached = False
        while time.time() - wait_start_time < 10.0:
            if all(self.pos_ctrl.isReached(i) for i in range(4)):
                all_reached = True
                break
            time.sleep(0.1)

        if all_reached:
            print(f"{self.device_name}: 电机已复位到零点，状态重置为0。")
        else:
            print(f"{self.device_name}: 电机复位超时或未能全部到达零点。")

    def cleanup(self):
        """清理资源"""
        print(f"{self.device_name}: 开始清理资源...")

        # 停止机械臂运动
        if self.is_arm_moving:
            self.stop_robot_arm_movement()

        # 注意：不在这里删除单个机械臂对象，将在DualArmMotorController中统一调用RoboticArm.rm_destroy()
        print(f"{self.device_name}: 机械臂将在统一清理中断开连接。")

        # 停止电机控制器
        if hasattr(self, 'pos_ctrl') and self.pos_ctrl:
            print(f"{self.device_name}: 通知电机控制器停止...")
            self.pos_ctrl.stop = True

            if hasattr(self.pos_ctrl, 't1_ref') and self.pos_ctrl.t1_ref and self.pos_ctrl.t1_ref.is_alive():
                print(f"{self.device_name}: 等待电机控制器线程完成...")
                self.pos_ctrl.t1_ref.join(timeout=5.0)
                if self.pos_ctrl.t1_ref.is_alive():
                    print(f"{self.device_name}: 电机控制器线程未能及时结束。")
                else:
                    print(f"{self.device_name}: 电机控制器线程已结束。")


class DualArmMotorController:
    def __init__(self):
        rospy.init_node('dual_arm_motor_controller', anonymous=True)

        # =================================================================
        # 🔧 USB设备固定映射配置 - 解决ttyUSB序号变化问题
        # =================================================================

        # 使用by-path路径（基于物理USB端口位置，永远不变）
        device_config = {
            'left_device': {
                'serial_port': '/dev/serial/by-path/pci-0000:00:14.0-usb-0:2.1:1.0-port0',  # USB端口1.1 -> ttyUSB0
                'robot_ip': '169.254.128.19',
                'description': 'Left Device (USB端口1.1) - 自定义限位参数'
            },
            'right_device': {
                'serial_port': '/dev/serial/by-path/pci-0000:00:14.0-usb-0:2.3:1.0-port0',  # USB端口1.3 -> ttyUSB1
                'robot_ip': '169.254.128.18',
                'description': 'Right Device (USB端口1.3) - 默认限位参数'
            }
        }

        # 备用配置（如果by-path不可用，自动回退到传统方式）
        fallback_config = {
            'left_device': {'serial_port': '/dev/ttyUSB0', 'robot_ip': '169.254.128.19'},
            'right_device': {'serial_port': '/dev/ttyUSB1', 'robot_ip': '169.254.128.18'}
        }

        # 检查by-path路径是否存在
        import os
        use_by_path = all(os.path.exists(device_config[dev]['serial_port']) for dev in device_config)

        if use_by_path:
            print("✅ 使用by-path固定路径配置（推荐）")
            final_config = device_config
        else:
            print("⚠️  by-path路径不可用，使用传统配置")
            final_config = fallback_config

        # 创建两个机械臂和电机控制器
        # Left Device：使用自定义限位参数
        self.left_controller = ArmMotorController(
            "Left Device (USB端口1.1)",
            final_config['left_device']['robot_ip'], 8080,
            final_config['left_device']['serial_port'],  # 使用固定路径
            scale_factor_arm=3.0,
            max_delta_arm=0.1,
            motor_step=100,
            thread_mode=rm_thread_mode_e.RM_TRIPLE_MODE_E,
            motor_limits=LEFT_DEVICE_MOTOR_LIMITS,
            is_left_device=True
            
        )

        # Right Device：使用默认限位参数
        self.right_controller = ArmMotorController(
            "Right Device (USB端口1.3)",
            final_config['right_device']['robot_ip'], 8080,
            final_config['right_device']['serial_port'],  # 使用固定路径
            scale_factor_arm=3.0,
            max_delta_arm=0.1,
            motor_step=100,
            thread_mode=None,
            motor_limits=None,  # 使用默认限位参数
            is_left_device=False
        )

        self.step = 8192
        self.stop = False
        self.q_pressed_for_exit = False

        # 订阅左设备话题
        rospy.Subscriber('/left_device/phantom/pose', PoseStamped, self.left_controller.phantom_pose_callback)
        rospy.Subscriber('/left_device/phantom/joint_states', JointState, self.left_controller.phantom_joint_state_callback)
        rospy.Subscriber('/left_device/phantom/button', OmniButtonEvent, self.left_controller.phantom_button_callback)

        # 订阅右设备话题
        rospy.Subscriber('/right_device/phantom/pose', PoseStamped, self.right_controller.phantom_pose_callback)
        rospy.Subscriber('/right_device/phantom/joint_states', JointState, self.right_controller.phantom_joint_state_callback)
        rospy.Subscriber('/right_device/phantom/button', OmniButtonEvent, self.right_controller.phantom_button_callback)

        self.ros_thread = threading.Thread(target=rospy.spin)
        self.ros_thread.daemon = True
        self.ros_thread.start()

        pygame.init()
        pygame.display.set_mode((400, 300))
        pygame.display.set_caption("Dual Arm Motor Control")

        print("================================================================")
        print("🤖 双设备系统已初始化")
        print("================================================================")
        print("📍 设备映射配置：")
        print(f"   Left Device:  {final_config['left_device']['serial_port']}")
        print(f"                 机械臂IP: {final_config['left_device']['robot_ip']}")
        print(f"   Right Device: {final_config['right_device']['serial_port']}")
        print(f"                 机械臂IP: {final_config['right_device']['robot_ip']}")
        print("================================================================")
        print("🎮 控制说明：")
        print("左设备控制：")
        print("  🤏 机械臂控制：按下灰色按钮")
        print("  ⚙️  电机控制：roll/pitch控制电机0&1，白色按钮状态切换电机2&3")
        print("    ⚪ 按一下白色按钮 → 电机2&3状态切换（0↔1）")
        print("    📍 状态0：电机2&3在零点")
        print("    📍 状态1：电机2正极限，电机3负极限（与右设备相反运动）")
        print("  ⌨️  键盘控制：数字键1-4控制左设备电机2&3")
        print("右设备控制：")
        print("  🤏 机械臂控制：按下灰色按钮")
        print("  ⚙️  电机控制：roll/pitch控制电机0&1，白色按钮状态切换电机2&3")
        print("    ⚪ 按一下白色按钮 → 电机2&3状态切换（0↔1）")
        print("    📍 状态0：电机2&3在零点")
        print("    📍 状态1：电机2负极限，电机3正极限（标准运动）")
        print("  ⌨️  键盘控制：数字键5-8控制右设备电机2&3")
        print("  按Q键重置所有电机到零点并退出")
        print("================================================================")
        print("🛡️  防误触机制：")
        print("   - 灰色按钮控制机械臂时，白色按钮电机控制被禁用")
        print("   - 系统会打印警告信息以提醒用户")
        print("================================================================")
        print("🔧 电机限位配置：")
        print(f"   Left Device (USB端口1.1): {LEFT_DEVICE_MOTOR_LIMITS}")
        print(f"   Right Device (USB端口1.3): 默认限位参数")
        print("================================================================")

        self.run()

    def on_key_event_pygame(self, event):
        key_name = pygame.key.name(event.key)
        changed_target = False

        # 左设备控制键 (数字键盘1-4)
        if key_name == '1':
            self.left_controller.motor_target_positions[2] += self.step
            self.left_controller.motor_target_positions[3] += self.step
            changed_target = True
        elif key_name == '2':
            self.left_controller.motor_target_positions[2] -= self.step
            self.left_controller.motor_target_positions[3] -= self.step
            changed_target = True
        elif key_name == '3':
            self.left_controller.motor_target_positions[2] -= self.step
            self.left_controller.motor_target_positions[3] += self.step
            changed_target = True
        elif key_name == '4':
            self.left_controller.motor_target_positions[2] += self.step
            self.left_controller.motor_target_positions[3] -= self.step
            changed_target = True

        # 右设备控制键 (数字键盘5-8)
        elif key_name == '5':
            self.right_controller.motor_target_positions[2] += self.step
            self.right_controller.motor_target_positions[3] += self.step
            changed_target = True
        elif key_name == '6':
            self.right_controller.motor_target_positions[2] -= self.step
            self.right_controller.motor_target_positions[3] -= self.step
            changed_target = True
        elif key_name == '7':
            self.right_controller.motor_target_positions[2] -= self.step
            self.right_controller.motor_target_positions[3] += self.step
            changed_target = True
        elif key_name == '8':
            self.right_controller.motor_target_positions[2] += self.step
            self.right_controller.motor_target_positions[3] -= self.step
            changed_target = True

        elif key_name == 'q':
            print("按下 Q，准备重置所有电机到零点并退出...")
            self.left_controller.reset_motors_to_zero()
            self.right_controller.reset_motors_to_zero()

            print("准备关闭双设备控制系统...")
            self.q_pressed_for_exit = True
            self.stop = True
            changed_target = False

        if changed_target:
            print(f"左设备电机目标位置更新为: {self.left_controller.motor_target_positions}")
            print(f"右设备电机目标位置更新为: {self.right_controller.motor_target_positions}")

    def run(self):
        try:
            main_loop_rate = rospy.Rate(200)  # 200Hz控制频率

            while not self.stop and not rospy.is_shutdown():
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        print("Pygame QUIT 事件收到。")
                        self.stop = True
                    elif event.type == pygame.KEYDOWN:
                        self.on_key_event_pygame(event)

                if self.stop:
                    break

                # 更新两个控制器
                self.left_controller.update()
                self.right_controller.update()

                main_loop_rate.sleep()

        except KeyboardInterrupt:
            print("DualArmMotorController: 检测到 KeyboardInterrupt (Ctrl-C)。正在关闭...")
            self.stop = True
        finally:
            print("DualArmMotorController: 进入 finally 清理块。")
            self.cleanup()

    def cleanup(self):
        print("DualArmMotorController: 开始清理资源...")
        self.stop = True

        # 清理左控制器
        if hasattr(self, 'left_controller'):
            self.left_controller.cleanup()

        # 清理右控制器
        if hasattr(self, 'right_controller'):
            self.right_controller.cleanup()

        # 断开所有机械臂连接，销毁线程
        try:
            RoboticArm.rm_destroy()
            print("DualArmMotorController: 所有机械臂连接已断开，线程已销毁。")
        except Exception as e:
            print(f"DualArmMotorController: 销毁机械臂连接时出错: {e}")

        if rospy and not rospy.is_shutdown():
            print("DualArmMotorController: 请求 ROS 关闭...")
            rospy.signal_shutdown("DualArmMotorController 正常退出")

        if hasattr(self, 'ros_thread') and self.ros_thread.is_alive():
            self.ros_thread.join(timeout=1.0)

        pygame.quit()
        print("DualArmMotorController: Pygame 已退出。双机械臂电机控制程序结束。")


if __name__ == "__main__":
    try:
        controller = DualArmMotorController()
    except Exception as e:
        print(f"主程序启动 DualArmMotorController 时发生严重错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("dual_arm_motor_control.py 程序执行完毕。")