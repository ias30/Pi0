# 数据采集系统文档索引

## 📋 文档导航

### 🚀 快速开始
- **[ARM_SWAP_QUICK_GUIDE.md](ARM_SWAP_QUICK_GUIDE.md)** - 机械臂映射配置快速指南（推荐先看）
- **[check_arm_mapping.py](check_arm_mapping.py)** - 验证工具：检查Episode文件的映射配置

### 📖 详细文档
- **[ARM_SWAP_CONFIGURATION.md](ARM_SWAP_CONFIGURATION.md)** - 机械臂映射配置完整说明
- **[FINAL_SUMMARY_ARM_SWAP.md](FINAL_SUMMARY_ARM_SWAP.md)** - 映射功能修改总结

### 🔧 工具脚本
- **[camera_collector.py](camera_collector.py)** - 相机数据采集（可单独测试）
- **[check_arm_mapping.py](check_arm_mapping.py)** - 映射配置检查工具
- **[dual_arm_with_data_collection.py](dual_arm_with_data_collection.py)** - 完整系统主程序

## 🎯 当前系统配置

### 相机配置（三相机）
- **camera_high**: RealSense (RGB + Depth) - 序列号 031522071209
- **camera_left_wrist**: 手腕相机 (RGB only) - /dev/video2
- **camera_right_wrist**: 手腕相机 (RGB only) - /dev/video4

### 机械臂映射（交换模式，默认）
```
物理设备                      →  数据标签
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
left_device  (169.254.128.18)  →  right_arm + right_motors
right_device (169.254.128.19)  →  left_arm  + left_motors
```

### 采集频率
- 主采集循环: 30 Hz
- 机械臂采集: 60 Hz（后台线程）
- 相机采集: 30 Hz（独立线程）
- 电机采集: 30 Hz（独立线程）

## 🚀 快速操作

### 运行完整系统
```bash
python dual_arm_with_data_collection.py
```

**控制键：**
- `R` - 开始录制
- `C` - 停止录制
- `S` - 显示状态
- `Backspace` - 删除最近的Episode
- `Q` - 退出系统

### 测试相机采集
```bash
python camera_collector.py
```

### 检查Episode映射
```bash
python check_arm_mapping.py <episode文件.h5>
```

## 📊 数据结构

```
episode_YYYYMMDD_HHMMSS.h5
├── observations/
│   ├── global_timestamps         [N] float64
│   ├── cameras/
│   │   ├── camera_high/          (RealSense)
│   │   │   ├── color             [N, 480, 640, 3] uint8
│   │   │   ├── depth             [N, 480, 640] uint16
│   │   │   └── local_timestamps  [N] float64
│   │   ├── camera_left_wrist/    (手腕相机)
│   │   │   ├── color             [N, 480, 640, 3] uint8
│   │   │   └── local_timestamps  [N] float64
│   │   └── camera_right_wrist/   (手腕相机)
│   │       ├── color             [N, 480, 640, 3] uint8
│   │       └── local_timestamps  [N] float64
│   ├── arms/
│   │   ├── left_arm/
│   │   │   ├── joint_positions      [N, 6] float64
│   │   │   ├── end_effector_poses   [N, 6] float64
│   │   │   └── local_timestamps     [N] float64
│   │   └── right_arm/ (同left_arm结构)
│   └── motors/
│       ├── left_motors/
│       │   ├── positions         [N, 4] int32
│       │   ├── states            [N, 4] int32
│       │   └── local_timestamps  [N] float64
│       └── right_motors/ (同left_motors结构)
└── metadata/
    ├── swap_arms                 (bool) 映射模式
    ├── device_mapping_description (string) 描述
    ├── physical_left_device_ip   (string) 169.254.128.18
    ├── physical_right_device_ip  (string) 169.254.128.19
    ├── version                   (string) 4.0_triple_camera
    ├── episode_start_time        (float64)
    ├── episode_end_time          (float64)
    └── observation_count         (int)
```

## 🔧 修改配置

### 切换机械臂映射模式

编辑 `dual_arm_with_data_collection.py`（约134行）：
```python
# 交换模式（默认）
self.data_collector = DataCollector("data_collection_episodes", swap_arms=True)

# 直接模式
self.data_collector = DataCollector("data_collection_episodes", swap_arms=False)
```

### 修改相机设备

编辑 `camera_collector.py` 中的 `TripleCameraCollector.__init__()`：
```python
# 手腕相机设备索引
self.camera_left_wrist = FFmpegWristCamera("camera_left_wrist", 
                                           device_index=2)  # 修改这里
self.camera_right_wrist = FFmpegWristCamera("camera_right_wrist", 
                                            device_index=4)  # 修改这里
```

## 📖 Python读取数据示例

```python
import h5py
import numpy as np

# 打开文件
with h5py.File('episode_20251020_120000.h5', 'r') as f:
    # 读取映射配置
    metadata = f['metadata']
    swap_arms = metadata.attrs['swap_arms']
    print(f"映射模式: {'交换' if swap_arms else '直接'}")
    
    # 读取数据（数据结构始终一致）
    obs = f['observations']
    
    # 相机数据
    camera_high_color = obs['cameras']['camera_high']['color'][:]  # [N, 480, 640, 3]
    camera_high_depth = obs['cameras']['camera_high']['depth'][:]  # [N, 480, 640]
    camera_left_wrist = obs['cameras']['camera_left_wrist']['color'][:]  # [N, 480, 640, 3]
    camera_right_wrist = obs['cameras']['camera_right_wrist']['color'][:]  # [N, 480, 640, 3]
    
    # 机械臂数据
    left_arm_joints = obs['arms']['left_arm']['joint_positions'][:]  # [N, 6]
    left_arm_poses = obs['arms']['left_arm']['end_effector_poses'][:]  # [N, 6]
    right_arm_joints = obs['arms']['right_arm']['joint_positions'][:]  # [N, 6]
    right_arm_poses = obs['arms']['right_arm']['end_effector_poses'][:]  # [N, 6]
    
    # 电机数据
    left_motors = obs['motors']['left_motors']['positions'][:]  # [N, 4]
    right_motors = obs['motors']['right_motors']['positions'][:]  # [N, 4]
    
    # 时间戳
    global_timestamps = obs['global_timestamps'][:]  # [N]
    left_arm_timestamps = obs['arms']['left_arm']['local_timestamps'][:]  # [N]
    
    print(f"采集了 {len(global_timestamps)} 帧数据")
```

## ⚠️ 重要提醒

1. **映射配置在初始化时设定**，录制期间不能更改
2. **数据标签始终一致**，分析代码无需根据映射模式调整
3. **依赖metadata判断映射**，不要依赖文件名
4. **相机数据不受映射影响**，始终按物理位置记录

## 🐛 故障排查

### 相机连接失败
```bash
# 查看可用设备
v4l2-ctl --list-devices

# 测试FFmpeg采集
ffmpeg -f v4l2 -input_format mjpeg -framerate 30 \
       -video_size 640x480 -i /dev/video2 \
       -f rawvideo -pix_fmt bgr24 -frames 10 /dev/null
```

### 机械臂连接失败
- 检查IP地址是否正确（169.254.128.18 和 169.254.128.19）
- 确认网络连接正常
- 查看日志中的详细错误信息

### 数据采集频率低
- 检查CPU负载
- 确认磁盘I/O性能
- 查看错误日志

## 📞 技术支持

- 遇到问题？先查看对应的详细文档
- 运行 `python camera_collector.py` 单独测试相机
- 运行 `python check_arm_mapping.py` 检查映射配置
- 查看系统输出的日志信息

## 🔧 最近修复

### Wrist 相机帧重复问题修复（2025-10-23）
**问题**：wrist 相机出现超过 80% 的帧重复，导致回放时严重不同步

**根本原因**：OpenCV 的 `VideoCapture.read()` 从内部缓冲区读取帧，当读取速度慢于采集速度时会累积旧帧

**解决方案**：
1. 设置 OpenCV 缓冲区大小为 1 (`CAP_PROP_BUFFERSIZE=1`)
2. 每次读取时主动清空缓冲区（双重读取，丢弃旧帧）
3. 添加线程安全的数据缓存机制 (`get_latest_data()` 方法)
4. 使用 MJPG 格式提高性能

**测试工具**：
```bash
# 测试帧重复修复效果
python test_code/test_frame_duplication_fix.py
```

**预期效果**：wrist 相机帧重复率从 >80% 降至 <5%

## 📝 版本信息

- **当前版本**: 4.1_triple_camera_frame_fix
- **最后更新**: 2025-10-23
- **主要特性**:
  - ✅ 三相机数据采集（1个RealSense + 2个手腕相机）
  - ✅ 灵活的机械臂映射配置
  - ✅ 30Hz高频数据采集
  - ✅ 完整的metadata记录
  - ✅ 统一的数据结构
  - ✅ **修复 wrist 相机帧重复问题**

---

**🎉 系统已准备就绪，开始采集数据吧！**

