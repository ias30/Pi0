# Realman π₀ 实时推理脚本使用说明

## 📋 概述

`realman_inference.py` 是一个用于 Realman 双臂机器人的实时推理脚本。它使用训练好的 π₀ 模型进行在线推理，采用滚动规划策略（预测 H 步，执行 k 步，然后重新规划）。

## 🔧 系统要求

### 硬件要求
- Realman 双臂机器人
  - 左臂：IP 169.254.128.18
  - 右臂：IP 169.254.128.19
- 3个相机
  - camera_high (index=0): 全局相机
  - camera_left_wrist (index=6): 左腕相机
  - camera_right_wrist (index=8): 右腕相机

### 软件依赖
```bash
# 主要依赖（应该已经安装）
- JAX
- OpenPI
- LeRobot
- OpenCV (cv2)
- PIL
- numpy
- pandas

# 机器人控制库
- Robotic_Arm (Realman SDK)
```

## 📂 文件结构

```
examples/
├── realman_inference.py          # 主推理脚本
└── REALMAN_INFERENCE_README.md   # 本文档

data_collection_pi0/              # 数据采集模块（需要）
├── camera_collector.py
├── Robotic_Arm/
└── ...

checkpoints/                      # 模型检查点目录
└── pi0_realman/
    └── your_checkpoint/
        └── 14999/

assets/                           # 归一化统计数据
└── pi0_realman/
    └── realman_dataset/
        └── norm_stats.json
```

## 🚀 使用方法

### 基本用法

```bash
python examples/realman_inference.py \
    --checkpoint checkpoints/pi0_realman/realman_finetune_v1/14999 \
    --norm-stats assets/pi0_realman/realman_dataset/norm_stats.json \
    --output inference_actions.csv
```

### 完整参数说明

```bash
python examples/realman_inference.py \
    --config-name pi0_realman \                    # 配置名称（默认：pi0_realman）
    --checkpoint <path/to/checkpoint> \            # 模型检查点路径（必需）
    --norm-stats <path/to/norm_stats.json> \       # 归一化统计文件路径
    --output inference_actions.csv \               # 输出CSV文件路径
    --speed 20 \                                   # 机器人速度 (1-100，默认：20)
    --steps-to-execute 3                           # 每次执行的步数（默认：3）
```

### 参数详解

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--config-name` | str | `pi0_realman` | 训练配置名称 |
| `--checkpoint` | str | **必需** | 模型检查点目录路径 |
| `--norm-stats` | str | `assets/pi0_realman/realman_dataset/norm_stats.json` | 归一化统计文件 |
| `--output` | str | `inference_actions.csv` | 保存执行动作的CSV文件 |
| `--speed` | int | 20 | 机器人运动速度（1-100） |
| `--steps-to-execute` | int | 3 | 每次重规划前执行的动作步数 |

## 📊 数据流程

### 1. 输入数据处理

```
采集数据 → 预处理 → 模型输入

相机图像：
BGR(480,640,3) → RGB → resize_with_pad(224,224) → CHW(3,224,224) → 归一化[-1,1]

关节角度：
deg(12,) → rad(12,) → 归一化 → state(12,)
  ├─ [0:6]: 右臂 (169.254.128.19)
  └─ [6:12]: 左臂 (169.254.128.18)
```

### 2. 模型推理

```
输入: {state: (12,), images: dict, prompt: str}
     ↓
  π₀ 模型
     ↓
输出: actions (action_horizon, 12)  [归一化的 rad]
```

### 3. 输出动作处理

```
模型输出 → 后处理 → 执行

模型输出: delta_actions(H, 12) [归一化的 delta]
     ↓
反归一化: delta_actions(H, 12) [未归一化的 delta, rad]
     ↓
Delta → 绝对角度: absolute = delta + current_state
     ↓
absolute_actions(H, 12) [rad]
     ↓
转换单位: deg(H, 12)
     ↓
执行前 k 步 (k=3)
     ├─ [0:6] → 右臂 (169.254.128.19)
     └─ [6:12] → 左臂 (169.254.128.18)
```

**⚠️ 重要**：模型输出的是 **delta actions**（相对于当前 state 的变化量），而不是绝对角度。在推理时必须将 delta 加上当前 state 才能得到目标绝对角度。这一转换由 `AbsoluteActions` transform 自动完成，但前提是必须同时传入 `state` 和 `actions`。

## 💾 输出文件格式

### CSV 文件结构

执行的动作会保存到指定的 CSV 文件中，格式如下：

```csv
joint_0_right_waist,joint_1_right_shoulder,joint_2_right_elbow,joint_3_right_forearm_roll,joint_4_right_wrist_angle,joint_5_right_wrist_rotate,joint_6_left_waist,joint_7_left_shoulder,joint_8_left_elbow,joint_9_left_forearm_roll,joint_10_left_wrist_angle,joint_11_left_wrist_rotate,timestamp
80.212,40.867,66.459,-45.413,-17.694,0.480,-79.574,-44.629,-71.258,70.081,14.242,42.442,1699123456.789
...
```

**列说明**：
- 列 0-5: 右臂关节角度（度）- 对应 169.254.128.19
- 列 6-11: 左臂关节角度（度）- 对应 169.254.128.18
- 列 12: Unix 时间戳

## 🎮 运行时控制

### 启动流程

1. **系统初始化**
   - 加载模型和配置
   - 连接机器人（左臂 + 右臂）
   - 初始化3个相机

2. **移动到初始位置**
   ```
   右臂: [80.212, 40.867, 66.459, -45.413, -17.694, 0.480]
   左臂: [-79.574, -44.629, -71.258, 70.081, 14.242, 42.442]
   ```

3. **进入推理循环**
   - 采集当前状态（关节角度 + 相机图像）
   - 运行模型推理
   - 执行前 k 步动作
   - 保存执行的动作到 CSV
   - 显示相机画面

### 停止方式

- **正常停止**: 按 `Ctrl+C`
- 系统会自动执行清理：
  - 停止相机
  - 断开机器人连接
  - 关闭可视化窗口
  - 保存所有数据

## 📺 可视化界面

脚本运行时会显示一个窗口，实时显示3个相机的画面：

```
+----------------+----------------+----------------+
|   High Camera  | Left Wrist Cam | Right Wrist Cam|
|    (全局相机)   |   (左腕相机)    |   (右腕相机)    |
+----------------+----------------+----------------+
```

## ⚙️ 核心参数调优

### 速度（`--speed`）

- **范围**: 1-100
- **推荐值**:
  - 开发测试: 10-20（慢速，安全）
  - 正常推理: 20-40（中速）
  - 快速执行: 50-80（快速，需确保安全）

### 执行步数（`--steps-to-execute`）

- **作用**: 控制重规划频率
- **权衡**:
  - **k 小** (例如 1-3): 重规划频繁，响应快，但推理开销大
  - **k 大** (例如 5-10): 推理开销小，但响应慢
- **推荐值**: 3（平衡性能和响应）

## 🔍 故障排查

### 常见问题

#### 1. 无法连接机器人

**错误信息**: `Failed to connect to left/right arm`

**解决方案**:
```bash
# 检查网络连接
ping 169.254.128.18
ping 169.254.128.19

# 检查机器人电源和网线
# 确保 IP 地址配置正确
```

#### 2. 相机连接失败

**错误信息**: `Some cameras failed to connect`

**解决方案**:
```bash
# 检查相机索引
ls /dev/video*

# 确认相机可用
v4l2-ctl --list-devices

# 检查相机权限
sudo chmod 666 /dev/video*
```

#### 3. 模型加载失败

**错误信息**: `FileNotFoundError` 或 `Failed to load checkpoint`

**解决方案**:
```bash
# 检查检查点路径
ls -la checkpoints/pi0_realman/your_checkpoint/14999/

# 检查归一化统计文件
ls -la assets/pi0_realman/realman_dataset/norm_stats.json

# 确保路径正确，没有拼写错误
```

#### 4. 动作异常

**症状**: 机器人动作不符合预期

**可能原因**:
- 归一化统计文件不匹配
- 检查点版本不对
- 初始位置不正确

**解决方案**:
```bash
# 1. 确认使用正确的 norm_stats.json（与训练数据集对应）
# 2. 确认检查点与配置匹配
# 3. 检查初始位置设置是否正确
```

## 📝 示例运行日志

```
================================================================================
🚀 Initializing Realman π₀ Real-time Inference System
================================================================================
INFO:__main__:Loading config: pi0_realman
INFO:__main__:Initializing model: pi0
INFO:__main__:Loading checkpoint: checkpoints/pi0_realman/realman_finetune_v1/14999
INFO:__main__:Loading normalization stats: assets/pi0_realman/realman_dataset/norm_stats.json
INFO:__main__:🤖 Initializing robot arms...
INFO:__main__:✅ Left arm connected: 169.254.128.18
INFO:__main__:✅ Right arm connected: 169.254.128.19
INFO:__main__:🦾 Moving to initial positions...
INFO:__main__:  ✅ Right arm at initial position
INFO:__main__:  ✅ Left arm at initial position
INFO:__main__:📷 Initializing cameras...
INFO:__main__:✅ Cameras initialized
INFO:__main__:📝 CSV file prepared: inference_actions.csv
INFO:__main__:✅ Initialization complete!
================================================================================
🎮 Starting inference loop
  Action horizon: 10
  Steps to execute: 3
  Robot speed: 20
  Output CSV: inference_actions.csv
  Press Ctrl+C to stop
================================================================================

============================================================
Iteration 1
============================================================
INFO:__main__:📊 Getting current state...
INFO:__main__:🧠 Running inference...
INFO:__main__:  Inference took 0.234s
INFO:__main__:  Predicted 10 steps
INFO:__main__:🎯 Executing first 3 steps...
INFO:__main__:  Step 1/3: Executing action...
INFO:__main__:    ✅ Executed in 1.234s
INFO:__main__:  Step 2/3: Executing action...
INFO:__main__:    ✅ Executed in 1.198s
INFO:__main__:  Step 3/3: Executing action...
INFO:__main__:    ✅ Executed in 1.205s

INFO:__main__:📈 Iteration 1 complete:
INFO:__main__:  Total time: 3.901s
INFO:__main__:  Inference time: 0.234s
INFO:__main__:  Execution time: 3.667s
INFO:__main__:  Effective frequency: 0.26 Hz

...
```

## 🔬 技术细节

### Delta Actions 处理 ⚠️ **关键**

模型训练时使用的是 **delta actions**（`use_delta_joint_actions=True`），这意味着：

1. **训练时**：
   ```python
   # 数据预处理 (DeltaActions transform)
   delta_action = absolute_action - current_state
   # 模型学习预测 delta
   ```

2. **推理时**：
   ```python
   # 模型输出 delta
   delta_actions = model(observation)
   
   # 必须转换回绝对角度 (AbsoluteActions transform)
   absolute_actions = delta_actions + current_state
   
   # ❌ 错误做法：
   denorm_dict = {"actions": predicted_actions}
   # 这样会导致只反归一化 delta，但不会加上 current_state！
   
   # ✅ 正确做法：
   denorm_dict = {
       "actions": predicted_actions,
       "state": current_state  # 必须传入！
   }
   # AbsoluteActions transform 会自动执行: actions += state
   ```

3. **为什么这很重要**：
   - 如果忘记传入 `state`，得到的将是 delta 值而不是绝对角度
   - 执行 delta 值会导致机器人行为异常（幅值偏差/漂移）
   - 累积误差会导致机器人越来越偏离预期轨迹

4. **代码中的实现**：
   ```python
   # realman_inference.py line 376-383
   denorm_dict = {
       "actions": predicted_actions,
       "state": transformed["state"]  # 关键：传入当前状态
   }
   denorm_result = self.output_transform(denorm_dict)
   ```

### 初始位置配置

初始位置在代码中硬编码，如需修改，请编辑 `realman_inference.py` 中的：

```python
self.initial_right_angles = [80.212, 40.867, 66.459, -45.413, -17.694, 0.480]
self.initial_left_angles = [-79.5739974975586, -44.62900161743164, -71.25800323486328, 
                            70.08100128173828, 14.241999626159668, 42.44200134277344]
```

### 固定 Prompt

当前使用固定的语言指令：
```python
self.prompt = "Let the forceps go along the black S shaped path"
```

如需修改，请编辑代码中的 `self.prompt` 变量。

### 阻塞执行模式

当前使用阻塞执行模式（`rm_movej` 的最后一个参数为 `True`），即等待每个动作完成后再执行下一个。

如需修改为非阻塞模式，请编辑 `_execute_action` 方法：

```python
# 阻塞模式（当前）
self.right_arm.rm_movej(right_action_deg, self.speed, 0, 0, True)

# 非阻塞模式
self.right_arm.rm_movej(right_action_deg, self.speed, 0, 0, False)
```

## 📚 相关文档

- [OpenPI 训练文档](../../docs/training.md)
- [数据采集说明](../../data_collection_pi0/README.md)
- [离线推理脚本](offline_inference.py)
- [数据转换脚本](aloha_real/convert_realman_data_to_lerobot.py)

## 🤝 贡献与支持

如遇到问题或有改进建议，请：
1. 检查本文档的故障排查部分
2. 查看相关日志输出
3. 提交 Issue 或 Pull Request

## 📄 许可证

与 OpenPI 主项目保持一致。

