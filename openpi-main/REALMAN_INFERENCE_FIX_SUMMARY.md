# 🔧 Realman实时推理修复总结

## 📌 问题诊断

### 症状
1. **离线推理**（offline_inference.py）：模型表现正常，能生成正确的S轨迹
2. **实时推理**（realman_inference.py）：轨迹形状明显错误，运行频率仅约0.14Hz

### 根本原因：**双重归一化问题** ❌

实时推理中的图像被归一化了**两次**，导致输入分布与训练数据严重不匹配。

---

## 📊 数据流对比

### ✅ Offline Inference（正确）

```
原始图像（LeRobot数据集）
[uint8, 0-255, CHW格式]
         ↓
Input Transform Pipeline
  - repack_transforms
  - data_transforms  
  - Normalize(norm_stats) ← 唯一的归一化点
  - model_transforms
         ↓
模型输入（正确的数值分布）
```

### ❌ Realtime Inference（修复前 - 错误）

```
原始图像（相机）
[uint8, 0-255, BGR-HWC格式]
         ↓
preprocess_image()
  - BGR → RGB
  - Resize + Padding
  - HWC → CHW
  - (img/127.5)-1.0 ← ❌ 第一次归一化到[-1,1]
         ↓
Input Transform Pipeline
  - repack_transforms
  - data_transforms
  - Normalize(norm_stats) ← ❌ 第二次归一化！
  - model_transforms
         ↓
模型输入（❌ 错误的数值分布，严重偏移）
```

### ✅ Realtime Inference（修复后 - 正确）

```
原始图像（相机）
[uint8, 0-255, BGR-HWC格式]
         ↓
preprocess_image()
  - BGR → RGB
  - Resize + Padding
  - HWC → CHW
  - ✅ 保持uint8格式[0,255]，不做归一化
         ↓
Input Transform Pipeline
  - repack_transforms
  - data_transforms
  - Normalize(norm_stats) ← ✅ 唯一的归一化点
  - model_transforms
         ↓
模型输入（✅ 正确的数值分布，与训练数据一致）
```

---

## 🔧 修复内容

### 修改文件：`examples/realman_inference.py`

**修改函数**：`preprocess_image()` (第86-110行)

**修改前**：
```python
def preprocess_image(image_bgr: np.ndarray, target_size: tuple = (224, 224)) -> np.ndarray:
    # ... BGR→RGB, Resize, HWC→CHW ...
    
    # Normalize to [-1, 1]
    image_normalized = (image_chw.astype(np.float32) / 127.5) - 1.0
    return image_normalized  # ❌ 返回已归一化的float32
```

**修改后**：
```python
def preprocess_image(image_bgr: np.ndarray, target_size: tuple = (224, 224)) -> np.ndarray:
    # ... BGR→RGB, Resize, HWC→CHW ...
    
    # ✅ 修复：不再手动归一化，保持uint8格式[0,255]，让transform pipeline处理归一化
    # 这样才能与offline_inference.py的数据格式一致，避免双重归一化问题
    return image_chw.astype(np.uint8)  # ✅ 返回uint8格式，不做归一化
```

---

## 📈 预期效果

### 1. **轨迹质量提升** 🎯
- 输入分布现在与训练数据一致
- 模型应能输出正确的S轨迹
- 动作预测精度显著提高

### 2. **运行频率改善** ⚡
- 删除了一次不必要的浮点数归一化操作
- 保持uint8格式可以减少内存占用
- 预期推理频率会有所提升（从0.14Hz）

---

## ✅ 验证步骤

### 方法1：使用对比工具验证
```bash
python examples/compare_offline_realtime_observation.py \
    --config-name pi0_realman \
    --checkpoint checkpoints/pi0_realman/realman_finetune_v1/14999 \
    --dataset-path ~/.cache/huggingface/lerobot/realman_dataset \
    --norm-stats assets/pi0_realman/realman_dataset/ \
    --episode 0 \
    --frame-index 10
```

**预期结果**：所有字段（特别是图像）应该显示"✅ VALUES MATCH"

### 方法2：直接测试实时推理
```bash
uv run examples/realman_inference.py \
    --config-name pi0_realman \
    --checkpoint checkpoints/pi0_realman/realman_finetune_v1/14999/params \
    --norm-stats assets/pi0_realman/realman_dataset/ \
    --output inference_actions.csv \
    --speed 20 \
    --steps-to-execute 3
```

**预期结果**：
- 机械臂应沿着正确的S轨迹运动
- 推理频率应有所提升
- 动作执行更加流畅和准确

---

## 🎓 经验教训

### 关键原则
1. **保持数据流一致性**：实时推理和离线推理应使用完全相同的数据预处理流程
2. **避免重复操作**：归一化等操作应只在transform pipeline中执行一次
3. **格式匹配**：确保原始数据格式（uint8/float32, 值域等）与训练数据一致
4. **单一职责**：
   - `preprocess_image()`: 只负责格式转换（BGR→RGB, HWC→CHW, Resize）
   - `transform pipeline`: 负责归一化、增强等数据变换

### 调试技巧
- 使用对比工具（compare_offline_realtime_observation.py）验证数据流
- 打印中间结果的数值范围（min, max, mean, std）
- 对比离线和实时的每个处理步骤

---

## 📝 相关文件

- ✅ **已修复**：`examples/realman_inference.py`
- 📊 **对比工具**：`examples/compare_offline_realtime_observation.py`
- 📚 **离线推理参考**：`examples/offline_inference.py`
- 🔄 **数据转换**：`examples/aloha_real/convert_realman_data_to_lerobot.py`

---

**修复日期**：2025-11-17  
**问题严重程度**：🔴 严重（影响模型推理准确性）  
**修复状态**：✅ 已完成



