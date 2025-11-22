# 🔧 Tensor 转换修复

## 问题

运行对比工具时遇到错误：
```
AttributeError: 'Tensor' object has no attribute 'astype'. Did you mean: 'dtype'?
```

## 原因

从 LeRobot 数据集加载的数据是 **PyTorch Tensor** 对象，而不是 numpy array。在 `chw_to_rgb_uint8` 函数中直接使用了 numpy 的 `.astype()` 方法导致错误。

## 修复内容

在 `examples/compare_offline_realtime_observation.py` 的 `create_realtime_frame()` 函数中添加了 Tensor 到 numpy array 的转换：

### 1. State 转换（第216-224行）
```python
# Extract state (should be in radians)
# Convert to numpy if it's a Tensor
state = offline_frame["observation.state"]
if hasattr(state, 'numpy'):
    state = state.numpy()
elif hasattr(state, '__array__'):
    state = np.array(state)
else:
    state = np.asarray(state)
```

### 2. Image 转换（第233-244行）
```python
def chw_to_rgb_uint8(img_chw):
    # Convert to numpy if it's a Tensor
    if hasattr(img_chw, 'numpy'):
        img_chw = img_chw.numpy()
    elif hasattr(img_chw, '__array__'):
        img_chw = np.array(img_chw)
    else:
        img_chw = np.asarray(img_chw)
    
    img_hwc = np.transpose(img_chw, (1, 2, 0))  # CHW -> HWC
    img_denorm = (img_hwc + 1.0) * 127.5  # [-1, 1] -> [0, 255]
    return np.clip(img_denorm, 0, 255).astype(np.uint8)
```

## 转换逻辑

使用三层检查确保兼容性：
1. `hasattr(x, 'numpy')` - PyTorch Tensor 有 `.numpy()` 方法
2. `hasattr(x, '__array__')` - 实现了 array 协议的对象
3. `np.asarray()` - 通用 numpy 转换

## 验证

现在可以正常运行对比工具：
```bash
python examples/compare_offline_realtime_observation.py \
    --config-name pi0_realman \
    --checkpoint checkpoints/pi0_realman/realman_finetune_v1/14999 \
    --dataset-path ~/.cache/huggingface/lerobot/realman_dataset \
    --norm-stats assets/pi0_realman/realman_dataset/ \
    --episode 0 \
    --frame-index 10
```

## 状态

✅ **已修复** - 对比工具现在可以正确处理 PyTorch Tensor 输入

---

**修复日期**：2025-11-17  
**相关问题**：AttributeError with Tensor.astype()



