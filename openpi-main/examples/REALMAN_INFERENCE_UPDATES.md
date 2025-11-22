# Realman Inference 更新说明

## 🔧 关键修复（2025-01-XX）

### 问题 1: Delta Actions 处理错误 ✅ 已修复

**问题描述**：
- 模型训练时使用 `use_delta_joint_actions=True`，输出的是相对于当前 state 的变化量（delta）
- 原代码在调用 `output_transform` 时只传入了 `actions`，没有传入 `state`
- 导致 `AbsoluteActions` transform 无法正确将 delta 转换为绝对角度
- 结果：机器人执行的是 delta 值而不是绝对角度，导致幅值偏差/漂移

**修复方法**：
```python
# ❌ 错误（旧代码）
denorm_dict = {"actions": predicted_actions}
denorm_result = self.output_transform(denorm_dict)

# ✅ 正确（新代码）
denorm_dict = {
    "actions": predicted_actions,
    "state": transformed["state"]  # 必须传入当前状态！
}
denorm_result = self.output_transform(denorm_dict)
```

### 问题 2: 模型加载方式错误 ✅ 已修复

**问题描述**：
- 原代码使用 `model.build_model(rng)` + `checkpoint_loader.load(model)`
- 这种方式在新版本中不再支持，会导致类型错误

**修复方法**：
```python
# ❌ 错误（旧代码）
model = train_config.model.build_model(rng)
checkpoint_loader = weight_loaders.CheckpointWeightLoader(str(checkpoint_path))
model = checkpoint_loader.load(model)

# ✅ 正确（新代码）
loaded_params = _model.restore_params(checkpoint_path, restore_type=np.ndarray)
model = train_config.model.load(loaded_params)
```

### 问题 3: 模型推理接口错误 ✅ 已修复

**问题描述**：
- 原代码直接调用 `model(observation)`
- Pi0 模型需要使用 `model.sample_actions()` 方法进行推理

**修复方法**：
```python
# ❌ 错误（旧代码）
predicted_actions = self.model(observation)

# ✅ 正确（新代码）
self.rng, inference_rng = jax.random.split(self.rng)
predicted_actions = self.model.sample_actions(
    inference_rng,
    observation,
    num_steps=10  # 扩散模型采样步数
)
```

### 问题 4: Transform 名称错误 ✅ 已修复

**问题描述**：
- 使用了 `Denormalize` 而不是正确的 `Unnormalize`

**修复方法**：
```python
# ❌ 错误（旧代码）
_transforms.Denormalize(self.norm_stats, ...)

# ✅ 正确（新代码）
_transforms.Unnormalize(self.norm_stats, ...)
```

### 问题 5: 数据类型转换 ✅ 已修复

**问题描述**：
- 模型需要 JAX arrays 作为输入
- Transform 需要 NumPy arrays 进行处理

**修复方法**：
```python
# 1. 输入模型前：numpy → JAX
batch = jax.tree.map(lambda x: np.expand_dims(x, axis=0), transformed)
batch = jax.tree.map(lambda x: jnp.asarray(x), batch)  # 转为 JAX array

# 2. 模型输出后：JAX → numpy
state_np = np.array(transformed["state"])
action_np = np.array(predicted_actions[step_idx])
```

## 📝 完整的修改列表

### `realman_inference.py`

1. **导入模块**：
   - ❌ 移除：`import openpi.training.weight_loaders as weight_loaders`
   - ✅ 保留：`import openpi.models.model as _model`

2. **`_load_model` 方法**（第 195-211 行）：
   - 使用 `_model.restore_params()` 加载参数
   - 使用 `train_config.model.load()` 创建模型

3. **`_create_transforms` 方法**（第 222-239 行）：
   - `Denormalize` → `Unnormalize`

4. **`_run_inference` 方法**（第 350-403 行）：
   - 添加 JAX array 转换
   - 使用 `model.sample_actions()` 进行推理
   - 为每个 action step 正确传入 state
   - 处理 2D actions 维度

5. **命令行参数**：
   - Checkpoint 路径：指向 `params` 子目录
   - Norm stats 路径：指向目录而不是文件

### `offline_inference.py`

1. **`load_model_and_config` 函数**：
   - 与 `realman_inference.py` 相同的模型加载方式

2. **`create_transforms` 函数**：
   - `Denormalize` → `Unnormalize`

3. **`run_inference` 函数**：
   - 添加 RNG 管理
   - 添加 JAX array 转换
   - 使用 `model.sample_actions()`
   - 正确处理 state 传递

## 🎯 使用更新后的代码

### 命令行示例

```bash
uv run examples/realman_inference.py \
    --config-name pi0_realman \
    --checkpoint checkpoints/pi0_realman/realman_finetune_v1/14999/params \
    --norm-stats assets/pi0_realman/realman_dataset/ \
    --output inference_actions.csv \
    --speed 20 \
    --steps-to-execute 3
```

**注意路径变化**：
- Checkpoint：`14999` → `14999/params`
- Norm stats：`norm_stats.json` → `realman_dataset/` (目录)

### 验证修复

运行后应该看到：
1. ✅ 模型成功加载（无 AssertionError）
2. ✅ 推理正常运行（使用 `sample_actions`）
3. ✅ 机器人执行正确的绝对角度（不再漂移）

## ⚠️ 重要提醒

### Delta Actions 的数学原理

```
训练时：
  input_state: [s₀, s₁, ..., s₁₁]
  target_action: [a₀, a₁, ..., a₁₁]
  delta_action = target_action - input_state  # DeltaActions transform
  model learns to predict delta_action

推理时：
  current_state: [s₀, s₁, ..., s₁₁]
  predicted_delta = model(observation)
  absolute_action = predicted_delta + current_state  # AbsoluteActions transform
  execute(absolute_action)
```

### 如果忘记传入 state 会怎样？

```python
# ❌ 忘记传入 state
denorm_dict = {"actions": predicted_delta}
result = output_transform(denorm_dict)
# result["actions"] 仍然是 delta，不是绝对角度！

# 执行 delta 值：
execute(delta)  # 机器人会执行错误的动作
# 例如：current = [10°, 20°, ...]，delta = [2°, 3°, ...]
# 应该执行：[12°, 23°, ...]
# 实际执行：[2°, 3°, ...]  ← 错误！
```

## 🔍 调试技巧

如果推理结果仍然异常，检查：

1. **验证 delta 转换**：
   ```python
   # 在 _run_inference 中添加调试输出
   print(f"Current state (rad): {state_np}")
   print(f"Predicted delta (rad): {predicted_actions[0]}")
   print(f"Absolute action (rad): {absolute_actions[0]}")
   ```

2. **检查数值范围**：
   - State: 应该在合理的关节角度范围内（例如 -π 到 π）
   - Delta: 应该是较小的值（例如 -0.5 到 0.5 rad）
   - Absolute: 应该接近 state（state + small delta）

3. **对比训练数据**：
   - 检查 norm_stats.json 中的统计数据
   - 确保推理时使用的归一化参数与训练一致

## 📚 参考资料

- `src/openpi/transforms.py` - DeltaActions 和 AbsoluteActions 实现
- `src/openpi/training/config.py` line 292 - `use_delta_joint_actions` 配置
- `examples/offline_inference.py` - 正确的推理流程参考

---

**更新日期**: 2025-01-XX  
**影响文件**: `realman_inference.py`, `offline_inference.py`  
**测试状态**: ✅ 待验证














