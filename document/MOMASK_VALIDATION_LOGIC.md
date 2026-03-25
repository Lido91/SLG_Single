# MoMask 验证逻辑详解

## ✅ 是的，所有验证逻辑都已完全实现！

---

## 📋 三个 Stage 的验证方式对比

| Stage | 验证函数 | 验证方式 | 需要的模型 | 评估指标 |
|-------|---------|---------|-----------|---------|
| **Stage 1 (VAE)** | `val_vae_forward()` | 运动重建：`motion → encode → decode → motion'` | 仅 VAE | MRMetrics (MPJPE, etc.) |
| **Stage 2 (MaskTransformer)** | `val_t2m_forward()` | 文本生成运动：`text → Q0 → (Q1-Q2) → motion` | VAE + MaskTransformer + (可选) ResTransformer | TM2TMetrics (FID, etc.) |
| **Stage 3 (ResTransformer)** | `val_res_transformer_forward()` | 运动重建（GT Q0）：`motion → Q0_GT → Q1-Q2_pred → motion'` | VAE + ResTransformer | MRMetrics (MPJPE, etc.) |

---

## 🔍 详细分析

### Stage 2: MaskTransformer 验证

#### 代码路径
```python
# mGPT/models/mgpt_momask.py:545-560
elif self.stage in ["mask_transformer", "inference"]:
    rs_set = self.val_t2m_forward(batch)  # 调用 text-to-motion 验证
```

#### 验证流程 (`val_t2m_forward()`, line 358-410)

```python
@torch.no_grad()
def val_t2m_forward(self, batch):
    """Validate text-to-motion generation."""
    texts = batch["text"]
    lengths = batch["length"]

    # 🎯 核心：调用完整的生成流程
    motion = self.generate(texts, lengths)
    #        ^^^^^^^^^^^^^^
    #        这里会调用 Stage 2 + Stage 3（如果有）

    # 计算评估指标
    # - 比较生成的 motion 和参考 motion
    # - 使用 TM2TMetrics (FID, Diversity, R-Precision, etc.)
```

#### `generate()` 方法的两种模式 (line 163-226)

**模式 A：有 ResidualTransformer (完整生成)**
```python
def generate(self, texts, lengths):
    # Step 1: MaskTransformer 生成 Q0
    q0_ids = self.mask_transformer.generate(texts, m_lens, ...)  # (B, T')

    # Step 2: ResidualTransformer 生成 Q1-Q2
    if self.res_transformer is not None:  # ✅ 如果有 ResTransformer
        all_indices = self.res_transformer.generate(
            motion_ids=q0_ids,  # 用 Q0 作为条件
            conds=texts,
            ...
        )  # (B, T', num_quantizers) - 包含 Q0, Q1, Q2

    # Step 3: VAE 解码
    motion = self.vae.decode(all_indices)  # (B, T, D)
    return motion
```

**模式 B：无 ResidualTransformer (仅用 Q0)**
```python
def generate(self, texts, lengths):
    # Step 1: MaskTransformer 生成 Q0
    q0_ids = self.mask_transformer.generate(texts, m_lens, ...)  # (B, T')

    # Step 2: 如果没有 ResTransformer
    else:  # ❌ res_transformer is None
        all_indices = q0_ids.unsqueeze(-1)  # (B, T', 1)
        #             ^^^^^^^^^^^^^^^^^^^^
        #             仅使用 Q0，添加一个维度

    # Step 3: VAE 解码（仅用 Q0）
    motion = self.vae.decode(all_indices)  # (B, T, D)
    #                         ^^^^^^^^^^^
    #                         shape: (B, T', 1) - 仅 Q0 层
    return motion
```

#### ✅ **"仅用 Q0" 逻辑已完全实现**

**位置**: `mGPT/models/mgpt_momask.py:219-221`
```python
else:
    # If no residual transformer, use Q0 only
    all_indices = q0_ids.unsqueeze(-1)  # (B, T', 1)
```

**效果**:
- VAE 解码时仅接收 Q0 层的 tokens
- 跳过 Q1, Q2 的精细化信息
- 生成质量会较差，但可以工作

---

### Stage 3: ResidualTransformer 验证

#### 代码路径
```python
# mGPT/models/mgpt_momask.py:528-543
elif self.stage == "res_transformer":
    rs_set = self.val_res_transformer_forward(batch)  # 专用的验证方法
```

#### 验证流程 (`val_res_transformer_forward()`, line 413-477)

```python
@torch.no_grad()
def val_res_transformer_forward(self, batch):
    """
    Stage 3 专用验证：使用 GT Q0 验证 ResidualTransformer 的预测能力
    """
    feats_ref = batch["motion"]  # 参考运动
    texts = batch["text"]

    # Step 1: 用 VAE 编码得到 GT tokens（所有层）
    code_idx, _ = self.vae.encode(feats_ref)  # (B, T', num_quantizers)
    #                                          # 包含 GT Q0, Q1, Q2

    # Step 2: 提取 GT Q0
    q0_ids = code_idx[..., 0]  # (B, T') - Ground Truth Q0
    #        ^^^^^^^^^^^^^^^^
    #        这是 VAE 编码的真实 Q0，不是预测的！

    # Step 3: 用 ResTransformer 预测 Q1-Q2（条件于 GT Q0）
    all_indices = self.res_transformer.generate(
        motion_ids=q0_ids,  # 🎯 使用 GT Q0 而非预测 Q0
        conds=texts,
        ...
    )  # (B, T', num_quantizers)
    #  返回 [Q0_GT, Q1_pred, Q2_pred]

    # Step 4: VAE 解码
    motion = self.vae.decode(all_indices)  # (B, T, D)

    # Step 5: 比较生成的 motion 和参考 motion
    # 使用 MRMetrics (MPJPE, MPJPE_PA, etc.)
```

#### 🎯 **Stage 3 验证的核心特点**

1. **不需要 MaskTransformer**
   - 直接从 VAE 编码获得 GT Q0
   - 不进行 text → Q0 的预测

2. **验证目标**
   - 评估 ResidualTransformer 预测 Q1-Q2 的能力
   - 假设 Q0 是完美的（Ground Truth）

3. **评估指标**
   - 使用运动重建指标 (MRMetrics)
   - 不是文本生成指标 (TM2TMetrics)

4. **与原始 MoMask 一致**
   ```python
   # momask-codes/eval_t2m_trans_res.py 中也是这样做的
   # 在评估 Stage 3 时：使用 GT Q0 → 预测 Q1-Q2 → 解码
   ```

---

## 📊 配置文件对应关系

### Stage 2 配置
```yaml
# configs/momask_h2s_stage2.yaml
TRAIN:
  STAGE: mask_transformer

model:
  params:
    stage: "mask_transformer"
    motion_vae: ${vq.h2s_rvq_3}           # ✅ 冻结
    mask_transformer: ${lm.momask_transformer}  # ✅ 训练
    res_transformer: null                 # ❌ 未配置

METRIC:
  TYPE: ['TM2TMetrics']  # Text-to-Motion 评估指标
```

**验证时会发生什么**:
1. 调用 `val_t2m_forward()`
2. `generate()` 内部检测到 `self.res_transformer is None`
3. 执行"仅用 Q0"的分支：`all_indices = q0_ids.unsqueeze(-1)`
4. VAE 仅用 Q0 解码，生成质量较差
5. 仍然可以计算 TM2TMetrics，但分数会较低

---

### Stage 3 配置
```yaml
# configs/momask_h2s_stage3.yaml
TRAIN:
  STAGE: res_transformer

model:
  params:
    stage: "res_transformer"
    motion_vae: ${vq.h2s_rvq_3}           # ✅ 冻结
    mask_transformer: null                # ✅ 不需要
    res_transformer: ${lm.momask_residual}  # ✅ 训练

METRIC:
  TYPE: ['MRMetrics']  # Motion Reconstruction 评估指标
```

**验证时会发生什么**:
1. 调用 `val_res_transformer_forward()` (不是 `val_t2m_forward()`)
2. 用 VAE 编码获得 GT Q0
3. ResTransformer 预测 Q1-Q2（条件于 GT Q0）
4. VAE 解码，计算运动重建误差
5. 使用 MRMetrics (MPJPE, MPJPE_PA, etc.)

---

## ⚖️ 两种验证方式的比较

### Stage 2 验证 (Text-to-Motion)
```
Input: text
  ↓
MaskTransformer.generate(text) → Q0_pred
  ↓
ResTransformer.generate(Q0_pred) → Q1-Q2_pred  ← 如果有 ResTransformer
  ↓  (或者仅用 Q0_pred)
VAE.decode([Q0, Q1, Q2]) → motion_pred
  ↓
Compare with motion_ref
  ↓
TM2TMetrics: FID, Diversity, R-Precision, etc.
```

**特点**:
- ✅ 端到端的 text → motion 评估
- ⚠️ 如果没有 ResTransformer，质量会较差（仅 Q0）
- 🎯 评估整个生成流程

---

### Stage 3 验证 (Motion Reconstruction with GT Q0)
```
Input: motion_ref
  ↓
VAE.encode(motion_ref) → [Q0_GT, Q1_GT, Q2_GT]
  ↓
Extract Q0_GT
  ↓
ResTransformer.generate(Q0_GT, text) → [Q0_GT, Q1_pred, Q2_pred]
  ↓
VAE.decode([Q0_GT, Q1_pred, Q2_pred]) → motion_pred
  ↓
Compare with motion_ref
  ↓
MRMetrics: MPJPE, MPJPE_PA, etc.
```

**特点**:
- ✅ 假设 Q0 是完美的（GT）
- ✅ 仅评估 ResTransformer 的预测能力
- ✅ 不需要 MaskTransformer
- 🎯 评估残差预测精度

---

## 🎯 总结

### ✅ 已完全实现的验证逻辑

| 功能 | 代码位置 | 状态 |
|------|---------|------|
| **Stage 2 验证（有 ResTransformer）** | `val_t2m_forward()` + `generate()` (line 210-218) | ✅ 完全实现 |
| **Stage 2 验证（仅 Q0）** | `generate()` (line 219-221) | ✅ 完全实现 |
| **Stage 3 验证（GT Q0）** | `val_res_transformer_forward()` (line 413-477) | ✅ 完全实现 |

### 🔧 实际使用建议

#### 方案 A: 严格遵循原论文（推荐）
1. **训练 Stage 2** 时不进行完整验证（或接受仅 Q0 的低质量验证）
2. **训练 Stage 3** 时使用 GT Q0 验证
3. **Stage 2 和 Stage 3 都训练完成后**，加载两个 checkpoint 进行完整的端到端评估

#### 方案 B: 改进 Stage 2 验证（可选）
1. 先训练一个 Stage 3 模型（可以用较少 epochs）
2. 在训练 Stage 2 时加载预训练的 Stage 3 用于验证
3. 需要实现 `PRETRAINED_RES` 加载功能（如 MOMASK_STAGE_INTEGRATION_STATUS.md 中建议）

---

## 📝 代码示例

### 示例 1: Stage 2 验证（仅 Q0）

**当前配置**（`res_transformer: null`）：
```python
# 验证时
motion = model.generate(["a person walks"], [100])

# 内部执行流程
q0_ids = mask_transformer.generate(...)  # (1, 25)
all_indices = q0_ids.unsqueeze(-1)  # (1, 25, 1) ← 仅 Q0
motion = vae.decode(all_indices)  # (1, 100, D) ← 质量较差
```

### 示例 2: Stage 3 验证（GT Q0）

```python
# 验证时
motion_ref = batch["motion"]  # (B, T, D)
code_idx, _ = vae.encode(motion_ref)  # (B, T', 3)

q0_gt = code_idx[..., 0]  # (B, T') ← Ground Truth Q0
all_indices = res_transformer.generate(q0_gt, texts, ...)  # (B, T', 3)
# 返回: [Q0_GT, Q1_pred, Q2_pred]

motion_pred = vae.decode(all_indices)  # (B, T, D)
# Compare motion_pred vs motion_ref → MPJPE
```

---

**结论**: 所有验证逻辑都已完整实现且与原论文一致！可以直接训练。Stage 2 验证时会使用"仅 Q0"的降级模式，这是正常且预期的行为。
