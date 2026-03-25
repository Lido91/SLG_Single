# MoMask Stage 集成现状报告

## ✅ 已实现的功能

### 1. 训练流程 (Training)

#### Stage 2: Masked Transformer 训练
```yaml
# configs/momask_h2s_stage2.yaml
TRAIN:
  STAGE: mask_transformer
  PRETRAINED_VAE: <VAE checkpoint>  # 已实现：加载冻结的 VAE

model:
  params:
    motion_vae: ${vq.h2s_rvq_3}           # ✅ 已配置
    mask_transformer: ${lm.momask_transformer}  # ✅ 已配置
    res_transformer: null                 # ✅ Stage 2 不需要
```

**训练流程** (`mgpt_momask.py:260-282`):
```python
def train_mask_transformer_forward(self, batch):
    # 1. 使用冻结的 VAE 编码 motion → tokens
    with torch.no_grad():
        code_idx, _ = self.vae.encode(feats)  # (B, T', num_quantizers)

    # 2. 训练 MaskTransformer 仅预测 Q0
    q0_ids = code_idx[..., 0]  # (B, T')
    ce_loss, pred_ids, acc = self.mask_transformer(q0_ids, texts, m_lens)
```

**状态**: ✅ **完全实现**

---

#### Stage 3: Residual Transformer 训练
```yaml
# configs/momask_h2s_stage3.yaml
TRAIN:
  STAGE: res_transformer
  PRETRAINED_VAE: <VAE checkpoint>  # 已实现：加载冻结的 VAE

model:
  params:
    motion_vae: ${vq.h2s_rvq_3}           # ✅ 已配置
    mask_transformer: null                # ✅ Stage 3 训练不需要
    res_transformer: ${lm.momask_residual}  # ✅ 已配置
```

**训练流程** (`mgpt_momask.py:284-305`):
```python
def train_res_transformer_forward(self, batch):
    # 1. 使用冻结的 VAE 编码 motion → tokens
    with torch.no_grad():
        code_idx, _ = self.vae.encode(feats)  # (B, T', num_quantizers)

    # 2. 训练 ResidualTransformer 预测 Q1-Q2（条件于 GT Q0）
    ce_loss, pred_ids, acc = self.res_transformer(code_idx, texts, m_lens)
```

**关键点**:
- ✅ **不需要加载 MaskTransformer checkpoint**
- ✅ **训练时使用 Ground Truth Q0** (从 VAE 编码得到)
- ✅ **随机采样量化器层** (q ∈ [1, num_quantizers))

**状态**: ✅ **完全实现**

---

### 2. 验证流程 (Validation)

#### Stage 2 验证
```python
# mgpt_momask.py:358-410
@torch.no_grad()
def val_t2m_forward(self, batch):
    # 完整的 text → motion 生成流程
    motion = self.generate(texts, lengths)
    # 需要 mask_transformer 和 res_transformer
```

**问题**: ❌ **Stage 2 验证需要 ResidualTransformer，但配置中设为 null**

**影响**: Stage 2 训练时无法进行 text-to-motion 验证

---

#### Stage 3 验证
```python
# mgpt_momask.py:413-477
@torch.no_grad()
def val_res_transformer_forward(self, batch):
    # 使用 GT Q0，生成 Q1-Q2，然后解码
    code_idx, _ = self.vae.encode(feats_ref)
    q0_ids = code_idx[..., 0]  # GT Q0
    all_indices = self.res_transformer.generate(q0_ids, texts, m_lens)
    motion = self.vae.decode(all_indices)
```

**状态**: ✅ **完全实现** (不需要 MaskTransformer)

---

### 3. 推理流程 (Inference)

```python
# mgpt_momask.py:163-226
@torch.no_grad()
def generate(self, texts, lengths):
    # Stage 2: 生成 Q0
    q0_ids = self.mask_transformer.generate(...)

    # Stage 3: 生成 Q1-Q2
    all_indices = self.res_transformer.generate(q0_ids, ...)

    # Decode
    motion = self.vae.decode(all_indices)
```

**需要**:
- ✅ `self.vae` (frozen)
- ✅ `self.mask_transformer` (frozen)
- ✅ `self.res_transformer` (frozen)

**状态**: ✅ **代码已实现**，但需要配置支持

---

## ❌ 缺失的功能

### 1. Stage 2 验证时缺少 ResidualTransformer

**问题描述**:
- Stage 2 验证调用 `val_t2m_forward()` → `self.generate()`
- `generate()` 需要 `self.res_transformer`，但 Stage 2 配置中设为 `null`

**错误**:
```python
# mgpt_momask.py:211
if self.res_transformer is not None:
    all_indices = self.res_transformer.generate(...)
else:
    # 仅使用 Q0，质量很差
    all_indices = q0_ids.unsqueeze(-1)
```

**解决方案**:

#### 方案 A: 使用预训练的 ResidualTransformer (推荐)
```yaml
# configs/momask_h2s_stage2.yaml
TRAIN:
  STAGE: mask_transformer
  PRETRAINED_VAE: <VAE checkpoint>
  PRETRAINED_RES: <Residual Transformer checkpoint>  # ❌ 新增：尚未实现

model:
  params:
    res_transformer: ${lm.momask_residual}  # 改为非 null
```

**需要实现**:
1. 在 `train.py` 中添加 `PRETRAINED_RES` 加载逻辑
2. 在 `mgpt_momask._setup_training_stage()` 中冻结 `res_transformer`

#### 方案 B: Stage 2 验证时仅用 Q0 (次优)
- 保持当前实现：`all_indices = q0_ids.unsqueeze(-1)`
- 验证指标会较差，但可以训练

**推荐**: **方案 A**（更符合原论文评估方式）

---

### 2. 推理配置缺少 checkpoint 加载路径

**当前配置** (`momask_h2s_inference.yaml`):
```yaml
model:
  params:
    stage: "inference"
    motion_vae: ${vq.h2s_rvq_3}
    mask_transformer: ${lm.momask_transformer}
    res_transformer: ${lm.momask_residual}
```

**问题**: ❌ **只定义了架构，没有指定 checkpoint 路径**

**需要添加**:
```yaml
TRAIN:
  PRETRAINED_VAE: <VAE checkpoint path>
  PRETRAINED_MASK: <Stage 2 checkpoint path>  # ❌ 尚未实现
  PRETRAINED_RES: <Stage 3 checkpoint path>   # ❌ 尚未实现
```

**需要实现**:
1. 在 `load_checkpoint.py` 中添加：
```python
def load_pretrained_mask_transformer(cfg, model, logger=None):
    """加载 Stage 2 MaskTransformer checkpoint"""
    state_dict = torch.load(cfg.TRAIN.PRETRAINED_MASK, ...)['state_dict']
    mask_dict = OrderedDict()
    for k, v in state_dict.items():
        if "mask_transformer" in k:
            name = k.replace("mask_transformer.", "")
            mask_dict[name] = v
    model.mask_transformer.load_state_dict(mask_dict, strict=True)
    return model

def load_pretrained_res_transformer(cfg, model, logger=None):
    """加载 Stage 3 ResidualTransformer checkpoint"""
    state_dict = torch.load(cfg.TRAIN.PRETRAINED_RES, ...)['state_dict']
    res_dict = OrderedDict()
    for k, v in state_dict.items():
        if "res_transformer" in k:
            name = k.replace("res_transformer.", "")
            res_dict[name] = v
    model.res_transformer.load_state_dict(res_dict, strict=True)
    return model
```

2. 在 `train.py` 中调用：
```python
if cfg.TRAIN.PRETRAINED_MASK:
    load_pretrained_mask_transformer(cfg, model, logger)
if cfg.TRAIN.PRETRAINED_RES:
    load_pretrained_res_transformer(cfg, model, logger)
```

---

## 📋 实现计划

### Phase 1: 添加 checkpoint 加载功能 (必需)

1. **扩展 `load_checkpoint.py`**:
   - [x] `load_pretrained_vae()` (已实现)
   - [ ] `load_pretrained_mask_transformer()`
   - [ ] `load_pretrained_res_transformer()`

2. **修改 `train.py`**:
   ```python
   # 添加在 load_pretrained_vae() 之后
   if cfg.TRAIN.PRETRAINED_MASK:
       load_pretrained_mask_transformer(cfg, model, logger)
   if cfg.TRAIN.PRETRAINED_RES:
       load_pretrained_res_transformer(cfg, model, logger)
   ```

3. **更新配置文件**:
   ```yaml
   # configs/momask_h2s_stage2.yaml
   TRAIN:
     PRETRAINED_RES: ""  # 可选：用于验证

   # configs/momask_h2s_inference.yaml
   TRAIN:
     PRETRAINED_MASK: experiments/mgpt/MoMask_H2S_Stage2/checkpoints/best.ckpt
     PRETRAINED_RES: experiments/mgpt/MoMask_H2S_Stage3/checkpoints/best.ckpt
   ```

---

### Phase 2: 改进验证流程 (可选)

#### 选项 A: Stage 2 加载预训练的 ResidualTransformer
```python
# mgpt_momask._setup_training_stage()
elif self.stage == 'mask_transformer':
    # Freeze VAE
    if self.vae is not None:
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False

    # Freeze ResidualTransformer (if provided for validation)
    if self.res_transformer is not None:
        self.res_transformer.eval()
        for p in self.res_transformer.parameters():
            p.requires_grad = False
```

#### 选项 B: 单独的验证模式
- Stage 2 训练时跳过 text-to-motion 验证
- 仅在 Stage 3 完成后进行完整评估

---

## 🎯 总结

### ✅ 已完全实现
1. **Stage 2 训练**: 使用冻结 VAE，训练 MaskTransformer
2. **Stage 3 训练**: 使用冻结 VAE，训练 ResidualTransformer（使用 GT Q0）
3. **Stage 3 验证**: 使用 GT Q0 + 预测 Q1-Q2 进行运动重建评估

### ❌ 需要实现
1. **Checkpoint 加载功能**:
   - `load_pretrained_mask_transformer()`
   - `load_pretrained_res_transformer()`
   - 在 `train.py` 中集成

2. **推理配置完善**:
   - 添加 `PRETRAINED_MASK` 和 `PRETRAINED_RES` 路径
   - 更新 `momask_h2s_inference.yaml`

3. **(可选) Stage 2 验证改进**:
   - 加载预训练的 ResidualTransformer 用于验证
   - 或者跳过 text-to-motion 验证

---

## 🚀 快速开始

### 当前可以做的事情

#### 1. 训练 Stage 2 (无完整验证)
```bash
python train.py --cfg configs/momask_h2s_stage2.yaml --nodebug
```
- ✅ 训练正常
- ⚠️ 验证仅使用 Q0（质量较差）

#### 2. 训练 Stage 3
```bash
python train.py --cfg configs/momask_h2s_stage3.yaml --nodebug
```
- ✅ 训练正常
- ✅ 验证使用 GT Q0 + 预测 Q1-Q2（运动重建指标）

#### 3. 推理 (需先实现 checkpoint 加载)
```bash
# 需要先实现 Phase 1
python train.py --cfg configs/momask_h2s_inference.yaml --nodebug
```

---

## 📌 原始 MoMask vs MotionGPT 对比

| 功能 | 原始 MoMask | MotionGPT 实现 | 状态 |
|------|------------|---------------|------|
| **Stage 2 训练** | ✅ 完整 | ✅ 完整 | ✅ 一致 |
| **Stage 3 训练** | ✅ 完整 | ✅ 完整 | ✅ 一致 |
| **Stage 2 验证** | ⚠️ 需要预训练 Stage 3 | ⚠️ 当前仅 Q0 | ⚠️ 需改进 |
| **Stage 3 验证** | ✅ 使用 GT Q0 | ✅ 使用 GT Q0 | ✅ 一致 |
| **推理** | ✅ 加载两个 checkpoint | ❌ 尚未实现加载逻辑 | ❌ 待实现 |
| **Checkpoint 管理** | 手动加载 | PyTorch Lightning | 🔧 框架差异 |

---

**结论**: 训练流程已完全实现且与原论文一致，**唯一缺失的是推理时的 checkpoint 加载功能**（Phase 1 必须实现）。Stage 2 验证改进（Phase 2）是可选的优化项。
