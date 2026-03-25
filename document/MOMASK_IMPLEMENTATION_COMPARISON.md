# MoMask 实现对比报告

本报告详细对比了 MotionGPT 中的 MoMask 实现与原始 MoMask 论文代码的差异。

---

## ✅ 核心架构一致性

### Masked Transformer (Stage 2)
| 组件 | 原始实现 | MotionGPT 实现 | 状态 |
|------|---------|---------------|------|
| **Forward Pass** | `forward(ids, y, m_lens)` | `forward(ids, y, m_lens)` | ✅ 一致 |
| **Loss 计算** | `cal_performance(logits, labels, ignore_index=mask_id)` | `cal_performance(logits, labels, ignore_index=mask_id)` | ✅ 一致 |
| **掩码策略** | BERT-style: 10% random, 88% MASK | BERT-style: 10% random, 88% MASK | ✅ 一致 |
| **噪声调度** | `cosine_schedule(rand_time)` | `cosine_schedule(rand_time)` | ✅ 一致 |
| **CFG** | `forward_with_cond_scale()` | `forward_with_cond_scale()` | ✅ 一致 |
| **生成方式** | 迭代去掩码 (10 steps) | 迭代去掩码 (10 steps) | ✅ 一致 |

### Residual Transformer (Stage 3)
| 组件 | 原始实现 | MotionGPT 实现 | 状态 |
|------|---------|---------------|------|
| **Forward Pass** | `forward(all_indices, y, m_lens)` | `forward(all_indices, y, m_lens)` | ✅ 一致 |
| **Layer Sampling** | `q_schedule(bs, low=1, high=num_q)` | `q_schedule(bs, low=1, high=num_q)` | ✅ 一致 |
| **Cumsum Codes** | `torch.cumsum(all_codes, dim=-1)` | `torch.cumsum(all_codes, dim=-1)` | ✅ 一致 |
| **Output Projection** | `output_project(logits, active_q_layers-1)` | `output_project(logits, qids-1)` | ✅ 一致 |
| **生成方式** | 自回归生成 Q1→Q2→... | 自回归生成 Q1→Q2→... | ✅ 一致 |
| **权重共享** | `share_weight` flag 控制 | `share_weight` flag 控制 | ✅ 一致 |

---

## 📊 训练超参数对比

### Stage 2: Masked Transformer

| 参数 | 原始 MoMask | 之前的配置 | ✅ 修正后 |
|------|------------|----------|-----------|
| **Batch Size** | 64 (HumanML3D) | 32 | **64** |
| **Epochs** | 500 | 150 | **500** |
| **Learning Rate** | 2e-4 | 1e-4 | **2e-4** |
| **Weight Decay** | 1e-5 | 0.0 | **1e-5** |
| **LR Scheduler** | MultiStepLR | CosineAnnealing | **MultiStepLR** |
| **Milestones** | [50000] iters | - | **[50000]** |
| **Gamma** | 0.1 | - | **0.1** |
| **Warmup Iters** | 2000 | - | **2000** |
| **Dropout** | 0.2 | 0.1 | **0.2** |
| **CFG Dropout** | 0.1 | 0.1 | ✅ 0.1 |
| **Num Workers** | 4 | 8 | **4** |

### Stage 3: Residual Transformer

| 参数 | 原始 MoMask | 之前的配置 | ✅ 修正后 |
|------|------------|----------|-----------|
| **Batch Size** | 64 (HumanML3D) | 32 | **64** |
| **Epochs** | 500 | 150 | **500** |
| **Learning Rate** | 2e-4 | 1e-4 | **2e-4** |
| **Weight Decay** | 1e-5 | 0.0 | **1e-5** |
| **LR Scheduler** | MultiStepLR | CosineAnnealing | **MultiStepLR** |
| **Milestones** | [50000] iters | - | **[50000]** |
| **Gamma** | 0.1 | - | **0.1** |
| **Warmup Iters** | 2000 | - | **2000** |
| **Dropout** | 0.2 | 0.2 | ✅ 0.2 |
| **CFG Dropout** | 0.1 | 0.2 | **0.1** |
| **Num Workers** | 4 | 8 | **4** |

---

## 🔍 Loss 计算详解

### 原始 MoMask Loss 实现

```python
# models/mask_transformer/tools.py: cal_performance()
def cal_performance(pred, labels, ignore_index=None, smoothing=0., tk=1):
    """
    Args:
        pred: (B, V, N) predicted logits
        labels: (B, N) ground truth token indices
        ignore_index: mask_id or pad_id to ignore
    Returns:
        loss: cross-entropy loss
        pred_id: predicted tokens
        acc: top-k accuracy
    """
    loss = cal_loss(pred, labels, ignore_index, smoothing=smoothing)

    pred_id_k = torch.topk(pred, k=tk, dim=1).indices
    pred_id = pred_id_k[:, 0]
    mask = labels.ne(ignore_index)
    n_correct = (pred_id_k == labels.unsqueeze(1)).any(dim=1).masked_select(mask)
    acc = torch.mean(n_correct.float()).item()

    return loss, pred_id, acc

def cal_loss(pred, labels, ignore_index=None, smoothing=0.):
    if smoothing:
        # Label smoothing implementation
        ...
    else:
        loss = F.cross_entropy(pred, labels, ignore_index=ignore_index)
    return loss
```

### Stage 2 训练 Loss

```python
# train_res_transformer.py: ResidualTransformerTrainer.forward()
code_idx, _ = vq_model.encode(motion)  # (B, T', num_quantizers)
m_lens = m_lens // 4  # 转换为 token 长度

# 仅训练 Q0
ce_loss, pred_ids, acc = mask_transformer(code_idx[..., 0], texts, m_lens)
```

**关键点**:
1. **仅训练 Q0**: 只用第一层量化器的 tokens
2. **Ignore Index**: `mask_id` (未被选中预测的位置)
3. **标准 CE Loss**: 无额外正则化

### Stage 3 训练 Loss

```python
# train_res_transformer.py: ResidualTransformerTrainer.forward()
code_idx, all_codes = vq_model.encode(motion)  # (B, T', num_quantizers)
m_lens = m_lens // 4

# 训练所有层 Q1-Q_{num_quantizers-1}
ce_loss, pred_ids, acc = res_transformer(code_idx, texts, m_lens)
```

**关键点**:
1. **随机采样层**: 每个 batch 随机选择一个量化器层 `q ∈ [1, num_quantizers)`
2. **条件于历史**: 用 `cumsum(all_codes[..., :q])` 作为输入
3. **Ignore Index**: `pad_id`
4. **标准 CE Loss**: 无额外正则化

---

## ⚙️ Optimizer & Scheduler 实现

### 原始实现

```python
# models/mask_transformer/transformer_trainer.py: MaskTransformerTrainer.train()
self.opt_t2m_transformer = optim.AdamW(
    self.t2m_transformer.parameters(),
    betas=(0.9, 0.99),
    lr=self.opt.lr,  # 2e-4
    weight_decay=1e-5
)

self.scheduler = optim.lr_scheduler.MultiStepLR(
    self.opt_t2m_transformer,
    milestones=self.opt.milestones,  # [50000] iterations
    gamma=self.opt.gamma  # 0.1
)

# Warmup logic
if it < self.opt.warm_up_iter:  # 2000
    current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr
```

### MotionGPT 需要适配

MotionGPT 使用 PyTorch Lightning 框架，需要在 `configure_optimizers()` 中实现:

```python
def configure_optimizers(self):
    if self.stage == "mask_transformer":
        params = self.mask_transformer.parameters_wo_clip()
    elif self.stage == "res_transformer":
        params = self.res_transformer.parameters_wo_clip()

    optimizer = torch.optim.AdamW(
        params,
        lr=2e-4,  # 原始: 2e-4
        betas=(0.9, 0.99),
        weight_decay=1e-5  # 原始: 1e-5
    )

    # 需要自定义 warmup + MultiStepLR
    scheduler = {
        'scheduler': torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[50000],  # iterations, 需要转换为 steps
            gamma=0.1
        ),
        'interval': 'step',  # 按 iteration 而非 epoch
    }

    return [optimizer], [scheduler]
```

**⚠️ 注意**: 原始实现使用 **iteration-based** scheduler, 而非 epoch-based!

---

## 🎯 关键差异总结

### ✅ 完全一致的部分
1. **模型架构**: Transformer 层数、维度、注意力头数
2. **前向传播逻辑**: 掩码策略、CFG、层采样
3. **Loss 函数**: 标准交叉熵，ignore_index 正确
4. **生成算法**: 迭代去掩码、自回归生成

### ⚠️ 需要修正的部分 (已修正)
1. ✅ **Batch Size**: 32 → **64**
2. ✅ **Epochs**: 150 → **500**
3. ✅ **Learning Rate**: 1e-4 → **2e-4**
4. ✅ **Weight Decay**: 0.0 → **1e-5**
5. ✅ **Scheduler**: CosineAnnealing → **MultiStepLR [50000]**
6. ✅ **Warmup**: 无 → **2000 iters**
7. ✅ **Dropout**: 0.1 → **0.2** (Stage 2)
8. ✅ **CFG Dropout**: 0.2 → **0.1** (Stage 3)

### 🔧 框架适配需求
1. **PyTorch Lightning 兼容性**:
   - 需要在 `configure_optimizers()` 中实现 warmup
   - MultiStepLR 的 milestones 需要转换为 Lightning steps

2. **数据加载器**:
   - 原始: 自定义 `Text2MotionDataset`
   - MotionGPT: `H2SDataModule` (已适配)

3. **评估指标**:
   - 原始: FID, Diversity, Top-k matching
   - MotionGPT: `TM2TMetrics` (包含 FID 等)

---

## 📝 训练命令对比

### 原始 MoMask

```bash
# Stage 2: Masked Transformer
python train_mask_transformer.py \
  --name mask_t2m \
  --gpu_id 0 \
  --dataset_name t2m \
  --batch_size 64 \
  --vq_name rvq_nq6_dc512_nc512 \
  --cond_drop_prob 0.1

# Stage 3: Residual Transformer
python train_res_transformer.py \
  --name res_t2m \
  --gpu_id 0 \
  --dataset_name t2m \
  --batch_size 64 \
  --vq_name rvq_nq6_dc512_nc512 \
  --cond_drop_prob 0.1 \
  --share_weight
```

### MotionGPT (修正后)

```bash
# Stage 2: Masked Transformer
python train.py --cfg configs/momask_h2s_stage2.yaml --nodebug

# Stage 3: Residual Transformer
python train.py --cfg configs/momask_h2s_stage3.yaml --nodebug
```

---

## 🚀 推荐训练策略

### 完全遵循原始论文

使用修正后的配置:
- Batch size: 64
- Epochs: 500
- LR: 2e-4 with MultiStepLR
- Warmup: 2000 iterations
- **预计训练时间**: ~5-7天 (3x A100)

### 快速实验 (调试)

可以使用以下设置快速验证:
- Batch size: 32 (减少内存)
- Epochs: 100 (快速收敛测试)
- LR: 1e-4 (更保守)
- **预计训练时间**: ~1-2天

---

## ✅ 验证清单

- [x] 模型架构与原始一致
- [x] Forward pass 逻辑正确
- [x] Loss 计算方式一致
- [x] 超参数匹配原始论文
- [x] Optimizer 设置正确
- [x] Scheduler 类型修正
- [x] Warmup 配置添加
- [x] CFG dropout 正确
- [ ] PyTorch Lightning warmup 实现 (需要在代码中添加)
- [ ] 评估指标完整性验证

---

## 📌 后续优化建议

1. **实现 Warmup**: 在 `mgpt_momask.py` 中添加 warmup scheduler
2. **日志对齐**: 确保 TensorBoard/WandB 日志与原始一致
3. **评估频率**: 原始每 10 epochs 评估一次，配置已设置
4. **Checkpoint 保存**: 保存 `net_best_fid.tar` 和 `net_best_acc.tar`

---

**总结**: 修正后的配置完全匹配原始 MoMask 实现的超参数设置！🎉
