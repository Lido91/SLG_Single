# MoMask Teacher Forcing 分析

## 🎯 答案：是的，MoMask 使用 Teacher Forcing！

但使用方式与传统自回归模型不同，MoMask 采用**并行 teacher forcing**（类似 BERT）。

---

## 📊 两种模式对比

### Stage 2: Masked Transformer - **并行 Teacher Forcing**

#### 训练过程（完全 teacher forcing）

```python
# models/mask_transformer/transformer.py: MaskTransformer.forward()

# 输入：ground truth tokens
ids = code_idx[..., 0]  # (B, T') - 从 VAE 编码得到的真实 Q0 tokens

# === Teacher Forcing 核心 ===
# 1. 随机选择要预测的位置（mask）
rand_mask_probs = cosine_schedule(rand_time)  # 随机掩码比例
mask = batch_randperm < num_token_masked      # 选择要被掩码的位置

# 2. 创建训练目标（ground truth）
labels = torch.where(mask, ids, self.mask_id)  # 预测位置 = GT, 其他 = mask_id

# 3. 创建输入（应用 BERT-style masking）
x_ids = ids.clone()
# 10% 替换为随机 token
x_ids = torch.where(mask_rid, rand_id, x_ids)
# 88% 替换为 MASK token
x_ids = torch.where(mask_mid, self.mask_id, x_ids)
# 2% 保持原 token (teacher forcing!)

# 4. 前向传播
logits = self.trans_forward(x_ids, cond_vector, ...)

# 5. 计算 loss（仅在被掩码位置）
ce_loss = F.cross_entropy(logits, labels, ignore_index=self.mask_id)
```

**关键点**:
- ✅ **使用 ground truth tokens** 作为输入
- ✅ **并行预测所有被掩码位置**（非自回归）
- ✅ **未被掩码的位置看到真实 token**（teacher forcing）
- ❌ **不是逐步生成**（与 GPT 不同）

#### 推理过程（迭代去掩码，无 teacher forcing）

```python
# models/mask_transformer/transformer.py: MaskTransformer.generate()

# 从全掩码开始
ids = torch.full((B, T'), self.mask_id)

for timestep in range(timesteps):  # 通常 10 步
    # 1. 预测所有位置
    logits = self.forward_with_cond_scale(ids, ...)
    pred_ids = sample(logits)

    # 2. 保留高置信度预测，重新掩码低置信度
    num_token_masked = round(rand_mask_prob * seq_len)
    is_mask = (confidence_ranks < num_token_masked)
    ids = torch.where(is_mask, self.mask_id, pred_ids)

    # 3. 逐步减少掩码比例（cosine schedule）
```

**关键点**:
- ❌ **不使用 ground truth**
- ✅ **迭代去掩码** (类似 MaskGIT)
- ✅ **每步都是并行预测**

---

### Stage 3: Residual Transformer - **完全 Teacher Forcing**

#### 训练过程

```python
# models/mask_transformer/transformer.py: ResidualTransformer.forward()

# 输入：ALL ground truth tokens from VAE
all_indices = code_idx  # (B, T', num_quantizers) - GT tokens

# === Teacher Forcing 核心 ===
# 1. 随机选择要预测的量化器层
active_q_layers = q_schedule(bs, low=1, high=num_q)  # 例如随机选 Q2

# 2. 使用 GT 前置层作为条件
token_embed = self.token_embed_weight  # 使用 GT tokens 的 embeddings
all_codes = token_embed.gather(1, all_indices[..., :-1])  # GT Q0, Q1, ... embeddings

# 3. 累加 GT 历史（teacher forcing）
cumsum_codes = torch.cumsum(all_codes, dim=-1)  # GT cumulative sum
history_sum = cumsum_codes[..., active_q_layers - 1]  # 例如 sum(Q0_GT + Q1_GT)

# 4. 预测目标层
active_indices = all_indices[torch.arange(bs), :, active_q_layers]  # 例如 Q2_GT
logits = self.trans_forward(history_sum, active_q_layers, ...)

# 5. Loss（预测 Q2，条件于 GT Q0+Q1）
ce_loss = F.cross_entropy(logits, active_indices, ignore_index=self.pad_id)
```

**关键点**:
- ✅ **完全使用 ground truth 前置层**（teacher forcing）
- ✅ **条件分布**: P(Q_i | Q_0^GT, Q_1^GT, ..., Q_{i-1}^GT, text)
- ✅ **训练时看到完美历史**
- ⚠️ **exposure bias 风险**（训练/推理分布不匹配）

#### 推理过程（自回归，无 teacher forcing）

```python
# models/mask_transformer/transformer.py: ResidualTransformer.generate()

# 输入：Q0 from MaskTransformer（预测值，非 GT）
motion_ids = mask_transformer.generate(...)  # (B, T') - 预测的 Q0

all_indices = [motion_ids]
history_sum = 0

for i in range(1, num_quantizers):  # Q1, Q2, ...
    # === 无 Teacher Forcing ===
    # 1. 使用预测的前置层（非 GT）
    token_embed = self.token_embed_weight[i-1]
    token_embed = repeat(token_embed, 'c d -> b c d', b=batch_size)
    gathered_ids = repeat(motion_ids, 'b n -> b n d', d=token_embed.shape[-1])
    history_sum += token_embed.gather(1, gathered_ids)  # 累加预测 embeddings

    # 2. 预测当前层
    logits = self.forward_with_cond_scale(history_sum, i, ...)
    pred_ids = sample(logits)

    # 3. 追加到历史（下一层会用到）
    motion_ids = pred_ids
    all_indices.append(pred_ids)

# 最终：all_indices = [Q0_pred, Q1_pred, Q2_pred]
```

**关键点**:
- ❌ **不使用 ground truth**
- ✅ **自回归生成** Q0 → Q1 → Q2
- ⚠️ **错误累积**（Q0 错误会影响 Q1, Q2）

---

## 🔍 与传统模型对比

### GPT（自回归 teacher forcing）

```python
# 训练
for t in range(seq_len):
    pred[t] = model(x[:t])      # 输入 GT 历史
    loss += CE(pred[t], x[t])   # 预测下一个

# 推理
for t in range(seq_len):
    pred[t] = model(pred[:t])   # 输入预测历史
```

**Teacher forcing**: ✅ 训练时使用 GT 历史
**并行性**: ❌ 串行生成

### BERT（并行 masked LM）

```python
# 训练
masked_x = apply_mask(x)        # 15% MASK, 10% random, 75% keep
pred = model(masked_x)          # 并行预测所有位置
loss = CE(pred[masked], x[masked])

# 推理
# BERT 本身不做生成，仅用于理解任务
```

**Teacher forcing**: ✅ 未被掩码位置看到 GT
**并行性**: ✅ 并行预测

### MoMask Stage 2（并行 masked LM + 迭代解码）

```python
# 训练（类似 BERT）
masked_x = apply_bert_mask(x)   # 88% MASK, 10% random, 2% keep
pred = model(masked_x)
loss = CE(pred[masked], x[masked])

# 推理（类似 MaskGIT）
x = all_masks
for step in range(10):
    pred = model(x)
    x = update_confident_tokens(x, pred)
```

**Teacher forcing**: ✅ 训练时部分位置看到 GT
**并行性**: ✅ 训练和推理都并行

### MoMask Stage 3（层级自回归）

```python
# 训练（完全 teacher forcing）
q_target = random_choice([1, 2])
history = sum(GT_embeddings[:q_target])  # GT Q0 + Q1
pred = model(history)
loss = CE(pred, GT[q_target])

# 推理（自回归，无 teacher forcing）
Q0 = stage2_generate()
Q1 = model(Q0_embedding)
Q2 = model(Q0_embedding + Q1_embedding)
```

**Teacher forcing**: ✅ 训练时使用 GT 历史
**并行性**: ❌ 推理时串行（层级）

---

## ⚠️ Exposure Bias 问题

### Stage 2: Masked Transformer

**训练-推理 gap**: 较小
- 训练：88% 位置被 MASK（看不到 GT）
- 推理：从全 MASK 开始
- ✅ **分布较接近**

### Stage 3: Residual Transformer

**训练-推理 gap**: **较大** ⚠️
- 训练：条件于 **GT** 前置层 `P(Q_i | Q_0^GT, ..., Q_{i-1}^GT)`
- 推理：条件于 **预测** 前置层 `P(Q_i | Q_0^pred, ..., Q_{i-1}^pred)`
- ❌ **分布不匹配**

**原论文解决方案**:
1. **Quantizer Dropout** (训练 VAE 时):
   ```python
   quantize_dropout_prob: 0.2  # 20% 随机 dropout 某些层
   ```
   - 让 decoder 适应不完整的量化器层
   - 减少对高层的依赖

2. **CFG Dropout** (训练 Transformer 时):
   ```python
   cond_drop_prob: 0.1  # 10% 随机丢弃文本条件
   ```
   - 增强模型鲁棒性

---

## 📋 总结表

| 模型 | Teacher Forcing | 训练并行性 | 推理并行性 | Exposure Bias |
|------|----------------|-----------|-----------|---------------|
| **GPT** | ✅ 完全 | ❌ 串行 | ❌ 串行 | 中等 |
| **BERT** | ✅ 部分 | ✅ 并行 | N/A | N/A |
| **MoMask Stage 2** | ✅ 部分 (2%) | ✅ 并行 | ✅ 并行 | 较小 |
| **MoMask Stage 3** | ✅ 完全 | ✅ 并行 | ❌ 层级串行 | 较大 |

---

## 💡 关键发现

1. **Stage 2 采用 BERT-style teacher forcing**:
   - 训练时 2% 位置保留真实 token
   - 98% 位置看到 MASK 或随机 token
   - **并行预测**，非自回归

2. **Stage 3 采用完全 teacher forcing**:
   - 训练时 100% 使用 GT 前置层
   - 推理时使用预测前置层
   - **存在 exposure bias**

3. **缓解 exposure bias 的策略**:
   - RVQ-VAE 的 quantizer dropout (0.2)
   - Transformer 的 CFG dropout (0.1)
   - 这些已在我们的配置中正确设置 ✅

---

**结论**: MoMask **确实使用 teacher forcing**，但方式比传统自回归模型更复杂，结合了并行预测和层级生成的优点。
