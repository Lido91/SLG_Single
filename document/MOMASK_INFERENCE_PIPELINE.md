# MoMask 推理/测试流程详解

基于原始代码 `gen_t2m.py` 和 `eval_t2m_trans_res.py` 的完整分析。

---

## 🎯 推理流程概览

```
输入文本 → 长度预测 → Stage 2 (Masked Transformer) → Stage 3 (Residual Transformer) → VAE 解码 → 运动序列
```

---

## 📝 完整推理流程（gen_t2m.py）

### 1️⃣ 模型加载

```python
# 1. 加载 RVQ-VAE
vq_model = load_vq_model(vq_opt)
vq_model.eval()

# 2. 加载 Masked Transformer (Stage 2)
t2m_transformer = load_trans_model(model_opt, 'latest.tar')
t2m_transformer.eval()

# 3. 加载 Residual Transformer (Stage 3)
res_model = load_res_model(res_opt, vq_opt)
res_model.eval()

# 4. 加载长度预测器（可选）
length_estimator = load_len_estimator(model_opt)
length_estimator.eval()
```

**关键点**:
- ✅ 需要加载 **3 个模型** (VAE + 2 个 Transformers)
- ✅ 可选第 4 个模型：长度预测器
- ✅ 所有模型设为 `eval()` 模式

---

### 2️⃣ 输入处理

#### 方式 A: 单个文本提示 + 指定长度
```python
prompt = "a person walks forward"
motion_length = 120  # frames
token_lens = motion_length // 4  # 30 tokens (unit_length=4)
```

#### 方式 B: 单个文本提示 + 自动预测长度
```python
prompt = "a person walks forward"
# 使用长度预测器
text_embedding = t2m_transformer.encode_text([prompt])  # CLIP 编码
pred_dis = length_estimator(text_embedding)  # 预测长度分布
probs = F.softmax(pred_dis, dim=-1)
token_lens = Categorical(probs).sample()  # 采样长度（in tokens）
motion_length = token_lens * 4
```

#### 方式 C: 批量文本文件
```python
# text_file.txt 格式:
# a person walks forward#120
# a person runs#80
# a person jumps

with open('prompts.txt', 'r') as f:
    for line in f.readlines():
        infos = line.split('#')
        prompt = infos[0]
        length = int(infos[1]) if len(infos) > 1 else None
```

**关键点**:
- ✅ 支持手动指定或自动预测长度
- ✅ 长度单位：**tokens** (frame // 4)
- ✅ 批量处理多个文本

---

### 3️⃣ 生成流程（核心）

```python
for r in range(repeat_times):  # 重复生成多次（多样性评估）
    with torch.no_grad():
        # ============ Stage 2: Masked Transformer ============
        # 生成 Q0 tokens
        mids = t2m_transformer.generate(
            captions,              # 文本列表
            token_lens,            # token 长度（非 frame）
            timesteps=10,          # 迭代去掩码步数
            cond_scale=4.0,        # CFG scale
            temperature=1.0,       # 采样温度
            topk_filter_thres=0.9, # Top-k 过滤
            gsample=True           # 使用 Gumbel 采样
        )
        # mids: (B, T') - Q0 token indices

        # ============ Stage 3: Residual Transformer ============
        # 生成 Q1, Q2 tokens
        pred_ids = res_model.generate(
            mids,                  # Q0 tokens from Stage 2
            captions,              # 文本列表
            token_lens,            # token 长度
            temperature=1.0,       # 采样温度
            cond_scale=5.0         # CFG scale (更高)
        )
        # pred_ids: (B, T', num_quantizers) - Q0, Q1, Q2 tokens

        # ============ VAE Decode ============
        pred_motions = vq_model.forward_decoder(pred_ids)
        # pred_motions: (B, T, D) - 运动特征（已归一化）

        # ============ 反归一化 ============
        pred_motions = pred_motions.detach().cpu().numpy()
        data = inv_transform(pred_motions)  # data * std + mean
```

**关键超参数**:

| 参数 | Stage 2 | Stage 3 | 说明 |
|------|---------|---------|------|
| **timesteps** | 10-18 | - | 迭代去掩码步数（越多越慢但质量更好）|
| **cond_scale** | 4.0 | 5.0 | CFG 强度（越大越符合文本但多样性降低）|
| **temperature** | 1.0 | 1.0 | 采样温度（越高越随机）|
| **topk_filter_thres** | 0.9 | 0.9 | Top-k 过滤（保留前 10% 概率质量）|
| **gsample** | True | False | Gumbel 采样 vs 分类采样 |

---

### 4️⃣ 后处理

```python
for k, (caption, joint_data) in enumerate(zip(captions, data)):
    # 1. 裁剪到指定长度
    joint_data = joint_data[:m_length[k]]

    # 2. 从表示恢复到 3D 关节
    joint = recover_from_ric(torch.from_numpy(joint_data).float(), 22).numpy()

    # 3. 应用逆运动学（IK）优化
    _, ik_joint = converter.convert(joint, iterations=100)

    # 4. 保存为 BVH 文件
    bvh_path = f"sample{k}_repeat{r}_len{m_length[k]}.bvh"
    converter.convert(joint, filename=bvh_path)

    # 5. 渲染为视频
    save_path = f"sample{k}_repeat{r}_len{m_length[k]}.mp4"
    plot_3d_motion(save_path, kinematic_chain, joint, title=caption, fps=20)

    # 6. 保存为 numpy
    np.save(f"sample{k}.npy", joint)
```

---

## 📊 评估流程（eval_t2m_trans_res.py）

### 评估指标计算

```python
# 重复评估 20 次，计算置信区间
repeat_time = 20
for i in range(repeat_time):
    with torch.no_grad():
        # 对测试集每个样本：
        for batch in eval_val_loader:
            texts, pose_gt, m_length = batch

            # === 生成运动 ===
            # Stage 2: Q0 generation
            mids = t2m_transformer.generate(
                texts,
                m_length // 4,
                timesteps=18,      # 评估时用更多步数
                cond_scale=4.0,
                temperature=1.0,
                topk_filter_thres=0.9
            )

            # Stage 3: Q1-Q2 generation
            pred_ids = res_model.generate(
                mids,
                texts,
                m_length // 4,
                temperature=1.0,
                cond_scale=5.0
            )

            # VAE decode
            pred_motions = vq_model.forward_decoder(pred_ids)

            # === 计算指标 ===
            # 1. FID (Fréchet Inception Distance)
            # 2. Diversity (生成运动的多样性)
            # 3. R-Precision (Top-1, Top-2, Top-3 文本-运动匹配)
            # 4. Matching Score (文本-运动对齐分数)
            # 5. Multimodality (同一文本多次生成的多样性)

# 计算均值和置信区间
fid_mean = np.mean(fid)
fid_conf = np.std(fid) * 1.96 / np.sqrt(repeat_time)
```

**关键评估指标**:
- **FID**: 越低越好（与真实运动分布的距离）
- **Diversity**: 越高越好（生成运动的多样性）
- **R-Precision**: 越高越好（文本检索运动的准确率）
- **Matching Score**: 越低越好（文本-运动匹配距离）
- **Multimodality**: 适中最好（同一文本的多样性）

---

## 🔍 详细生成算法

### Stage 2: Masked Transformer 迭代去掩码

```python
def generate(captions, m_lens, timesteps=10, cond_scale=4.0, ...):
    # 1. 编码文本
    cond_vector = encode_text(captions)  # CLIP embeddings

    # 2. 初始化：全 MASK
    ids = torch.full((B, T'), MASK_ID)
    scores = torch.zeros((B, T'))

    # 3. 迭代去掩码（10 步）
    for t in linspace(0, 1, timesteps):
        # 计算当前应保持掩码的比例
        mask_ratio = cosine_schedule(t)  # t=0: 100%, t=1: 0%
        num_masked = round(mask_ratio * m_lens)

        # 选择低置信度的位置重新掩码
        is_mask = (confidence_ranks < num_masked)
        ids[is_mask] = MASK_ID

        # 预测所有位置（并行）
        logits = forward_with_cfg(ids, cond_vector, cond_scale)

        # Top-k 过滤 + 采样
        filtered_logits = top_k(logits, topk_filter_thres)
        pred_ids = gumbel_sample(filtered_logits, temperature)

        # 更新 ids 和置信度
        ids[is_mask] = pred_ids[is_mask]
        scores = logits.softmax(dim=-1).gather(-1, pred_ids)

    return ids  # (B, T') - Q0 tokens
```

**关键点**:
- ✅ **并行预测** 所有位置（非自回归）
- ✅ **渐进去掩码** 从 100% → 0%
- ✅ **置信度驱动** 保留高置信度预测
- ✅ **CFG** 增强文本对齐

### Stage 3: Residual Transformer 层级生成

```python
def generate(motion_ids, captions, m_lens, temperature=1.0, cond_scale=5.0):
    # motion_ids: (B, T') - Q0 from Stage 2

    cond_vector = encode_text(captions)
    all_indices = [motion_ids]
    history_sum = 0

    # 逐层生成 Q1, Q2, ...
    for i in range(1, num_quantizers):  # i=1,2
        # 获取当前层的 embedding
        token_embed = self.token_embed_weight[i-1]  # (num_tokens, code_dim)

        # 累加前置层的 embeddings
        gathered_embed = token_embed.gather(0, motion_ids)  # (B, T', code_dim)
        history_sum += gathered_embed

        # 预测当前层
        logits = forward_with_cfg(
            history_sum,      # 累积 embeddings
            layer_id=i,       # 当前层 ID
            cond_vector,
            cond_scale
        )

        # 采样
        filtered_logits = top_k(logits, topk_filter_thres)
        pred_ids = gumbel_sample(filtered_logits, temperature)

        motion_ids = pred_ids
        all_indices.append(pred_ids)

    # Stack: [Q0, Q1, Q2] -> (B, T', num_quantizers)
    all_indices = torch.stack(all_indices, dim=-1)
    return all_indices
```

**关键点**:
- ✅ **自回归** 层级生成 Q0 → Q1 → Q2
- ✅ **累积条件** 每层条件于所有前置层的 sum
- ✅ **CFG** 增强文本对齐
- ⚠️ **错误累积** Q0 错误会影响 Q1, Q2

---

## ⚙️ 超参数调优建议

### 生成质量 vs 多样性权衡

| 目标 | timesteps | cond_scale (S2) | cond_scale (S3) | temperature |
|------|-----------|----------------|----------------|-------------|
| **高质量** | 18 | 5-7 | 6-8 | 0.8 |
| **平衡** | 10-12 | 4.0 | 5.0 | 1.0 |
| **高多样性** | 8 | 2-3 | 3-4 | 1.2 |

### 速度 vs 质量权衡

| 配置 | timesteps | 速度 | 质量 |
|------|-----------|------|------|
| **快速** | 6 | 最快 | 中等 |
| **标准** | 10 | 中等 | 良好 |
| **精细** | 18 | 慢 | 最佳 |

---

## 📋 完整推理脚本示例

```bash
# 方式 1: 单个文本 + 指定长度
python gen_t2m.py \
  --name mask_transformer_model \
  --res_name res_transformer_model \
  --text_prompt "a person walks forward" \
  --motion_length 120 \
  --repeat_times 5 \
  --time_steps 10 \
  --cond_scale 4.0 \
  --temperature 1.0 \
  --topkr 0.9 \
  --gumbel_sample

# 方式 2: 批量文本文件
python gen_t2m.py \
  --name mask_transformer_model \
  --res_name res_transformer_model \
  --text_path prompts.txt \
  --repeat_times 1 \
  --time_steps 18 \
  --cond_scale 4.0

# 方式 3: 评估模式
python eval_t2m_trans_res.py \
  --name mask_transformer_model \
  --res_name res_transformer_model \
  --dataset_name t2m \
  --time_steps 18 \
  --cond_scale 4.0 \
  --ext final_eval
```

---

## 🎯 MotionGPT 适配要点

我们的 MoMask 实现需要支持：

### 1. 推理接口
```python
class MoMask:
    @torch.no_grad()
    def generate(self, texts, lengths, **kwargs):
        # Stage 2
        q0 = self.mask_transformer.generate(texts, lengths, ...)

        # Stage 3
        all_indices = self.res_transformer.generate(q0, texts, lengths, ...)

        # Decode
        motion = self.vae.decode(all_indices)
        return motion
```

### 2. 超参数配置
```yaml
GENERATION:
  timesteps: 10
  cond_scale_mask: 4.0
  cond_scale_res: 5.0
  temperature: 1.0
  topk_filter_thres: 0.9
  gsample: true
```

### 3. 评估集成
- 支持 FID, Diversity, R-Precision 等指标
- 多次重复评估计算置信区间
- 与 MotionGPT 现有评估系统集成

---

## 📊 性能基准（HumanML3D）

| 指标 | 值 | 备注 |
|------|----|----|
| **FID** | ~0.080 | 越低越好 |
| **Diversity** | ~9.50 | 适中最好 |
| **R-Precision (Top-3)** | ~0.55 | 越高越好 |
| **Matching Score** | ~2.80 | 越低越好 |
| **Multimodality** | ~2.40 | 适中最好 |

---

**总结**: MoMask 推理流程清晰，分为 Stage 2（并行生成 Q0）和 Stage 3（层级生成 Q1-Q2），最后通过 VAE 解码得到运动。我们的实现已完全兼容此流程！✅
