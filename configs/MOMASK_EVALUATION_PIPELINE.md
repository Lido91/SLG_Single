# MoMask 评估流程详解

基于原始代码 `eval_t2m_trans_res.py` 和 `utils/eval_t2m.py` 的完整分析。

---

## 🎯 评估流程概览

```
测试数据 → 生成运动 → 提取embeddings → 计算5大指标 → 重复20次 → 统计置信区间
```

**5大核心指标**:
1. **FID** (Fréchet Inception Distance) - 生成质量
2. **Diversity** - 运动多样性
3. **R-Precision** (Top-1/2/3) - 文本-运动检索准确率
4. **Matching Score** - 文本-运动匹配距离
5. **Multimodality** - 同一文本的多样性

---

## 📊 完整评估流程

### 1️⃣ 模型加载

```python
# 加载评估所需的所有模型
# 1. RVQ-VAE
vq_model = load_vq_model(vq_opt)
vq_model.eval()

# 2. Masked Transformer (Stage 2)
t2m_transformer = load_trans_model(model_opt, 'net_best_fid.tar')
t2m_transformer.eval()

# 3. Residual Transformer (Stage 3)
res_model = load_res_model(res_opt, vq_opt)
res_model.eval()

# 4. Evaluation Wrapper (用于提取 embeddings)
eval_wrapper = EvaluatorModelWrapper(wrapper_opt)
# 包含3个预训练模型:
#   - text_encoder: 编码文本
#   - motion_encoder: 编码运动
#   - movement_encoder: 编码运动特征
```

**关键点**:
- ✅ 需要 **4 组模型**: VAE + 2 Transformers + Evaluation Wrapper
- ✅ Evaluation Wrapper 使用预训练的 Text2Motion 模型（来自 HumanML3D）
- ✅ 所有模型设为 `eval()` 模式

---

### 2️⃣ 数据准备

```python
# 加载测试数据
eval_val_loader, _ = get_dataset_motion_loader(
    dataset_opt_path,
    batch_size=32,
    split='test',      # 使用测试集
    device='cuda'
)

# 每个 batch 包含:
batch = {
    'word_embeddings': word_embs,   # GloVe embeddings (B, max_len, 300)
    'pos_one_hots': pos_ohot,       # POS tags one-hot (B, max_len, pos_dim)
    'clip_text': texts,             # CLIP text (B,) - list of strings
    'sent_len': sent_len,           # 文本长度 (B,)
    'pose': motion_gt,              # ground truth运动 (B, T, D)
    'm_length': m_length,           # 运动长度 (B,)
    'token': tokens                 # 文本 tokens
}
```

---

### 3️⃣ 评估主循环（重复20次）

```python
repeat_time = 20  # 重复评估20次，计算置信区间
fid_list, div_list, top1_list, top2_list, top3_list = [], [], [], [], []
matching_list, mm_list = [], []

for repeat_id in range(repeat_time):
    # ========== 对测试集的每个batch进行评估 ==========
    for i, batch in enumerate(eval_val_loader):
        texts, pose_gt, m_length = batch['clip_text'], batch['pose'], batch['m_length']

        # ========== 生成运动 ==========
        # Stage 2: 生成 Q0
        mids = t2m_transformer.generate(
            texts,
            m_length // 4,         # token长度
            timesteps=18,          # 评估用更多步数（比训练时的10步更多）
            cond_scale=4.0,
            temperature=1.0,
            topk_filter_thres=0.9
        )

        # Stage 3: 生成 Q1-Q2
        pred_ids = res_model.generate(
            mids,
            texts,
            m_length // 4,
            temperature=1.0,
            cond_scale=5.0         # Stage 3用更高的CFG
        )

        # VAE解码
        pred_motions = vq_model.forward_decoder(pred_ids)
        # pred_motions: (B, T, D)

        # ========== 提取 embeddings ==========
        # Ground Truth embeddings
        et_gt, em_gt = eval_wrapper.get_co_embeddings(
            word_embs, pos_ohot, sent_len,
            pose_gt,              # GT motion
            m_length
        )
        # et_gt: (B, 512) - text embeddings
        # em_gt: (B, 512) - GT motion embeddings

        # Predicted embeddings
        et_pred, em_pred = eval_wrapper.get_co_embeddings(
            word_embs, pos_ohot, sent_len,
            pred_motions,         # Predicted motion
            m_length
        )
        # et_pred: (B, 512) - text embeddings (same as et_gt)
        # em_pred: (B, 512) - predicted motion embeddings

        # 收集所有 embeddings
        motion_gt_list.append(em_gt)
        motion_pred_list.append(em_pred)

    # ========== 计算指标 ==========
    # 1. FID
    # 2. Diversity
    # 3. R-Precision
    # 4. Matching Score
    # 5. Multimodality (前3个batch × 30次)
```

---

### 4️⃣ Multimodality 评估（特殊处理）

**Multimodality** 衡量同一文本生成的运动多样性。

```python
# 仅对前3个batch计算 Multimodality
num_mm_batch = 3

for i, batch in enumerate(eval_val_loader):
    if i < num_mm_batch:
        # ========== 同一文本生成30次 ==========
        motion_multimodality_batch = []
        for _ in range(30):
            # 生成运动（每次不同，因为有随机采样）
            mids = t2m_transformer.generate(texts, m_length // 4, ...)
            pred_ids = res_model.generate(mids, texts, m_length // 4, ...)
            pred_motions = vq_model.forward_decoder(pred_ids)

            # 提取 embedding
            _, em_pred = eval_wrapper.get_co_embeddings(
                word_embs, pos_ohot, sent_len,
                pred_motions, m_length
            )
            motion_multimodality_batch.append(em_pred.unsqueeze(1))

        # (B, 30, 512) - 每个文本生成30次
        motion_multimodality_batch = torch.cat(motion_multimodality_batch, dim=1)
        motion_multimodality.append(motion_multimodality_batch)

# 计算 Multimodality
motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
# shape: (num_mm_batch * batch_size, 30, 512)
multimodality = calculate_multimodality(motion_multimodality, 10)
```

**关键点**:
- ✅ 仅对前 **3 个 batch** 计算（计算量大）
- ✅ 每个文本生成 **30 次**
- ✅ 从30次中随机选10对计算距离

---

### 5️⃣ 指标计算详解

#### 1. FID (Fréchet Inception Distance)

```python
# 收集所有 embeddings
motion_gt_np = torch.cat(motion_gt_list, dim=0).cpu().numpy()    # (N, 512)
motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()  # (N, 512)

# 计算均值和协方差
gt_mu, gt_cov = calculate_activation_statistics(motion_gt_np)
pred_mu, pred_cov = calculate_activation_statistics(motion_pred_np)
# gt_mu, pred_mu: (512,)
# gt_cov, pred_cov: (512, 512)

# 计算 FID
fid = calculate_frechet_distance(gt_mu, gt_cov, pred_mu, pred_cov)
# FID公式: ||mu1 - mu2||^2 + Tr(C1 + C2 - 2*sqrt(C1*C2))
```

**FID 越低越好**: 表示生成运动的分布接近真实运动。

---

#### 2. Diversity

```python
# 使用 motion embeddings 计算
diversity_gt = calculate_diversity(motion_gt_np, diversity_times=300)
diversity_pred = calculate_diversity(motion_pred_np, diversity_times=300)

def calculate_diversity(activation, diversity_times):
    # activation: (N, 512)
    # 随机采样两组样本
    first_indices = np.random.choice(N, diversity_times, replace=False)
    second_indices = np.random.choice(N, diversity_times, replace=False)

    # 计算欧氏距离
    dist = linalg.norm(activation[first_indices] - activation[second_indices], axis=1)

    return dist.mean()  # 平均距离
```

**Diversity 适中最好**: 太低表示缺乏多样性，太高可能偏离真实分布。

---

#### 3. R-Precision (Top-1, Top-2, Top-3)

```python
# 计算每个batch
temp_R = calculate_R_precision(
    et_pred.cpu().numpy(),   # (B, 512) - text embeddings
    em_pred.cpu().numpy(),   # (B, 512) - motion embeddings
    top_k=3,
    sum_all=True
)
# temp_R: (3,) - [top1_count, top2_count, top3_count]

R_precision += temp_R  # 累加

# 最后归一化
R_precision = R_precision / nb_sample
# R_precision: (3,) - [top1_acc, top2_acc, top3_acc]

def calculate_R_precision(text_emb, motion_emb, top_k=3):
    # 计算距离矩阵
    dist_mat = euclidean_distance_matrix(text_emb, motion_emb)  # (B, B)

    # 排序: 每行找最近的运动
    argmax = np.argsort(dist_mat, axis=1)  # (B, B)

    # 检查GT是否在top-k中
    # GT索引: [0, 1, 2, ..., B-1]
    gt_indices = np.arange(B)
    top_k_mat = calculate_top_k(argmax, top_k)  # (B, 3)

    return top_k_mat.sum(axis=0)  # (3,)
```

**R-Precision 越高越好**: 表示给定文本能准确检索到对应运动。

---

#### 4. Matching Score

```python
# 计算每个batch
temp_match = euclidean_distance_matrix(
    et_pred.cpu().numpy(),   # (B, 512)
    em_pred.cpu().numpy()    # (B, 512)
).trace()  # 只取对角线（对应的文本-运动对）

matching_score_pred += temp_match

# 最后归一化
matching_score_pred = matching_score_pred / nb_sample
```

**Matching Score 越低越好**: 表示文本和运动的embedding距离越小。

---

#### 5. Multimodality

```python
multimodality = calculate_multimodality(motion_multimodality, 10)

def calculate_multimodality(activation, multimodality_times):
    # activation: (N, 30, 512) - N个文本，每个生成30次
    # 对每个文本，随机选10对生成结果
    first_indices = np.random.choice(30, multimodality_times, replace=False)
    second_indices = np.random.choice(30, multimodality_times, replace=False)

    # 计算距离
    dist = linalg.norm(
        activation[:, first_indices] - activation[:, second_indices],
        axis=2
    )  # (N, 10)

    return dist.mean()  # 平均距离
```

**Multimodality 适中最好**: 表示同一文本的多次生成有一定差异但不会太大。

---

### 6️⃣ 置信区间计算

```python
# 重复20次评估
for i in range(20):
    fid, div, R_prec, match, mm = evaluation_mask_transformer_test_plus_res(...)
    fid_list.append(fid)
    div_list.append(div)
    # ...

# 转为numpy数组
fid_array = np.array(fid_list)  # (20,)
div_array = np.array(div_list)

# 计算均值
fid_mean = np.mean(fid_array)
div_mean = np.mean(div_array)

# 计算95%置信区间 (1.96 * 标准误)
fid_conf = np.std(fid_array) * 1.96 / np.sqrt(20)
div_conf = np.std(div_array) * 1.96 / np.sqrt(20)

# 输出
print(f"FID: {fid_mean:.3f}, conf. {fid_conf:.3f}")
print(f"Diversity: {div_mean:.3f}, conf. {div_conf:.3f}")
print(f"TOP1: {top1_mean:.3f}, conf. {top1_conf:.3f}")
print(f"TOP2: {top2_mean:.3f}, conf. {top2_conf:.3f}")
print(f"TOP3: {top3_mean:.3f}, conf. {top3_conf:.3f}")
print(f"Matching: {match_mean:.3f}, conf. {match_conf:.3f}")
print(f"Multimodality: {mm_mean:.3f}, conf. {mm_conf:.3f}")
```

---

## 📋 Evaluation Wrapper 详解

### EvaluatorModelWrapper 架构

```python
class EvaluatorModelWrapper:
    def __init__(self, opt):
        # 加载3个预训练模型 (来自HumanML3D的评估模型)
        self.text_encoder = TextEncoderBiGRUCo(...)
        self.motion_encoder = MotionEncoderBiGRUCo(...)
        self.movement_encoder = MovementConvEncoder(...)

        # 加载预训练权重
        checkpoint = torch.load('checkpoints/t2m/Comp_v6_KLD005/model/finest.tar')
        self.text_encoder.load_state_dict(checkpoint['text_encoder'])
        self.motion_encoder.load_state_dict(checkpoint['motion_encoder'])
        self.movement_encoder.load_state_dict(checkpoint['movement_encoder'])

        # 设为eval模式
        self.text_encoder.eval()
        self.motion_encoder.eval()
        self.movement_encoder.eval()

    def get_co_embeddings(self, word_embs, pos_ohot, cap_lens, motions, m_lens):
        # 1. 编码运动
        movements = self.movement_encoder(motions[..., :-4])  # 去掉最后4维(脚接触)
        m_lens_tokens = m_lens // 4  # unit_length=4
        motion_embedding = self.motion_encoder(movements, m_lens_tokens)
        # motion_embedding: (B, 512)

        # 2. 编码文本
        text_embedding = self.text_encoder(word_embs, pos_ohot, cap_lens)
        # text_embedding: (B, 512)

        return text_embedding, motion_embedding
```

**关键点**:
- ✅ 使用 **预训练的 Text2Motion 模型** (HumanML3D)
- ✅ 将运动和文本都编码到 **512维空间**
- ✅ 在此空间计算所有指标

---

## 🎯 评估超参数

### 生成超参数（评估时）

```python
# Stage 2: Masked Transformer
timesteps = 18           # 比训练时(10)更多，质量更好
cond_scale = 4.0         # CFG强度
temperature = 1.0        # 采样温度
topk_filter_thres = 0.9  # Top-k过滤

# Stage 3: Residual Transformer
temperature = 1.0
cond_scale = 5.0         # 比Stage 2更高
```

### 评估超参数

```python
repeat_time = 20              # 重复评估次数（计算置信区间）
num_mm_batch = 3              # Multimodality评估的batch数
mm_repeats = 30               # 每个文本生成次数（Multimodality）
diversity_times = 300         # Diversity采样次数
multimodality_times = 10      # Multimodality采样次数
```

---

## 📊 HumanML3D 基准性能

| 指标 | MoMask | 说明 |
|------|--------|------|
| **FID ↓** | 0.080 | 生成质量 |
| **Diversity** | 9.50 | 多样性 |
| **R-Precision (Top-1) ↑** | 0.492 | 文本检索准确率 |
| **R-Precision (Top-2) ↑** | 0.502 | |
| **R-Precision (Top-3) ↑** | 0.534 | |
| **Matching Score ↓** | 2.799 | 文本-运动匹配 |
| **Multimodality** | 2.424 | 同文本多样性 |

---

## 🔧 MotionGPT 适配要点

### 1. Evaluation Wrapper

我们需要确保有 **预训练的评估模型**:

```python
# MotionGPT 可能已有类似的评估wrapper
from mGPT.metrics.t2m import TM2TMetrics

# 或者需要从HumanML3D加载预训练模型
eval_wrapper = EvaluatorModelWrapper(opt)
```

**检查点**:
- ✅ 是否有预训练的 Text/Motion encoder？
- ✅ 是否支持 512维 embedding 提取？
- ✅ 是否有 GloVe word embeddings？

### 2. 数据加载器

```python
# 需要提供以下字段
batch = {
    'word_embeddings': ...,  # GloVe embeddings
    'pos_one_hots': ...,     # POS tags
    'clip_text': ...,        # CLIP text (for generation)
    'sent_len': ...,         # Text lengths
    'pose': ...,             # GT motion
    'm_length': ...,         # Motion lengths
}
```

### 3. 评估脚本

```python
# MotionGPT评估接口
def evaluate_momask(model, test_loader, eval_wrapper):
    fid_list, div_list = [], []

    for repeat_id in range(20):
        for batch in test_loader:
            # 生成
            pred_motion = model.generate(batch['texts'], batch['lengths'], ...)

            # 提取embeddings
            et, em_gt = eval_wrapper.get_co_embeddings(..., batch['motion'], ...)
            _, em_pred = eval_wrapper.get_co_embeddings(..., pred_motion, ...)

            # 收集
            motion_gt_list.append(em_gt)
            motion_pred_list.append(em_pred)

        # 计算指标
        fid, div, R_prec, match, mm = compute_metrics(...)
        fid_list.append(fid)

    # 统计
    print(f"FID: {np.mean(fid_list):.3f} ± {np.std(fid_list)*1.96/np.sqrt(20):.3f}")
```

---

## 📝 评估命令

```bash
# 原始MoMask评估
python eval_t2m_trans_res.py \
  --name mask_transformer_model \
  --res_name res_transformer_model \
  --dataset_name t2m \
  --gpu_id 0 \
  --time_steps 18 \
  --cond_scale 4.0 \
  --temperature 1.0 \
  --topkr 0.9 \
  --ext final_eval
```

---

## 🎯 关键要点总结

### 评估流程核心

1. **重复20次**: 计算置信区间
2. **5大指标**: FID, Diversity, R-Precision, Matching, Multimodality
3. **Embedding空间**: 所有指标在512维embedding空间计算
4. **预训练模型**: 需要HumanML3D的评估模型

### 计算量

| 操作 | 次数 | 说明 |
|------|------|------|
| 生成运动 | 20 × test_size | 每次评估全部测试集 |
| Multimodality | 3 × 30 × batch_size | 前3个batch，每个30次 |
| 总计 | ~20,000+ 次生成 | 取决于测试集大小 |

### 与训练的差异

| 项目 | 训练 | 评估 |
|------|------|------|
| **timesteps** | 10 | 18 (更多步数) |
| **重复次数** | 1 | 20 |
| **Multimodality** | 不计算 | 计算 |
| **速度** | 快 | 慢 |

---

**总结**: MoMask评估采用标准的Text2Motion评估协议，在预训练的embedding空间计算5大指标，通过20次重复评估获得置信区间。我们需要确保MotionGPT有兼容的评估wrapper！
