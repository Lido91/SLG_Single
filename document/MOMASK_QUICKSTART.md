# MoMask 训练快速启动指南

本指南提供从零开始训练 MoMask 模型的完整步骤。

---

## 📋 前提条件检查

### 1. 环境依赖

```bash
# 检查 Python 版本
python --version  # 需要 Python 3.8+

# 检查 PyTorch
python -c "import torch; print(torch.__version__)"  # 需要 PyTorch 1.12+

# 检查 CUDA
python -c "import torch; print(torch.cuda.is_available())"  # 应返回 True

# 安装必要的包
pip install pytorch-lightning  # Lightning 框架
pip install einops            # 张量操作
pip install clip              # CLIP 文本编码器
```

### 2. 数据准备

检查 How2Sign 数据是否就绪：

```bash
# 检查数据路径
ls /data/hwu/slg_data/How2Sign/

# 应包含:
# - train/  (训练数据)
# - val/    (验证数据)
# - test/   (测试数据)
# - mean.pt, std.pt (归一化参数)
```

### 3. 预训练 RVQ-VAE 检查

```bash
# 检查 VAE 检查点是否存在
ls experiments/mgpt/DETO_RVQ_wholebody_3/checkpoints/

# 应该有:
# min-how2sign_MPJPE_PA_handepoch=489.ckpt
```

如果没有，需要先训练 RVQ-VAE：
```bash
python train.py --cfg configs/deto_h2s_rvq_3.yaml --nodebug
```

---

## 🚀 训练流程

MoMask 采用 **三阶段训练**，但我们可以**跳过 Stage 1**（使用已有的 RVQ-VAE）。

```
Stage 1: RVQ-VAE (✅ 已完成 - 使用 deto_h2s_rvq_3)
    ↓
Stage 2: Masked Transformer (⭐ 从这里开始)
    ↓
Stage 3: Residual Transformer
    ↓
推理/评估
```

---

## 🎯 Stage 2: Masked Transformer 训练

### 配置文件检查

```bash
# 查看配置
cat configs/momask_h2s_stage2.yaml

# 关键参数确认:
# - PRETRAINED_VAE: experiments/mgpt/DETO_RVQ_wholebody_3/checkpoints/min-how2sign_MPJPE_PA_handepoch=489.ckpt
# - BATCH_SIZE: 64
# - END_EPOCH: 500
# - LR: 2e-4
```

### 启动训练

```bash
# 方式 1: 基础训练（推荐新手）
python train.py \
  --cfg configs/momask_h2s_stage2.yaml \
  --nodebug

# 方式 2: 指定 GPU（多卡训练）
python train.py \
  --cfg configs/momask_h2s_stage2.yaml \
  --nodebug \
  --device 0,1,2

# 方式 3: 从检查点恢复训练
python train.py \
  --cfg configs/momask_h2s_stage2.yaml \
  --nodebug \
  --resume experiments/mgpt/MoMask_H2S_Stage2/checkpoints/latest.ckpt
```

### 训练监控

```bash
# 查看 WandB 日志（浏览器）
# 访问: https://wandb.ai/your-username/SLG

# 或使用 TensorBoard（如果配置了）
tensorboard --logdir experiments/mgpt/MoMask_H2S_Stage2/logs

# 查看最新检查点
ls -lht experiments/mgpt/MoMask_H2S_Stage2/checkpoints/
```

### 预期输出

```
Epoch 0:
  Train/loss: 4.5123, Train/acc: 0.1234, Train/lr: 0.0002
Epoch 1:
  Train/loss: 3.8234, Train/acc: 0.2456, Train/lr: 0.0002
  Val/loss: 4.1234, Val/acc: 0.2123
...
Epoch 100:
  Train/loss: 1.2345, Train/acc: 0.7234
  Val/loss: 1.5678, Val/acc: 0.6890
```

### 训练时间估计

| 硬件 | 批次大小 | 每 Epoch 时间 | 总时间 (500 epochs) |
|------|---------|--------------|-------------------|
| 1x A100 | 64 | ~30 min | ~10 天 |
| 3x A100 | 64 | ~12 min | ~4 天 |
| 8x A100 | 64 | ~6 min | ~2 天 |

### 何时停止训练？

观察验证准确率：
- ✅ **良好**: Val/acc > 70%
- ✅ **优秀**: Val/acc > 75%
- ⚠️ **可能过拟合**: Train/acc >> Val/acc (差距>10%)

---

## 🎯 Stage 3: Residual Transformer 训练

### 前提条件

确保 Stage 2 已完成训练：

```bash
# 检查 Stage 2 检查点
ls experiments/mgpt/MoMask_H2S_Stage2/checkpoints/

# 应该有 best.ckpt 或 latest.ckpt
```

### 启动训练

```bash
# Stage 3 训练
python train.py \
  --cfg configs/momask_h2s_stage3.yaml \
  --nodebug

# 多卡训练
python train.py \
  --cfg configs/momask_h2s_stage3.yaml \
  --nodebug \
  --device 0,1,2
```

### 预期输出

```
Epoch 0:
  Train/loss: 3.2123, Train/acc: 0.2345
Epoch 1:
  Train/loss: 2.8234, Train/acc: 0.3456
...
Epoch 100:
  Train/loss: 0.9876, Train/acc: 0.7890
  Val Metrics: MPJPE: 45.2, FID: 0.12
```

### 训练时间估计

与 Stage 2 类似：
- **3x A100**: ~4 天
- **8x A100**: ~2 天

### 验证指标

Stage 3 使用运动重建指标（用 GT Q0 预测 Q1-Q2）：
- ✅ **MPJPE**: < 50mm
- ✅ **FID**: < 0.15

---

## 🔍 推理测试

### Stage 2 推理（仅 Q0）

```bash
# 测试 Stage 2 模型
python test.py \
  --cfg configs/momask_h2s_stage2.yaml \
  --checkpoint experiments/mgpt/MoMask_H2S_Stage2/checkpoints/best.ckpt
```

### 完整推理（Stage 2 + Stage 3）

需要手动加载两个模型（暂时）：

```python
# inference_script.py
import torch
from mGPT.models.mgpt_momask import MoMask

# 加载配置
model = MoMask.load_from_checkpoint(
    'experiments/mgpt/MoMask_H2S_Stage2/checkpoints/best.ckpt'
)

# 手动加载 Stage 3
stage3_ckpt = torch.load('experiments/mgpt/MoMask_H2S_Stage3/checkpoints/best.ckpt')
model.res_transformer.load_state_dict(stage3_ckpt['state_dict'], strict=False)

# 推理
texts = ["a person walks forward", "a person waves"]
lengths = [120, 80]
motions = model.generate(texts, lengths)
```

---

## 📊 评估

### 快速评估

```bash
# 评估 Stage 2（仅生成质量）
python test.py \
  --cfg configs/momask_h2s_stage2.yaml \
  --checkpoint experiments/mgpt/MoMask_H2S_Stage2/checkpoints/best.ckpt \
  --split test

# 评估 Stage 3（重建质量）
python test.py \
  --cfg configs/momask_h2s_stage3.yaml \
  --checkpoint experiments/mgpt/MoMask_H2S_Stage3/checkpoints/best.ckpt \
  --split test
```

### 完整评估（FID, Diversity, R-Precision）

需要等待完整评估流程集成（参考 `MOMASK_EVALUATION_PIPELINE.md`）。

---

## ⚠️ 常见问题

### 1. OOM (Out of Memory)

```bash
# 解决方案 1: 减小批次大小
# 编辑 configs/momask_h2s_stage2.yaml
TRAIN:
  BATCH_SIZE: 32  # 从 64 改为 32

# 解决方案 2: 使用梯度累积
# 在配置中添加
TRAIN:
  ACCUMULATE_GRAD_BATCHES: 2  # 有效批次 = 32 * 2 = 64
```

### 2. CLIP 加载失败

```bash
# 错误: ModuleNotFoundError: No module named 'clip'
pip install git+https://github.com/openai/CLIP.git

# 如果网络问题，手动下载
git clone https://github.com/openai/CLIP.git
cd CLIP && pip install -e .
```

### 3. VAE 检查点路径错误

```bash
# 错误: FileNotFoundError: experiments/mgpt/DETO_RVQ_wholebody_3/...

# 检查实际路径
find experiments -name "*rvq*.ckpt" -type f

# 更新配置文件中的 PRETRAINED_VAE 路径
```

### 4. 数据加载慢

```bash
# 增加 workers
TRAIN:
  NUM_WORKERS: 16  # 根据 CPU 核心数调整
```

### 5. 训练不收敛

检查学习率是否合适：
```yaml
# 如果 loss 震荡，降低学习率
TRAIN:
  OPTIM:
    params:
      lr: 1e-4  # 从 2e-4 降低到 1e-4
```

---

## 🛠️ 调试模式

### 开启调试（快速验证）

```bash
# 使用小批次和少量 epoch 快速验证流程
python train.py \
  --cfg configs/momask_h2s_stage2.yaml \
  --debug \
  --max_epochs 2 \
  --limit_train_batches 10 \
  --limit_val_batches 5
```

### 检查数据加载

```python
# test_dataloader.py
from mGPT.data.H2S import H2SDataModule

dm = H2SDataModule(cfg)
dm.setup('fit')
batch = next(iter(dm.train_dataloader()))

print("Batch keys:", batch.keys())
print("Motion shape:", batch['motion'].shape)
print("Text sample:", batch['text'][0])
print("Length:", batch['length'][0])
```

---

## 📈 训练监控检查表

### 每 10 个 Epoch 检查

- [ ] Loss 是否下降？
- [ ] Accuracy 是否提升？
- [ ] 验证指标是否改善？
- [ ] 学习率是否正常？
- [ ] GPU 利用率是否充分？
- [ ] 是否有 NaN 或 Inf？

### 每 50 个 Epoch

- [ ] 可视化生成样本
- [ ] 检查过拟合（Train vs Val）
- [ ] 保存中间检查点
- [ ] 评估生成质量

---

## 🎯 快速开始模板

### 最小化训练脚本

```bash
#!/bin/bash
# train_momask.sh

# Stage 2: Masked Transformer
echo "=== Starting Stage 2 Training ==="
python train.py \
  --cfg configs/momask_h2s_stage2.yaml \
  --nodebug \
  --device 0,1,2

# 等待 Stage 2 完成...

# Stage 3: Residual Transformer
echo "=== Starting Stage 3 Training ==="
python train.py \
  --cfg configs/momask_h2s_stage3.yaml \
  --nodebug \
  --device 0,1,2

echo "=== Training Complete ==="
```

### 后台运行（推荐）

```bash
# 使用 nohup 后台运行
nohup bash train_momask.sh > train.log 2>&1 &

# 查看日志
tail -f train.log

# 查看进程
ps aux | grep python
```

### 使用 screen/tmux

```bash
# 创建 screen 会话
screen -S momask_train

# 运行训练
python train.py --cfg configs/momask_h2s_stage2.yaml --nodebug

# 分离: Ctrl+A, D
# 重新连接: screen -r momask_train
```

---

## 📝 训练日志示例

### 正常训练日志

```
[2024-02-20 10:00:00] Initializing MoMask Stage 2...
[2024-02-20 10:00:05] Loading RVQ-VAE from experiments/mgpt/DETO_RVQ_wholebody_3/...
[2024-02-20 10:00:10] VAE loaded successfully! Freezing weights...
[2024-02-20 10:00:15] Loading CLIP ViT-B/32...
[2024-02-20 10:00:20] CLIP loaded successfully!
[2024-02-20 10:00:25] Starting training...

Epoch 0:   0%|          | 0/1250 [00:00<?, ?it/s]
Epoch 0:  10%|█         | 125/1250 [05:00<45:00,  0.42it/s, loss=4.23, acc=0.15]
Epoch 0:  20%|██        | 250/1250 [10:00<40:00,  0.42it/s, loss=3.89, acc=0.18]
...
Epoch 0: 100%|██████████| 1250/1250 [50:00<00:00,  0.42it/s, loss=2.34, acc=0.35]

Validation: 100%|██████████| 50/50 [02:00<00:00,  0.42it/s]
Val/loss: 2.56, Val/acc: 0.32

Saved checkpoint: experiments/mgpt/MoMask_H2S_Stage2/checkpoints/epoch=0.ckpt
```

---

## ✅ 完整训练检查表

### 开始前

- [ ] 检查 GPU 可用性
- [ ] 确认数据路径正确
- [ ] 确认 VAE 检查点存在
- [ ] 安装所有依赖包
- [ ] 配置 WandB（可选）

### Stage 2 训练

- [ ] 启动训练脚本
- [ ] 验证第一个 batch 正常
- [ ] 监控前 10 个 epoch
- [ ] 确认 loss 下降趋势
- [ ] 保存最佳检查点

### Stage 3 训练

- [ ] 确认 Stage 2 完成
- [ ] 更新配置中的检查点路径
- [ ] 启动 Stage 3 训练
- [ ] 监控重建指标
- [ ] 保存最终模型

### 推理测试

- [ ] 加载两个阶段模型
- [ ] 测试文本生成
- [ ] 可视化结果
- [ ] 计算评估指标

---

## 🚀 开始训练！

最简单的启动命令：

```bash
# 1. 进入项目目录
cd /home/student/hwu/Workplace/MotionGPT

# 2. 检查 VAE
ls experiments/mgpt/DETO_RVQ_wholebody_3/checkpoints/

# 3. 开始 Stage 2 训练
python train.py --cfg configs/momask_h2s_stage2.yaml --nodebug

# 4. 等待完成后，开始 Stage 3
python train.py --cfg configs/momask_h2s_stage3.yaml --nodebug
```

就这么简单！祝训练顺利！🎉