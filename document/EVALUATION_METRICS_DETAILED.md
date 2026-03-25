# Evaluation Metrics: Detailed Technical Documentation

This document provides a comprehensive explanation of how evaluation metrics are computed in the MotionGPT sign language generation framework.

---

## Table of Contents

1. [Metric Configuration](#1-metric-configuration)
2. [MRMetrics: Motion Reconstruction Metrics](#2-mrmetrics-motion-reconstruction-metrics)
3. [TM2TMetrics: Text/Audio-to-Motion Metrics](#3-tm2tmetrics-textaudio-to-motion-metrics)
4. [Metric Initialization](#4-metric-initialization)
5. [Metric Update Pipeline](#5-metric-update-pipeline)
6. [Metric Computation and Logging](#6-metric-computation-and-logging)

---

## 1. Metric Configuration

### 1.1 Configuration via YAML Files

Metrics are configured in YAML config files under the `METRIC` section:

**File:** `configs/default.yaml`
```yaml
METRIC:
  TASK: 't2m'
  FORCE_IN_METER: True           # Convert units to millimeters
  DIST_SYNC_ON_STEP: True        # Sync across distributed processes
  MM_NUM_SAMPLES: 100
  MM_NUM_REPEATS: 30
  MM_NUM_TIMES: 10
  DIVERSITY_TIMES: 300
```

**Example Configurations:**

For VAE stage (reconstruction):
```yaml
# configs/deto_h2s.yaml
METRIC:
  TYPE: ["MRMetrics"]  # Motion Reconstruction metrics
```

For LM stage (generation):
```yaml
# configs/deto_h2s_rvq_hierarchical_tf.yaml
METRIC:
  TYPE: ["TM2TMetrics"]  # Text-to-motion metrics with DTW
```

### 1.2 Metric Selection in Model

The metric type is passed to the model via config:

**File:** `configs/default.yaml:57`
```yaml
model:
  params:
    metrics_dict: ${METRIC.TYPE}  # Injects metric list into model
```

---

## 2. MRMetrics: Motion Reconstruction Metrics

Used during **VAE training** to measure reconstruction quality.

**File:** `mGPT/metrics/mr.py`

### 2.1 Initialization

```python
class MRMetrics(Metric):
    def __init__(self,
                 njoints,
                 jointstype: str = "mmm",
                 force_in_meter: bool = True,
                 align_root: bool = True,
                 dist_sync_on_step=True,
                 **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        # Body part indices from SMPL-X model
        self.joint_part2idx = smpl_x.joint_part2idx
        self.vertex_part2idx = smpl_x.vertex_part2idx
        self.J_regressor = smpl_x.J_regressor

        # State variables for each dataset (how2sign, youtube3d)
        self.add_state("how2sign_count", default=torch.tensor(0))
        self.add_state("youtube3d_count", default=torch.tensor(0))

        # MPVPE metrics (vertex-level)
        self.add_state("how2sign_MPVPE_PA_all", default=torch.tensor([0.0]))
        self.add_state("how2sign_MPVPE_PA_hand", default=torch.tensor([0.0]))
        # ... (similar for lhand, rhand, face)

        # MPJPE metrics (joint-level)
        self.add_state("how2sign_MPJPE_PA_body", default=torch.tensor([0.0]))
        self.add_state("how2sign_MPJPE_PA_hand", default=torch.tensor([0.0]))
        # ... (same pattern for youtube3d)
```

### 2.2 Metrics Computed

| Metric Category | Description | Body Parts |
|----------------|-------------|------------|
| **MPVPE** | Mean Per-Vertex Position Error (no alignment) | all, hand, lhand, rhand, face |
| **MPVPE_PA** | MPVPE with Procrustes Alignment | all, hand, lhand, rhand, face |
| **MPJPE** | Mean Per-Joint Position Error (no alignment) | body, hand |
| **MPJPE_PA** | MPJPE with Procrustes Alignment | body, hand |

### 2.3 Update Function - Core Computation

**File:** `mGPT/metrics/mr.py:176-274`

```python
def update(self,
           feats_rst: Tensor,      # Reconstructed features
           feats_ref: Tensor,      # Reference features
           joints_rst: Tensor,     # Reconstructed joints
           joints_ref: Tensor,     # Reference joints
           vertices_rst: Tensor,   # Reconstructed vertices
           vertices_ref: Tensor,   # Reference vertices
           lengths: List[int],     # Sequence lengths
           src: List[str],         # Dataset source ('how2sign' or 'youtube3d')
           name: List[str]):       # Sample names

    # Reshape from (B*T, N, 3) to (B, T, N, 3)
    B = len(lengths)
    BT, N = joints_rst.shape[:2]
    joints_rst = joints_rst.reshape(B, BT//B, N, 3)
    joints_ref = joints_ref.reshape(B, BT//B, N, 3)
    vertices_rst = vertices_rst.reshape(B, BT//B, N, 3)
    vertices_ref = vertices_ref.reshape(B, BT//B, N, 3)

    # Move to CPU to avoid DDP errors
    joints_rst = joints_rst.detach().cpu()
    joints_ref = joints_ref.detach().cpu()
    vertices_rst = vertices_rst.detach().cpu()
    vertices_ref = vertices_ref.detach().cpu()

    for i in range(len(lengths)):
        cur_len = lengths[i]
        data_src = src[i] if src[i] in ['how2sign', 'youtube3d'] else 'how2sign'

        # Update frame count
        setattr(self, f'{data_src}_count',
                cur_len + getattr(self, f'{data_src}_count'))

        # Extract current sequence
        mesh_gt = vertices_ref[i, :cur_len, ...]
        mesh_out = vertices_rst[i, :cur_len, ...]

        # ===== 1. MPVPE_PA (All vertices with Procrustes Alignment) =====
        mesh_out_align = rigid_align_torch_batch(mesh_out, mesh_gt)
        value = torch.mean(
            torch.sqrt(torch.sum((mesh_out_align - mesh_gt) ** 2, dim=-1)),
            dim=-1
        ).sum()
        setattr(self, f"{data_src}_MPVPE_PA_all",
                getattr(self, f"{data_src}_MPVPE_PA_all") + value)

        # ===== 2. MPVPE (All vertices with root alignment only) =====
        # Align by pelvis joint
        pelvis_idx = smpl_x.J_regressor_idx['pelvis']
        mesh_out_align = (mesh_out -
                         joints_rst[i, :cur_len, pelvis_idx:pelvis_idx+1] +
                         joints_ref[i, :cur_len, pelvis_idx:pelvis_idx+1])
        value = torch.mean(
            torch.sqrt(torch.sum((mesh_out_align - mesh_gt) ** 2, dim=-1)),
            dim=-1
        ).sum()
        setattr(self, f"{data_src}_MPVPE_all",
                getattr(self, f"{data_src}_MPVPE_all") + value)

        # ===== 3. Hand-specific MPVPE_PA =====
        # Left hand
        mesh_gt_lhand = mesh_gt[:, smpl_x.hand_vertex_idx['left_hand'], :]
        mesh_out_lhand = mesh_out[:, smpl_x.hand_vertex_idx['left_hand'], :]
        mesh_out_lhand_align = rigid_align_torch_batch(mesh_out_lhand, mesh_gt_lhand)
        lhand_pa = torch.mean(
            torch.sqrt(torch.sum((mesh_out_lhand_align - mesh_gt_lhand) ** 2, dim=-1)),
            dim=-1
        ).sum()

        # Right hand
        mesh_gt_rhand = mesh_gt[:, smpl_x.hand_vertex_idx['right_hand'], :]
        mesh_out_rhand = mesh_out[:, smpl_x.hand_vertex_idx['right_hand'], :]
        mesh_out_rhand_align = rigid_align_torch_batch(mesh_out_rhand, mesh_gt_rhand)
        rhand_pa = torch.mean(
            torch.sqrt(torch.sum((mesh_out_rhand_align - mesh_gt_rhand) ** 2, dim=-1)),
            dim=-1
        ).sum()

        # Update metrics
        setattr(self, f"{data_src}_MPVPE_PA_lhand",
                getattr(self, f"{data_src}_MPVPE_PA_lhand") + lhand_pa)
        setattr(self, f"{data_src}_MPVPE_PA_rhand",
                getattr(self, f"{data_src}_MPVPE_PA_rhand") + rhand_pa)
        setattr(self, f"{data_src}_MPVPE_PA_hand",
                getattr(self, f"{data_src}_MPVPE_PA_hand") + (lhand_pa + rhand_pa) / 2.0)

        # ===== 4. Hand MPVPE (wrist-aligned) =====
        lwrist_idx = smpl_x.J_regressor_idx['lwrist']
        rwrist_idx = smpl_x.J_regressor_idx['rwrist']

        mesh_out_lhand_align = (mesh_out_lhand -
                               joints_rst[i, :cur_len, lwrist_idx:lwrist_idx+1] +
                               joints_ref[i, :cur_len, lwrist_idx:lwrist_idx+1])
        mesh_out_rhand_align = (mesh_out_rhand -
                               joints_rst[i, :cur_len, rwrist_idx:rwrist_idx+1] +
                               joints_ref[i, :cur_len, rwrist_idx:rwrist_idx+1])

        lhand = torch.mean(
            torch.sqrt(torch.sum((mesh_out_lhand_align - mesh_gt_lhand) ** 2, dim=-1)),
            dim=-1
        ).sum()
        rhand = torch.mean(
            torch.sqrt(torch.sum((mesh_out_rhand_align - mesh_gt_rhand) ** 2, dim=-1)),
            dim=-1
        ).sum()

        setattr(self, f"{data_src}_MPVPE_lhand",
                getattr(self, f"{data_src}_MPVPE_lhand") + lhand)
        setattr(self, f"{data_src}_MPVPE_rhand",
                getattr(self, f"{data_src}_MPVPE_rhand") + rhand)
        setattr(self, f"{data_src}_MPVPE_hand",
                getattr(self, f"{data_src}_MPVPE_hand") + (lhand + rhand) / 2.0)

        # ===== 5. Face MPVPE_PA =====
        mesh_gt_face = mesh_gt[:, smpl_x.face_vertex_idx, :]
        mesh_out_face = mesh_out[:, smpl_x.face_vertex_idx, :]
        mesh_out_face_align = rigid_align_torch_batch(mesh_out_face, mesh_gt_face)
        value = torch.mean(
            torch.sqrt(torch.sum((mesh_out_face_align - mesh_gt_face) ** 2, dim=-1)),
            dim=-1
        ).sum()
        setattr(self, f"{data_src}_MPVPE_PA_face",
                getattr(self, f"{data_src}_MPVPE_PA_face") + value)

        # ===== 6. MPJPE_PA (Joint-level metrics) =====
        # Body joints (14 joints regressed from vertices)
        joint_gt_body = torch.matmul(smpl_x.j14_regressor, mesh_gt)
        joint_out_body = torch.matmul(smpl_x.j14_regressor, mesh_out)
        joint_out_body_align = rigid_align_torch_batch(joint_out_body, joint_gt_body)
        value = torch.mean(
            torch.sqrt(torch.sum((joint_out_body_align - joint_gt_body) ** 2, dim=-1)),
            dim=-1
        ).sum()
        setattr(self, f"{data_src}_MPJPE_PA_body",
                getattr(self, f"{data_src}_MPJPE_PA_body") + value)

        # Body MPJPE (pelvis-aligned)
        joint_out_body_align = (joint_out_body -
                               joints_rst[i, :cur_len, pelvis_idx:pelvis_idx+1] +
                               joints_ref[i, :cur_len, pelvis_idx:pelvis_idx+1])
        value = torch.mean(
            torch.sqrt(torch.sum((joint_out_body_align - joint_gt_body) ** 2, dim=-1)),
            dim=-1
        ).sum()
        setattr(self, f"{data_src}_MPJPE_body",
                getattr(self, f"{data_src}_MPJPE_body") + value)

        # ===== 7. Hand MPJPE_PA =====
        # Regress hand joints from vertices
        joint_gt_lhand = torch.matmul(smpl_x.orig_hand_regressor['left'], mesh_gt)
        joint_out_lhand = torch.matmul(smpl_x.orig_hand_regressor['left'], mesh_out)
        joint_out_lhand_align = rigid_align_torch_batch(joint_out_lhand, joint_gt_lhand)

        joint_gt_rhand = torch.matmul(smpl_x.orig_hand_regressor['right'], mesh_gt)
        joint_out_rhand = torch.matmul(smpl_x.orig_hand_regressor['right'], mesh_out)
        joint_out_rhand_align = rigid_align_torch_batch(joint_out_rhand, joint_gt_rhand)

        value = (torch.mean(
                    torch.sqrt(torch.sum((joint_out_lhand_align - joint_gt_lhand) ** 2, dim=-1)),
                    dim=-1
                ).sum() +
                torch.mean(
                    torch.sqrt(torch.sum((joint_out_rhand_align - joint_gt_rhand) ** 2, dim=-1)),
                    dim=-1
                ).sum()) / 2.0
        setattr(self, f"{data_src}_MPJPE_PA_hand",
                getattr(self, f"{data_src}_MPJPE_PA_hand") + value)

        # Hand MPJPE (wrist-aligned)
        joint_out_lhand_align = (joint_out_lhand -
                                joints_rst[i, :cur_len, lwrist_idx:lwrist_idx+1] +
                                joints_ref[i, :cur_len, lwrist_idx:lwrist_idx+1])
        joint_out_rhand_align = (joint_out_rhand -
                                joints_rst[i, :cur_len, rwrist_idx:rwrist_idx+1] +
                                joints_ref[i, :cur_len, rwrist_idx:rwrist_idx+1])

        value = (torch.mean(
                    torch.sqrt(torch.sum((joint_out_lhand_align - joint_gt_lhand) ** 2, dim=-1)),
                    dim=-1
                ).sum() +
                torch.mean(
                    torch.sqrt(torch.sum((joint_out_rhand_align - joint_gt_rhand) ** 2, dim=-1)),
                    dim=-1
                ).sum()) / 2.0
        setattr(self, f"{data_src}_MPJPE_hand",
                getattr(self, f"{data_src}_MPJPE_hand") + value)
```

### 2.4 Rigid Alignment (Procrustes Analysis)

**File:** `mGPT/utils/human_models.py:265-324`

The Procrustes alignment removes global translation, rotation, and scale differences:

```python
def rigid_transform_3D_torch_batch(P, Q):
    """
    Computes optimal rotation and translation to align P to Q using Kabsch algorithm.

    Args:
        P: (B, N, 3) - predicted points
        Q: (B, N, 3) - ground truth points

    Returns:
        c: scale factor
        R: rotation matrix (B, 3, 3)
        t: translation vector (B, 3)
    """
    _, n, dim = P.shape

    # Step 1: Compute centroids
    centroid_P = torch.mean(P, dim=1, keepdims=True)  # (B, 1, 3)
    centroid_Q = torch.mean(Q, dim=1, keepdims=True)  # (B, 1, 3)

    # Step 2: Center the points
    p = P - centroid_P  # (B, N, 3)
    q = Q - centroid_Q  # (B, N, 3)

    # Step 3: Compute covariance matrix
    H = torch.matmul(p.transpose(1, 2), q) / n  # (B, 3, 3)

    # Step 4: SVD decomposition
    U, S, Vt = torch.linalg.svd(H)  # (B, 3, 3)

    # Step 5: Compute rotation (handle reflection)
    d = torch.det(torch.matmul(Vt.transpose(1, 2), U.transpose(1, 2)))
    flip = d < 0.0
    if flip.any().item():
        Vt_new = Vt.clone()
        Vt_new[flip, 2, :] *= -1
        R = torch.matmul(Vt_new.transpose(1, 2), U.transpose(1, 2))
    else:
        R = torch.matmul(Vt.transpose(1, 2), U.transpose(1, 2))

    # Step 6: Compute scale
    varP = torch.var(P, dim=1, unbiased=False).sum(dim=1)  # (B,)
    c = 1 / varP * S.sum(dim=1)  # (B,)
    c = c.unsqueeze(1).unsqueeze(2)  # (B, 1, 1)

    # Step 7: Compute translation
    t = centroid_Q - centroid_P  # (B, 1, 3)

    return c, R, t


def rigid_align_torch_batch(P, Q):
    """Apply rigid alignment transformation."""
    c, R, t = rigid_transform_3D_torch_batch(P, Q)
    # Apply: P_aligned = c * R * P + t
    P2 = torch.matmul(c * R, P.transpose(1, 2)).transpose(1, 2) + t.transpose(1, 2)
    return P2
```

### 2.5 Compute Final Metrics

**File:** `mGPT/metrics/mr.py:150-174`

```python
def compute(self, sanity_flag):
    """Compute final averaged metrics."""
    if self.force_in_meter:
        factor = 1000.0  # Convert to millimeters
    else:
        factor = 1.0

    mr_metrics = {}

    for name in self.MR_metrics:
        # Use appropriate count based on dataset
        if name.startswith('youtube3d_'):
            count = getattr(self, 'youtube3d_count')
        else:
            count = getattr(self, 'how2sign_count')

        # Average over all frames
        mr_metrics[name] = getattr(self, name) / max(count, 1e-6)

        # Convert position errors to millimeters
        if 'MPVPE' in name or 'MPJPE' in name:
            mr_metrics[name] = mr_metrics[name] * factor

    for name, v in mr_metrics.items():
        print(name, ': ', v)

    self.reset()  # Reset all states for next epoch

    return mr_metrics
```

---

## 3. TM2TMetrics: Text/Audio-to-Motion Metrics

Used during **LM stage** for text-to-motion or audio-to-motion generation evaluation.

**File:** `mGPT/metrics/t2m.py`

### 3.1 Key Differences from MRMetrics

1. **Uses DTW (Dynamic Time Warping)** to handle variable-length sequences
2. **Supports multiple datasets**: how2sign, csl, phoenix, youtube3d
3. **Per-sequence metrics** stored in `name2scores` dictionary

### 3.2 Initialization

```python
class TM2TMetrics(Metric):
    def __init__(self,
                 cfg,
                 dataname='humanml3d',
                 top_k=3,
                 R_size=32,
                 diversity_times=300,
                 dist_sync_on_step=True,
                 **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.cfg = cfg
        self.dataname = dataname
        self.name = "MPJPE, MPVPE DTW"

        # State counters for each dataset
        self.add_state("how2sign_count_seq", default=torch.tensor(0))
        self.add_state("csl_count_seq", default=torch.tensor(0))
        self.add_state("phoenix_count_seq", default=torch.tensor(0))
        self.add_state("youtube3d_count_seq", default=torch.tensor(0))

        # DTW-based MPJPE metrics
        self.add_state("how2sign_DTW_MPJPE_PA_lhand", default=torch.tensor([0.0]))
        self.add_state("how2sign_DTW_MPJPE_PA_rhand", default=torch.tensor([0.0]))
        self.add_state("how2sign_DTW_MPJPE_PA_body", default=torch.tensor([0.0]))
        # ... (similar for csl, phoenix, youtube3d)

        self.MR_metrics = [
            "how2sign_DTW_MPJPE_PA_lhand", "how2sign_DTW_MPJPE_PA_rhand",
            "how2sign_DTW_MPJPE_PA_body",
            "csl_DTW_MPJPE_PA_lhand", "csl_DTW_MPJPE_PA_rhand",
            "csl_DTW_MPJPE_PA_body",
            "phoenix_DTW_MPJPE_PA_lhand", "phoenix_DTW_MPJPE_PA_rhand",
            "phoenix_DTW_MPJPE_PA_body",
            "youtube3d_DTW_MPJPE_PA_lhand", "youtube3d_DTW_MPJPE_PA_rhand",
            "youtube3d_DTW_MPJPE_PA_body"
        ]

        # Store per-sample scores
        self.name2scores = defaultdict(dict)
```

### 3.3 Update Function with DTW

**File:** `mGPT/metrics/t2m.py:120-187`

```python
@torch.no_grad()
def update(self,
           feats_rst: Tensor,
           feats_ref: Tensor,
           joints_rst: Tensor,
           joints_ref: Tensor,
           vertices_rst: Tensor,
           vertices_ref: Tensor,
           lengths: List[int],       # Reference lengths
           lengths_rst: List[int],   # Generated lengths (may differ!)
           split: str,
           src: List[str],
           name: List[str]):

    # Reshape tensors
    B = len(lengths)
    BT, N = joints_rst.shape[:2]
    joints_rst = joints_rst.reshape(B, BT//B, N, 3)
    joints_ref = joints_ref.reshape(B, BT//B, N, 3)
    vertices_rst = vertices_rst.reshape(B, BT//B, N, 3)
    vertices_ref = vertices_ref.reshape(B, BT//B, N, 3)

    # Convert to numpy for DTW computation
    joints_rst = joints_rst.detach().cpu().numpy()
    joints_ref = joints_ref.detach().cpu().numpy()
    vertices_rst = vertices_rst.detach().cpu()
    vertices_ref = vertices_ref.detach().cpu()

    # Only compute DTW for validation/test
    if split in ['val', 'test']:
        part_lst = ['body', 'lhand', 'rhand']

        for i in range(len(lengths)):
            cur_len = lengths[i]         # Ground truth length
            rst_len = lengths_rst[i]     # Generated length
            mesh_gt = vertices_ref[i, :cur_len]
            mesh_out = vertices_rst[i, :rst_len]
            joints_rst_cur = joints_rst[i, :rst_len]   # (T_gen, N, 3)
            joints_ref_cur = joints_ref[i, :cur_len]   # (T_ref, N, 3)
            data_src = src[i]
            cur_name = name[i]

            setattr(self, f"{data_src}_count_seq",
                    getattr(self, f"{data_src}_count_seq") + 1)

            # ===== 1. Body DTW-MPJPE (actually DTW-JPE with align_idx=0) =====
            joint_idx = self.joint_part2idx['upper_body']

            # Distance function: root-aligned L2 distance
            dist_func = partial(l2_dist_align, wanted=joint_idx, align_idx=0)

            # Compute DTW
            value = dtw(joints_rst_cur, joints_ref_cur, dist_func)[0]

            setattr(self, f'{data_src}_DTW_MPJPE_PA_body',
                    getattr(self, f'{data_src}_DTW_MPJPE_PA_body') + value)
            self.name2scores[cur_name][f'{data_src}_DTW_MPJPE_PA_body'] = value

            # ===== 2. Left Hand DTW-MPJPE =====
            # Regress hand joints from vertices
            joint_gt_lhand = torch.matmul(
                smpl_x.orig_hand_regressor['left'],
                mesh_gt
            ).float().numpy()
            joint_out_lhand = torch.matmul(
                smpl_x.orig_hand_regressor['left'],
                mesh_out
            ).float().numpy()

            # DTW with root alignment at index 0 (wrist)
            dist_func = partial(l2_dist_align, align_idx=0)
            value = dtw(joint_out_lhand, joint_gt_lhand, dist_func)[0]

            setattr(self, f"{data_src}_DTW_MPJPE_PA_lhand",
                    getattr(self, f"{data_src}_DTW_MPJPE_PA_lhand") + value)
            self.name2scores[cur_name][f"{data_src}_DTW_MPJPE_PA_lhand"] = value

            # ===== 3. Right Hand DTW-MPJPE =====
            joint_gt_rhand = torch.matmul(
                smpl_x.orig_hand_regressor['right'],
                mesh_gt
            ).float().numpy()
            joint_out_rhand = torch.matmul(
                smpl_x.orig_hand_regressor['right'],
                mesh_out
            ).float().numpy()

            dist_func = partial(l2_dist_align, align_idx=0)
            value = dtw(joint_out_rhand, joint_gt_rhand, dist_func)[0]

            setattr(self, f"{data_src}_DTW_MPJPE_PA_rhand",
                    getattr(self, f"{data_src}_DTW_MPJPE_PA_rhand") + value)
            self.name2scores[cur_name][f"{data_src}_DTW_MPJPE_PA_rhand"] = value
```

### 3.4 DTW Algorithm Implementation

**File:** `mGPT/metrics/dtw.py:14-64`

```python
def dtw(x, y, dist, warp=1, w=inf, s=1.0):
    """
    Computes Dynamic Time Warping (DTW) of two sequences.

    Args:
        x: (N1, M) - generated sequence
        y: (N2, M) - reference sequence
        dist: distance function dist(x[i], y[j]) -> float
        warp: how many shifts are computed
        w: window size (Sakoe-Chiba band)
        s: diagonal preference weight

    Returns:
        distance: minimum DTW distance
        C: cost matrix
        D: accumulated cost matrix
        path: optimal warping path
    """
    r, c = len(x), len(y)

    # Initialize cost matrix
    if not isinf(w):
        D0 = full((r + 1, c + 1), inf)
        for i in range(1, r + 1):
            D0[i, max(1, i - w):min(c + 1, i + w + 1)] = 0
        D0[0, 0] = 0
    else:
        D0 = zeros((r + 1, c + 1))
        D0[0, 1:] = inf
        D0[1:, 0] = inf

    D1 = D0[1:, 1:]  # Working area

    # Step 1: Compute pairwise distances
    for i in range(r):
        for j in range(c):
            if (isinf(w) or (max(0, i - w) <= j <= min(c, i + w))):
                D1[i, j] = dist(x[i], y[j])  # Call distance function

    C = D1.copy()  # Store original costs

    # Step 2: Accumulate costs using dynamic programming
    for i in range(r):
        jrange = range(c) if isinf(w) else range(max(0, i - w), min(c, i + w + 1))
        for j in jrange:
            # Minimum of: diagonal, vertical, horizontal
            min_list = [D0[i, j]]  # Diagonal (preferred)
            for k in range(1, warp + 1):
                i_k = min(i + k, r)
                j_k = min(j + k, c)
                min_list += [D0[i_k, j] * s,   # Vertical (penalized by s)
                            D0[i, j_k] * s]    # Horizontal (penalized by s)
            D1[i, j] += min(min_list)

    # Step 3: Backtrack to find optimal path
    path = _traceback(D0)

    return D1[-1, -1], C, D1, path  # Return minimum distance
```

### 3.5 Distance Function for DTW

**File:** `mGPT/metrics/dtw.py:84-97`

```python
def l2_dist_align(x, y, wanted=None, align_idx=None):
    """
    Compute L2 distance between two frames with optional alignment.

    Args:
        x, y: (N, 3) numpy arrays - two frames of joints/vertices
        wanted: indices of joints to use
        align_idx: if None, use Procrustes; if int, align to that joint

    Returns:
        dist: mean L2 distance
    """
    # Alignment
    if align_idx is None:
        # Procrustes alignment
        x = rigid_align(x, y)
    else:
        # Root alignment (e.g., pelvis or wrist)
        x = x - x[align_idx:align_idx+1] + y[align_idx:align_idx+1]

    # Select subset of joints if specified
    if wanted is not None:
        x = x[wanted]
        y = y[wanted]

    # Compute mean L2 distance
    dist = np.mean(np.sqrt(((x - y)**2).sum(axis=1)))

    return dist
```

**Note:** In the current implementation:
- `align_idx=0` is used, which performs **root alignment** (not Procrustes)
- Despite the metric name "DTW_MPJPE_PA", it actually computes **DTW-JPE** when `align_idx=0`

### 3.6 Compute Final Metrics

**File:** `mGPT/metrics/t2m.py:103-116`

```python
@torch.no_grad()
def compute(self, sanity_flag):
    """Compute final averaged metrics."""
    mr_metrics = {}

    for name in self.metrics:
        # Extract dataset name
        d = name.split('_')[0]  # e.g., 'how2sign', 'youtube3d'

        # Average over sequences
        mr_metrics[name] = (getattr(self, name) /
                           max(getattr(self, f'{d}_count_seq'), 1e-6))

    for name, v in mr_metrics.items():
        print(name, ': ', v)

    self.reset()

    return mr_metrics
```

---

## 4. Metric Initialization

### 4.1 BaseMetrics Class

**File:** `mGPT/metrics/base.py:10-57`

```python
class BaseMetrics(nn.Module):
    def __init__(self, cfg, datamodule, debug, metrics_dict=None, **kwargs):
        super().__init__()

        njoints = datamodule.njoints
        data_name = datamodule.name

        # Get metric list from config
        metrics_to_init = metrics_dict if metrics_dict else cfg.METRIC.TYPE

        # Initialize TM2TMetrics if configured
        if 'TM2TMetrics' in metrics_to_init and data_name in ["humanml3d", "kit"]:
            self.TM2TMetrics = TM2TMetrics(
                cfg=cfg,
                dataname=data_name,
                diversity_times=30 if debug else cfg.METRIC.DIVERSITY_TIMES,
                dist_sync_on_step=cfg.METRIC.DIST_SYNC_ON_STEP,
            )

        # Initialize MRMetrics if configured
        if 'MRMetrics' in metrics_to_init:
            self.MRMetrics = MRMetrics(
                njoints=njoints,
                jointstype=cfg.DATASET.JOINT_TYPE,
                dist_sync_on_step=cfg.METRIC.DIST_SYNC_ON_STEP,
            )

        # Initialize other metrics (M2TMetrics, MMMetrics, PredMetrics)
        # ...
```

### 4.2 Model Configuration

**File:** `mGPT/models/base.py:212-213`

```python
def configure_metrics(self):
    """Called during model initialization."""
    self.metrics = BaseMetrics(datamodule=self.datamodule, **self.hparams)
```

---

## 5. Metric Update Pipeline

### 5.1 Training/Validation Step

**File:** `mGPT/models/mgpt.py:510-618`

```python
def allsplit_step(self, split: str, batch, batch_idx):
    """Main training/validation/test step."""

    lengths = batch['length']
    src = batch['src']      # Dataset source
    name = batch['name']    # Sample names

    # ========== TRAINING ==========
    if split == "train":
        if self.hparams.stage == "vae":
            rs_set = self.train_vae_forward(batch)
            loss = self._losses['losses_train'].update(rs_set)
        elif self.hparams.stage in ["lm_instruct", "lm_pretrain", "lm_rvq_hierarchical"]:
            rs_set = self.train_lm_forward(batch)
            loss = self._losses['losses_train'].update(rs_set)

    # ========== VALIDATION/TEST ==========
    elif split in ["val", "test"]:
        if self.hparams.stage == "vae":
            # VAE evaluation
            rs_set = self.val_vae_forward(batch, split)

            # Update MRMetrics
            getattr(self.metrics, 'MRMetrics').update(
                feats_rst=rs_set["m_rst"],
                feats_ref=rs_set["m_ref"],
                joints_rst=rs_set["joints_rst"],
                joints_ref=rs_set["joints_ref"],
                vertices_rst=rs_set["vertices_rst"],
                vertices_ref=rs_set["vertices_ref"],
                lengths=lengths,
                src=src,
                name=name
            )

        elif self.hparams.stage in ["lm_instruct", "lm_pretrain", "lm_rvq_hierarchical"]:
            if self.hparams.task in ["t2m", "a2m"]:
                # Text/Audio-to-Motion evaluation
                rs_set = self.val_t2m_forward(batch)

                # Get configured metric
                metric_name = (self.hparams.metrics_dict[0]
                              if self.hparams.metrics_dict else 'MRMetrics')

                if hasattr(self.metrics, metric_name):
                    if metric_name == 'MRMetrics':
                        # MRMetrics: uses generated lengths
                        getattr(self.metrics, metric_name).update(
                            feats_rst=rs_set["m_rst"],
                            feats_ref=rs_set["m_ref"],
                            joints_rst=rs_set["joints_rst"],
                            joints_ref=rs_set["joints_ref"],
                            vertices_rst=rs_set["vertices_rst"],
                            vertices_ref=rs_set["vertices_ref"],
                            lengths=rs_set['lengths_rst'],  # Generated lengths
                            src=src,
                            name=name
                        )
                    else:  # TM2TMetrics
                        # TM2TMetrics: passes both reference and generated lengths
                        getattr(self.metrics, metric_name).update(
                            feats_rst=rs_set["m_rst"],
                            feats_ref=rs_set["m_ref"],
                            joints_rst=rs_set["joints_rst"],
                            joints_ref=rs_set["joints_ref"],
                            vertices_rst=rs_set["vertices_rst"],
                            vertices_ref=rs_set["vertices_ref"],
                            lengths=lengths,                # Reference lengths
                            lengths_rst=rs_set['lengths_rst'],  # Generated lengths
                            split=split,
                            src=src,
                            name=name
                        )

    return loss if split == "train" else rs_set
```

---

## 6. Metric Computation and Logging

### 6.1 Validation Epoch End

**File:** `mGPT/models/base.py:80-94`

```python
def on_validation_epoch_end(self):
    """Called at the end of validation epoch."""
    # Step and loss logging
    dico = self.step_log_dict()
    dico.update(self.loss_log_dict('train'))
    dico.update(self.loss_log_dict('val'))

    # Compute and log metrics
    dico.update(self.metrics_log_dict())

    # Write to logger (wandb, tensorboard)
    if not self.trainer.sanity_checking:
        self.log_dict(dico, sync_dist=True, rank_zero_only=True)
```

### 6.2 Metrics Log Dictionary

**File:** `mGPT/models/base.py:174-193`

```python
def metrics_log_dict(self):
    """Compute and format metrics for logging."""

    # Get metric list
    if self.trainer.datamodule.is_mm and "TM2TMetrics" in self.hparams.metrics_dict:
        metrics_dicts = ['MMMetrics']
    else:
        metrics_dicts = self.hparams.metrics_dict

    # Compute all metrics
    metrics_log_dict = {}
    for metric in metrics_dicts:
        # Call compute() method
        metrics_dict = getattr(self.metrics, metric).compute(
            sanity_flag=self.trainer.sanity_checking
        )

        # Format for logging
        metrics_log_dict.update({
            f"Metrics/{metric}": value.item()
            for metric, value in metrics_dict.items()
        })

    return metrics_log_dict
```

### 6.3 Test Epoch End with Score Saving

**File:** `mGPT/models/base.py:96-139`

```python
def on_test_epoch_end(self):
    """Called at the end of test epoch."""

    # Print per-sample scores for LM stage
    if 'lm' in self.hparams.stage:
        name2scores = getattr(self.metrics.TM2TMetrics, 'name2scores')
        metrics = [
            "how2sign_DTW_MPJPE_PA_lhand", "how2sign_DTW_MPJPE_PA_rhand",
            "how2sign_DTW_MPJPE_PA_body",
            "csl_DTW_MPJPE_PA_lhand", "csl_DTW_MPJPE_PA_rhand",
            "csl_DTW_MPJPE_PA_body",
            "phoenix_DTW_MPJPE_PA_lhand", "phoenix_DTW_MPJPE_PA_rhand",
            "phoenix_DTW_MPJPE_PA_body",
            "youtube3d_DTW_MPJPE_PA_lhand", "youtube3d_DTW_MPJPE_PA_rhand",
            "youtube3d_DTW_MPJPE_PA_body"
        ]

        # Aggregate per-sample scores
        scores, count = {}, {}
        for m in metrics:
            scores[m] = count[m] = 0

        for name, value_dict in name2scores.items():
            for n, val in value_dict.items():
                scores[n] = scores[n] + val
                count[n] = count[n] + 1

        for k in scores.keys():
            scores[k] = scores[k] / max(count[k], 1)

        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        print('rank: ', rank, scores)

    # Log aggregated metrics
    dico = self.metrics_log_dict()
    if not self.trainer.sanity_checking:
        self.log_dict(dico, sync_dist=True, rank_zero_only=True)

    # Save per-sample scores to JSON
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    save_dir = os.path.join(self.output_dir, f'{self.hparams.cfg.TEST.SPLIT}_rank_{rank}')
    os.makedirs(save_dir, exist_ok=True)

    if 'lm' in self.hparams.stage:
        with open(os.path.join(save_dir, 'test_scores.json'), 'w') as f:
            json.dump(getattr(self.metrics.TM2TMetrics, 'name2scores'), f)
    elif 'vae' in self.hparams.stage:
        with open(os.path.join(save_dir, 'test_scores.json'), 'w') as f:
            json.dump(getattr(self.metrics.MRMetrics, 'name2scores'), f)
```

---

## Summary

### Metric Types by Stage

| Training Stage | Metric Type | Key Metrics | Alignment Method |
|---------------|-------------|-------------|------------------|
| **VAE** | MRMetrics | MPVPE, MPVPE_PA, MPJPE, MPJPE_PA | Procrustes + Root |
| **LM** | TM2TMetrics | DTW_MPJPE_PA (body, lhand, rhand) | DTW + Root |

### Key Computational Steps

1. **MRMetrics (VAE stage)**:
   - Frame-level comparison (no DTW)
   - Procrustes alignment for PA metrics
   - Root joint alignment for non-PA metrics
   - Separate metrics for all vertices, body joints, hands, face

2. **TM2TMetrics (LM stage)**:
   - DTW alignment between generated and reference sequences
   - Root alignment within each frame pair
   - Separate metrics per dataset (how2sign, csl, phoenix, youtube3d)
   - Per-sample scores stored for analysis

### Units

- All position errors (MPVPE, MPJPE) are in **millimeters** when `FORCE_IN_METER: True`
- DTW distances are accumulated over the optimal warping path

### Output Files

- **Logged metrics**: Written to wandb/tensorboard during training
- **Per-sample scores**: Saved to `{output_dir}/{split}_rank_{rank}/test_scores.json`
- **Format**: JSON dictionary mapping sample names to metric values

---

## References

- **Procrustes Alignment**: Kabsch algorithm - https://hunterheidenreich.com/posts/kabsch_algorithm/
- **DTW Implementation**: Based on https://github.com/pierre-rouanet/dtw
- **SMPL-X Model**: Body model used for vertex and joint regressors
