from typing import List
from collections import defaultdict

import torch
from torch import Tensor
from torchmetrics import Metric
from torch.nn.functional import smooth_l1_loss

from .utils import *
from mGPT.utils.human_models import rigid_align, rigid_align_torch_batch, smpl_x


# motion reconstruction metric
class MRMetrics(Metric):

    def __init__(self,
                 njoints,
                 jointstype: str = "mmm",
                 force_in_meter: bool = True,
                 align_root: bool = True,
                 dist_sync_on_step=True,
                 **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.name = 'Motion Reconstructions'
        self.name2scores = defaultdict(dict)
        self.jointstype = jointstype
        self.align_root = align_root
        self.force_in_meter = force_in_meter

        self.joint_part2idx = smpl_x.joint_part2idx
        self.vertex_part2idx = smpl_x.vertex_part2idx
        self.smplx_part2idx = {'upper_body': list(range(30)), 'lhand': list(range(30, 75)), 'rhand': list(range(75, 120)), 'hand': list(range(30, 120)), 'face': list(range(120, 133))}
        self.J_regressor = smpl_x.J_regressor

        self.add_state("how2sign_count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("how2sign_count_seq",
                       default=torch.tensor(0),
                       dist_reduce_fx="sum")

        self.add_state("how2sign_MPVPE_PA_all",
                       default=torch.tensor([0.0]),
                       dist_reduce_fx="sum")
        self.add_state("how2sign_MPVPE_PA_hand",
                       default=torch.tensor([0.0]),
                       dist_reduce_fx="sum")
        self.add_state("how2sign_MPVPE_PA_lhand",
                       default=torch.tensor([0.0]),
                       dist_reduce_fx="sum")
        self.add_state("how2sign_MPVPE_PA_rhand",
                       default=torch.tensor([0.0]),
                       dist_reduce_fx="sum")
        self.add_state("how2sign_MPVPE_PA_face",
                       default=torch.tensor([0.0]),
                       dist_reduce_fx="sum")

        self.add_state("how2sign_MPVPE_all",
                       default=torch.tensor([0.0]),
                       dist_reduce_fx="sum")
        self.add_state("how2sign_MPVPE_hand",
                       default=torch.tensor([0.0]),
                       dist_reduce_fx="sum")
        self.add_state("how2sign_MPVPE_lhand",
                       default=torch.tensor([0.0]),
                       dist_reduce_fx="sum")
        self.add_state("how2sign_MPVPE_rhand",
                       default=torch.tensor([0.0]),
                       dist_reduce_fx="sum")
        self.add_state("how2sign_MPVPE_face",
                       default=torch.tensor([0.0]),
                       dist_reduce_fx="sum")
        
        self.add_state("how2sign_MPJPE_PA_body",
                       default=torch.tensor([0.0]),
                       dist_reduce_fx="sum")
        self.add_state("how2sign_MPJPE_PA_hand",
                       default=torch.tensor([0.0]),
                       dist_reduce_fx="sum")

        self.add_state("how2sign_MPJPE_body",
                       default=torch.tensor([0.0]),
                       dist_reduce_fx="sum")
        self.add_state("how2sign_MPJPE_hand",
                       default=torch.tensor([0.0]),
                       dist_reduce_fx="sum")
        
        m = ["MPVPE_PA_all", "MPVPE_PA_hand", "MPVPE_PA_lhand", "MPVPE_PA_rhand", "MPVPE_PA_face",
            "MPJPE_PA_body", "MPJPE_PA_hand", "MPJPE_body", "MPJPE_hand",
            "MPVPE_all", "MPVPE_hand", "MPVPE_lhand", "MPVPE_rhand", "MPVPE_face"]
        self.MR_metrics = []
        for m_ in m:
            self.MR_metrics.append(f'how2sign_{m_}')

        # All metric
        self.metrics = self.MR_metrics

    def compute(self, sanity_flag):
        if self.force_in_meter:
            factor = 1000.0
        else:
            factor = 1.0

        mr_metrics = {}

        for name in self.MR_metrics:
            mr_metrics[name] = getattr(self, name) / max(getattr(self, 'how2sign_count'), 1e-6)
            if 'MPVPE' in name or 'MPJPE' in name:
                mr_metrics[name] = mr_metrics[name] * factor

        for name, v in mr_metrics.items():
            print(name, ': ', v)
        
        # Reset
        self.reset()
        
        return mr_metrics

    def update(self, 
               feats_rst: Tensor, feats_ref: Tensor,
               joints_rst: Tensor, joints_ref: Tensor,
               vertices_rst: Tensor, vertices_ref: Tensor,
               lengths: List[int], src: List[str], name: List[str]):
        assert joints_rst.shape == joints_ref.shape
        # assert joints_rst.dim() == 4
        # (bs, seq, njoint=22, 3)

        # print(lengths, joints_rst.shape)
        B = len(lengths)
        BT, N = joints_rst.shape[:2]
        joints_rst = joints_rst.reshape(B, BT//B, N, 3)
        joints_ref = joints_ref.reshape(B, BT//B, N, 3)
        BT, N = vertices_rst.shape[:2]
        vertices_rst = vertices_rst.reshape(B, BT//B, N, 3)
        vertices_ref = vertices_ref.reshape(B, BT//B, N, 3)

        # avoid cuda error of DDP in pampjpe
        joints_rst = joints_rst.detach().cpu()
        joints_ref = joints_ref.detach().cpu()
        vertices_rst = vertices_rst.detach().cpu()
        vertices_ref = vertices_ref.detach().cpu()
        
        for i in range(len(lengths)):
            cur_len = lengths[i]
            data_src = 'how2sign'
            setattr(self, 'how2sign_count', cur_len + getattr(self, 'how2sign_count'))
            
            mesh_gt = vertices_ref[i, :cur_len, ...]
            mesh_out = vertices_rst[i, :cur_len, ...]
            mesh_out_align = rigid_align_torch_batch(mesh_out, mesh_gt)
            value = torch.mean(torch.sqrt(torch.sum((mesh_out_align - mesh_gt) ** 2, dim=-1)), dim=-1).sum()
            setattr(self, f"{data_src}_MPVPE_PA_all", getattr(self, f"{data_src}_MPVPE_PA_all") + value)

            mesh_out_align = mesh_out - joints_rst[i, :cur_len, smpl_x.J_regressor_idx['pelvis']:smpl_x.J_regressor_idx['pelvis']+1] + joints_ref[i, :cur_len, smpl_x.J_regressor_idx['pelvis']:smpl_x.J_regressor_idx['pelvis']+1]
            setattr(self, f"{data_src}_MPVPE_all", getattr(self, f"{data_src}_MPVPE_PA_all") + torch.mean(torch.sqrt(torch.sum((mesh_out_align - mesh_gt) ** 2, dim=-1)), dim=-1).sum())

            mesh_gt_lhand = mesh_gt[:, smpl_x.hand_vertex_idx['left_hand'], :]
            mesh_out_lhand = mesh_out[:, smpl_x.hand_vertex_idx['left_hand'], :]
            mesh_gt_rhand = mesh_gt[:, smpl_x.hand_vertex_idx['right_hand'], :]
            mesh_out_rhand = mesh_out[:, smpl_x.hand_vertex_idx['right_hand'], :]
            mesh_out_lhand_align = rigid_align_torch_batch(mesh_out_lhand, mesh_gt_lhand)
            mesh_out_rhand_align = rigid_align_torch_batch(mesh_out_rhand, mesh_gt_rhand)
            lhand_pa = torch.mean(torch.sqrt(torch.sum((mesh_out_lhand_align - mesh_gt_lhand) ** 2, dim=-1)), dim=-1).sum()
            rhand_pa = torch.mean(torch.sqrt(torch.sum((mesh_out_rhand_align - mesh_gt_rhand) ** 2, dim=-1)), dim=-1).sum()
            setattr(self, f"{data_src}_MPVPE_PA_lhand", getattr(self, f"{data_src}_MPVPE_PA_lhand") + lhand_pa)
            setattr(self, f"{data_src}_MPVPE_PA_rhand", getattr(self, f"{data_src}_MPVPE_PA_rhand") + rhand_pa)
            setattr(self, f"{data_src}_MPVPE_PA_hand",
                    getattr(self, f"{data_src}_MPVPE_PA_hand") + (lhand_pa + rhand_pa) / 2.0)

            mesh_out_lhand_align = mesh_out_lhand - joints_rst[i, :cur_len, smpl_x.J_regressor_idx['lwrist']:smpl_x.J_regressor_idx['lwrist']+1] + joints_ref[i, :cur_len, smpl_x.J_regressor_idx['lwrist']:smpl_x.J_regressor_idx['lwrist']+1]
            mesh_out_rhand_align = mesh_out_rhand - joints_rst[i, :cur_len, smpl_x.J_regressor_idx['rwrist']:smpl_x.J_regressor_idx['rwrist']+1] + joints_ref[i, :cur_len, smpl_x.J_regressor_idx['rwrist']:smpl_x.J_regressor_idx['rwrist']+1]
            lhand = torch.mean(torch.sqrt(torch.sum((mesh_out_lhand_align - mesh_gt_lhand) ** 2, dim=-1)), dim=-1).sum()
            rhand = torch.mean(torch.sqrt(torch.sum((mesh_out_rhand_align - mesh_gt_rhand) ** 2, dim=-1)), dim=-1).sum()
            setattr(self, f"{data_src}_MPVPE_lhand", getattr(self, f"{data_src}_MPVPE_lhand") + lhand)
            setattr(self, f"{data_src}_MPVPE_rhand", getattr(self, f"{data_src}_MPVPE_rhand") + rhand)
            setattr(self, f"{data_src}_MPVPE_hand",
                    getattr(self, f"{data_src}_MPVPE_hand") + (lhand + rhand) / 2.0)
            
            mesh_gt_face = mesh_gt[:, smpl_x.face_vertex_idx, :]
            mesh_out_face = mesh_out[:, smpl_x.face_vertex_idx, :]
            mesh_out_face_align = rigid_align_torch_batch(mesh_out_face, mesh_gt_face)
            setattr(self, f"{data_src}_MPVPE_PA_face", getattr(self, f"{data_src}_MPVPE_PA_face") + torch.mean(torch.sqrt(torch.sum((mesh_out_face_align - mesh_gt_face) ** 2, dim=-1)), dim=-1).sum())
            mesh_out_face_align = mesh_out_face - joints_rst[i, :cur_len, smpl_x.J_regressor_idx['neck']:smpl_x.J_regressor_idx['neck']+1] + joints_ref[i, :cur_len, smpl_x.J_regressor_idx['neck']:smpl_x.J_regressor_idx['neck']+1]
            setattr(self, f"{data_src}_MPVPE_face", getattr(self, f"{data_src}_MPVPE_face") + torch.mean(torch.sqrt(torch.sum((mesh_out_face_align - mesh_gt_face) ** 2, dim=-1)), dim=-1).sum())

            joint_gt_body = torch.matmul(smpl_x.j14_regressor, mesh_gt)
            joint_out_body = torch.matmul(smpl_x.j14_regressor, mesh_out)
            joint_out_body_align = rigid_align_torch_batch(joint_out_body, joint_gt_body)
            setattr(self, f"{data_src}_MPJPE_PA_body", getattr(self, f"{data_src}_MPJPE_PA_body") + torch.mean(torch.sqrt(torch.sum((joint_out_body_align - joint_gt_body) ** 2, dim=-1)), dim=-1).sum())
            joint_out_body_align = joint_out_body - joints_rst[i, :cur_len, smpl_x.J_regressor_idx['pelvis']:smpl_x.J_regressor_idx['pelvis']+1] + joints_ref[i, :cur_len, smpl_x.J_regressor_idx['pelvis']:smpl_x.J_regressor_idx['pelvis']+1]
            setattr(self, f"{data_src}_MPJPE_body", getattr(self, f"{data_src}_MPJPE_body") + torch.mean(torch.sqrt(torch.sum((joint_out_body_align - joint_gt_body) ** 2, dim=-1)), dim=-1).sum())
            
            joint_gt_lhand = torch.matmul(smpl_x.orig_hand_regressor['left'], mesh_gt)
            joint_out_lhand = torch.matmul(smpl_x.orig_hand_regressor['left'], mesh_out)
            joint_out_lhand_align = rigid_align_torch_batch(joint_out_lhand, joint_gt_lhand)
            joint_gt_rhand = torch.matmul(smpl_x.orig_hand_regressor['right'], mesh_gt)
            joint_out_rhand = torch.matmul(smpl_x.orig_hand_regressor['right'], mesh_out)
            joint_out_rhand_align = rigid_align_torch_batch(joint_out_rhand, joint_gt_rhand)
            value = (torch.mean(torch.sqrt(torch.sum((joint_out_lhand_align - joint_gt_lhand) ** 2, dim=-1)), dim=-1).sum() +
                    torch.mean(torch.sqrt(torch.sum((joint_out_rhand_align - joint_gt_rhand) ** 2, dim=-1)), dim=-1).sum()) / 2.
            setattr(self, f"{data_src}_MPJPE_PA_hand", getattr(self, f"{data_src}_MPJPE_PA_hand") + value)

            joint_out_lhand_align = joint_out_lhand - joints_rst[i, :cur_len, smpl_x.J_regressor_idx['lwrist']:smpl_x.J_regressor_idx['lwrist']+1] + joints_ref[i, :cur_len, smpl_x.J_regressor_idx['lwrist']:smpl_x.J_regressor_idx['lwrist']+1]
            joint_out_rhand_align = joint_out_rhand - joints_rst[i, :cur_len, smpl_x.J_regressor_idx['rwrist']:smpl_x.J_regressor_idx['rwrist']+1] + joints_ref[i, :cur_len, smpl_x.J_regressor_idx['rwrist']:smpl_x.J_regressor_idx['rwrist']+1]
            value = (torch.mean(torch.sqrt(torch.sum((joint_out_lhand_align - joint_gt_lhand) ** 2, dim=-1)), dim=-1).sum() +
                    torch.mean(torch.sqrt(torch.sum((joint_out_rhand_align - joint_gt_rhand) ** 2, dim=-1)), dim=-1).sum()) / 2.
            setattr(self, f"{data_src}_MPJPE_hand", getattr(self, f"{data_src}_MPJPE_hand") + value)

            # diff_joint = joints_ref[i, :cur_len, ...] - joints_rst[i, :cur_len, ...]
            # joint_idx = self.joint_part2idx['upper_body']
            # diff_joint_part = diff_joint[:, joint_idx, :]
            # setattr(self, f'{data_src}_MPJPE_body', getattr(self, f'{data_src}_MPJPE_body') + torch.mean(torch.sqrt(torch.sum(torch.square(diff_joint_part), dim=-1)), dim=-1).sum())

            # joint_idx = self.joint_part2idx['hand']
            # diff_joint_part = diff_joint[:, joint_idx, :]
            # setattr(self, f'{data_src}_MPJPE_hand', getattr(self, f'{data_src}_MPJPE_hand') + torch.mean(torch.sqrt(torch.sum(torch.square(diff_joint_part), dim=-1)), dim=-1).sum())
