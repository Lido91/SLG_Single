import torch
import torch.nn as nn
from .base import BaseLosses


class CommitLoss(nn.Module):
    """
    Useless Wrapper
    """
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, commit, commit2, **kwargs):
        return commit


class GPTLosses(BaseLosses):
    
    def __init__(self, cfg, stage, num_joints, **kwargs):
        # Save parameters
        self.stage = stage
        recons_loss = cfg.LOSS.ABLATION.RECONS_LOSS

        # Define losses
        losses = []
        params = {}
        if stage == "vae":
            losses.append("recons_feature")
            params['recons_feature'] = cfg.LOSS.LAMBDA_FEATURE

            losses.append("recons_velocity")
            params['recons_velocity'] = cfg.LOSS.LAMBDA_VELOCITY

            losses.append("vq_commit")
            params['vq_commit'] = cfg.LOSS.LAMBDA_COMMIT

            # VQ-Style losses (contrastive + MI)
            losses.append("vqstyle_con")
            params['vqstyle_con'] = cfg.LOSS.get('LAMBDA_CONTRASTIVE', 0.0)
            losses.append("vqstyle_mi")
            params['vqstyle_mi'] = cfg.LOSS.get('LAMBDA_MI', 0.0)

            # Cumulative alignment loss (Align before Fuse)
            losses.append("vqstyle_align")
            params['vqstyle_align'] = cfg.LOSS.get('LAMBDA_ALIGN', 0.0)

            # LG-VQ language-guided codebook loss
            losses.append("vqstyle_lgvq")
            params['vqstyle_lgvq'] = cfg.LOSS.get('LAMBDA_LGVQ', 0.0)

            # Raw InfoNCE loss (for wandb monitoring only, weight=0)
            losses.append("vqstyle_nce")
            params['vqstyle_nce'] = 0.0

            # Codebook perplexity (monitoring only, weight=0)
            losses.append("vq_perplexity")
            params['vq_perplexity'] = 0.0
        elif stage in ["lm_pretrain", "lm_instruct", "lm_rvq_hierarchical"]:
            losses.append("gpt_loss")
            params['gpt_loss'] = cfg.LOSS.LAMBDA_CLS
            # Add per-decoder losses and accuracies for hierarchical RVQ-GPT logging
            # These are registered for all LM stages but only used if model outputs them
            # 3-layer losses and accuracies
            for i in range(3):
                losses.append(f"gpt_loss_q{i}")
                params[f'gpt_loss_q{i}'] = 0.0  # Weight 0 (already included in gpt_loss)
                losses.append(f"gpt_acc_q{i}")
                params[f'gpt_acc_q{i}'] = 0.0
            # 6-layer losses and accuracies (will be used if model has them)
            for i in range(3, 6):
                losses.append(f"gpt_loss_q{i}")
                params[f'gpt_loss_q{i}'] = 0.0
                losses.append(f"gpt_acc_q{i}")
                params[f'gpt_acc_q{i}'] = 0.0
        elif stage == "lamp":
            # LaMP pretraining with 3 losses (SOKE-aligned, no LM loss)
            losses.append("lamp_ptc")     # Point-Text Contrastive
            params['lamp_ptc'] = cfg.LOSS.get('LAMBDA_PTC', 1.0)
            losses.append("lamp_ptm")     # Point-Text Matching
            params['lamp_ptm'] = cfg.LOSS.get('LAMBDA_PTM', 1.0)
            losses.append("lamp_gen")     # Generation (T2M)
            params['lamp_gen'] = cfg.LOSS.get('LAMBDA_GEN', 1.0)
        elif stage == "lm_masked_t2m":
            # Masked Transformer T2M
            losses.append("masked_ce")    # Cross-entropy for masked prediction
            params['masked_ce'] = cfg.LOSS.get('LAMBDA_MASKED_CE', 1.0)
            losses.append("masked_acc")   # Token prediction accuracy
            params['masked_acc'] = 0.0    # Just for logging

        # Define loss functions & weights
        losses_func = {}
        for loss in losses:
            if loss.split('_')[0] == 'recons':
                if recons_loss == "l1":
                    losses_func[loss] = nn.L1Loss
                elif recons_loss == "l2":
                    losses_func[loss] = nn.MSELoss
                elif recons_loss == "l1_smooth":
                    losses_func[loss] = nn.SmoothL1Loss
            elif loss.split('_')[0] in ['lamp', 'masked', 'vqstyle']:
                # LaMP, Masked T2M, and VQ-Style losses use CommitLoss wrapper
                losses_func[loss] = CommitLoss
            elif loss.split('_')[1] in [
                    'commit', 'loss', 'gpt', 'm2t2m', 't2m2t', 'acc',
                    'perplexity'
            ]:
                losses_func[loss] = CommitLoss
            elif loss.split('_')[1] in ['cls', 'lm']:
                losses_func[loss] = nn.CrossEntropyLoss
            else:
                raise NotImplementedError(f"Loss {loss} not implemented.")

        super().__init__(cfg, losses, params, losses_func, num_joints,
                         **kwargs)

    def update(self, rs_set):
        '''Update the losses'''
        total: float = 0.0

        if self.stage in ["vae"]:
            total += self._update_loss("recons_feature", rs_set['m_rst'],
                                       rs_set['m_ref'])
            # total += self._update_loss("recons_joints", rs_set['joints_rst'], rs_set['joints_ref'])
            nfeats = rs_set['m_rst'].shape[-1]
            if nfeats in [263, 135 + 263]:
                if nfeats == 135 + 263:
                    vel_start = 135 + 4
                elif nfeats == 263:
                    vel_start = 4
                total += self._update_loss(
                    "recons_velocity",
                    rs_set['m_rst'][..., vel_start:(self.num_joints - 1) * 3 +
                                    vel_start],
                    rs_set['m_ref'][..., vel_start:(self.num_joints - 1) * 3 +
                                    vel_start])
            else:
                if self._params['recons_velocity'] != 0.0:
                    raise NotImplementedError(
                        "Velocity not implemented for nfeats = {})".format(nfeats))
            total += self._update_loss("vq_commit", rs_set['loss_commit'],
                                       rs_set['loss_commit'])

            # VQ-Style losses (only present when using RVQVaeVQStyle)
            if 'loss_con' in rs_set:
                total += self._update_loss("vqstyle_con", rs_set['loss_con'],
                                           rs_set['loss_con'])
            if 'loss_mi' in rs_set:
                total += self._update_loss("vqstyle_mi", rs_set['loss_mi'],
                                           rs_set['loss_mi'])

            # Cumulative alignment loss (only present when using RVQVaeAlign)
            if 'loss_align' in rs_set:
                total += self._update_loss("vqstyle_align", rs_set['loss_align'],
                                           rs_set['loss_align'])

            # LG-VQ loss (only present when using RVQVaeLGVQ)
            if 'loss_lgvq' in rs_set:
                total += self._update_loss("vqstyle_lgvq", rs_set['loss_lgvq'],
                                           rs_set['loss_lgvq'])

            # Raw InfoNCE loss for monitoring (not added to total, already in lgvq)
            if 'loss_nce' in rs_set:
                self._update_loss("vqstyle_nce", rs_set['loss_nce'],
                                  rs_set['loss_nce'])

            # Codebook perplexity for monitoring (not added to total)
            if 'perplexity' in rs_set:
                self._update_loss("vq_perplexity", rs_set['perplexity'],
                                  rs_set['perplexity'])

        if self.stage in ["lm_pretrain", "lm_instruct", "lm_rvq_hierarchical"]:
            total += self._update_loss("gpt_loss", rs_set['outputs'].loss,
                                       rs_set['outputs'].loss)
            # Log per-decoder losses and accuracies for hierarchical RVQ-GPT
            # Dynamically log losses and accuracies for available quantizers (3 or 6)
            # Works for all LM stages - only logs if model outputs these attributes
            for i in range(6):
                loss_attr = f'loss_q{i}'
                acc_attr = f'acc_q{i}'
                if hasattr(rs_set['outputs'], loss_attr):
                    self._update_loss(f"gpt_loss_q{i}",
                                      getattr(rs_set['outputs'], loss_attr),
                                      getattr(rs_set['outputs'], loss_attr))
                if hasattr(rs_set['outputs'], acc_attr):
                    self._update_loss(f"gpt_acc_q{i}",
                                      getattr(rs_set['outputs'], acc_attr),
                                      getattr(rs_set['outputs'], acc_attr))

        if self.stage == "lamp":
            # LaMP pretraining losses (SOKE-aligned: 3 losses, no LM)
            total += self._update_loss("lamp_ptc", rs_set['outputs'].loss_ptc,
                                      rs_set['outputs'].loss_ptc)
            total += self._update_loss("lamp_ptm", rs_set['outputs'].loss_ptm,
                                      rs_set['outputs'].loss_ptm)
            total += self._update_loss("lamp_gen", rs_set['outputs'].loss_gen,
                                      rs_set['outputs'].loss_gen)

        if self.stage == "lm_masked_t2m":
            # Masked Transformer T2M losses
            total += self._update_loss("masked_ce", rs_set['ce_loss'],
                                      rs_set['ce_loss'])
            self._update_loss("masked_acc", rs_set['acc'],
                             rs_set['acc'])

        # Update the total loss
        self.total += total.detach()
        self.count += 1

        return total
