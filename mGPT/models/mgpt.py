import numpy as np
import os
import random
import torch
import torch.nn.functional as F
import time
from mGPT.config import instantiate_from_config
from os.path import join as pjoin
from mGPT.losses.mgpt import GPTLosses
from mGPT.models.base import BaseModel
from .base import BaseModel
import json
import mGPT.render.matplot.plot_3d_global as plot_3d


class MotionGPT(BaseModel):
    """
    Stage 1 Motion Tokenizer
    Stage 2 Motion-language pretrian
    Stage 3 Motion-language instruction tuning
    """

    def __init__(self,
                 cfg,
                 datamodule,
                 lm,
                 motion_vae,
                 codebook_size=512,
                 stage='vae',
                 debug=True,
                 condition='text',
                 task='t2m',
                 metrics_dict=['TM2TMetrics'],
                 text_encoder_type=None,
                 speech_encoder_type=None,
                 **kwargs):

        self.save_hyperparameters(ignore='datamodule', logger=False)
        self.datamodule = datamodule
        super().__init__()

        # Instantiate motion tokenizer
        if motion_vae != None:
            self.vae = instantiate_from_config(motion_vae)

        # Instantiate motion-language model
        # For LaMP stage, no vq_model needed (encoding done in trainer)
        if self.hparams.stage == "lamp":
            if condition == "audio":
                from mGPT.archs.lamp import LaMPSpeech
                lamp_params = {k: v for k, v in lm.get('params', {}).items()}
                self.lm = LaMPSpeech(**lamp_params)
            else:
                from mGPT.archs.lamp import LaMP
                lamp_params = {k: v for k, v in lm.get('params', {}).items()}
                self.lm = LaMP(**lamp_params)
        else:
            # Override use_speech and encoder types based on train config's condition field
            if 'params' in lm:
                if condition == 'audio':
                    lm['params']['use_speech'] = True
                    if speech_encoder_type is not None:
                        lm['params']['speech_encoder_type'] = speech_encoder_type
                elif condition == 'text':
                    lm['params']['use_speech'] = False
                    if text_encoder_type is not None:
                        lm['params']['text_encoder_type'] = text_encoder_type
            self.lm = instantiate_from_config(lm)

        # Freeze the motion tokenizer for lm/lamp training
        if 'lm' in self.hparams.stage or self.hparams.stage == 'lamp':
            self.vae.training = False
            for p in self.vae.parameters():
                p.requires_grad = False

        # Instantiate the losses
        self._losses = torch.nn.ModuleDict({
            split: GPTLosses(cfg, self.hparams.stage, self.datamodule.njoints)
            for split in ["losses_train", "losses_test", "losses_val"]
        })

        # Data transform
        self.feats2joints = datamodule.feats2joints

        # Count codebook frequency
        self.codePred = []
        self.codeFrequency = torch.zeros((self.hparams.codebook_size, ))

    def forward(self, batch, task="t2m"):
        texts = batch["text"]
        lengths_ref = batch["length"]

        # Forward
        # texts = ['Generate motion: ' + text for text in texts]
        outputs, output_texts = self.lm.generate_direct(texts, do_sample=True)

        # Motion Decode
        feats_rst_lst = []
        lengths = []
        max_len = 0

        for i in range(len(texts)):
            if task == "pred":
                motion = self.vae.decode(
                    torch.cat((batch["motion"][i], outputs[i])))
            elif task in ["t2m", "m2t", "inbetween"]:
                motion = self.vae.decode(outputs[i])
                # motion = self.datamodule.denormalize(motion)
                lengths.append(motion.shape[1])
            else:
                raise NotImplementedError

            if motion.shape[1] > max_len:
                max_len = motion.shape[1]

            if task in ["t2m", "m2t", "pred"]:
                feats_rst_lst.append(motion)

            elif task == "inbetween":
                motion = torch.cat(
                    (batch["motion_heading"][i][None],
                     motion[:, lengths_ref[i] // 4:lengths_ref[i] // 4 * 3,
                            ...], batch["motion_tailing"][i][None]),
                    dim=1)
                feats_rst_lst.append(motion)

        feats_rst = torch.zeros(
            (len(feats_rst_lst), max_len, motion.shape[-1])).to(self.device)

        # padding and concat
        for i in range(len(feats_rst_lst)):
            feats_rst[i, :feats_rst_lst[i].shape[1], ...] = feats_rst_lst[i]

        # Recover joints for evaluation
        joints_rst = self.feats2joints(feats_rst)

        # return set
        outputs = {
            "texts": output_texts,
            "feats": feats_rst,
            "joints": joints_rst,
            "length": lengths
        }

        return outputs

    def train_lm_forward(self, batch):
        tokens_ref = batch["motion"]
        texts = batch.get("text", None)
        lengths = batch["length"]
        tasks = batch.get("tasks", None)
        all_captions = batch.get('all_captions', None)
        audio_waveforms = batch.get("audio", None)
        speech_feats = batch.get("speech_feats", None)
        speech_mask = batch.get("speech_mask", None)

        if self.hparams.condition == 'caption' and all_captions is not None and texts is not None:
            texts = [random.choice(all_captions[i]) for i in range(len(texts))]

        # LLM Forward (supports text, speech waveform, and precomputed speech modes)
        if speech_feats is not None:
            # Precomputed speech feature mode (skip HuBERT)
            outputs = self.lm(texts=None, audio_waveforms=None,
                              motion_tokens=tokens_ref, lengths=lengths, tasks=tasks,
                              speech_feats=speech_feats, speech_mask=speech_mask)
        elif audio_waveforms is not None:
            # Speech-driven mode (raw waveform)
            outputs = self.lm(texts=None, audio_waveforms=audio_waveforms,
                              motion_tokens=tokens_ref, lengths=lengths, tasks=tasks)
        else:
            # Text-driven mode (original)
            outputs = self.lm(texts=texts, audio_waveforms=None,
                              motion_tokens=tokens_ref, lengths=lengths, tasks=tasks)
        return {'outputs': outputs}

    @torch.no_grad()
    def val_t2m_forward(self, batch):
        # SOKE's approach: batch["motion"] contains raw poses from Text2MotionDatasetEval
        feats_ref = batch["motion"]  # Raw motion features (B, T, 133) from dataset
        texts = batch["text"]
        lengths = batch["length"]
        tasks = None
        B = feats_ref.shape[0]

        if self.trainer.datamodule.is_mm:
            texts = texts * self.hparams.cfg.METRIC.MM_NUM_REPEATS
            feats_ref = feats_ref.repeat_interleave(
                self.hparams.cfg.METRIC.MM_NUM_REPEATS, dim=0)
            lengths = lengths * self.hparams.cfg.METRIC.MM_NUM_REPEATS
            instructions = pjoin(self.datamodule.hparams.data_root,
                                 'template_instructions.json')
            instructions = json.load(open(instructions, 'r'))
            tasks = [instructions["Text-to-Motion"]["caption"]] * len(texts)

        if self.hparams.condition == 'caption':
            tasks = [{
                'input': ['<Caption_Placeholder>'],
                'output': ['']
            }] * len(texts)

        if self.hparams.cfg.DATASET.TASK_PATH:
            instructions = pjoin(self.hparams.cfg.DATASET.TASK_PATH)
            instructions = json.load(open(instructions, 'r'))
            tasks = [instructions["Text-to-Motion"]["t2m"]] * len(texts)

        # Generate motion tokens from text or audio (val/test uses raw audio + HuBERT)
        audio_waveforms = batch.get("audio", None)
        outputs_tokens = self.lm.generate_conditional(texts=texts,
                                                      lengths=lengths,
                                                      stage='test',
                                                      tasks=tasks,
                                                      audio_waveforms=audio_waveforms)

        # SOKE's approach: allocate based on max GENERATED token length
        max_len = max(map(len, outputs_tokens))
        C = self.datamodule.nfeats
        feats_rst = torch.zeros(B, max_len * 4, C).to(feats_ref.device)  # Each token = 4 frames

        # Decode generated tokens and track lengths
        lengths_rst = []
        for i in range(B):
            outputs_tokens[i] = torch.clamp(outputs_tokens[i], 0, self.hparams.codebook_size - 1, out=None)

            if len(outputs_tokens[i]) > 1:
                # Check if tokens have multiple quantizers (e.g., from HierarchicalRVQGPT)
                if outputs_tokens[i].dim() == 2 and outputs_tokens[i].shape[-1] > 1:
                    # Multi-quantizer tokens (T, n_quantizers) - use decode_partial
                    motion = self.vae.decode_partial(outputs_tokens[i])
                else:
                    # Single quantizer tokens (T,) - use standard decode
                    motion = self.vae.decode(outputs_tokens[i])
            else:
                motion = torch.zeros(1, 4, C).to(feats_ref.device)

            feats_rst[i:i+1, :motion.shape[1], :] = motion
            lengths_rst.append(motion.shape[1])

        # Recover joints for evaluation
        vertices_ref, joints_ref = self.feats2joints(feats_ref)
        vertices_rst, joints_rst = self.feats2joints(feats_rst)

        # Renorm for evaluation
        feats_ref = self.datamodule.renorm4t2m(feats_ref)
        feats_rst = self.datamodule.renorm4t2m(feats_rst)

        # return set
        rs_set = {
            "m_ref": feats_ref,
            "m_rst": feats_rst,
            "joints_ref": joints_ref,
            "joints_rst": joints_rst,
            "lengths_rst": lengths_rst,  # Generated motion lengths
            "vertices_ref": vertices_ref,
            "vertices_rst": vertices_rst,
        }

        return rs_set

    @torch.no_grad()
    def val_m2t_forward(self, batch):
        self.hparams.metrics_dict = []

        feats_ref = batch["motion"]
        texts = batch["text"]
        lengths = batch["length"]
        all_captions = batch['all_captions']

        # Motion Encode
        motion_tokens = []
        lengths_tokens = []
        for i in range(len(feats_ref)):
            motion_token, _ = self.vae.encode(feats_ref[i:i + 1])
            motion_tokens.append(motion_token[0])
            lengths_tokens.append(motion_token.shape[1])

        # Forward
        outputs = self.lm.generate_conditional(motion_tokens=motion_tokens,
                                               lengths=lengths_tokens,
                                               task="m2t",
                                               stage='test')

        # return set
        rs_set = {
            "m_ref": feats_ref,
            "t_ref": all_captions,
            # "t_ref": texts,
            "t_pred": outputs,
            "length": lengths
        }

        return rs_set

    @torch.no_grad()
    def val_m2m_forward(self, batch, task="pred"):
        feats_ref = batch["motion"]
        lengths = batch["length"]

        # Motion Encode
        motion_tokens = []
        lengths_tokens = []
        for i in range(len(feats_ref)):
            motion_token, _ = self.vae.encode(feats_ref[i:i + 1])
            motion_tokens.append(motion_token[0])

        # Forward
        outputs = self.lm.generate_conditional(motion_tokens=motion_tokens,
                                               lengths=lengths,
                                               task=task,
                                               stage='test')

        # Motion Decode
        feats_rst = torch.zeros_like(feats_ref)
        min_len = lengths.copy()

        for i in range(len(lengths)):
            outputs[i] = torch.clamp(outputs[i],
                                     0,
                                     self.hparams.codebook_size - 1,
                                     out=None)

            if len(outputs[i]) > 1:
                motion = self.vae.decode(outputs[i])
            else:
                motion = torch.zeros_like(feats_ref[i:i + 1, ...])

            min_len[i] = min(motion.shape[1], lengths[i])

            # Cut Motion
            feats_rst[i:i + 1, :min_len[i], ...] = motion[:, :lengths[i]]

        # Recover joints for evaluation
        joints_ref = self.feats2joints(feats_ref)
        joints_rst = self.feats2joints(feats_rst)

        # Renorm for evaluation
        feats_ref = self.datamodule.renorm4t2m(feats_ref)
        feats_rst = self.datamodule.renorm4t2m(feats_rst)

        # return set
        rs_set = {
            "m_ref": feats_ref,
            "m_rst": feats_rst,
            "joints_ref": joints_ref,
            "joints_rst": joints_rst,
            "length": min_len
            # "length": lengths
        }

        return rs_set

    def train_vae_forward(self, batch):
        # batch detach
        feats_ref = batch["motion"]

        # feats2joints returns (vertices, joints) for H2S dataset, just joints for others
        feats2joints_result_ref = self.feats2joints(feats_ref)
        if isinstance(feats2joints_result_ref, tuple):
            _, joints_ref = feats2joints_result_ref
        else:
            joints_ref = feats2joints_result_ref

        # motion encode & decode
        loss_lgvq = torch.tensor(0.0, device=feats_ref.device)

        loss_nce = torch.tensor(0.0, device=feats_ref.device)
        loss_lgvq = torch.tensor(0.0, device=feats_ref.device)
        # Check for Speech LG-VQ VAE
        if hasattr(self.vae, 'speech_dim'):
            audio_waveforms = batch.get("audio", None)
            audio_lengths = batch.get("audio_lengths", None)
            speech_feats = batch.get("speech_feats", None)
            speech_mask = batch.get("speech_mask", None)
            feats_rst, loss_commit, perplexity, motion_emb, speech_emb = self.vae(
                feats_ref, audio_waveforms=audio_waveforms, audio_lengths=audio_lengths,
                speech_feats=speech_feats, speech_mask=speech_mask)

            # Compute NCE loss with all_gather for DDP
            if motion_emb is not None and speech_emb is not None:
                if self.trainer.world_size > 1:
                    motion_emb = self.all_gather(motion_emb, sync_grads=True).flatten(0, 1)
                    speech_emb = self.all_gather(speech_emb, sync_grads=True).flatten(0, 1)

                logits = (motion_emb @ speech_emb.t()) / 0.07
                labels = torch.arange(logits.shape[0], device=logits.device)
                loss_nce = F.cross_entropy(logits, labels)
                loss_lgvq = loss_nce

            loss_con = torch.tensor(0.0, device=feats_ref.device)
            loss_mi = torch.tensor(0.0, device=feats_ref.device)
            loss_align = torch.tensor(0.0, device=feats_ref.device)
        # Check for CLIP LG-VQ VAE (has nce_weight but no speech_encoder)
        elif hasattr(self.vae, 'nce_weight'):
            texts = batch.get("text", None)
            clip_text_features = batch.get("clip_text_features", None)
            feats_rst, loss_commit, perplexity, motion_emb, text_emb = self.vae(
                feats_ref, texts=texts, clip_text_features=clip_text_features)

            # Compute NCE loss with all_gather for DDP
            if motion_emb is not None and text_emb is not None:
                # Gather embeddings from all GPUs for full negative pool
                if self.trainer.world_size > 1:
                    motion_emb = self.all_gather(motion_emb, sync_grads=True).flatten(0, 1)
                    text_emb = self.all_gather(text_emb, sync_grads=True).flatten(0, 1)

                # InfoNCE
                logits = (motion_emb @ text_emb.t()) / 0.07  # [B_total, B_total]
                labels = torch.arange(logits.shape[0], device=logits.device)
                loss_nce = F.cross_entropy(logits, labels)
                loss_lgvq = loss_nce  # Weight controlled by LAMBDA_LGVQ in config

            loss_con = torch.tensor(0.0, device=feats_ref.device)
            loss_mi = torch.tensor(0.0, device=feats_ref.device)
            loss_align = torch.tensor(0.0, device=feats_ref.device)
        # Check for Align VAE (has lambda_align attribute)
        elif hasattr(self.vae, 'lambda_align'):
            texts = batch.get("text", None)
            lengths = batch.get("length", None)
            feats_rst, loss_commit, perplexity, loss_align = self.vae(feats_ref, texts=texts, lengths=lengths)
            loss_con = torch.tensor(0.0, device=feats_ref.device)
            loss_mi = torch.tensor(0.0, device=feats_ref.device)
        # Check for VQ-Style VAE (has content_cutoff attribute)
        elif hasattr(self.vae, 'content_cutoff'):
            names = batch.get("name", None)
            feats_rst, loss_commit, perplexity, loss_con, loss_mi = self.vae(feats_ref, names=names)
            loss_align = torch.tensor(0.0, device=feats_ref.device)
        else:
            feats_rst, loss_commit, perplexity = self.vae(feats_ref)
            loss_con = torch.tensor(0.0, device=feats_ref.device)
            loss_mi = torch.tensor(0.0, device=feats_ref.device)
            loss_align = torch.tensor(0.0, device=feats_ref.device)

        feats2joints_result_rst = self.feats2joints(feats_rst)
        if isinstance(feats2joints_result_rst, tuple):
            _, joints_rst = feats2joints_result_rst
        else:
            joints_rst = feats2joints_result_rst

        # return set
        rs_set = {
            "m_ref": feats_ref,
            "joints_ref": joints_ref,
            "m_rst": feats_rst,
            "joints_rst": joints_rst,
            "loss_commit": loss_commit,
            "perplexity": perplexity,
            "loss_con": loss_con,
            "loss_mi": loss_mi,
            "loss_align": loss_align,
            "loss_lgvq": loss_lgvq,
            "loss_nce": loss_nce,
        }
        return rs_set

    @torch.no_grad()
    def val_vae_forward(self, batch, split="train"):
        # Detach batch
        feats_ref = batch["motion"]
        lengths = batch["length"]

        # Repeat for multimodal evaluation
        if self.trainer.datamodule.is_mm:
            feats_ref = feats_ref.repeat_interleave(
                self.hparams.cfg.METRIC.MM_NUM_REPEATS, dim=0)
            lengths = lengths * self.hparams.cfg.METRIC.MM_NUM_REPEATS

        # Motion encode & decode
        feats_rst = torch.zeros_like(feats_ref)

        for i in range(len(feats_ref)):
            if lengths[i] == 0:
                continue
            vae_out = self.vae(feats_ref[i:i + 1, :lengths[i]])
            feats_pred = vae_out[0]
            feats_rst[i:i + 1, :feats_pred.shape[1], :] = feats_pred

            code_pred, _ = self.vae.encode(feats_ref[i:i + 1, :lengths[i]])

            # codeFre_pred = torch.bincount(code_pred[0],
            #                               minlength=self.hparams.codebook_size).to(
            #                                   self.codeFrequency.device)
            # self.codePred.append(code_pred[0])
            # self.codeFrequency += codeFre_pred

        # np.save('../memData/results/codeFrequency.npy',
        #         self.codeFrequency.cpu().numpy())

        # Recover joints for evaluation
        # feats2joints returns (vertices, joints) for H2S dataset
        feats2joints_result_ref = self.feats2joints(feats_ref)
        feats2joints_result_rst = self.feats2joints(feats_rst)

        # Handle both return formats: (vertices, joints) or just joints
        if isinstance(feats2joints_result_ref, tuple):
            vertices_ref, joints_ref = feats2joints_result_ref
            vertices_rst, joints_rst = feats2joints_result_rst
        else:
            joints_ref = feats2joints_result_ref
            joints_rst = feats2joints_result_rst
            vertices_ref = None
            vertices_rst = None

        # Renorm for evaluation
        feats_ref = self.datamodule.renorm4t2m(feats_ref)
        feats_rst = self.datamodule.renorm4t2m(feats_rst)

        # Return set
        rs_set = {
            "m_ref": feats_ref,
            "joints_ref": joints_ref,
            "m_rst": feats_rst,
            "joints_rst": joints_rst,
            "vertices_ref": vertices_ref,
            "vertices_rst": vertices_rst,
            "length": lengths,
        }

        return rs_set

    def train_lamp_forward(self, batch):
        """LaMP pretraining forward pass (SOKE-aligned: online encoding in trainer)."""
        feats_ref = batch["motion"]  # (B, T, nfeats=133)

        # Online encoding with frozen VQ-VAE
        with torch.no_grad():
            # Encode: [B, T, 133] → [B, 133, T] → encoder → [B, 512, T']
            motion_input = feats_ref.permute(0, 2, 1)
            motion_embeds = self.vae.encoder(motion_input)  # [B, 512, T']
            motion_embeds = motion_embeds.permute(0, 2, 1)  # [B, T', 512]

            # Quantize to get token IDs (for GEN loss)
            token_ids, _ = self.vae.encode(feats_ref)  # [B, T', num_quantizers]
            token_ids = token_ids[:, :, 0]  # [B, T'] — first quantizer only

        if self.hparams.condition == "audio":
            audio_waveforms = batch["audio"]
            outputs, cond_feat, motion_feat = self.lm(motion_embeds, audio_waveforms, token_ids=token_ids)
        else:
            texts = batch["text"]
            outputs, cond_feat, motion_feat = self.lm(motion_embeds, texts, token_ids=token_ids)

        return {'outputs': outputs, 'text_feat': cond_feat, 'motion_feat': motion_feat}

    def train_masked_t2m_forward(self, batch):
        """Masked Transformer T2M training forward pass."""
        tokens_ref = batch["motion"]  # (B, T, num_quantizers) motion tokens
        texts = batch["text"]
        lengths = batch["length"]

        # Convert frame lengths to token lengths (divide by 4)
        token_lengths = [l // 4 for l in lengths]
        token_lengths = torch.tensor(token_lengths, device=tokens_ref.device)

        # Use first quantizer only (coarse codes)
        if tokens_ref.dim() == 3:
            tokens_ref = tokens_ref[:, :, 0]  # (B, T)

        # Forward through Masked Transformer
        ce_loss, pred_id, acc = self.lm(tokens_ref, texts, token_lengths)

        return {'ce_loss': ce_loss, 'pred_id': pred_id, 'acc': acc}

    @torch.no_grad()
    def val_masked_t2m_forward(self, batch):
        """Masked Transformer T2M validation forward pass."""
        feats_ref = batch["motion"]  # Raw motion features (B, T, nfeats)
        texts = batch["text"]
        lengths = batch["length"]
        B = feats_ref.shape[0]

        if self.trainer.datamodule.is_mm:
            texts = texts * self.hparams.cfg.METRIC.MM_NUM_REPEATS
            feats_ref = feats_ref.repeat_interleave(
                self.hparams.cfg.METRIC.MM_NUM_REPEATS, dim=0)
            lengths = lengths * self.hparams.cfg.METRIC.MM_NUM_REPEATS

        # Generate motion tokens using masked transformer
        outputs_tokens = self.lm.generate_conditional(
            texts=texts,
            lengths=lengths,
            stage='test'
        )  # List of (T_i,) tensors

        # Decode tokens to motion
        feats_rst = torch.zeros_like(feats_ref)
        lengths_rst = []

        for i in range(len(outputs_tokens)):
            if len(outputs_tokens[i]) > 0:
                # Decode tokens (single quantizer)
                motion = self.vae.decode_q0_only(outputs_tokens[i])
                actual_len = min(motion.shape[1], lengths[i])
                feats_rst[i, :actual_len, :] = motion[0, :actual_len, :]
                lengths_rst.append(actual_len)
            else:
                lengths_rst.append(0)

        # Recover joints for evaluation
        joints_ref = self.feats2joints(feats_ref)
        joints_rst = self.feats2joints(feats_rst)

        # Get vertices if available
        feats2joints_result_ref = self.feats2joints(feats_ref)
        feats2joints_result_rst = self.feats2joints(feats_rst)

        if isinstance(feats2joints_result_ref, tuple):
            vertices_ref, joints_ref = feats2joints_result_ref
            vertices_rst, joints_rst = feats2joints_result_rst
        else:
            joints_ref = feats2joints_result_ref
            joints_rst = feats2joints_result_rst
            vertices_ref = torch.zeros_like(feats_ref)
            vertices_rst = torch.zeros_like(feats_rst)

        # Renorm for evaluation
        feats_ref = self.datamodule.renorm4t2m(feats_ref)
        feats_rst = self.datamodule.renorm4t2m(feats_rst)

        return {
            "m_ref": feats_ref,
            "m_rst": feats_rst,
            "joints_ref": joints_ref,
            "joints_rst": joints_rst,
            "vertices_ref": vertices_ref,
            "vertices_rst": vertices_rst,
            "lengths_rst": lengths_rst
        }

    def allsplit_step(self, split: str, batch, batch_idx):
        # Compute the losses
        loss = None

        lengths = batch['length']
        src = batch['src']
        name = batch['name']
        # print('task: ', self.hparams.task)
        if self.hparams.stage == "vae" and split in ["train", "val"]:
            rs_set = self.train_vae_forward(batch)
            loss = self._losses['losses_' + split].update(rs_set)
        elif self.hparams.stage in ["lm_instruct", "lm_pretrain", "lm_rvq_hierarchical"] and split in ["train"]:
            rs_set = self.train_lm_forward(batch)
            loss = self._losses['losses_' + split].update(rs_set)
        elif self.hparams.stage == "lamp" and split in ["train", "val"]:
            rs_set = self.train_lamp_forward(batch)
            loss = self._losses['losses_' + split].update(rs_set)
        elif self.hparams.stage == "lm_masked_t2m" and split in ["train"]:
            rs_set = self.train_masked_t2m_forward(batch)
            loss = self._losses['losses_' + split].update(rs_set)

        # Compute the metrics
        if split in ["val", "test"]:
            # Measure inference time
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            infer_start = time.time()

            if self.hparams.stage == "vae":
                rs_set = self.val_vae_forward(batch, split)
                getattr(self.metrics,'MRMetrics').update(
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
            elif self.hparams.stage in ["lm_instruct", "lm_pretrain", "lm_rl", "lm_rvq_hierarchical"]:
                if self.hparams.task in ["t2m", "a2m"]:
                    rs_set = self.val_t2m_forward(batch)
                    # Use the configured metric (MRMetrics or TM2TMetrics)
                    metric_name = self.hparams.metrics_dict[0] if self.hparams.metrics_dict else 'MRMetrics'
                    if hasattr(self.metrics, metric_name):
                        # MRMetrics only accepts: feats, joints, vertices, lengths, src, name
                        # TM2TMetrics accepts additional: lengths_rst, split
                        if metric_name == 'MRMetrics':
                            getattr(self.metrics, metric_name).update(
                                    feats_rst=rs_set["m_rst"],
                                    feats_ref=rs_set["m_ref"],
                                    joints_rst=rs_set["joints_rst"],
                                    joints_ref=rs_set["joints_ref"],
                                    vertices_rst=rs_set["vertices_rst"],
                                    vertices_ref=rs_set["vertices_ref"],
                                    lengths=rs_set['lengths_rst'],  # Use generated lengths for MR
                                    src=src,
                                    name=name
                                )
                        else:  # TM2TMetrics
                            getattr(self.metrics, metric_name).update(
                                    feats_rst=rs_set["m_rst"],
                                    feats_ref=rs_set["m_ref"],
                                    joints_rst=rs_set["joints_rst"],
                                    joints_ref=rs_set["joints_ref"],
                                    vertices_rst=rs_set["vertices_rst"],
                                    vertices_ref=rs_set["vertices_ref"],
                                    lengths=lengths,
                                    lengths_rst=rs_set['lengths_rst'],
                                    split=split,
                                    src=src,
                                    name=name
                                )
                elif self.hparams.task == "m2t":
                    rs_set_m2t = self.val_m2t_forward(batch)
                    getattr(self.metrics, 'M2TMetrics').update(
                        pred_texts=rs_set_m2t["t_pred"],
                        gt_texts=rs_set_m2t["t_ref"],
                        lengths=rs_set_m2t['length'],
                        src=src,
                    )
            elif self.hparams.stage == "lm_masked_t2m":
                rs_set = self.val_masked_t2m_forward(batch)
                # Use TM2TMetrics for masked transformer
                metric_name = self.hparams.metrics_dict[0] if self.hparams.metrics_dict else 'TM2TMetrics'
                if hasattr(self.metrics, metric_name):
                    if metric_name == 'MRMetrics':
                        getattr(self.metrics, metric_name).update(
                                feats_rst=rs_set["m_rst"],
                                feats_ref=rs_set["m_ref"],
                                joints_rst=rs_set["joints_rst"],
                                joints_ref=rs_set["joints_ref"],
                                vertices_rst=rs_set["vertices_rst"],
                                vertices_ref=rs_set["vertices_ref"],
                                lengths=rs_set['lengths_rst'],
                                src=src,
                                name=name
                            )
                    else:  # TM2TMetrics
                        getattr(self.metrics, metric_name).update(
                                feats_rst=rs_set["m_rst"],
                                feats_ref=rs_set["m_ref"],
                                joints_rst=rs_set["joints_rst"],
                                joints_ref=rs_set["joints_ref"],
                                vertices_rst=rs_set["vertices_rst"],
                                vertices_ref=rs_set["vertices_ref"],
                                lengths=lengths,
                                lengths_rst=rs_set['lengths_rst'],
                                split=split,
                                src=src,
                                name=name
                            )

            # Record inference time
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            infer_end = time.time()
            self.times.append(infer_end - infer_start)
            self.infer_sample_counts.append(len(batch['length']))

        if split in ["test"]:
            if self.hparams.stage == "vae":
                # return rs_set["joints_rst"], rs_set["joints_ref"], rs_set["vertices_rst"], rs_set["vertices_ref"], rs_set["m_ref"], rs_set["m_rst"], batch["length"]
                return {'name': name, 'feats_ref': rs_set["m_ref"], 'feats_rst': rs_set['m_rst'], 'lengths': batch['length'], 'lengths_rst': batch['length'], 'text': batch['text']}
            elif "lm" in self.hparams.stage or self.hparams.stage == "lamp":
                # return rs_set["joints_rst"], rs_set["joints_ref"], rs_set["vertices_rst"], rs_set["vertices_ref"], rs_set["m_ref"], rs_set["m_rst"], \
                    # rs_set_m2t["t_pred"], rs_set_m2t["t_ref"], batch["length"]
                return {'name': name, 'feats_ref': rs_set["m_ref"], 'feats_rst': rs_set['m_rst'], 'lengths': batch['length'], 'lengths_rst': rs_set['lengths_rst'], 'text': batch['text']}
               
        return loss