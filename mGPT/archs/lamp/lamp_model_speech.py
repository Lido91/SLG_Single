"""
LaMPSpeech: Speech-conditioned LaMP (Language-Motion Pretraining)

Replaces BERT text conditioning with HuBERT-Large (1024D) speech features.
Same 3 training objectives as text LaMP: PTC + PTM + GEN.
"""
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from .qformer_base import QFormer_Base
from .QFormer_output import QFormer_Output
from .basemodel import all_gather_with_grad, concat_all_gather
from mGPT.archs.speech_encoder import SpeechEncoder


class LaMPSpeech(QFormer_Base):
    """
    LaMPSpeech: Speech-conditioned Language-Motion Pretraining with QFormer.

    Same architecture as LaMP but replaces BERT text with HuBERT-Large speech features.
    3 losses: PTC (contrastive), PTM (matching), GEN (generation).
    """

    def __init__(
        self,
        nfeats=133,
        num_query_token=49,
        cross_attention_freq=2,
        embed_dim=512,
        motion_encoder_dim=512,
        num_tokens=512,
        max_motion_tokens=100,
        speech_encoder_type='hubert-large',
    ):
        super().__init__()

        self.nfeats = nfeats
        self.num_tokens = num_tokens
        self.embed_dim = embed_dim

        # Speech encoder (frozen)
        self.speech_encoder = SpeechEncoder(speech_encoder_type, freeze=True)
        speech_dim = self.speech_encoder.output_dim  # 1024 for hubert-large

        # Speech projection heads
        self.audio_proj = nn.Linear(speech_dim, embed_dim)          # PTC: speech → contrastive space
        self.audio_to_qformer = nn.Linear(speech_dim, 768)          # PTM: speech → QFormer hidden size
        self.audio_to_motion = nn.Linear(speech_dim, 1408)          # GEN: speech → motion feature space

        # Motion projection: encoder output → QFormer input dimension
        motion_feature_dim = 1408
        self.motion_projection = nn.Parameter(torch.empty(motion_encoder_dim, motion_feature_dim))
        nn.init.normal_(self.motion_projection, std=motion_feature_dim ** -0.5)

        # Initialize QFormer with cross-attention to motion features
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, motion_feature_dim, cross_attention_freq
        )

        # Initialize BERT tokenizer (needed for QFormer token embedding sizing)
        self.tokenizer = self.init_tokenizer()
        self.Qformer.resize_token_embeddings(len(self.tokenizer))

        # Copy weights from BERT to query-specific parameters
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                if key_orig in state_dict:
                    param.data.copy_(state_dict[key_orig])

        # Projection heads
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)  # reused name for compat
        self.motion_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        # PTM head
        self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)

        # Motion token classifier for generation loss
        self.motion_cls = nn.Linear(self.Qformer.config.hidden_size, self.num_tokens, bias=False)

        # Positional embeddings for GEN loss (position-aware motion queries)
        self.max_motion_tokens = max_motion_tokens
        self.motion_pos_embed = nn.Embedding(max_motion_tokens, self.Qformer.config.hidden_size)

        # Learnable temperature for contrastive loss (used as exp(temp))
        self.temp = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        print(f"\n{'='*70}")
        print(f"{'LaMPSpeech Model Initialized':^70}")
        print(f"{'='*70}")
        print(f"{'Speech Encoder:':<25} {speech_encoder_type} ({speech_dim}D)")
        print(f"{'Motion Features:':<25} {nfeats}D")
        print(f"{'Motion Encoder Dim:':<25} {motion_encoder_dim}D")
        print(f"{'QFormer Feature Dim:':<25} {motion_feature_dim}D")
        print(f"{'Query Tokens:':<25} {num_query_token}")
        print(f"{'Embedding Dim:':<25} {embed_dim}D")
        print(f"{'Codebook Size:':<25} {num_tokens}")
        print(f"{'Losses:':<25} PTC + PTM + GEN (3)")
        print(f"{'='*70}\n")

    def parameters_wo_clip(self):
        """Return parameters excluding frozen components."""
        return (p for p in self.parameters() if p.requires_grad)

    def forward(self, motion_embeds, audio_waveforms, token_ids=None):
        """
        Forward pass with 3 training objectives.

        Args:
            motion_embeds: Motion embeddings [B, T', 512] from VQ-VAE encoder
            audio_waveforms: Raw audio waveforms [B, num_samples] at 16kHz
            token_ids: VQ token IDs [B, T'] for GEN loss

        Returns:
            QFormer_Output, speech_feat [B, embed_dim], motion_feats_pooled [B, embed_dim]
        """
        device = motion_embeds.device
        B = motion_embeds.shape[0]

        # ===== Speech Encoding =====
        speech_features, speech_mask = self.speech_encoder(audio_waveforms)  # [B, S, 1024], [B, S]

        # Pooled speech feature for PTC
        # Masked mean pooling
        speech_mask_f = speech_mask.unsqueeze(-1).float()  # [B, S, 1]
        speech_pooled = (speech_features * speech_mask_f).sum(dim=1) / speech_mask_f.sum(dim=1).clamp(min=1)  # [B, 1024]

        # ===== Motion Embedding Projection =====
        motion_embeds_proj = motion_embeds @ self.motion_projection  # [B, T', 1408]
        motion_atts = torch.ones(motion_embeds_proj.size()[:-1], dtype=torch.long).to(device)

        # ===== QFormer: Motion → Query Features =====
        query_tokens = self.query_tokens.expand(B, -1, -1)  # [B, 49, 768]
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=motion_embeds_proj,
            encoder_attention_mask=motion_atts,
            use_cache=True,
            return_dict=True,
        )

        # Motion features from query output
        motion_feats = F.normalize(
            self.motion_proj(query_output.last_hidden_state), dim=-1
        )  # [B, 49, embed_dim]

        motion_feats_pooled = F.normalize(
            torch.mean(motion_feats, dim=1), dim=-1
        )  # [B, embed_dim]

        # ===== Speech Feature for PTC =====
        speech_feat = F.normalize(self.audio_proj(speech_pooled), dim=-1)  # [B, embed_dim]

        # ===== Loss 1: PTC (Point-Text Contrastive) — speech version =====
        motion_feats_all = concat_all_gather(motion_feats)
        speech_feat_all = concat_all_gather(speech_feat)

        logit_scale = self.temp.exp()  # exp(ln(1/0.07)) ≈ 14.3
        sim_q2t = torch.matmul(
            motion_feats.unsqueeze(1), speech_feat_all.unsqueeze(-1)
        ).squeeze(-1)  # [B, B*GPUs, 49]
        sim_p2t = logit_scale * sim_q2t

        sim_t2q = torch.matmul(
            speech_feat.unsqueeze(1).unsqueeze(1), motion_feats_all.permute(0, 2, 1)
        ).squeeze(-2)  # [B, B*GPUs, 49]
        sim_t2p = logit_scale * sim_t2q

        # Contrastive targets with rank offset for multi-GPU
        bs = B
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0
        targets = torch.arange(bs, device=device) + rank * bs

        loss_p2t = F.cross_entropy(sim_p2t.max(-1)[0], targets, label_smoothing=0.1)
        loss_t2p = F.cross_entropy(sim_t2p.max(-1)[0], targets, label_smoothing=0.1)
        loss_ptc = (loss_p2t + loss_t2p) / 2

        # ===== Loss 2: PTM (Point-Text Matching) — speech version =====
        # Project speech to QFormer hidden size as "pseudo text embeddings"
        speech_qformer = self.audio_to_qformer(speech_features)  # [B, S, 768]

        speech_qformer_all = concat_all_gather(speech_qformer)
        speech_mask_all = concat_all_gather(speech_mask)
        motion_embeds_world = all_gather_with_grad(motion_embeds_proj)

        with torch.no_grad():
            # rank already computed above for PTC targets
            sim_t2p_max = sim_t2p.max(-1)[0].clone()
            sim_p2t_max = sim_p2t.max(-1)[0].clone()

            for i in range(bs):
                sim_t2p_max[i, rank * bs + i] = -10000
                sim_p2t_max[i, rank * bs + i] = -10000

            weights_t2p = F.softmax(sim_t2p_max, dim=1)
            weights_p2t = F.softmax(sim_p2t_max, dim=1)

        # Select negative motion for each speech
        motion_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2p[b], 1).item()
            motion_embeds_neg.append(motion_embeds_world[neg_idx])
        motion_embeds_neg = torch.stack(motion_embeds_neg, dim=0)

        # Select negative speech for each motion
        speech_qformer_neg = []
        speech_mask_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_p2t[b], 1).item()
            speech_qformer_neg.append(speech_qformer_all[neg_idx])
            speech_mask_neg.append(speech_mask_all[neg_idx])
        speech_qformer_neg = torch.stack(speech_qformer_neg, dim=0)
        speech_mask_neg = torch.stack(speech_mask_neg, dim=0)

        # Create triplets: (pos motion, pos speech), (neg motion, pos speech), (pos motion, neg speech)
        # Feed speech embeddings directly into QFormer as input_embeds (bypassing token embedding lookup)
        speech_emb_all = torch.cat([speech_qformer, speech_qformer, speech_qformer_neg], dim=0)  # [3B, S, 768]
        speech_att_all = torch.cat([speech_mask, speech_mask, speech_mask_neg], dim=0)  # [3B, S]

        query_tokens_ptm = self.query_tokens.expand(speech_emb_all.shape[0], -1, -1)
        query_atts_ptm = torch.ones(query_tokens_ptm.size()[:-1], dtype=torch.long).to(device)

        # Concatenate query attention and speech attention
        # QFormer expects: [query_atts, text_atts] for the attention_mask
        attention_mask_all = torch.cat([query_atts_ptm, speech_att_all], dim=1)

        motion_embeds_all = torch.cat([motion_embeds_proj, motion_embeds_neg, motion_embeds_proj], dim=0)
        motion_atts_all = torch.ones(motion_embeds_all.size()[:-1], dtype=torch.long).to(device)

        # QFormer with both motion cross-attention and speech as "text" input
        output_ptm = self.Qformer.bert(
            query_embeds=query_tokens_ptm,
            inputs_embeds=speech_emb_all,  # Use inputs_embeds instead of input_ids
            attention_mask=attention_mask_all,
            encoder_hidden_states=motion_embeds_all,
            encoder_attention_mask=motion_atts_all,
            return_dict=True,
        )

        vl_embeddings = output_ptm.last_hidden_state[:, :query_tokens_ptm.size(1), :]
        vl_output = self.itm_head(vl_embeddings)  # [3B, 49, 2]
        logits = vl_output.mean(dim=1)  # [3B, 2]

        ptm_labels = torch.cat(
            [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)], dim=0
        ).to(device)
        loss_ptm = F.cross_entropy(logits, ptm_labels)

        # ===== Loss 3: GEN (Generation) — speech version =====
        if token_ids is not None:
            motion_targets = token_ids  # [B, T']
        else:
            raise ValueError("token_ids must be provided for GEN loss")

        T_motion = min(motion_targets.shape[1], self.max_motion_tokens)
        motion_targets = motion_targets[:, :T_motion]

        # Create position-aware queries for each motion token position
        pos_ids = torch.arange(T_motion, device=device).unsqueeze(0).expand(B, -1)  # [B, T_motion]
        motion_queries = self.motion_pos_embed(pos_ids)  # [B, T_motion, 768]

        # Project speech to motion feature space as conditioning
        speech_motion = self.audio_to_motion(speech_features)  # [B, S, 1408]

        # QFormer: position queries cross-attend to speech features
        gen_output = self.Qformer.bert(
            query_embeds=motion_queries,
            encoder_hidden_states=speech_motion,
            encoder_attention_mask=speech_mask,
            use_cache=True,
            return_dict=True,
        )

        # Predict motion token at each position
        prediction = self.motion_cls(gen_output.last_hidden_state)  # [B, T_motion, num_tokens]

        loss_fct = nn.CrossEntropyLoss(reduction='mean', label_smoothing=0.1)
        loss_gen = loss_fct(prediction.reshape(-1, self.num_tokens), motion_targets.reshape(-1))

        # ===== Total Loss =====
        total_loss = loss_ptc + loss_ptm + loss_gen

        return QFormer_Output(
            loss=total_loss,
            loss_ptc=loss_ptc,
            loss_ptm=loss_ptm,
            loss_lm=torch.tensor(0.0, device=device),
            loss_gen=loss_gen
        ), speech_feat, motion_feats_pooled
