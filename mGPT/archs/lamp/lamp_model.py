"""
LaMP (Language-Motion Pretraining) Model

Aligned with SOKE implementation (single VQ-VAE, online encoding).

Architecture:
    Motion Embeddings (from external VQ-VAE encoder) → QFormer (learnable queries) → Motion Features
                                          ↓
    Text → BERT Tokenizer → Text Embeddings
                                          ↓
                         3 Training Objectives:
                         - loss_ptc: Point-Text Contrastive
                         - loss_ptm: Point-Text Matching
                         - loss_gen: Generation (T2M token prediction)
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


class LaMP(QFormer_Base):
    """
    LaMP: Language-Motion Pretraining with QFormer.
    Aligned with SOKE implementation — 3 losses (PTC + PTM + GEN), no LM loss.
    Accepts pre-computed motion embeddings from external VQ-VAE encoder.
    """

    def __init__(
        self,
        nfeats=133,
        num_query_token=49,
        cross_attention_freq=2,
        embed_dim=512,
        max_txt_len=32,
        motion_encoder_dim=512,
        num_tokens=512,
        max_motion_tokens=100,
    ):
        super().__init__()

        self.nfeats = nfeats
        self.num_tokens = num_tokens
        self.embed_dim = embed_dim
        self.max_txt_len = max_txt_len

        # Motion projection: encoder output → QFormer input dimension
        motion_feature_dim = 1408  # QFormer expects this dimension
        self.motion_projection = nn.Parameter(torch.empty(motion_encoder_dim, motion_feature_dim))
        nn.init.normal_(self.motion_projection, std=motion_feature_dim ** -0.5)

        # Initialize QFormer with cross-attention to motion features
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, motion_feature_dim, cross_attention_freq
        )

        # Initialize BERT tokenizer
        self.tokenizer = self.init_tokenizer()

        # Resize token embeddings for special tokens
        self.Qformer.resize_token_embeddings(len(self.tokenizer))

        # Copy weights from BERT to query-specific parameters
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                if key_orig in state_dict:
                    param.data.copy_(state_dict[key_orig])

        # Projection heads
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.motion_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        # PTM (Point-Text Matching) head
        self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)

        # Text projection for T2M generation
        self.text_projection = nn.Parameter(torch.empty(self.Qformer.config.hidden_size, motion_feature_dim))
        nn.init.normal_(self.text_projection, std=motion_feature_dim ** -0.5)

        # Motion token classifier for generation loss
        self.motion_cls = nn.Linear(self.Qformer.config.hidden_size, self.num_tokens, bias=False)

        # Positional embeddings for GEN loss (position-aware motion queries)
        self.max_motion_tokens = max_motion_tokens
        self.motion_pos_embed = nn.Embedding(max_motion_tokens, self.Qformer.config.hidden_size)

        # Learnable temperature for contrastive loss (used as exp(temp))
        self.temp = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        print(f"\n{'='*70}")
        print(f"{'LaMP Model Initialized (SOKE-aligned)':^70}")
        print(f"{'='*70}")
        print(f"{'Motion Features:':<25} {nfeats}D")
        print(f"{'Motion Encoder Dim:':<25} {motion_encoder_dim}D")
        print(f"{'QFormer Feature Dim:':<25} {motion_feature_dim}D")
        print(f"{'Query Tokens:':<25} {num_query_token}")
        print(f"{'Embedding Dim:':<25} {embed_dim}D")
        print(f"{'Codebook Size:':<25} {num_tokens}")
        print(f"{'Max Text Length:':<25} {max_txt_len}")
        print(f"{'Losses:':<25} PTC + PTM + GEN (3)")
        print(f"{'='*70}\n")

    def parameters_wo_clip(self):
        """Return parameters excluding CLIP/frozen components (SOKE compatibility)."""
        # All parameters are trainable in LaMP (no frozen CLIP encoder)
        return self.parameters()

    def forward(self, motion_embeds, text, token_ids=None):
        """
        Forward pass with 3 training objectives (SOKE-aligned).

        Args:
            motion_embeds: Motion embeddings [B, T', 512] from VQ-VAE encoder
            text: List of text descriptions [B]
            token_ids: VQ token IDs [B, T'] for GEN loss (optional, from VQ-VAE quantizer)

        Returns:
            QFormer_Output: Contains loss and individual loss components
            text_feat: Text features for monitoring [B, embed_dim]
            motion_feats: Motion features for monitoring [B, embed_dim]
        """
        device = motion_embeds.device
        B = motion_embeds.shape[0]

        # ===== Motion Embedding Projection =====
        # motion_embeds: [B, T', 512] — already encoded by external VQ-VAE encoder
        # Project to QFormer dimension
        motion_embeds_proj = motion_embeds @ self.motion_projection  # [B, T', 1408]

        motion_atts = torch.ones(motion_embeds_proj.size()[:-1], dtype=torch.long).to(device)  # [B, T']

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

        # Aggregate motion features (mean pooling)
        motion_feats_pooled = F.normalize(
            torch.mean(motion_feats, dim=1), dim=-1
        )  # [B, embed_dim]

        # ===== Text Encoding =====
        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(device)

        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )

        text_feat = F.normalize(
            self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        )  # [B, embed_dim] - CLS token

        # ===== Loss 1: Point-Text Contrastive (PTC) — SOKE style =====
        motion_feats_all = concat_all_gather(motion_feats)  # [B*GPUs, 49, embed_dim]
        text_feat_all = concat_all_gather(text_feat)  # [B*GPUs, embed_dim]

        # Motion-to-text similarity
        logit_scale = self.temp.exp()  # exp(ln(1/0.07)) ≈ 14.3
        sim_q2t = torch.matmul(
            motion_feats.unsqueeze(1), text_feat_all.unsqueeze(-1)
        ).squeeze(-1)  # [B, B*GPUs, 49]
        sim_p2t = logit_scale * sim_q2t  # [B, B*GPUs, 49]

        # Text-to-motion similarity
        sim_t2q = torch.matmul(
            text_feat.unsqueeze(1).unsqueeze(1), motion_feats_all.permute(0, 2, 1)
        ).squeeze(-2)  # [B, B*GPUs, 49]
        sim_t2p = logit_scale * sim_t2q  # [B, B*GPUs, 49]

        # Contrastive targets with rank offset for multi-GPU
        bs = B
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0
        targets = torch.arange(bs, device=device) + rank * bs

        # Max over queries at cross_entropy time (SOKE style)
        loss_p2t = F.cross_entropy(sim_p2t.max(-1)[0], targets, label_smoothing=0.1)
        loss_t2p = F.cross_entropy(sim_t2p.max(-1)[0], targets, label_smoothing=0.1)
        loss_ptc = (loss_p2t + loss_t2p) / 2

        # ===== Loss 2: Point-Text Matching (PTM) — SOKE style =====
        text_input_ids_world = concat_all_gather(text_tokens.input_ids)
        text_attention_mask_world = concat_all_gather(text_tokens.attention_mask)
        motion_embeds_world = all_gather_with_grad(motion_embeds_proj)

        with torch.no_grad():
            # rank already computed above for PTC targets
            sim_t2p_max = sim_t2p.max(-1)[0].clone()  # [B, B*GPUs]
            sim_p2t_max = sim_p2t.max(-1)[0].clone()  # [B, B*GPUs]

            # Mask out diagonal with rank awareness
            for i in range(bs):
                sim_t2p_max[i, rank * bs + i] = -10000
                sim_p2t_max[i, rank * bs + i] = -10000

            # Sample negative pairs based on similarity
            weights_t2p = F.softmax(sim_t2p_max, dim=1)
            weights_p2t = F.softmax(sim_p2t_max, dim=1)

        # Select negative motion for each text
        motion_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2p[b], 1).item()
            motion_embeds_neg.append(motion_embeds_world[neg_idx])
        motion_embeds_neg = torch.stack(motion_embeds_neg, dim=0)

        # Select negative text for each motion
        text_ids_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_p2t[b], 1).item()
            text_ids_neg.append(text_input_ids_world[neg_idx])
            text_atts_neg.append(text_attention_mask_world[neg_idx])
        text_ids_neg = torch.stack(text_ids_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        # Create triplets: (pos motion, pos text), (neg motion, pos text), (pos motion, neg text)
        text_ids_all = torch.cat([text_tokens.input_ids, text_tokens.input_ids, text_ids_neg], dim=0)
        text_atts_all = torch.cat([text_tokens.attention_mask, text_tokens.attention_mask, text_atts_neg], dim=0)

        query_tokens_ptm = self.query_tokens.expand(text_ids_all.shape[0], -1, -1)
        query_atts_ptm = torch.ones(query_tokens_ptm.size()[:-1], dtype=torch.long).to(device)
        attention_mask_all = torch.cat([query_atts_ptm, text_atts_all], dim=1)

        motion_embeds_all = torch.cat([motion_embeds_proj, motion_embeds_neg, motion_embeds_proj], dim=0)
        motion_atts_all = torch.ones(motion_embeds_all.size()[:-1], dtype=torch.long).to(device)

        # QFormer with both motion and text
        output_ptm = self.Qformer.bert(
            text_ids_all,
            query_embeds=query_tokens_ptm,
            attention_mask=attention_mask_all,
            encoder_hidden_states=motion_embeds_all,
            encoder_attention_mask=motion_atts_all,
            return_dict=True,
        )

        vl_embeddings = output_ptm.last_hidden_state[:, :query_tokens_ptm.size(1), :]
        vl_output = self.itm_head(vl_embeddings)  # [3*B, 49, 2]
        logits = vl_output.mean(dim=1)  # [3*B, 2]

        # Labels: first B are positive, next 2*B are negative
        ptm_labels = torch.cat(
            [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)], dim=0
        ).to(device)
        loss_ptm = F.cross_entropy(logits, ptm_labels)

        # ===== Loss 3: Generation (T2M) =====
        # Use position-aware motion queries so query[i] corresponds to motion_token[i]
        if token_ids is not None:
            motion_targets = token_ids  # [B, T']
        else:
            raise ValueError("token_ids must be provided for GEN loss")

        T_motion = min(motion_targets.shape[1], self.max_motion_tokens)
        motion_targets = motion_targets[:, :T_motion]

        # Create position-aware queries for each motion token position
        pos_ids = torch.arange(T_motion, device=device).unsqueeze(0).expand(B, -1)  # [B, T_motion]
        motion_queries = self.motion_pos_embed(pos_ids)  # [B, T_motion, 768]

        # Encode text as conditioning
        decoder_input_ids = text_tokens.input_ids.clone()
        decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
        text_embeds = self.Qformer.bert.embeddings(decoder_input_ids)
        text_embeds = text_embeds @ self.text_projection  # [B, seq_len, 1408]
        text_atts = text_tokens.attention_mask.clone()

        # QFormer: position queries cross-attend to text features
        gen_output = self.Qformer.bert(
            query_embeds=motion_queries,
            encoder_hidden_states=text_embeds,
            encoder_attention_mask=text_atts,
            use_cache=True,
            return_dict=True,
        )

        # Predict motion token at each position
        prediction = self.motion_cls(gen_output.last_hidden_state)  # [B, T_motion, num_tokens]

        loss_fct = nn.CrossEntropyLoss(reduction='mean', label_smoothing=0.1)
        loss_gen = loss_fct(prediction.reshape(-1, self.num_tokens), motion_targets.reshape(-1))

        # ===== Total Loss (3 losses, no LM) =====
        total_loss = loss_ptc + loss_ptm + loss_gen

        return QFormer_Output(
            loss=total_loss,
            loss_ptc=loss_ptc,
            loss_ptm=loss_ptm,
            loss_lm=torch.tensor(0.0, device=device),  # Fixed 0 for output format compatibility
            loss_gen=loss_gen
        ), text_feat, motion_feats_pooled

    def encode_text(self, text):
        """Encode text to features (for downstream tasks)."""
        device = next(self.parameters()).device

        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(device)

        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )

        text_feat = F.normalize(
            self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        )

        return text_feat

    def encode_motion(self, motion_embeds):
        """
        Encode motion embeddings to features (for downstream tasks).

        Args:
            motion_embeds: Motion embeddings [B, T', 512] from VQ-VAE encoder

        Returns:
            motion_features: [B, embed_dim] normalized motion features
        """
        device = motion_embeds.device
        B = motion_embeds.shape[0]

        # Project to QFormer dimension
        motion_embeds_proj = motion_embeds @ self.motion_projection  # [B, T', 1408]
        motion_atts = torch.ones(motion_embeds_proj.size()[:-1], dtype=torch.long).to(device)

        # QFormer
        query_tokens = self.query_tokens.expand(B, -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=motion_embeds_proj,
            encoder_attention_mask=motion_atts,
            use_cache=True,
            return_dict=True,
        )

        motion_feats = F.normalize(
            self.motion_proj(query_output.last_hidden_state), dim=-1
        )
        motion_feats_pooled = F.normalize(torch.mean(motion_feats, dim=1), dim=-1)

        return motion_feats_pooled
