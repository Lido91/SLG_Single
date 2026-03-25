# Motion architectures
from .mgpt_vq import VQVae
from .mgpt_rvq import RVQVae
from .mgpt_hrvq import HRVQVae
from .mgpt_lm import MLM

# FSQ-based architectures
from .mgpt_fsq import FSQVae
from .mgpt_fsq_rvq import FSQRVQVae

# VQ-Style RVQ-VAE (contrastive + MI losses)
from .mgpt_rvq_vqstyle import RVQVaeVQStyle

# Align RVQ-VAE (cumulative text alignment via ITC)
from .mgpt_rvq_align import RVQVaeAlign

# LG-VQ RVQ-VAE (Language-Guided Codebook Learning)
from .mgpt_rvq_lgvq import RVQVaeLGVQ

# Speech LG-VQ RVQ-VAE (Speech-Guided Codebook Learning)
from .mgpt_rvq_lgvq_speech import RVQVaeLGVQSpeech

# Hierarchical RVQ-GPT
from .mgpt_rvq_hierarchical import HierarchicalRVQGPT

# Multi-head Text2VQPoseGPT
from .mgpt_multihead_t2vqp import Text2VQPoseGPT

# MoMask transformers
from .mask_transformer import MaskTransformer
from .residual_transformer import ResidualTransformer

# Text-to-Motion Transformer (UniMuMo-style)
from .tm_transformer import MotionLM, MotionTransformer, ScaledEmbedding, LMOutput
from .tm_conditioner import T5Conditioner, CFGDropout, ConditionProvider
from .tm_codebook_patterns import (
    Pattern, CodebooksPatternProvider, DelayedPatternProvider,
    ParallelPatternProvider, FlattenPatternProvider, get_pattern_provider
)
from .tm_model import TextToMotionLM, TMModelOutput, build_text_to_motion_model

__all__ = [
    'VQVae',
    'RVQVae',
    'RVQVaeVQStyle',
    'RVQVaeAlign',
    'RVQVaeLGVQ',
    'RVQVaeLGVQSpeech',
    'HRVQVae',
    'MLM',
    'FSQVae',
    'FSQRVQVae',
    'HierarchicalRVQGPT',
    'Text2VQPoseGPT',
    'MaskTransformer',
    'ResidualTransformer',
    # Text-to-Motion Transformer
    'MotionLM',
    'MotionTransformer',
    'ScaledEmbedding',
    'LMOutput',
    'T5Conditioner',
    'CFGDropout',
    'ConditionProvider',
    'Pattern',
    'CodebooksPatternProvider',
    'DelayedPatternProvider',
    'ParallelPatternProvider',
    'FlattenPatternProvider',
    'get_pattern_provider',
    'TextToMotionLM',
    'TMModelOutput',
    'build_text_to_motion_model',
]
