"""
LaMP (Language-Motion Pretraining) Architecture

Adapted from the original LaMP codebase for sign language generation.
Integrates with MotionGPT's RVQ-VAE tokenizer.
"""

from .lamp_model import LaMP
from .lamp_model_speech import LaMPSpeech

__all__ = ['LaMP', 'LaMPSpeech']
