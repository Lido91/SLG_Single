"""
Vector Quantization modules for MotionGPT

Includes:
- FSQ: Finite Scalar Quantization (no learned codebook)
"""

from .FSQ import FSQ

__all__ = ['FSQ']
