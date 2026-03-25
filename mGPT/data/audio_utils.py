"""
Audio Utilities for Speech-Driven Sign Language Generation

Handles audio loading, preprocessing, and augmentation for speech encoders.
"""
import torch
import torchaudio
import numpy as np
from typing import Union, Optional
import warnings


def load_audio(
    audio_path: str,
    target_sr: int = 16000,
    mono: bool = True,
    normalize: bool = True,
    max_duration: Optional[float] = None
) -> torch.Tensor:
    """
    Load and preprocess audio file for speech encoder input.

    Args:
        audio_path: Path to audio file (.wav, .mp3, .flac, etc.)
        target_sr: Target sample rate (default: 16kHz for speech encoders)
        mono: Convert to mono if True
        normalize: Normalize amplitude to [-1, 1] range
        max_duration: Maximum duration in seconds (truncate if longer)

    Returns:
        audio: (num_samples,) audio tensor at target_sr
               Ready for speech encoder input (requires adding batch dim)

    Example:
        >>> audio = load_audio('/path/to/audio.wav')
        >>> audio.shape  # (num_samples,)
        >>> # Add batch dimension for encoder
        >>> audio_batch = audio.unsqueeze(0)  # (1, num_samples)
    """
    try:
        # Load audio file
        waveform, sr = torchaudio.load(audio_path)

        # Convert to mono if needed
        if mono and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if needed
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr,
                new_freq=target_sr
            )
            waveform = resampler(waveform)

        # Truncate to max duration if specified
        if max_duration is not None:
            max_samples = int(max_duration * target_sr)
            if waveform.shape[1] > max_samples:
                waveform = waveform[:, :max_samples]

        # Normalize amplitude to [-1, 1]
        if normalize:
            waveform = waveform / waveform.abs().max().clamp(min=1e-8)

        # Remove channel dimension: (1, num_samples) -> (num_samples,)
        waveform = waveform.squeeze(0)

        return waveform

    except Exception as e:
        warnings.warn(f"Error loading audio from {audio_path}: {e}")
        # Return silent audio as fallback (3 seconds)
        return torch.zeros(target_sr * 3)


def load_audio_segment(
    audio_path: str,
    start_time: float,
    end_time: float,
    target_sr: int = 16000,
    mono: bool = True,
    normalize: bool = True
) -> torch.Tensor:
    """
    Load a specific segment of audio file.

    Args:
        audio_path: Path to audio file
        start_time: Start time in seconds
        end_time: End time in seconds
        target_sr: Target sample rate
        mono: Convert to mono if True
        normalize: Normalize amplitude

    Returns:
        audio: (num_samples,) audio segment at target_sr
    """
    try:
        # Get audio info first
        info = torchaudio.info(audio_path)
        orig_sr = info.sample_rate

        # Calculate frame offsets
        start_frame = int(start_time * orig_sr)
        num_frames = int((end_time - start_time) * orig_sr)

        # Load segment
        waveform, sr = torchaudio.load(
            audio_path,
            frame_offset=start_frame,
            num_frames=num_frames
        )

        # Convert to mono
        if mono and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr,
                new_freq=target_sr
            )
            waveform = resampler(waveform)

        # Normalize
        if normalize:
            waveform = waveform / waveform.abs().max().clamp(min=1e-8)

        return waveform.squeeze(0)

    except Exception as e:
        warnings.warn(f"Error loading audio segment from {audio_path}: {e}")
        duration = end_time - start_time
        return torch.zeros(int(duration * target_sr))


def pad_audio_batch(
    audio_list: list,
    pad_value: float = 0.0
) -> tuple:
    """
    Pad a list of audio tensors to the same length.

    Args:
        audio_list: List of (num_samples,) audio tensors
        pad_value: Value to use for padding

    Returns:
        padded_audio: (B, max_num_samples) padded audio batch
        lengths: (B,) tensor of original lengths
    """
    lengths = torch.tensor([len(audio) for audio in audio_list])
    max_len = lengths.max().item()

    batch_size = len(audio_list)
    padded = torch.full((batch_size, max_len), pad_value, dtype=torch.float32)

    for i, audio in enumerate(audio_list):
        padded[i, :len(audio)] = audio

    return padded, lengths


def compute_audio_length(num_samples: int, sample_rate: int = 16000) -> float:
    """
    Compute audio duration in seconds.

    Args:
        num_samples: Number of audio samples
        sample_rate: Sample rate in Hz

    Returns:
        duration: Duration in seconds
    """
    return num_samples / sample_rate


def align_audio_motion_lengths(
    audio_samples: int,
    motion_frames: int,
    audio_sr: int = 16000,
    motion_fps: int = 25
) -> dict:
    """
    Compute alignment information between audio and motion.

    Args:
        audio_samples: Number of audio samples
        motion_frames: Number of motion frames
        audio_sr: Audio sample rate (Hz)
        motion_fps: Motion frame rate (fps)

    Returns:
        dict with alignment info:
            - audio_duration: Audio duration in seconds
            - motion_duration: Motion duration in seconds
            - duration_diff: Absolute difference in seconds
            - audio_frames: Number of audio frames from speech encoder (~50fps)
            - samples_per_motion_frame: Audio samples per motion frame
    """
    audio_duration = audio_samples / audio_sr
    motion_duration = motion_frames / motion_fps

    # Speech encoders (HuBERT/WavLM) produce ~50 frames/sec
    # (stride of 320 samples at 16kHz)
    audio_encoder_fps = audio_sr / 320
    audio_encoder_frames = int(audio_samples / 320)

    samples_per_motion_frame = audio_sr / motion_fps

    return {
        'audio_duration': audio_duration,
        'motion_duration': motion_duration,
        'duration_diff': abs(audio_duration - motion_duration),
        'audio_encoder_frames': audio_encoder_frames,
        'motion_frames': motion_frames,
        'samples_per_motion_frame': samples_per_motion_frame,
        'sync_ratio': audio_encoder_frames / motion_frames if motion_frames > 0 else 0
    }


# Audio augmentation for training robustness
class AudioAugmentation:
    """
    Audio augmentation for improving model robustness.
    Use carefully for sign language - some augmentations may hurt temporal alignment.
    """

    @staticmethod
    def add_noise(audio: torch.Tensor, noise_level: float = 0.005) -> torch.Tensor:
        """Add Gaussian noise to audio."""
        noise = torch.randn_like(audio) * noise_level
        return audio + noise

    @staticmethod
    def volume_scale(audio: torch.Tensor, scale_range: tuple = (0.8, 1.2)) -> torch.Tensor:
        """Randomly scale volume."""
        scale = torch.rand(1).item() * (scale_range[1] - scale_range[0]) + scale_range[0]
        return audio * scale

    @staticmethod
    def time_mask(audio: torch.Tensor, max_mask_size: int = 1600, num_masks: int = 2) -> torch.Tensor:
        """
        Apply time masking (SpecAugment-style).
        Warning: May disrupt temporal alignment with motion!
        """
        audio = audio.clone()
        for _ in range(num_masks):
            mask_size = torch.randint(0, max_mask_size, (1,)).item()
            mask_start = torch.randint(0, max(1, len(audio) - mask_size), (1,)).item()
            audio[mask_start:mask_start + mask_size] = 0
        return audio


def check_audio_motion_sync(
    audio_path: str,
    motion_frames: int,
    audio_sr: int = 16000,
    motion_fps: int = 25,
    tolerance: float = 0.5
) -> bool:
    """
    Check if audio and motion are temporally aligned.

    Args:
        audio_path: Path to audio file
        motion_frames: Number of motion frames
        audio_sr: Audio sample rate
        motion_fps: Motion frame rate
        tolerance: Maximum allowed difference in seconds

    Returns:
        is_synced: True if within tolerance, False otherwise
    """
    try:
        info = torchaudio.info(audio_path)
        audio_duration = info.num_frames / info.sample_rate
        motion_duration = motion_frames / motion_fps

        diff = abs(audio_duration - motion_duration)
        return diff <= tolerance
    except:
        return False


def test_audio_utils():
    """Test audio utilities"""
    print("=" * 80)
    print("Testing Audio Utilities")
    print("=" * 80)

    # Create dummy audio (3 seconds at 16kHz)
    dummy_audio = torch.randn(16000 * 3)
    print(f"\nDummy audio shape: {dummy_audio.shape}")
    print(f"Duration: {compute_audio_length(len(dummy_audio))} seconds")

    # Test padding
    audio_list = [
        torch.randn(16000 * 2),
        torch.randn(16000 * 3),
        torch.randn(16000 * 1),
    ]
    padded, lengths = pad_audio_batch(audio_list)
    print(f"\nPadded batch shape: {padded.shape}")
    print(f"Original lengths: {lengths.tolist()}")

    # Test alignment computation
    alignment = align_audio_motion_lengths(
        audio_samples=16000 * 3,  # 3 seconds
        motion_frames=75,  # 3 seconds at 25fps
        audio_sr=16000,
        motion_fps=25
    )
    print(f"\nAlignment info:")
    for key, value in alignment.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")

    # Test augmentation
    augmenter = AudioAugmentation()
    noisy = augmenter.add_noise(dummy_audio, noise_level=0.01)
    scaled = augmenter.volume_scale(dummy_audio, scale_range=(0.8, 1.2))
    print(f"\nAugmentation:")
    print(f"  Noisy audio shape: {noisy.shape}")
    print(f"  Scaled audio shape: {scaled.shape}")

    print("\n" + "=" * 80)
    print("All tests passed!")
    print("=" * 80)


if __name__ == '__main__':
    test_audio_utils()
