"""
Speech Encoder Module for Sign Language Generation
Supports HuBERT, WavLM, Whisper, and wav2vec2 encoders
"""
import torch
import torch.nn as nn
from typing import Tuple, Optional
from transformers import (
    HubertModel,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Model,
    Wav2Vec2Processor,
    WhisperModel,
    WhisperFeatureExtractor,
)


SPEECH_ENCODER_CONFIGS = {
    'hubert-base': {
        'dim': 768,
        'model': 'facebook/hubert-base-ls960',
        'processor': 'facebook/hubert-base-ls960',
        'description': 'HuBERT Base trained on LibriSpeech 960h'
    },
    'hubert-large': {
        'dim': 1024,
        'model': 'facebook/hubert-large-ll60k',
        'processor': 'facebook/hubert-large-ll60k',
        'description': 'HuBERT Large trained on Libri-Light 60k hours'
    },
    'hubert-xlarge': {
        'dim': 1280,
        'model': 'facebook/hubert-xlarge-ll60k',
        'processor': 'facebook/hubert-xlarge-ll60k',
        'description': 'HuBERT XLarge trained on Libri-Light 60k hours'
    },
    'wav2vec2-base': {
        'dim': 768,
        'model': 'facebook/wav2vec2-base-960h',
        'processor': 'facebook/wav2vec2-base-960h',
        'description': 'Wav2Vec2 Base trained on LibriSpeech 960h'
    },
    'wav2vec2-large': {
        'dim': 1024,
        'model': 'facebook/wav2vec2-large-xlsr-53',
        'processor': 'facebook/wav2vec2-large-xlsr-53',
        'description': 'Wav2Vec2 Large trained on 53 languages'
    },
    'wavlm-base': {
        'dim': 768,
        'model': 'microsoft/wavlm-base',
        'processor': 'microsoft/wavlm-base',
        'description': 'WavLM Base trained on 94k hours'
    },
    'wavlm-large': {
        'dim': 1024,
        'model': 'microsoft/wavlm-large',
        'processor': 'microsoft/wavlm-large',
        'description': 'WavLM Large trained on 94k hours'
    },
    'whisper-base': {
        'dim': 512,
        'model': 'openai/whisper-base',
        'processor': 'openai/whisper-base',
        'description': 'Whisper Base multilingual'
    },
    'whisper-medium': {
        'dim': 1024,
        'model': 'openai/whisper-medium',
        'processor': 'openai/whisper-medium',
        'description': 'Whisper Medium multilingual'
    },
    'whisper-large-v3': {
        'dim': 1280,
        'model': 'openai/whisper-large-v3',
        'processor': 'openai/whisper-large-v3',
        'description': 'Whisper Large v3 multilingual (128 mel bins)'
    },
}


class SpeechEncoder(nn.Module):
    """
    Speech encoder wrapper for various pretrained models.
    Encodes raw audio waveforms into frame-level speech representations.

    Supports:
    - HuBERT (facebook/hubert-base-ls960, hubert-large-ll60k)
    - WavLM (microsoft/wavlm-base, wavlm-large)
    - Wav2Vec2 (facebook/wav2vec2-base-960h, wav2vec2-large-xlsr-53)
    - Whisper (openai/whisper-base, whisper-large-v3)

    Args:
        encoder_type: Model type (e.g., 'hubert-large', 'wavlm-large')
        freeze: Whether to freeze encoder weights (default: True)

    Input:
        audio_waveforms: (B, num_samples) raw audio at 16kHz

    Output:
        audio_features: (B, seq_len, dim) frame-level embeddings
        attention_mask: (B, seq_len) mask for valid frames
    """

    def __init__(self, encoder_type: str = 'hubert-large', freeze: bool = True):
        super().__init__()

        if encoder_type not in SPEECH_ENCODER_CONFIGS:
            raise ValueError(
                f"Unknown encoder type: {encoder_type}. "
                f"Available: {list(SPEECH_ENCODER_CONFIGS.keys())}"
            )

        self.encoder_type = encoder_type
        self.config = SPEECH_ENCODER_CONFIGS[encoder_type]
        self.output_dim = self.config['dim']
        self.freeze = freeze

        print(f"Loading {encoder_type}: {self.config['description']}")
        print(f"Output dimension: {self.output_dim}D")

        # Load model and processor based on encoder type
        if 'hubert' in encoder_type:
            self._load_hubert()
        elif 'wavlm' in encoder_type:
            self._load_wavlm()
        elif 'wav2vec2' in encoder_type:
            self._load_wav2vec2()
        elif 'whisper' in encoder_type:
            self._load_whisper()
        else:
            raise ValueError(f"Unsupported encoder type: {encoder_type}")

        # Freeze encoder if specified
        if freeze:
            self._freeze_encoder()
            print(f"Encoder frozen: {freeze}")

    def _load_hubert(self):
        """Load HuBERT model and feature extractor"""
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.config['processor']
        )
        self.model = HubertModel.from_pretrained(self.config['model'])
        self.encoder_mode = 'waveform'  # Processes raw waveform

    def _load_wavlm(self):
        """Load WavLM model and feature extractor"""
        try:
            from transformers import WavLMModel
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
                self.config['processor']
            )
            self.model = WavLMModel.from_pretrained(self.config['model'])
            self.encoder_mode = 'waveform'
        except ImportError:
            raise ImportError(
                "WavLM requires transformers>=4.26. "
                "Please upgrade: pip install transformers>=4.26"
            )

    def _load_wav2vec2(self):
        """Load Wav2Vec2 model and processor"""
        self.processor = Wav2Vec2Processor.from_pretrained(
            self.config['processor']
        )
        self.model = Wav2Vec2Model.from_pretrained(self.config['model'])
        self.encoder_mode = 'waveform'

    def _load_whisper(self):
        """Load Whisper model and feature extractor"""
        self.processor = WhisperFeatureExtractor.from_pretrained(
            self.config['processor']
        )
        self.model = WhisperModel.from_pretrained(self.config['model'])
        self.encoder_mode = 'mel_spectrogram'  # Whisper uses mel spectrogram

    def _freeze_encoder(self):
        """Freeze all encoder parameters"""
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def train(self, mode: bool = True):
        """Override to keep frozen encoder in eval mode."""
        if self.freeze:
            return super().train(False)
        return super().train(mode)

    def forward(
        self,
        audio_waveforms: torch.Tensor,
        return_attention_mask: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Encode audio waveforms to speech features.

        Args:
            audio_waveforms: (B, num_samples) raw audio at 16kHz
            return_attention_mask: Whether to return attention mask

        Returns:
            audio_features: (B, seq_len, dim) frame-level embeddings
                - For HuBERT/WavLM/Wav2Vec2: ~50 frames/sec (320 samples/frame)
                - For Whisper: 1500 frames for 30sec audio (~50 frames/sec)
            attention_mask: (B, seq_len) mask for valid frames (1=valid, 0=padding)
                - None if return_attention_mask=False
        """
        device = next(self.model.parameters()).device

        # Move audio to correct device if needed
        if audio_waveforms.device != device:
            audio_waveforms = audio_waveforms.to(device)

        # Process based on encoder type
        if self.encoder_mode == 'waveform':
            # HuBERT, WavLM, Wav2Vec2: process raw waveform
            audio_features, attention_mask = self._encode_waveform(audio_waveforms)
        elif self.encoder_mode == 'mel_spectrogram':
            # Whisper: process mel spectrogram
            audio_features, attention_mask = self._encode_whisper(audio_waveforms)
        else:
            raise ValueError(f"Unknown encoder mode: {self.encoder_mode}")

        if return_attention_mask:
            return audio_features, attention_mask
        else:
            return audio_features

    def _encode_waveform(
        self,
        audio_waveforms: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode waveform using HuBERT/WavLM/Wav2Vec2.

        Args:
            audio_waveforms: (B, num_samples) at 16kHz

        Returns:
            features: (B, seq_len, dim) where seq_len ≈ num_samples / 320
            attention_mask: (B, seq_len)
        """
        device = audio_waveforms.device

        # Convert to numpy for processor (handles batching and padding)
        # Note: processor expects list of 1D arrays or 2D array
        if audio_waveforms.dim() == 1:
            audio_waveforms = audio_waveforms.unsqueeze(0)

        # Pass individual arrays so the processor normalizes each sample
        # independently, avoiding statistics corruption from zero-padding.
        audio_np = audio_waveforms.cpu().numpy()
        inputs = self.processor(
            [audio_np[i] for i in range(audio_np.shape[0])],
            sampling_rate=16000,
            return_tensors='pt',
            padding=True  # Pad to max length in batch
        )

        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Forward pass through encoder
        if self.freeze:
            with torch.no_grad():
                outputs = self.model(
                    inputs['input_values'],
                    attention_mask=inputs.get('attention_mask'),
                    return_dict=True
                )
        else:
            outputs = self.model(
                inputs['input_values'],
                attention_mask=inputs.get('attention_mask'),
                return_dict=True
            )

        # Extract features and mask
        audio_features = outputs.last_hidden_state  # (B, seq_len, dim)

        # Get attention mask (from processor or compute from feature lengths)
        if 'attention_mask' in inputs:
            # Downsample attention mask to match feature length
            # HuBERT/Wav2Vec2: 320 samples -> 1 frame (stride=320)
            input_mask = inputs['attention_mask']  # (B, num_samples)
            feature_length = audio_features.shape[1]

            # Simple downsampling (take every 320th element)
            # More accurate: compute actual feature lengths
            attention_mask = self._downsample_mask(
                input_mask,
                feature_length
            )
        else:
            # No padding, all frames valid
            attention_mask = torch.ones(
                audio_features.shape[:2],
                dtype=torch.long,
                device=device
            )

        return audio_features.float(), attention_mask

    def _encode_whisper(
        self,
        audio_waveforms: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode audio using Whisper encoder.

        Args:
            audio_waveforms: (B, num_samples) at 16kHz

        Returns:
            features: (B, 1500, dim) for 30sec audio
            attention_mask: (B, 1500) - usually all ones (Whisper pads internally)
        """
        device = audio_waveforms.device

        if audio_waveforms.dim() == 1:
            audio_waveforms = audio_waveforms.unsqueeze(0)

        # Whisper processor converts to mel spectrogram
        inputs = self.processor(
            audio_waveforms.cpu().numpy(),
            sampling_rate=16000,
            return_tensors='pt'
        )

        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Forward pass through encoder only (not decoder)
        if self.freeze:
            with torch.no_grad():
                outputs = self.model.encoder(
                    inputs['input_features'],
                    return_dict=True
                )
        else:
            outputs = self.model.encoder(
                inputs['input_features'],
                return_dict=True
            )

        audio_features = outputs.last_hidden_state  # (B, 1500, dim)

        # Whisper has fixed-length output, create all-ones mask
        attention_mask = torch.ones(
            audio_features.shape[:2],
            dtype=torch.long,
            device=device
        )

        return audio_features.float(), attention_mask

    def _downsample_mask(
        self,
        input_mask: torch.Tensor,
        target_length: int
    ) -> torch.Tensor:
        """
        Downsample input mask to match feature length.

        Args:
            input_mask: (B, num_samples) mask for input audio
            target_length: Desired sequence length

        Returns:
            downsampled_mask: (B, target_length)
        """
        batch_size = input_mask.shape[0]
        device = input_mask.device

        # Compute how many input samples correspond to each feature
        # For HuBERT/Wav2Vec2: stride = 320 samples per frame
        stride = input_mask.shape[1] // target_length

        # Downsample by taking every stride-th element
        # A frame is valid if ANY of its input samples are valid
        downsampled = torch.zeros(
            batch_size,
            target_length,
            dtype=torch.long,
            device=device
        )

        for i in range(target_length):
            start_idx = i * stride
            end_idx = min(start_idx + stride, input_mask.shape[1])
            # Frame is valid if any sample in its window is valid
            downsampled[:, i] = input_mask[:, start_idx:end_idx].any(dim=1).long()

        return downsampled

    def get_output_length(self, input_length: int) -> int:
        """
        Compute output sequence length given input audio length.

        Args:
            input_length: Number of audio samples

        Returns:
            output_length: Number of feature frames
        """
        if self.encoder_mode == 'waveform':
            # HuBERT/WavLM/Wav2Vec2: stride of 320 samples
            return input_length // 320
        elif self.encoder_mode == 'mel_spectrogram':
            # Whisper: fixed length of 1500 frames for 30sec (480k samples)
            # For shorter audio, still returns 1500 frames (padded)
            return 1500
        else:
            raise ValueError(f"Unknown encoder mode: {self.encoder_mode}")


def test_speech_encoder():
    """Test function for speech encoder"""
    print("=" * 80)
    print("Testing Speech Encoder")
    print("=" * 80)

    # Create dummy audio (3 seconds at 16kHz)
    batch_size = 2
    audio_length = 16000 * 3  # 3 seconds
    dummy_audio = torch.randn(batch_size, audio_length)

    # Test HuBERT
    print("\n1. Testing HuBERT Large:")
    print("-" * 40)
    encoder = SpeechEncoder('hubert-large', freeze=True)
    features, mask = encoder(dummy_audio)
    print(f"Input shape: {dummy_audio.shape}")
    print(f"Output features shape: {features.shape}")
    print(f"Attention mask shape: {mask.shape}")
    print(f"Output dimension: {encoder.output_dim}D")
    print(f"Expected seq_len: ~{encoder.get_output_length(audio_length)} frames")

    # Test memory
    print(f"\nMemory usage:")
    print(f"  Features: {features.numel() * 4 / 1024 / 1024:.2f} MB (fp32)")
    print(f"  Features: {features.numel() * 2 / 1024 / 1024:.2f} MB (fp16)")

    print("\n" + "=" * 80)
    print("Test completed successfully!")
    print("=" * 80)


if __name__ == '__main__':
    test_speech_encoder()
