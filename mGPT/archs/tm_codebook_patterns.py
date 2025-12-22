"""
Codebook Pattern Providers for RVQ Motion Codes
Adapted from UniMuMo's codebooks_patterns.py

Key Features:
- DelayedPatternProvider for interleaving multiple codebooks
- Pattern-based sequence building and reverting
- Support for 6 codebooks (MotionGPT RVQ)
"""

from collections import namedtuple
from dataclasses import dataclass
from functools import lru_cache
import logging
import typing as tp
from abc import ABC, abstractmethod

import torch


LayoutCoord = namedtuple('LayoutCoord', ['t', 'q'])  # (timestep, codebook index)
PatternLayout = tp.List[tp.List[LayoutCoord]]  # Sequence of coordinates
logger = logging.getLogger(__name__)


@dataclass
class Pattern:
    """
    Base implementation of a pattern over a sequence with multiple codebooks.

    The codebook pattern consists in a layout, defining for each sequence step
    the list of coordinates of each codebook timestep in the resulting interleaved sequence.

    Key methods:
    - build_pattern_sequence: Maps [B, K, T] codes to interleaved [B, K, S] sequence
    - revert_pattern_sequence: Maps [B, K, S] back to [B, K, T] original alignment
    - revert_pattern_logits: Maps logits back to original alignment
    """
    layout: PatternLayout
    timesteps: int
    n_q: int

    def __post_init__(self):
        assert len(self.layout) > 0
        assert self.layout[0] == []
        self._validate_layout()
        self._build_reverted_sequence_scatter_indexes = lru_cache(100)(
            self._build_reverted_sequence_scatter_indexes
        )
        self._build_pattern_sequence_scatter_indexes = lru_cache(100)(
            self._build_pattern_sequence_scatter_indexes
        )
        logger.debug(f"New pattern, time steps: {self.timesteps}, sequence steps: {len(self.layout)}")

    def _validate_layout(self):
        """Validate the pattern layout."""
        q_timesteps = {q: 0 for q in range(self.n_q)}
        for s, seq_coords in enumerate(self.layout):
            if len(seq_coords) > 0:
                qs = set()
                for coord in seq_coords:
                    qs.add(coord.q)
                    last_q_timestep = q_timesteps[coord.q]
                    assert coord.t >= last_q_timestep, \
                        f"Past timesteps found for codebook {coord.q} at step {s}"
                    q_timesteps[coord.q] = coord.t
                assert len(qs) == len(seq_coords), \
                    f"Multiple entries for same codebook at step {s}"

    @property
    def num_sequence_steps(self) -> int:
        return len(self.layout) - 1

    @property
    def max_delay(self) -> int:
        max_t_in_seq_coords = 0
        for seq_coords in self.layout[1:]:
            for coords in seq_coords:
                max_t_in_seq_coords = max(max_t_in_seq_coords, coords.t + 1)
        return max_t_in_seq_coords - self.timesteps

    @property
    def valid_layout(self) -> PatternLayout:
        valid_step = len(self.layout) - self.max_delay
        return self.layout[:valid_step]

    def _build_pattern_sequence_scatter_indexes(
        self,
        timesteps: int,
        n_q: int,
        keep_only_valid_steps: bool,
        device: tp.Union[torch.device, str] = 'cpu'
    ) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        """Build scatter indexes for pattern sequence construction."""
        assert n_q == self.n_q
        assert timesteps <= self.timesteps

        ref_layout = self.valid_layout if keep_only_valid_steps else self.layout
        indexes = torch.zeros(n_q, len(ref_layout), dtype=torch.long).numpy()
        mask = torch.zeros(n_q, len(ref_layout), dtype=torch.bool).numpy()
        indexes[:] = n_q * timesteps  # Fill with special token index

        for s, sequence_coords in enumerate(ref_layout):
            for coords in sequence_coords:
                if coords.t < timesteps:
                    indexes[coords.q, s] = coords.t + coords.q * timesteps
                    mask[coords.q, s] = 1

        indexes = torch.from_numpy(indexes).to(device)
        mask = torch.from_numpy(mask).to(device)
        return indexes, mask

    def build_pattern_sequence(
        self,
        z: torch.Tensor,
        special_token: int,
        keep_only_valid_steps: bool = False
    ) -> tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build interleaved sequence from input codes.

        Args:
            z: Input tensor [B, K, T]
            special_token: Special token for padding
            keep_only_valid_steps: Only include fully defined steps

        Returns:
            values: Interleaved sequence [B, K, S]
            indexes: Scatter indexes [K, S]
            mask: Valid position mask [K, S]
        """
        B, K, T = z.shape
        indexes, mask = self._build_pattern_sequence_scatter_indexes(
            T, K, keep_only_valid_steps=keep_only_valid_steps, device=str(z.device)
        )
        z = z.view(B, -1)
        z = torch.cat([z, torch.zeros_like(z[:, :1]) + special_token], dim=1)
        values = z[:, indexes.view(-1)]
        values = values.view(B, K, indexes.shape[-1])
        return values, indexes, mask

    def _build_reverted_sequence_scatter_indexes(
        self,
        sequence_steps: int,
        n_q: int,
        keep_only_valid_steps: bool = False,
        is_model_output: bool = False,
        device: tp.Union[torch.device, str] = 'cpu'
    ) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        """Build scatter indexes for reverting pattern sequence."""
        ref_layout = self.valid_layout if keep_only_valid_steps else self.layout
        timesteps = self.timesteps
        assert n_q == self.n_q
        assert sequence_steps <= len(ref_layout)

        if is_model_output:
            ref_layout = ref_layout[1:]

        indexes = torch.zeros(n_q, timesteps, dtype=torch.long).numpy()
        mask = torch.zeros(n_q, timesteps, dtype=torch.bool).numpy()
        indexes[:] = n_q * sequence_steps

        for s, sequence_codes in enumerate(ref_layout):
            if s < sequence_steps:
                for code in sequence_codes:
                    if code.t < timesteps:
                        indexes[code.q, code.t] = s + code.q * sequence_steps
                        mask[code.q, code.t] = 1

        indexes = torch.from_numpy(indexes).to(device)
        mask = torch.from_numpy(mask).to(device)
        return indexes, mask

    def revert_pattern_sequence(
        self,
        s: torch.Tensor,
        special_token: int,
        keep_only_valid_steps: bool = False
    ) -> tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Revert interleaved sequence back to original alignment.

        Args:
            s: Interleaved sequence [B, K, S]
            special_token: Special token value
            keep_only_valid_steps: Only use fully defined steps

        Returns:
            values: Original alignment [B, K, T]
            indexes: Scatter indexes [K, T]
            mask: Valid position mask [K, T]
        """
        B, K, S = s.shape
        indexes, mask = self._build_reverted_sequence_scatter_indexes(
            S, K, keep_only_valid_steps, is_model_output=False, device=str(s.device)
        )
        s = s.view(B, -1)
        s = torch.cat([s, torch.zeros_like(s[:, :1]) + special_token], dim=1)
        values = s[:, indexes.view(-1)]
        values = values.view(B, K, indexes.shape[-1])
        return values, indexes, mask

    def revert_pattern_logits(
        self,
        logits: torch.Tensor,
        special_token: float,
        keep_only_valid_steps: bool = False
    ) -> tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Revert model logits from pattern sequence to original alignment.

        Args:
            logits: Model logits [B, card, K, S]
            special_token: Special token value (usually NaN)
            keep_only_valid_steps: Only use fully defined steps

        Returns:
            values: Original alignment logits [B, card, K, T]
            indexes: Scatter indexes [K, T]
            mask: Valid position mask [K, T]
        """
        B, card, K, S = logits.shape
        indexes, mask = self._build_reverted_sequence_scatter_indexes(
            S, K, keep_only_valid_steps, is_model_output=True, device=logits.device
        )
        logits = logits.reshape(B, card, -1)
        logits = torch.cat([logits, torch.zeros_like(logits[:, :, :1]) + special_token], dim=-1)
        values = logits[:, :, indexes.view(-1)]
        values = values.view(B, card, K, indexes.shape[-1])
        return values, indexes, mask


class CodebooksPatternProvider(ABC):
    """
    Abstract base class for codebook pattern providers.

    Args:
        n_q: Number of codebooks
        cached: Whether to cache patterns
    """
    def __init__(self, n_q: int, cached: bool = True):
        assert n_q > 0
        self.n_q = n_q
        if cached:
            self.get_pattern = lru_cache(100)(self.get_pattern)

    @abstractmethod
    def get_pattern(self, timesteps: int) -> Pattern:
        """Build pattern for given number of timesteps."""
        raise NotImplementedError()


class DelayedPatternProvider(CodebooksPatternProvider):
    """
    Provider for delayed pattern across codebooks.

    Each codebook is delayed relative to the previous one, creating an
    interleaved sequence that maintains temporal dependencies.

    Example with timesteps=4, n_q=3, delays=[0,1,2]:
        Input:  [[1, 2, 3, 4],
                 [1, 2, 3, 4],
                 [1, 2, 3, 4]]
        Output: [[S, 1, 2, 3, 4],
                 [S, S, 1, 2, 3],
                 [S, S, S, 1, 2]]
        (S = special token)

    Args:
        n_q: Number of codebooks
        delays: Delay for each codebook. Default: [0, 1, 2, ..., n_q-1]
        flatten_first: Flatten first N timesteps
        empty_initial: Prepend N empty coordinate lists
    """
    def __init__(
        self,
        n_q: int,
        delays: tp.Optional[tp.List[int]] = None,
        flatten_first: int = 0,
        empty_initial: int = 0
    ):
        super().__init__(n_q)
        if delays is None:
            delays = list(range(n_q))
        self.delays = delays
        self.flatten_first = flatten_first
        self.empty_initial = empty_initial
        assert len(self.delays) == self.n_q
        assert sorted(self.delays) == self.delays

    def get_pattern(self, timesteps: int) -> Pattern:
        """Build delayed pattern for given timesteps."""
        out: PatternLayout = [[]]
        max_delay = max(self.delays)

        if self.empty_initial:
            out += [[] for _ in range(self.empty_initial)]

        if self.flatten_first:
            for t in range(min(timesteps, self.flatten_first)):
                for q in range(self.n_q):
                    out.append([LayoutCoord(t, q)])

        for t in range(self.flatten_first, timesteps + max_delay):
            v = []
            for q, delay in enumerate(self.delays):
                t_for_q = t - delay
                if t_for_q >= self.flatten_first:
                    v.append(LayoutCoord(t_for_q, q))
            out.append(v)

        return Pattern(out, n_q=self.n_q, timesteps=timesteps)


class ParallelPatternProvider(DelayedPatternProvider):
    """
    Pattern provider with no delays (all codebooks in parallel).

    This is equivalent to DelayedPatternProvider with delays=[0]*n_q.
    """
    def __init__(self, n_q: int):
        super().__init__(n_q, [0] * n_q)


class FlattenPatternProvider(CodebooksPatternProvider):
    """
    Pattern provider that flattens all codebooks sequentially.

    Each timestep's codebooks are emitted one after another before
    moving to the next timestep.

    Example with timesteps=3, n_q=2:
        Input:  [[1, 2, 3],
                 [1, 2, 3]]
        Output: [[S, 1, S, 2, S, 3],
                 [S, S, 1, S, 2, S]]
    """
    def __init__(self, n_q: int):
        super().__init__(n_q)

    def get_pattern(self, timesteps: int) -> Pattern:
        out: PatternLayout = [[]]
        for t in range(timesteps):
            for q in range(self.n_q):
                out.append([LayoutCoord(t, q)])
        return Pattern(out, n_q=self.n_q, timesteps=timesteps)


def get_pattern_provider(
    n_q: int,
    pattern_type: str = 'delayed',
    delays: tp.Optional[tp.List[int]] = None,
) -> CodebooksPatternProvider:
    """
    Factory function to create pattern providers.

    Args:
        n_q: Number of codebooks
        pattern_type: 'delayed', 'parallel', or 'flatten'
        delays: Custom delays for 'delayed' pattern

    Returns:
        CodebooksPatternProvider instance
    """
    if pattern_type == 'delayed':
        return DelayedPatternProvider(n_q, delays=delays)
    elif pattern_type == 'parallel':
        return ParallelPatternProvider(n_q)
    elif pattern_type == 'flatten':
        return FlattenPatternProvider(n_q)
    else:
        raise ValueError(f"Unknown pattern type: {pattern_type}")
