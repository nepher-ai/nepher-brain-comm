# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Participant brain interface.

A brain turns an observation dict into an action chunk. The server owns
seeding, so ``act`` itself must not draw unseeded randomness.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class Brain(ABC):
    """Inference entry point loaded from a submission."""

    def setup(self, device: str, weights_dir: Path, config: dict) -> None:
        """Load weights. The default does nothing.

        Args:
            device: Torch device string for this replica, such as ``"cuda:0"``.
            weights_dir: Read-only directory of submitted weights.
            config: Parsed ``agent.yaml``.
        """

    @abstractmethod
    def begin_episode(self, episode_ids: list[str], seeds: list[int], instructions: list[str]) -> None:
        """Start a batch of episodes.

        Args:
            episode_ids: Stable id per environment.
            seeds: Per-environment seed.
            instructions: Natural-language instruction per environment.
        """

    @abstractmethod
    def act(self, obs: dict[str, np.ndarray], step: int) -> np.ndarray:
        """Return an action chunk.

        Args:
            obs: Arrays batched on axis 0. Each value has shape ``[num_envs, ...]``.
            step: Control step at which this chunk starts.

        Returns:
            Actions with shape ``[num_envs, horizon, action_dim]``.
        """

    def end_episode(self, episode_ids: list[str]) -> None:
        """Release per-episode state. The default does nothing."""
