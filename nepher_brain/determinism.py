# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Seeding for one brain replica. Torch is used only when it is installed."""

from __future__ import annotations

import os
import random

import numpy as np


def apply_determinism_flags() -> None:
    """Enable deterministic kernels once, before the model is loaded."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch = _torch()
    if torch is None:
        return
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def reseed(seed: int, step: int) -> None:
    """Reseed Python, numpy, and torch from the episode seed and the step."""
    mixed = (int(seed) * 1_000_003 + int(step)) & 0xFFFFFFFF
    random.seed(mixed)
    np.random.seed(mixed)
    torch = _torch()
    if torch is None:
        return
    torch.manual_seed(mixed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(mixed)


def apply_memory_fraction(max_gpu_mem_gb: float | None) -> None:
    """Cap this process to the declared GPU budget, when CUDA is available."""
    if max_gpu_mem_gb is None:
        return
    torch = _torch()
    if torch is None or not torch.cuda.is_available():
        return
    total = torch.cuda.get_device_properties(0).total_memory
    fraction = min(1.0, (float(max_gpu_mem_gb) * (1024 ** 3)) / total)
    torch.cuda.set_per_process_memory_fraction(fraction)


def _torch():
    try:
        import torch
    except ImportError:
        return None
    return torch
