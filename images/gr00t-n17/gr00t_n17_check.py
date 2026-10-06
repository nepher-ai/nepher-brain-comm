# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Architecture check for a GR00T N1.7 submission.

The hook compares the checkpoint config and tensor name/shape signature with
the reference sealed into this image. It does not judge how the weights were
trained. ``NEPHER_GROOT_SIGNATURE`` overrides the reference path in tests.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

REFERENCE = Path(__file__).with_name("reference_signature.json")


def check(submission: Path, config: dict) -> str | None:
    """Return an error string, or None when the checkpoint matches."""
    reference_path = Path(os.environ.get("NEPHER_GROOT_SIGNATURE", REFERENCE))
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    if not reference.get("sealed"):
        return "GR00T N1.7 reference signature is not sealed; refuse the submission"
    weights = Path(submission) / "weights"
    try:
        observed_config, tensors = load_signature(weights)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return f"cannot read checkpoint signature: {exc}"
    for key, expected in reference.get("config", {}).items():
        if observed_config.get(key) != expected:
            return f"config {key!r} does not match the GR00T N1.7 reference"
    expected_tensors = {name: list(shape) for name, shape in reference.get("tensors", {}).items()}
    if tensors != expected_tensors:
        return "tensor name/shape signature does not match the GR00T N1.7 reference"
    if not isinstance(config.get("embodiment_tag"), str) or not config["embodiment_tag"]:
        return "agent.yaml must set embodiment_tag to the base checkpoint's Franka tag"
    return None


def load_signature(weights: Path) -> tuple[dict, dict[str, list[int]]]:
    """Read config plus tensor shapes from a checkpoint directory."""
    config_path = weights / "config.json"
    if not config_path.is_file():
        raise ValueError("weights/config.json is missing")
    observed = json.loads(config_path.read_text(encoding="utf-8"))
    signature_path = weights / "signature.json"
    if signature_path.is_file():
        raw = json.loads(signature_path.read_text(encoding="utf-8"))
        return observed, {name: list(shape) for name, shape in raw.items()}
    extracted = _extract_from_files(weights)
    if extracted is None:
        raise ValueError("weights have no signature.json, safetensors, or torch checkpoint")
    return observed, extracted


def seal_from_directory(weights: Path, destination: Path) -> None:
    """Write a sealed reference from a base checkpoint. Run before publishing."""
    observed, tensors = load_signature(weights)
    payload = {"sealed": True, "config": {"model_type": observed.get("model_type")}, "tensors": tensors}
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _extract_from_files(weights: Path) -> dict[str, list[int]] | None:
    safetensors = sorted(weights.glob("*.safetensors"))
    if safetensors:
        from safetensors import safe_open

        shapes: dict[str, list[int]] = {}
        for path in safetensors:
            with safe_open(path, framework="numpy") as handle:
                for name in handle.keys():
                    shapes[name] = list(handle.get_slice(name).get_shape())
        return shapes
    checkpoints = sorted(weights.glob("*.pt"))
    if not checkpoints:
        return None
    import torch

    blob = torch.load(checkpoints[0], map_location="cpu", weights_only=True)
    tensors = blob.get("state_dict", blob) if isinstance(blob, dict) else blob
    return {name: list(value.shape) for name, value in tensors.items() if hasattr(value, "shape")}
