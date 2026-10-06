# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
import json
from pathlib import Path

_PATH = Path(__file__).resolve().parents[1] / "images" / "gr00t-n17" / "gr00t_n17_check.py"
_spec = importlib.util.spec_from_file_location("gr00t_n17_check", _PATH)
hook = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(hook)


def test_unsealed_reference_rejects(tmp_path: Path, monkeypatch):
    _submission(tmp_path, {"model_type": "gr00t_n1"}, {"w": [2]})
    monkeypatch.setenv(
        "NEPHER_GROOT_SIGNATURE",
        str(_reference(tmp_path, sealed=False, config={"model_type": "gr00t_n1"}, tensors={"w": [2]})),
    )
    assert hook.check(tmp_path, {"embodiment_tag": "franka"}) is not None


def test_matching_checkpoint_passes(tmp_path: Path, monkeypatch):
    _submission(tmp_path, {"model_type": "gr00t_n1"}, {"block.weight": [4, 4]})
    monkeypatch.setenv(
        "NEPHER_GROOT_SIGNATURE",
        str(_reference(tmp_path, sealed=True, config={"model_type": "gr00t_n1"}, tensors={"block.weight": [4, 4]})),
    )
    assert hook.check(tmp_path, {"embodiment_tag": "franka-joint"}) is None


def test_shape_mismatch_rejects(tmp_path: Path, monkeypatch):
    _submission(tmp_path, {"model_type": "gr00t_n1"}, {"block.weight": [4, 8]})
    monkeypatch.setenv(
        "NEPHER_GROOT_SIGNATURE",
        str(_reference(tmp_path, sealed=True, config={"model_type": "gr00t_n1"}, tensors={"block.weight": [4, 4]})),
    )
    message = hook.check(tmp_path, {"embodiment_tag": "franka-joint"})
    assert message is not None
    assert "signature" in message


def _submission(path: Path, config: dict, tensors: dict) -> None:
    weights = path / "weights"
    weights.mkdir(parents=True)
    (weights / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (weights / "signature.json").write_text(json.dumps(tensors), encoding="utf-8")


def _reference(path: Path, sealed: bool, config: dict, tensors: dict) -> Path:
    target = path / "reference.json"
    target.write_text(
        json.dumps({"sealed": sealed, "config": config, "tensors": tensors}),
        encoding="utf-8",
    )
    return target
