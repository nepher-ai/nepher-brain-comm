# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import textwrap
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from nepher_brain_comm.client import BrainClient, BrainTimeout
from nepher_brain_comm.serve import run_replica
from nepher_brain_comm.smoke import smoke

ROOT = Path(__file__).resolve().parents[1]
ZERO = ROOT / "examples" / "zero_brain"


def test_zero_brain_smoke(tmp_path: Path):
    _start(tmp_path, ZERO, entry=None)
    spec = smoke(tmp_path / "brain-0.sock", timeout_s=5)
    assert spec["horizon"] == 1
    assert spec["action_dim"] == 8


def test_act_is_deterministic_for_the_same_seed(tmp_path: Path):
    submission = _noise_submission(tmp_path / "submission")
    _start(tmp_path, submission, entry="noise:NoiseBrain")
    client = BrainClient(tmp_path / "brain-0.sock", timeout_s=5)
    client.connect()
    obs = {"probe": np.zeros((2, 3), dtype=np.float32)}
    client.begin_episode(["a", "b"], [7, 8], ["one", "two"])
    first = client.act(obs, step=1, seed=7)
    second = client.act(obs, step=1, seed=7)
    other = client.act(obs, step=2, seed=7)
    client.close()
    assert np.array_equal(first, second)
    assert not np.array_equal(first, other)
    assert first.shape == (2, 1, 4)


def test_act_timeout(tmp_path: Path):
    submission = _slow_submission(tmp_path / "submission")
    _start(tmp_path, submission, entry="slow:SlowBrain")
    client = BrainClient(tmp_path / "brain-0.sock", timeout_s=0.2)
    client.connect()
    with pytest.raises(BrainTimeout):
        client.act({"probe": np.zeros((1, 1), np.float32)}, step=0, seed=0)


def _start(socket_dir: Path, submission: Path, entry: str | None) -> None:
    args = argparse.Namespace(
        entry=entry,
        replicas=1,
        socket_dir=str(socket_dir),
        submission=str(submission),
        max_gb=None,
    )
    thread = threading.Thread(target=run_replica, args=(0, args), daemon=True)
    thread.start()
    deadline = time.time() + 5
    while time.time() < deadline:
        if (socket_dir / "brain-0.sock").exists():
            return
        time.sleep(0.02)
    raise RuntimeError("brain socket was not created")


def _noise_submission(path: Path) -> Path:
    _write_submission(
        path,
        "noise:NoiseBrain",
        '''
import numpy as np
from nepher_brain_comm.brain import Brain

class NoiseBrain(Brain):
    def setup(self, device, weights_dir, config):
        self.horizon = int(config["horizon"])
        self.action_dim = int(config["action_dim"])

    def begin_episode(self, episode_ids, seeds, instructions):
        return None

    def act(self, obs, step):
        count = next(iter(obs.values())).shape[0]
        return np.random.standard_normal((count, self.horizon, self.action_dim)).astype(np.float32)
''',
        horizon=1,
        action_dim=4,
    )
    return path


def _slow_submission(path: Path) -> Path:
    _write_submission(
        path,
        "slow:SlowBrain",
        '''
import time
import numpy as np
from nepher_brain_comm.brain import Brain

class SlowBrain(Brain):
    def setup(self, device, weights_dir, config):
        self.horizon = 1
        self.action_dim = 1

    def begin_episode(self, episode_ids, seeds, instructions):
        return None

    def act(self, obs, step):
        time.sleep(2)
        return np.zeros((1, 1, 1), dtype=np.float32)
''',
        horizon=1,
        action_dim=1,
    )
    return path


def _write_submission(path: Path, entry: str, source: str, horizon: int, action_dim: int) -> None:
    module, _class_name = entry.split(":")
    (path / "brain").mkdir(parents=True)
    (path / "weights").mkdir()
    (path / "train").mkdir()
    (path / "brain" / f"{module}.py").write_text(textwrap.dedent(source), encoding="utf-8")
    (path / "agent.yaml").write_text(
        f"entry: {entry}\nhorizon: {horizon}\naction_dim: {action_dim}\nweights: {{}}\n",
        encoding="utf-8",
    )
