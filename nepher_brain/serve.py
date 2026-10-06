# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Serve one brain replica per GPU.

``python -m nepher_brain.serve --replicas N --socket-dir /run/brain``
"""

from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from nepher_brain.brain import Brain
from nepher_brain.check import check_submission
from nepher_brain.determinism import apply_determinism_flags, apply_memory_fraction, reseed
from nepher_brain.protocol import PROTOCOL_VERSION, ProtocolError, recv_message, send_message
from nepher_brain.sockets import bind_listener


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    replica = os.environ.get("NEPHER_BRAIN_REPLICA")
    if replica is None and args.replicas > 1:
        _spawn_replicas(args)
        return
    index = int(replica) if replica is not None else 0
    if replica is None and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    run_replica(index, args)


def run_replica(index: int, args: argparse.Namespace) -> None:
    """Load the submission and serve ``brain-<index>.sock``."""
    apply_determinism_flags()
    config = check_submission(args.submission, max_gb=args.max_gb)
    entry = args.entry or config["entry"]
    brain_dir = Path(args.submission) / "brain"
    if str(brain_dir) not in sys.path:
        sys.path.insert(0, str(brain_dir))
    brain = _load_brain(entry)()
    device = "cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES", "0") != "-1" else "cpu"
    memory = config.get("max_gpu_mem_gb")
    apply_memory_fraction(float(memory) if memory is not None else None)
    brain.setup(device, Path(args.submission) / "weights", config)
    spec = {
        "type": "hello",
        "protocol_version": PROTOCOL_VERSION,
        "horizon": int(config["horizon"]),
        "action_dim": int(config["action_dim"]),
        "action_dtype": "float32",
    }
    listener = bind_listener(Path(args.socket_dir) / f"brain-{index}.sock")
    try:
        while True:
            conn, _addr = listener.accept()
            conn.settimeout(None)
            try:
                _serve_connection(conn, brain, spec)
            except (ConnectionError, ProtocolError, OSError):
                pass
            finally:
                conn.close()
    finally:
        listener.close()


def _serve_connection(conn, brain: Brain, spec: dict) -> None:
    while True:
        message = recv_message(conn)
        try:
            if not _dispatch(conn, brain, spec, message):
                return
        except Exception as exc:
            send_message(conn, {"type": "error", "message": str(exc)})


def _dispatch(conn, brain: Brain, spec: dict, message: dict) -> bool:
    """Handle one message. Return False when the connection should close."""
    kind = message.get("type")
    if kind == "hello":
        if message.get("protocol_version") != PROTOCOL_VERSION:
            send_message(conn, {"type": "error", "message": "protocol version mismatch"})
            return False
        send_message(conn, spec)
    elif kind == "begin_episode":
        brain.begin_episode(message["episode_ids"], message["seeds"], message["instructions"])
        send_message(conn, {"type": "begin_episode", "ok": True})
    elif kind == "act":
        reseed(int(message["seed"]), int(message["step"]))
        actions = np.asarray(brain.act(message["obs"], int(message["step"])), dtype=np.float32)
        _check_actions(actions, spec)
        send_message(conn, {"type": "act", "actions": actions})
    elif kind == "end_episode":
        brain.end_episode(list(message.get("episode_ids", [])))
        send_message(conn, {"type": "end_episode", "ok": True})
    elif kind == "close":
        send_message(conn, {"type": "close", "ok": True})
        return False
    else:
        send_message(conn, {"type": "error", "message": f"unknown message {kind!r}"})
        return False
    return True


def _check_actions(actions: np.ndarray, spec: dict) -> None:
    if actions.ndim != 3 or actions.shape[1] != spec["horizon"] or actions.shape[2] != spec["action_dim"]:
        raise ProtocolError(
            f"actions shape {tuple(actions.shape)} != "
            f"[num_envs, {spec['horizon']}, {spec['action_dim']}]"
        )


def _load_brain(entry: str) -> type[Brain]:
    module_name, class_name = entry.split(":")
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    if not isinstance(cls, type) or not issubclass(cls, Brain):
        raise TypeError(f"{entry} is not a Brain subclass")
    return cls


def _spawn_replicas(args: argparse.Namespace) -> None:
    procs = []
    command = [
        sys.executable,
        "-m",
        "nepher_brain.serve",
        "--replicas",
        str(args.replicas),
        "--socket-dir",
        args.socket_dir,
        "--submission",
        args.submission,
    ]
    if args.entry:
        command += ["--entry", args.entry]
    if args.max_gb is not None:
        command += ["--max-gb", str(args.max_gb)]
    for index in range(args.replicas):
        env = os.environ.copy()
        env["NEPHER_BRAIN_REPLICA"] = str(index)
        env["CUDA_VISIBLE_DEVICES"] = str(index)
        procs.append(subprocess.Popen(command, env=env))
    code = 0
    try:
        for proc in procs:
            status = proc.wait()
            if status != 0:
                code = status
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()
    raise SystemExit(code)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="nepher-brain serve")
    parser.add_argument("--entry", default=None, help="Override agent.yaml entry, as module:Class")
    parser.add_argument("--replicas", type=int, default=1)
    parser.add_argument("--socket-dir", default=os.environ.get("NEPHER_SOCKET_DIR", "/run/brain"))
    parser.add_argument("--submission", default=os.environ.get("NEPHER_SUBMISSION", "/submission"))
    parser.add_argument("--max-gb", type=float, default=None)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
