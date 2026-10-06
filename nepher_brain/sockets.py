# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unix domain sockets, with a TCP loopback file on Windows."""

from __future__ import annotations

import os
import socket
from pathlib import Path


def bind_listener(path: Path) -> socket.socket:
    """Bind ``path`` and return a listening socket.

    On Windows the file contains ``tcp://127.0.0.1:<port>`` and the socket is
    loopback TCP. Elsewhere ``path`` is a Unix socket.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    if os.name == "nt" or not hasattr(socket, "AF_UNIX"):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        path.write_text(f"tcp://127.0.0.1:{port}\n", encoding="utf-8")
        return listener
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(path))
    listener.listen(1)
    return listener


def connect(path: Path, timeout: float) -> socket.socket:
    """Connect to a socket created by :func:`bind_listener`."""
    path = Path(path)
    if path.is_file():
        text = path.read_text(encoding="utf-8").strip()
        if text.startswith("tcp://"):
            host, port_text = text[len("tcp://") :].rsplit(":", 1)
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            sock.connect((host, int(port_text)))
            return sock
    if os.name == "nt" or not hasattr(socket, "AF_UNIX"):
        raise FileNotFoundError(
            f"brain address file {path} is missing or is not a tcp:// address. "
            "Start nepher-brain serve with the same --socket-dir before evaluation."
        )
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    sock.connect(str(path))
    return sock
