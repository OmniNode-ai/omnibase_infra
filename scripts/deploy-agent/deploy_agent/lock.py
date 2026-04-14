# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Single-flight advisory lock for deploy-agent rebuild operations."""

from __future__ import annotations

import contextlib
import fcntl
import os
from pathlib import Path

from deploy_agent.events import DeployInProgressError

_LOCK_PATH = Path(os.environ.get("DEPLOY_AGENT_LOCK_PATH", "/tmp/deploy-agent.lock"))  # noqa: S108


@contextlib.contextmanager
def single_flight_lock():
    """Advisory exclusive lock. Raises DeployInProgressError if already held."""
    _LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd = _LOCK_PATH.open("w")
    try:
        fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fd.close()
        raise DeployInProgressError("Another deploy is already in flight")
    try:
        yield
    finally:
        fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
        fd.close()
