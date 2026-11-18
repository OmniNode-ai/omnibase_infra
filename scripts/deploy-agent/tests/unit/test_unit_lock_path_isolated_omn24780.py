# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A unit test never takes the host-global single-flight lock (OMN-24780).

``deploy_agent.lock`` defaults to one file under ``/tmp`` shared by every process
on the host. The deploy-agent CI job runs the suite on four xdist workers, and a
test that drives a job through the agent without patching the lock path takes that
shared file: a second worker doing the same at that moment is rejected with
``in_progress`` and the test it was running fails. On the merged head of dev a
follow-test lost that race (KeyError ``sibling_tags``) and failed CI Summary.
"""

from __future__ import annotations

import fcntl
import tempfile
from pathlib import Path

import pytest
from deploy_agent import lock as lock_mod
from deploy_agent.lock import single_flight_lock

#: The path the module resolves with no override, read at import time, before any
#: fixture runs (a fixture changing the module attribute cannot move it).
_HOST_GLOBAL_LOCK = lock_mod._LOCK_PATH


@pytest.mark.unit
def test_the_default_lock_path_is_the_host_global_file() -> None:
    """Positive control: the path this module captured at import is the shared file."""
    assert Path(tempfile.gettempdir(), "deploy-agent.lock") == _HOST_GLOBAL_LOCK


@pytest.mark.unit
def test_a_peer_holding_the_host_global_lock_does_not_reject_a_unit_test() -> None:
    """The race, made deterministic: a peer worker holds the shared file right now."""
    _HOST_GLOBAL_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with _HOST_GLOBAL_LOCK.open("w") as peer:
        fcntl.flock(peer.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            with single_flight_lock():
                pass  # must not raise DeployInProgressError
        finally:
            fcntl.flock(peer.fileno(), fcntl.LOCK_UN)


@pytest.mark.unit
def test_the_lock_path_lives_in_the_tests_own_directory(tmp_path: Path) -> None:
    assert lock_mod._LOCK_PATH.is_relative_to(tmp_path)
