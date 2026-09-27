# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19501 -- lane-lock re-entrancy is per THREAD inside this process.

``lane_lock`` reads ``ONEX_LANE_LOCK_HELD`` to decide "an outer process in my
ancestry already holds this project, do not acquire again". That token lives in
``os.environ``, which is process-wide. While the agent ran one thread of lane
work, "this process holds it" and "my caller holds it" were the same fact.

OMN-19501 adds a second thread: the settle worker re-acquires the dev lane lock
for the onex-api pin recreate while the job thread may be running the next
compose job. With a process-wide token that re-acquire is a silent NO-OP, so the
pin recreates a container in the middle of another job's compose run. The
extended TLA+ model reports exactly that (``MC_a1_env_token_reentrancy``:
``NoOverlapWithPin`` violated), and the save-and-restore of the previous token
value leaks a released project back into the token when two threads interleave.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import pytest
from deploy_agent.lane_lock_client import (
    ENV_LOCK_HELD,
    LaneLockContendedError,
    lane_lock,
)

pytestmark = pytest.mark.unit

WAIT_SECONDS = 10.0


@pytest.fixture(autouse=True)
def _isolated_lock_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lock_dir = tmp_path / "lane-locks"
    lock_dir.mkdir()
    monkeypatch.setenv("ONEX_LANE_LOCK_DIR", str(lock_dir))
    monkeypatch.delenv(ENV_LOCK_HELD, raising=False)


def _hold_in_thread(
    project: str, acquired: threading.Event, release: threading.Event
) -> threading.Thread:
    def _body() -> None:
        with lane_lock(project, lane="dev", ref="r", timeout=WAIT_SECONDS):
            acquired.set()
            release.wait(WAIT_SECONDS)

    thread = threading.Thread(target=_body, daemon=True)
    thread.start()
    return thread


def test_another_thread_holding_the_project_is_contention_not_reentrancy() -> None:
    """RED against the process-wide token: the second thread's acquire returns
    at once as a no-op instead of waiting for the holder."""
    acquired, release = threading.Event(), threading.Event()
    holder = _hold_in_thread("omnibase-infra", acquired, release)
    assert acquired.wait(WAIT_SECONDS)
    try:
        with pytest.raises(LaneLockContendedError):
            with lane_lock("omnibase-infra", lane="dev", ref="r2", timeout=0.5):
                pass
    finally:
        release.set()
        holder.join(WAIT_SECONDS)


def test_the_second_thread_gets_the_lock_once_the_holder_releases() -> None:
    acquired, release = threading.Event(), threading.Event()
    holder = _hold_in_thread("omnibase-infra", acquired, release)
    assert acquired.wait(WAIT_SECONDS)
    got_it = threading.Event()

    def _waiter() -> None:
        with lane_lock("omnibase-infra", lane="dev", ref="r2", timeout=WAIT_SECONDS):
            got_it.set()

    waiter = threading.Thread(target=_waiter, daemon=True)
    waiter.start()
    assert not got_it.wait(0.5), "the waiter got the lock while it was held"
    release.set()
    holder.join(WAIT_SECONDS)
    waiter.join(WAIT_SECONDS)
    assert got_it.is_set()


def test_the_same_thread_nested_is_still_reentrant() -> None:
    """``git_pull`` takes the lane lock inside the job's own hold; that nested
    call on the SAME thread must stay a no-op rather than deadlock."""
    with lane_lock("omnibase-infra", lane="dev", ref="r", timeout=1.0):
        with lane_lock("omnibase-infra", lane="dev", ref="r", timeout=0.5):
            assert "omnibase-infra" in os.environ[ENV_LOCK_HELD].split()


def test_an_ancestor_holding_the_project_is_still_reentrant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A token INHERITED from a parent process (this process never acquired the
    project) keeps its meaning: the parent holds it, do not acquire again."""
    monkeypatch.setenv(ENV_LOCK_HELD, "omnibase-infra")
    # A holder in ANOTHER process is modelled by the inherited token alone. The
    # call must neither take the flock (no holder sidecar is written) nor touch
    # the inherited token.
    from deploy_agent.lane_lock_client import resolve_lane_lock_module

    holder_file = resolve_lane_lock_module().holder_path("omnibase-infra")
    with lane_lock("omnibase-infra", lane="dev", ref="r", timeout=0.5):
        assert not holder_file.exists()
    assert os.environ[ENV_LOCK_HELD] == "omnibase-infra"


def test_interleaved_releases_do_not_leak_a_project_into_the_token() -> None:
    """RED against save-and-restore: thread A takes P, thread B takes Q, A
    releases (restoring "no token"), B releases (restoring "P"). The token then
    names P while nobody holds it, and every later acquire of P in this process
    is a silent no-op."""
    a_acquired, a_release = threading.Event(), threading.Event()
    b_acquired, b_release = threading.Event(), threading.Event()
    a = _hold_in_thread("omnibase-infra", a_acquired, a_release)
    assert a_acquired.wait(WAIT_SECONDS)
    b = _hold_in_thread("onex-lab-k3s", b_acquired, b_release)
    assert b_acquired.wait(WAIT_SECONDS)
    a_release.set()
    a.join(WAIT_SECONDS)
    b_release.set()
    b.join(WAIT_SECONDS)

    assert os.environ.get(ENV_LOCK_HELD, "").split() == []
