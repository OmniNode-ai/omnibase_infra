# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The deploy agent's half of the per-lane host lock (OMN-18124).

WHAT WAS UNPROTECTED, MEASURED
------------------------------

``Executor.git_pull`` runs ``git fetch --all --prune`` and then
``git reset --hard <ref>`` against ``REPO_DIR`` -- the SHARED deploy-source
clone -- under no lock at all.

That clone is also owned by ``scripts/runtime_build/refresh_dev_lane.sh``, which
since OMN-16729 holds a per-compose-project host lock across its whole build,
gate and readback critical section. That lock exists because two sanctioned
dev-lane refreshes collided on 2026-09-08. The agent is a third writer to the
same tree and participated in none of it.

The collision is in the clone's own reflog rather than in theory: the refresh
checked out ``origin/dev`` at 21:26:37 local and an agent job pulled it back to
``origin/main`` 166 seconds later; the pattern repeated at 23:06:50 and
23:31:49. For those windows a refresh that had already captured its pre-state
was building a tree the agent was concurrently rewriting. ``reset --hard`` also
discards uncommitted state in the shared tree with no warning.

WHY THE EXISTING LOCK, RESOLVED FROM THE EXISTING MODULE
--------------------------------------------------------

A lock only excludes the writers that take the SAME lock. A second lock file
here would serialise the agent against itself and nothing else, which is the
precise shape of non-protection that looks like protection in a review.

So this module does not rebuild the path rule. It loads
``scripts/runtime_build/lane_lock.py`` and calls its ``lock_path`` -- one owner
of the rule, one file per compose project, and a test asserts this module's
answer equals what that helper's own CLI prints for the same project.

The module is loaded from the AGENT'S OWN code clone, beside the agent package,
not from ``REPO_DIR``. ``REPO_DIR`` is the tree being mutated: reading the lock
implementation out of the thing you are about to ``reset --hard`` would mean the
rule changed underneath the lock that was protecting it.

RE-ENTRANCY, AND WHY IT IS AN ENVIRONMENT TOKEN
-----------------------------------------------

``ONEX_LANE_LOCK_HELD`` is the shell front end's existing convention: the
acquiring process appends the compose project and children inherit it. A nested
call that sees its own project already listed does not try to acquire again, so
an agent-initiated deploy that shells out to a locking script cannot deadlock
against its own parent. This module reads and writes the same token, in the same
shape, so the two directions interoperate -- an agent holding the lock is
visible to a script it launches, and a script holding it is visible here.

RE-ENTRANCY INSIDE THIS PROCESS IS PER THREAD (OMN-19501)
----------------------------------------------------------

The token is in ``os.environ``, which one process shares across all its
threads. While the agent did all of its lane work on one thread, "this process
holds the project" and "my caller holds it" were the same fact. OMN-19501 adds a
second thread: the settle worker re-acquires the dev lane lock for the onex-api
pin recreate while the job thread may be running the next compose job. Read as
a process-wide flag, the token made that re-acquire a silent no-op, and the pin
recreated a container in the middle of another job's compose run. The extended
TLA+ model on OMN-19501 reports exactly that (``MC_a1_env_token_reentrancy``).

So a project this PROCESS acquired is tracked in ``_OWNERS`` against the thread
holding it. The same thread asking again is re-entrant; another thread takes
the flock like any other writer and waits for it (``flock`` on a second open
file description conflicts inside one process). The token keeps its meaning
for CHILD processes, and a token this process did not write (inherited from a
parent) still means "an ancestor holds it". The token is edited by adding and
removing this project, never by restoring a saved value: with two threads
interleaving, a restore puts a project released by the other thread back into
the token, and every later acquire of it becomes a no-op.

WHAT THIS DOES NOT DO
---------------------

It never steals a lock. Contention is a bounded wait that ends in a refusal
naming the holder, exactly as ``lane_lock.sh`` documents. A lock is released by
closing the descriptor -- so it is released however this process exits, the
kernel included -- and there is no code path here that releases another
process's lock. The residual ``lane_lock.py`` records still applies: a surviving
child of a dead holder keeps the lane locked, which is diagnosable through the
holder sidecar rather than silent.
"""

from __future__ import annotations

import fcntl
import importlib.util
import logging
import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from types import ModuleType

logger = logging.getLogger(__name__)

ENV_LOCK_HELD = "ONEX_LANE_LOCK_HELD"

# Matches lane_lock.sh's own default. A bounded wait, never an indefinite one:
# an anonymous hang is what the holder sidecar exists to prevent.
DEFAULT_LANE_LOCK_TIMEOUT_SECONDS = 900.0

_POLL_INTERVAL_SECONDS = 0.25


class LaneLockContendedError(RuntimeError):
    """Raised when the bounded wait for a lane lock expired.

    Its own class so a caller can tell "another writer owns this lane's tree"
    from a build failure. Those are different facts and lead to different
    actions: one is retried later, the other is fixed.
    """


def _lane_lock_module_path() -> Path:
    """``scripts/runtime_build/lane_lock.py`` in the agent's own code clone."""
    return Path(__file__).resolve().parents[2] / "runtime_build" / "lane_lock.py"


@lru_cache(maxsize=1)
def resolve_lane_lock_module() -> ModuleType:
    """Load the repo's lane-lock helper as a module.

    Loaded by path rather than imported by name because ``scripts/`` is not a
    package and is not installed into the agent's environment; the helper is
    stdlib-only, so it loads under the interpreter already running.
    """
    path = _lane_lock_module_path()
    if not path.is_file():
        raise RuntimeError(
            f"lane lock helper not found at {path}. The deploy agent must take "
            "the SAME per-compose-project lock as "
            "scripts/runtime_build/refresh_dev_lane.sh, and a lock that is not "
            "that file excludes nobody. This path is resolved relative to the "
            "agent's own code clone, so a missing helper means that clone is "
            "incomplete."
        )

    spec = importlib.util.spec_from_file_location("onex_lane_lock", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load the lane lock helper at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def lane_lock_path(compose_project: str) -> str:
    """The lock file for one compose project, as the shell helper computes it."""
    module = resolve_lane_lock_module()
    return str(module.lock_path(compose_project))


def _held_projects() -> list[str]:
    return os.environ.get(ENV_LOCK_HELD, "").split()


#: OMN-19501. Projects THIS process holds the flock for, mapped to the thread
#: that holds it. Guarded by ``_OWNERS_GUARD``, which also serialises every edit
#: of the ``ONEX_LANE_LOCK_HELD`` token.
_OWNERS: dict[str, int] = {}
_OWNERS_GUARD = threading.Lock()


def _reentrant_hold(compose_project: str) -> str | None:
    """Why acquiring ``compose_project`` here would be a no-op, or ``None``.

    Re-entrant when THIS thread already holds it, or when an ancestor process
    does (the project is in the inherited token and this process never took
    it). Another thread of this process holding it is NOT re-entrancy: that
    caller must take the flock and wait.
    """
    with _OWNERS_GUARD:
        owner = _OWNERS.get(compose_project)
        if owner is not None:
            return "this thread" if owner == threading.get_ident() else None
        if compose_project in _held_projects():
            return "an outer process in this ancestry"
    return None


def _record_hold(compose_project: str) -> None:
    with _OWNERS_GUARD:
        _OWNERS[compose_project] = threading.get_ident()
        held = _held_projects()
        if compose_project not in held:
            os.environ[ENV_LOCK_HELD] = " ".join([*held, compose_project])


def _clear_hold(compose_project: str) -> None:
    with _OWNERS_GUARD:
        _OWNERS.pop(compose_project, None)
        remaining = [p for p in _held_projects() if p != compose_project]
        if remaining:
            os.environ[ENV_LOCK_HELD] = " ".join(remaining)
        else:
            os.environ.pop(ENV_LOCK_HELD, None)


@contextmanager
def lane_lock(
    compose_project: str,
    *,
    lane: str,
    ref: str,
    timeout: float = DEFAULT_LANE_LOCK_TIMEOUT_SECONDS,
) -> Iterator[None]:
    """Hold the per-compose-project host lock for the duration of the block.

    A no-op when an OUTER process in this ancestry already holds the project,
    which is the same re-entrancy rule ``lane_lock.sh`` applies -- and is what
    keeps a deploy that shells out to a locking script from deadlocking against
    itself. Also a no-op when THIS thread already holds it. Another thread of
    this process holding it is contention, not re-entrancy (OMN-19501).

    Raises :class:`LaneLockContendedError` when the bounded wait expires. The
    holder is named from the sidecar the helper maintains, so contention says
    WHO rather than hanging anonymously.
    """
    holder = _reentrant_hold(compose_project)
    if holder is not None:
        logger.info(
            "lane lock for %s already held by %s; not re-acquiring",
            compose_project,
            holder,
        )
        yield
        return

    module = resolve_lane_lock_module()
    path = Path(lane_lock_path(compose_project))
    path.parent.mkdir(parents=True, exist_ok=True)

    handle = path.open("a+")
    acquired = False
    try:
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise LaneLockContendedError(
                        f"lane lock for compose project {compose_project!r} is "
                        f"held by another writer after waiting {timeout:g}s. "
                        f"Holder: {module.describe_holder(compose_project)}. "
                        "The deploy was not started and the deploy-source "
                        "clone was not touched. This lock is never stolen -- "
                        "it is released when the holder's descriptor closes."
                    ) from None
                time.sleep(_POLL_INTERVAL_SECONDS)

        module.write_holder(compose_project, lane, ref, "deploy-agent")
        _record_hold(compose_project)
        logger.info(
            "lane lock acquired for %s (lane=%s ref=%s)", compose_project, lane, ref
        )
        yield
    finally:
        if acquired:
            _clear_hold(compose_project)
            try:
                module.holder_path(compose_project).unlink()
            except OSError:
                pass
        handle.close()
