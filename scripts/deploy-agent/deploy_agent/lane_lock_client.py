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
    itself.

    Raises :class:`LaneLockContendedError` when the bounded wait expires. The
    holder is named from the sidecar the helper maintains, so contention says
    WHO rather than hanging anonymously.
    """
    if compose_project in _held_projects():
        logger.info(
            "lane lock for %s already held by an outer process in this "
            "ancestry; not re-acquiring",
            compose_project,
        )
        yield
        return

    module = resolve_lane_lock_module()
    path = Path(lane_lock_path(compose_project))
    path.parent.mkdir(parents=True, exist_ok=True)

    handle = path.open("a+")
    acquired = False
    previous_token = os.environ.get(ENV_LOCK_HELD)
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
        os.environ[ENV_LOCK_HELD] = " ".join([*_held_projects(), compose_project])
        logger.info(
            "lane lock acquired for %s (lane=%s ref=%s)", compose_project, lane, ref
        )
        yield
    finally:
        if acquired:
            if previous_token is None:
                os.environ.pop(ENV_LOCK_HELD, None)
            else:
                os.environ[ENV_LOCK_HELD] = previous_token
            try:
                module.holder_path(compose_project).unlink()
            except OSError:
                pass
        handle.close()
