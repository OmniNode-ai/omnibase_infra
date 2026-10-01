# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Coordinate image builds with reconcile-host's canonical clone writer.

OMN-20154: the onex wrapper self-heals a below-floor workspace by invoking
reconcile-host.sh, whose deploy_source_ref.py reconcile checks out every
canonical clone with git checkout --force -B dev <sha>.
That includes the deploy agent's omnibase_infra build context.
A checkout resets the tracked workspace/sibling-vcs-provenance.json
placeholder after stage_workspace.sh has populated it.
The dev lane's second build reuses that staging without writing it again,
so an intervening reconcile tick copies empty provenance and fails the
compute_workspace_provenance.py verification step (OMN-13030).
Taking the shell's mkdir lock from staging through the last image build
prevents reconciliation from rewriting the context during either build.
The holder and reclaim rules follow reconcile-host.sh's host-scoped protocol.
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import socket
import time
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path

RECONCILE_HOST_LOCK_DIRNAME = ".onex-reconcile-host.lock"

logger = logging.getLogger(__name__)


class ReconcileHostLockTimeoutError(RuntimeError):
    """A build could not acquire reconcile-host's lock within its wait budget."""


def _read_holder(lock_dir: Path) -> dict[str, str]:
    try:
        lines = (lock_dir / "holder").read_text().splitlines()
    except (OSError, UnicodeError):
        return {}
    fields: dict[str, str] = {}
    for line in lines:
        key, separator, value = line.partition("=")
        if separator:
            # The shell reader returns the first line matching key=.
            fields.setdefault(key, value)
    return fields


def _lock_reclaim_reason(
    lock_dir: Path, holder: dict[str, str], *, host: str, stale_seconds: int
) -> str | None:
    try:
        holder_path = lock_dir / "holder"
        target = holder_path if holder_path.is_file() else lock_dir
        age = int(time.time()) - int(target.stat().st_mtime)
    except OSError:
        # An unreadable age is not evidence of staleness, even for a dead pid.
        return None

    pid = holder.get("pid", "")
    holder_host = holder.get("host", "")
    if not pid or not holder_host:
        if age > stale_seconds:
            return (
                f"no readable holder record and the lock is {age}s old "
                f"(bound {stale_seconds}s)"
            )
        return None
    if holder_host != host:
        if age > stale_seconds:
            return (
                f"holder pid {pid} is on host {holder_host}, not this one, "
                f"and the lock is {age}s old (bound {stale_seconds}s)"
            )
        return None
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return (
            f"holder pid {pid} on this host is not running, and the lock is {age}s old"
        )
    except (OSError, ValueError, OverflowError):
        # PermissionError means alive; other errors cannot prove the pid dead.
        return None
    return None


@contextlib.contextmanager
def hold_reconcile_host_lock(
    omni_home: str,
    *,
    purpose: str,
    wait_seconds: float = 900.0,
    poll_seconds: float = 5.0,
    stale_seconds: int = 3600,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> Iterator[bool]:
    """Hold the shared mkdir lock, waiting for live peers and reclaiming dead ones."""
    if not omni_home.strip():
        logger.warning(
            "OMNI_HOME is blank; the build context is not coordinated with reconcile-host"
        )
        yield False
        return

    if not Path(omni_home).is_dir():
        # reconcile-host.sh itself refuses a missing OMNI_HOME (INDETERMINATE),
        # so no reconcile can be writing clones under it either.
        logger.warning(
            "OMNI_HOME %s is not a directory; the build context is not "
            "coordinated with reconcile-host",
            omni_home,
        )
        yield False
        return

    lock_dir = Path(omni_home, RECONCILE_HOST_LOCK_DIRNAME)
    pid = os.getpid()
    host = socket.gethostname()
    start = monotonic()
    while True:
        try:
            lock_dir.mkdir()
        except FileExistsError:
            holder = _read_holder(lock_dir)
            reason = _lock_reclaim_reason(
                lock_dir, holder, host=host, stale_seconds=stale_seconds
            )
            if reason is not None:
                logger.warning(
                    "Reclaiming reconcile-host lock %s: %s", lock_dir, reason
                )
                shutil.rmtree(lock_dir, ignore_errors=True)
                try:
                    lock_dir.mkdir()
                except FileExistsError:
                    # A peer won the reclaim race. Wait as for any other holder.
                    pass
                else:
                    break
            remaining = wait_seconds - (monotonic() - start)
            if remaining <= 0:
                holder = _read_holder(lock_dir)
                raise ReconcileHostLockTimeoutError(
                    f"reconcile_host_lock_timeout: waited {wait_seconds}s for "
                    f"{lock_dir}; holder pid={holder.get('pid', '<unrecorded>')} "
                    f"host={holder.get('host', '<unrecorded>')} "
                    f"started_at={holder.get('started_at', '<unrecorded>')}"
                )
            sleep(min(poll_seconds, remaining))
        else:
            break

    started_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        (lock_dir / "holder").write_text(
            f"pid={pid}\nhost={host}\nstarted_at={started_at}\n"
            f"holder=deploy-agent\npurpose={purpose}\n"
        )
    except OSError:
        # A lock with no holder record reads as a live peer for an hour; never
        # leave one behind.
        shutil.rmtree(lock_dir, ignore_errors=True)
        raise
    try:
        logger.info("Acquired reconcile-host lock %s for %s", lock_dir, purpose)
        yield True
    finally:
        holder = _read_holder(lock_dir)
        if holder.get("pid") == str(pid) and holder.get("host") == host:
            shutil.rmtree(lock_dir, ignore_errors=True)
            logger.info("Released reconcile-host lock %s for %s", lock_dir, purpose)
        else:
            logger.warning(
                "Reconcile-host lock %s no longer names our pid %s and host %s; "
                "leaving it alone",
                lock_dir,
                pid,
                host,
            )
