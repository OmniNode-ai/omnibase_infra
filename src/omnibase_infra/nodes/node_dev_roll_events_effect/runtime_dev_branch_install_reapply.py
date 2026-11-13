# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The deploy agent's branch-install reapply entry (OMN-24013).

Runs the branch-install handler as a subprocess and, when the handler restarted
containers (a runtime start), publishes the dev runtime start event for the
reapply. The topics come from the node's contract through build_events_at; nothing
here holds a topic or a bound. A failed publication is logged and never fails or
delays the deploy.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from omnibase_infra.nodes.node_dev_roll_events_effect.runtime_dev_roll_events import (
    EnumDevStartCause,
    build_events_at,
)

if TYPE_CHECKING:
    from omnibase_infra.protocols.protocol_dev_roll_events import (
        ProtocolDevRollEvents,
    )

logger = logging.getLogger(__name__)

FRICTION = "friction_type=dev_event_publish_failed"
_ACTIVE_STATUSES = {"accepted", "in_progress"}


def active_job_id(state_dir: Path) -> str | None:
    """The correlation_id of the deploy agent job in flight, or None.

    The newest ``<state_dir>/<correlation_id>.json`` whose status is accepted or
    in_progress; unreadable or non-job files are skipped.
    """
    try:
        candidates = list(state_dir.glob("*.json"))
    except OSError:
        return None
    best: tuple[float, str] | None = None
    for path in candidates:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict) or data.get("status") not in _ACTIVE_STATUSES:
            continue
        correlation_id = data.get("correlation_id")
        if not isinstance(correlation_id, str) or not correlation_id:
            continue
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if best is None or mtime > best[0]:
            best = (mtime, correlation_id)
    return best[1] if best is not None else None


def restart_count(stdout: str) -> int:
    """The number of containers the handler restarted, from its stdout."""
    total = 0
    for line in stdout.splitlines():
        for prefix in ("RECEIPT ", "READBACK "):
            if line.startswith(prefix):
                try:
                    payload = json.loads(line[len(prefix) :])
                except ValueError:
                    continue
                if isinstance(payload, dict):
                    restarted = payload.get("restarted")
                    if isinstance(restarted, list):
                        total += len(restarted)
                break
    return total


def announce_reapply_start(
    state_dir: Path,
    stdout: str,
    *,
    repo_root: Path,
    build: Callable[[Path], ProtocolDevRollEvents] = build_events_at,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> bool:
    """Publish the reapply start event when the handler restarted containers."""
    if restart_count(stdout) <= 0:
        return False
    try:
        publisher = build(repo_root)
        job_id = active_job_id(state_dir) or f"branch-install-reapply/{uuid4()}"
        publisher.started(job_id, now(), EnumDevStartCause.BRANCH_INSTALL_REAPPLY)
    except Exception:
        logger.exception("branch-install reapply start event %s", FRICTION)
        return False
    return True


def main(
    argv: Sequence[str] | None = None,
    *,
    handler_main: Callable[[Sequence[str] | None], int] | None = None,
    announce: Callable[..., bool] = announce_reapply_start,
) -> int:
    """Run the branch-install handler, then announce its restarts."""
    if handler_main is None:
        from omnibase_infra.handlers import handler_dev_lane_branch_install

        handler_main = handler_dev_lane_branch_install.main
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--state-dir", default=None)
    parsed, _ = parser.parse_known_args(argv)
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        exit_code = handler_main(argv)
    sys.stdout.write(captured.getvalue())
    if parsed.state_dir is not None:
        announce(
            Path(parsed.state_dir),
            captured.getvalue(),
            repo_root=Path(__file__).resolve().parents[4],
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
