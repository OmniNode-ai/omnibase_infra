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
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from omnibase_infra.nodes.node_dev_roll_events_effect.runtime_dev_roll_events import (
    EnumDevStartCause,
    build_events_at,
    newest_dev_recreate,
    recreate_gate_at,
)

if TYPE_CHECKING:
    from omnibase_infra.protocols.protocol_dev_roll_events import (
        ProtocolDevRollEvents,
    )

logger = logging.getLogger(__name__)

FRICTION = "friction_type=dev_event_publish_failed"
#: The requester the recreate window sees for a reapply's restart (OMN-25912).
REAPPLY_REQUESTER = "agent/branch-install-reapply"
_ANNOUNCED_MARKER = "branch-installs/recreate-announced-for"
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


def window_holds_apply(
    state_dir: Path,
    *,
    repo_root: Path,
    build: Callable[..., Any] = recreate_gate_at,
) -> bool:
    """Whether the recreate window holds this reapply: its restart waits for the window.

    A held reapply announces the next recreate once per window (the marker file keeps
    the announced time across the agent's one-shot runs); a permitted one announces first.
    """
    marker = state_dir / _ANNOUNCED_MARKER
    try:
        announced_for = datetime.fromisoformat(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        announced_for = None
    gate = build(repo_root, newest_dev_recreate(state_dir), announced_for)
    permitted = bool(gate.allow(REAPPLY_REQUESTER, None))
    after = getattr(gate, "announced_for", None)
    if after is not None and after != announced_for:
        try:
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(after.isoformat(), encoding="utf-8")
        except OSError:
            logger.exception("recreate announcement marker %s", FRICTION)
    return not permitted


def main(
    argv: Sequence[str] | None = None,
    *,
    handler_main: Callable[[Sequence[str] | None], int] | None = None,
    announce: Callable[..., bool] = announce_reapply_start,
    window: Callable[..., bool] = window_holds_apply,
) -> int:
    """Run the branch-install handler, then announce its restarts."""
    if handler_main is None:
        from omnibase_infra.handlers import handler_dev_lane_branch_install

        handler_main = handler_dev_lane_branch_install.main
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--state-dir", default=None)
    parsed, _ = parser.parse_known_args(argv)
    mode = (list(argv) if argv is not None else sys.argv[1:])[:1]
    if (
        mode == ["apply"]
        and parsed.state_dir is not None
        and window(
            Path(parsed.state_dir), repo_root=Path(__file__).resolve().parents[4]
        )
    ):
        sys.stdout.write(
            "RECREATE-WINDOW-HOLD the reapply restart waits for the window\n"
        )
        return 0
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
