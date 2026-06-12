# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""disk_watermark_event.py — Build the typed disk-watermark bus event (OMN-13008).

Pure builder so the event schema is deterministic and unit-testable. Reads the
measured values from env (the shell measures via df) and emits one JSON object
on stdout for `rpk topic produce` to publish to onex.evt.infra.disk-watermark.v1.

The event is the alert authority: a downstream consumer (node_runtime_sweep
auto-ticket path) creates the Linear ticket on `severity=warning`, and operators
get the loud signal on `severity=critical`. This keeps a single ticket-creation
authority rather than letting every cron script talk to Linear directly.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime, timezone


def build_event(
    *,
    mount: str,
    used_pct: int,
    avail_kb: int,
    severity: str,
    warn_pct: int,
    crit_pct: int,
    host: str,
    topic: str,
    now: datetime | None = None,
) -> dict[str, object]:
    """Construct the typed disk-watermark event payload."""
    now = now or datetime.now(UTC)
    if severity not in {"warning", "critical"}:
        raise ValueError(f"severity must be 'warning' or 'critical', got {severity!r}")
    return {
        "schema_version": "1.0.0",
        "event_type": "disk-watermark",
        "topic": topic,
        "host": host,
        "mount": mount,
        "used_pct": used_pct,
        "avail_kb": avail_kb,
        "warn_pct": warn_pct,
        "crit_pct": crit_pct,
        "severity": severity,
        "emitted_at": now.isoformat(),
        # Stable dedupe key so the consumer can collapse repeated alerts for the
        # same host/mount/severity into one open ticket instead of N.
        "alert_key": f"disk-watermark:{host}:{mount}:{severity}",
        "message": (
            f"{host}:{mount} at {used_pct}% used (warn>={warn_pct}%, crit>={crit_pct}%) "
            f"severity={severity}"
        ),
    }


def main() -> int:
    event = build_event(
        mount=os.environ["MOUNT"],
        used_pct=int(os.environ["USED_PCT"]),
        avail_kb=int(os.environ["AVAIL_KB"]),
        severity=os.environ["SEVERITY"],
        warn_pct=int(os.environ["WARN_PCT"]),
        crit_pct=int(os.environ["CRIT_PCT"]),
        host=os.environ["HOSTNAME_TAG"],
        topic=os.environ["TOPIC"],
    )
    json.dump(event, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
