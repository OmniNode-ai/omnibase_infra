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

SCHEMA 2.0.0 (OMN-17872). The halt criterion moved from percent-used to absolute
free space, so the payload carries both readings against both of their declared
lines: `avail_gb` against `crit_free_gb`/`warn_free_gb`, and `used_pct` against
the now-advisory `warn_pct`. `halt_reason` names which of the three fired, so a
receipt that cites this guard says why it fired rather than only that it did.
The 1.0.0 field `crit_pct` is gone — there is no critical percentage any more.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime

_HALT_REASONS = frozenset(
    {
        "free_space_below_crit_floor",
        "free_space_below_warn_floor",
        "used_pct_advisory",
    }
)


def build_event(
    *,
    mount: str,
    used_pct: int,
    avail_kb: int,
    avail_gb: int,
    severity: str,
    halt_reason: str,
    warn_pct: int,
    warn_free_gb: int,
    crit_free_gb: int,
    host: str,
    topic: str,
    now: datetime | None = None,
) -> dict[str, object]:
    """Construct the typed disk-watermark event payload."""
    now = now or datetime.now(UTC)
    if severity not in {"warning", "critical"}:
        raise ValueError(f"severity must be 'warning' or 'critical', got {severity!r}")
    if halt_reason not in _HALT_REASONS:
        raise ValueError(
            f"halt_reason must be one of {sorted(_HALT_REASONS)}, got {halt_reason!r}"
        )
    if severity == "critical" and halt_reason != "free_space_below_crit_floor":
        raise ValueError(
            "only free space below the critical floor may be critical; "
            f"got severity=critical with halt_reason={halt_reason!r}"
        )
    return {
        "schema_version": "2.0.0",
        "event_type": "disk-watermark",
        "topic": topic,
        "host": host,
        "mount": mount,
        "avail_gb": avail_gb,
        "avail_kb": avail_kb,
        "crit_free_gb": crit_free_gb,
        "warn_free_gb": warn_free_gb,
        "used_pct": used_pct,
        "warn_pct": warn_pct,
        "severity": severity,
        "halt_reason": halt_reason,
        "emitted_at": now.isoformat(),
        # Stable dedupe key so the consumer can collapse repeated alerts for the
        # same host/mount/severity into one open ticket instead of N.
        "alert_key": f"disk-watermark:{host}:{mount}:{severity}",
        # Both numbers, both lines — a receipt quoting this message is complete.
        "message": (
            f"{host}:{mount} {avail_gb} GiB free "
            f"(crit floor {crit_free_gb} GiB, warn floor {warn_free_gb} GiB); "
            f"{used_pct}% used (advisory warn>={warn_pct}%) "
            f"severity={severity} reason={halt_reason}"
        ),
    }


def main() -> int:
    event = build_event(
        mount=os.environ["MOUNT"],
        used_pct=int(os.environ["USED_PCT"]),
        avail_kb=int(os.environ["AVAIL_KB"]),
        avail_gb=int(os.environ["AVAIL_GB"]),
        severity=os.environ["SEVERITY"],
        halt_reason=os.environ["HALT_REASON"],
        warn_pct=int(os.environ["WARN_PCT"]),
        warn_free_gb=int(os.environ["WARN_FREE_GB"]),
        crit_free_gb=int(os.environ["CRIT_FREE_GB"]),
        host=os.environ["HOSTNAME_TAG"],
        topic=os.environ["TOPIC"],
    )
    json.dump(event, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
