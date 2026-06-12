# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""lane_census_event.py — Build the typed lane-census-drift bus event (OMN-13011).

Pure builder so the drift-event schema is deterministic and unit-testable. Takes
the drift plan emitted by lane_census_plan.py and produces one JSON object on
stdout for `rpk topic produce` to publish to onex.evt.infra.lane-census-drift.v1.

The event is the alert authority: a downstream consumer (the runtime_sweep /
sweep auto-ticket path) creates the Linear ticket naming exactly what is missing
or extra. This keeps a single ticket-creation authority rather than letting every
cron script talk to Linear directly — the same pattern as the disk-watermark
event (OMN-13008).

The `alert_key` deduplicates: one open ticket per (host, lane, kind-set) so a
persistent outage does not spam a new ticket every reconcile tick.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from typing import Any

DRIFT_TOPIC = "onex.evt.infra.lane-census-drift.v1"

# A drift finding is a flat string mapping (lane, kind, container, detail, severity)
# as emitted by lane_census_plan.py. The plan envelope is loosely typed (str/Any)
# because it round-trips through JSON; the helpers below narrow it locally.
Finding = dict[str, str]
Plan = dict[str, Any]


def _findings(plan: Plan) -> list[Finding]:
    """Narrow the loosely-typed plan envelope to its findings list."""
    raw = plan.get("findings", [])
    return list(raw)


def _alert_key(host: str, plan: Plan) -> str:
    """Stable dedupe key over the (host, lane, kind, container) finding set."""
    signature = sorted(
        f"{f['lane']}:{f['kind']}:{f['container']}" for f in _findings(plan)
    )
    digest = hashlib.sha256("|".join(signature).encode("utf-8")).hexdigest()[:16]
    return f"lane-census-drift:{host}:{digest}"


def _ticket_title(plan: Plan) -> str:
    """One-line title naming exactly what is wrong."""
    findings = _findings(plan)
    if not findings:
        return "lane census: no drift"
    lanes = sorted({f["lane"] for f in findings})
    # Summarize by kind for a precise, greppable title.
    kinds: dict[str, int] = {}
    for f in findings:
        kinds[f["kind"]] = kinds.get(f["kind"], 0) + 1
    kind_summary = ", ".join(
        f"{count}x {kind}" for kind, count in sorted(kinds.items())
    )
    return f"fix(infra): lane drift [{', '.join(lanes)}] — {kind_summary}"


def _ticket_body(host: str, plan: Plan) -> str:
    findings = _findings(plan)
    lanes_checked: list[str] = list(plan.get("lanes_checked", []))
    lines = [
        "## Lane census drift detected",
        "",
        f"Host: `{host}`",
        f"Lanes checked: {', '.join(lanes_checked)}",
        "",
        "The desired-state lane census (deploy/lane-census/lane-manifest.yaml)",
        "does not match the live runtime. Drift items below name exactly what is",
        "missing or extra. This is the regression class the lane-census ratchet",
        "(OMN-13011) exists to catch — it would have fired on 2026-06-11 when prod",
        "runtime containers and the broker network were silently absent.",
        "",
        "## Findings",
        "",
    ]
    for f in findings:
        lines.append(
            f"- **[{f['severity']}] {f['kind']}** `{f['container']}` ({f['lane']}): {f['detail']}"
        )
    lines.append("")
    lines.append("## Action")
    lines.append(
        "Bring the lane back to declared state (compose up the absent containers / "
        "reattach the network), or update the lane manifest in the same PR if the "
        "desired state legitimately changed. Do not silence the check."
    )
    return "\n".join(lines)


def build_event(
    *,
    host: str,
    plan: Plan,
    topic: str = DRIFT_TOPIC,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Construct the typed lane-census-drift event payload."""
    now = now or datetime.now(UTC)
    findings = _findings(plan)
    severity = (
        "critical" if any(f["severity"] == "critical" for f in findings) else "warning"
    )
    return {
        "schema_version": "1.0.0",
        "event_type": "lane-census-drift",
        "topic": topic,
        "host": host,
        "emitted_at": now.isoformat(),
        "severity": severity,
        "lanes_checked": plan.get("lanes_checked", []),
        "drift_count": len(findings),
        "findings": findings,
        "alert_key": _alert_key(host, plan),
        "ticket_title": _ticket_title(plan),
        "ticket_body": _ticket_body(host, plan),
    }


def main() -> int:

    host = os.environ.get("LANE_CENSUS_HOST", "")
    if not host:
        # Fail-fast: never silently fabricate a host into the alert_key.
        print("ERROR: LANE_CENSUS_HOST must be set", file=sys.stderr)
        return 2
    plan = json.load(sys.stdin)
    event = build_event(host=host, plan=plan)
    json.dump(event, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
