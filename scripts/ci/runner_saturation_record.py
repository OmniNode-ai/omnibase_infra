#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runner saturation monitor (OMN-18031 G7).

THE QUESTION THIS ANSWERS: "how do we know if the lab runners start getting
slammed?" Once per-run routing is live the dangerous mode is SILENT -- the route
job correctly falls back to hosted on every run, every check is green, and the
only symptom is a GitHub Actions bill. No existing surface reports it:

  * ``runner-fleet-canary.yml`` watches REGISTRATIONS and listener liveness, and
    OMN-16030 deliberately stopped it failing on the offline count because that
    field goes stale under load. It is not a saturation monitor.
  * ``runner-routing-audit.yml`` is an hourly DRIFT gate whose red means "a
    variable changed". Hourly is too coarse for saturation, and OMN-16727's
    step-level masking would be inherited.

So this rides the EXISTING fleet canary scheduler (moved */15 -> */5) rather
than adding a third scheduler, which the consent row puts out of scope.

SUSTAINED STATE NEEDS NO NEW DATASTORE. The monitor reads its own previous N
record artifacts and counts CONSECUTIVE breaches from the newest sample
backwards. ``sustained_samples: 4`` at */5 is a 20-minute sustain, which is what
keeps the ALERT path from flapping on a signal measured swinging 65 -> 10 busy
of 88 inside two minutes.

FOUR CONDITIONS, and the third is the one the operator actually asked for:
  1. ``fleet_busy_sustained``            -- the fleet is genuinely hammered
  2. ``lab_load_sustained``              -- a lab host is over load or out of memory
  3. ``route_fallback_sustained``        -- routing keeps CHOOSING hosted because
     the lab was saturated. Green runs, quiet logs, a rising bill.
  4. ``lab_probe_unavailable_sustained`` -- the self-hosted lab-load probe could
     not start. The one reason a 3-minute self-hosted job cannot start is that
     the fleet has no free runner, so the MISSING probe is itself the signal.

CONDITION 3 EXCLUDES THE INERT REASONS ON PURPOSE. Until the seam is flipped
every run decides ``seam_ceiling_hosted``; counting that as a saturation
fallback would alert continuously from the moment this lands and get itself
muted long before the seam ever flips. ``fork_isolation`` and
``policy_allowlist`` are correct permanent hosted decisions on the same terms.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

DEFAULT_POLICY = Path("config/runner_routing_policy.yaml")

REQUIRED_ALERT_KEYS: tuple[str, ...] = (
    "busy_fraction_threshold",
    "lab_load_ratio_threshold",
    "sustained_samples",
    "sustained_min_span_seconds",
)

# The route reasons that ARE positive evidence the lab is under pressure.
# Condition 3 counts these and nothing else.
#
# THIS IS AN ALLOWLIST, AND THAT IS THE FIX FOR A MEASURED DEFECT. It shipped
# first as a denylist -- "any hosted reason except seam_ceiling_hosted,
# fork_isolation and policy_allowlist is saturation". A denylist FAILS OPEN:
# every reason nobody thought of silently becomes a saturation alert, and two
# already did.
#
#   `lab_unknown` is the reason on EVERY run until the monitor has produced its
#   first lab record for the route job to read -- i.e. from the moment this
#   lands. Measured 2026-09-07T12:22Z against a fully healthy fleet (88 online,
#   4 busy) and a healthy lab host (0.669x load, 66013 MiB free), the denylist
#   form raised `route_fallback_sustained`. That is precisely the "fires on
#   every sample from the day it lands and is muted long before it matters"
#   failure this module's own docstring claims to have avoided by excluding
#   `seam_ceiling_hosted` -- reintroduced one reason over.
#
#   `probe_error:*` is a probe FAULT, not lab pressure (`missing_token` is
#   permanent on any run without the fleet-status secret), and reporting it
#   under a saturation headline points the reader at the wrong system.
#
# An allowlist fails closed: a new reason raises no alert until someone decides
# it is one. It was also DUPLICATED into runner_route_decision.py, which has no
# consumer for it; that copy is deleted rather than kept in sync, because two
# definitions of one policy is the drift this repo's gates exist to stop.
SATURATION_FALLBACK_REASONS: frozenset[str] = frozenset(
    {"fleet_saturated", "fleet_degraded", "lab_saturated"}
)


@dataclass(frozen=True)
class Alert:
    condition: str
    detail: str


def select_latest_route_artifact(
    artifacts: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Pick the newest non-expired ``runner-route-decision-*`` artifact from a
    repo-wide ``GET /actions/artifacts`` listing.

    OMN-18031 G7 gap: ``ROUTE_REASON`` was hardcoded to ``"unknown"`` in
    ``dev-lane-liveness.yml`` because nothing aggregated the route job's own
    decision artifacts into this workflow, so condition 3
    (``route_fallback_sustained``, the alert the operator actually asked for)
    had no input and could never fire. This is the pure selection half; the
    network read (``gh api .../actions/artifacts``) is a workflow step, kept
    out of this function so the selection logic is testable without a live
    API call.

    Returns ``None`` when no matching, unexpired artifact exists -- the
    caller's fallback to ``"unknown"`` is then the same safe default as
    before this landed, not a new failure mode.
    """
    candidates = [
        item
        for item in artifacts
        if isinstance(item, dict)
        and isinstance(item.get("name"), str)
        and item["name"].startswith("runner-route-decision-")
        and not item.get("expired", False)
        and isinstance(item.get("created_at"), str)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda item: str(item["created_at"]))


def extract_route_reason(payload: Any) -> str | None:
    """Pull ``reason`` out of a downloaded ``runner-route-decision.json``.

    Returns ``None`` on any shape other than a decision record carrying a
    non-empty string reason -- malformed or unexpected content must fall back
    to the caller's ``"unknown"`` default, never be laundered into an empty
    string that then fails to match anything in ``SATURATION_FALLBACK_REASONS``
    while LOOKING like a real value in the record.
    """
    if not isinstance(payload, dict):
        return None
    reason = payload.get("reason")
    return reason if isinstance(reason, str) and reason else None


def load_alert_policy(path: Path) -> dict[str, Any]:
    """Load ``route.saturation_alert``, failing on any missing threshold."""
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    route = loaded.get("route")
    if not isinstance(route, dict):
        raise KeyError(f"{path} is missing the required 'route:' section")
    section = route.get("saturation_alert")
    if not isinstance(section, dict):
        raise KeyError(
            f"{path} is missing the required 'route.saturation_alert:' section"
        )
    for key in REQUIRED_ALERT_KEYS:
        if key not in section:
            raise KeyError(f"{path} saturation_alert is missing required key {key!r}")
    return section


def build_record(
    *,
    fleet: dict[str, Any],
    lab: dict[str, Any],
    route_reason: str,
    sampled_at: str | None = None,
) -> dict[str, Any]:
    """Assemble one ``runner_saturation_record/v1`` sample.

    An absent lab probe is recorded as ``unavailable`` -- NEVER as a healthy
    zero. The probe job is self-hosted and shares fate with the fleet, so its
    absence carries information and must not be laundered into "0.0 load".
    """
    online = fleet.get("online") if isinstance(fleet, dict) else None
    busy = fleet.get("busy") if isinstance(fleet, dict) else None
    fleet_ok = isinstance(fleet, dict) and fleet.get("ok") and isinstance(online, int)
    busy_fraction = (
        (busy / online) if (fleet_ok and isinstance(busy, int) and online) else None
    )

    lab_ok = isinstance(lab, dict) and bool(lab.get("ok"))
    hosts = lab.get("hosts") if lab_ok else None

    return {
        "schema": "runner_saturation_record/v1",
        "sampled_at": sampled_at or datetime.now(UTC).isoformat(),
        "fleet": {
            "online": online if fleet_ok else None,
            "busy": busy if fleet_ok else None,
            "busy_fraction": busy_fraction,
        },
        "lab": {
            "probe": "ok" if lab_ok else "unavailable",
            "hosts": hosts if isinstance(hosts, list) else [],
        },
        "route": {"recent_reason": route_reason},
    }


def _span_seconds(window: list[dict[str, Any]], count: int) -> float:
    """Wall-clock span covered by the newest ``count`` samples.

    Returns 0.0 when the timestamps cannot be read, which fails CLOSED (no
    alert): an unreadable clock is not evidence a breach was sustained.
    """
    if count < 2:
        return 0.0
    stamps: list[datetime] = []
    for sample in window[:count]:
        raw = sample.get("sampled_at") if isinstance(sample, dict) else None
        if not isinstance(raw, str):
            return 0.0
        try:
            stamps.append(datetime.fromisoformat(raw.replace("Z", "+00:00")))
        except ValueError:
            return 0.0
    return abs((max(stamps) - min(stamps)).total_seconds())


def _consecutive(window: list[dict[str, Any]], predicate: Any) -> int:
    """Count consecutive samples satisfying ``predicate`` from the newest back.

    A malformed sample is skipped rather than crashing the monitor -- a watcher
    that dies on one bad artifact stops watching entirely -- but it also BREAKS
    the run, because an unreadable sample is not proof the breach continued.
    """
    count = 0
    for sample in window:
        if not isinstance(sample, dict):
            break
        try:
            if predicate(sample):
                count += 1
                continue
        except (TypeError, ValueError, KeyError, AttributeError):
            break
        break
    return count


def evaluate_alerts(
    window: list[dict[str, Any]], policy: dict[str, Any]
) -> list[Alert]:
    """Return the alerts a window of samples justifies. Newest sample first."""
    sustained = int(policy["sustained_samples"])
    min_span = float(policy["sustained_min_span_seconds"])
    busy_threshold = float(policy["busy_fraction_threshold"])
    ratio_threshold = float(policy["lab_load_ratio_threshold"])
    mem_floor = int(policy.get("min_lab_free_mem_mib", 4096))
    alerts: list[Alert] = []

    if not window:
        return alerts

    def busy_breach(sample: dict[str, Any]) -> bool:
        fraction = (sample.get("fleet") or {}).get("busy_fraction")
        return isinstance(fraction, (int, float)) and float(fraction) >= busy_threshold

    def lab_breach(sample: dict[str, Any]) -> bool:
        lab = sample.get("lab") or {}
        if lab.get("probe") != "ok":
            return False
        for host in lab.get("hosts") or []:
            if not isinstance(host, dict):
                continue
            ratio = host.get("ratio")
            free_mem = host.get("free_mem_mib")
            if isinstance(ratio, (int, float)) and float(ratio) >= ratio_threshold:
                return True
            if isinstance(free_mem, int) and free_mem < mem_floor:
                return True
        return False

    def route_breach(sample: dict[str, Any]) -> bool:
        reason = (sample.get("route") or {}).get("recent_reason")
        if not isinstance(reason, str):
            return False
        # ALLOWLIST, not a denylist: only reasons that are positive evidence of
        # lab pressure count. See SATURATION_FALLBACK_REASONS for the measured
        # defect that a denylist produced (`lab_unknown` alerting on a fully
        # healthy fleet and a fully healthy lab, from the day this lands).
        return reason in SATURATION_FALLBACK_REASONS

    def probe_missing(sample: dict[str, Any]) -> bool:
        return (sample.get("lab") or {}).get("probe") == "unavailable"

    def is_sustained(run: int) -> bool:
        """A breach must be both long ENOUGH and last long ENOUGH.

        Sample count alone is not duration: these records arrive as fast as the
        GitHub-hosted queue drains the canary, not at the cron's cadence.
        """
        return run >= sustained and _span_seconds(window, run) >= min_span

    busy_run = _consecutive(window, busy_breach)
    if is_sustained(busy_run):
        newest = (window[0].get("fleet") or {}).get("busy_fraction")
        alerts.append(
            Alert(
                "fleet_busy_sustained",
                f"fleet busy_fraction {newest} >= {busy_threshold} for {busy_run} consecutive samples",
            )
        )

    lab_run = _consecutive(window, lab_breach)
    if is_sustained(lab_run):
        alerts.append(
            Alert(
                "lab_load_sustained",
                f"a lab host has been at or above ratio {ratio_threshold} "
                f"(or below {mem_floor} MiB free) for {lab_run} consecutive samples",
            )
        )

    route_run = _consecutive(window, route_breach)
    if is_sustained(route_run):
        reason = (window[0].get("route") or {}).get("recent_reason")
        alerts.append(
            Alert(
                "route_fallback_sustained",
                f"routing has fallen back to hosted ({reason}) for {route_run} consecutive "
                "samples -- runs stay green while Actions spend rises",
            )
        )

    probe_run = _consecutive(window, probe_missing)
    if is_sustained(probe_run):
        alerts.append(
            Alert(
                "lab_probe_unavailable_sustained",
                f"the self-hosted lab-load probe has not started for {probe_run} consecutive "
                "samples -- the usual cause is that no runner is free",
            )
        )

    return alerts


def write_step_summary(alerts: list[Alert]) -> None:
    """Record the verdict on the run itself, so the evidence survives whether or
    not the notification does."""
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write("## Runner saturation monitor (OMN-18031)\n\n")
        if not alerts:
            handle.write("No sustained breach; fleet and lab within thresholds.\n")
            return
        handle.write("| Condition | Detail |\n|-----------|--------|\n")
        for alert in alerts:
            handle.write(f"| {alert.condition} | {alert.detail} |\n")


def post_slack_alert(alerts: list[Alert]) -> None:
    """Post through the SAME channel secret the fleet canary already uses.

    BEST-EFFORT BY DESIGN, and it can never fail the monitor. A missing token is
    a no-op, and every transport error is swallowed: the conditions are also
    emitted as ``::warning::`` annotations and written into the uploaded
    artifact, so the evidence survives a dead webhook. A watcher that goes red
    because Slack was down stops being read, which is the failure mode
    OMN-16030 already paid for once on the canary this rides.

    The recipient is inherited, not chosen here -- it is whatever
    ``SLACK_CHANNEL_ID`` already points at for the fleet canary.
    """
    token = os.environ.get("SLACK_BOT_TOKEN")
    channel = os.environ.get("SLACK_CHANNEL_ID")
    if not token or not channel:
        print("[saturation] Slack not configured; annotation + artifact only")
        return
    detail = " | ".join(f"{a.condition}: {a.detail}" for a in alerts)
    # fmt: off
    server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")  # url-authority-ok: Actions-injected GitHub web base used only to build a human-readable run link in an alert message; it addresses no ONEX service and is never fetched.
    # fmt: on
    run_url = (
        f"{server}/{os.environ.get('GITHUB_REPOSITORY', '')}"
        f"/actions/runs/{os.environ.get('GITHUB_RUN_ID', '')}"
    )
    payload = json.dumps(
        {
            "channel": channel,
            "text": f"*[RUNNER SATURATION - SUSTAINED]* {detail}. Run: {run_url}",
        }
    ).encode("utf-8")
    # fmt: off
    slack_url = "https://slack.com/api/chat.postMessage"  # url-authority-ok: fixed public Slack Web API method, no ONEX routing authority -- same contract as the fleet canary's existing chat.postMessage call in scripts/ci/runner_fleet_canary.sh
    # fmt: on
    request = urllib.request.Request(  # noqa: S310 -- fixed https Slack Web API literal above
        slack_url,
        data=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310
            response.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(
            f"[saturation] Slack post failed ({type(exc).__name__}); annotation + artifact stand"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--fleet-json", type=Path, required=True)
    parser.add_argument("--lab-json", type=Path, default=None)
    parser.add_argument("--route-reason", default="unknown")
    parser.add_argument(
        "--history", type=Path, default=None, help="dir of prior record artifacts"
    )
    parser.add_argument(
        "--out", type=Path, default=Path("runner-saturation-record.json")
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    policy = load_alert_policy(args.policy)

    def _read(path: Path | None) -> dict[str, Any]:
        if path is None or not path.exists():
            return {"ok": False, "error": "artifact_missing"}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"ok": False, "error": "unreadable"}
        return (
            payload
            if isinstance(payload, dict)
            else {"ok": False, "error": "unreadable"}
        )

    record = build_record(
        fleet=_read(args.fleet_json),
        lab=_read(args.lab_json),
        route_reason=args.route_reason,
    )

    history: list[dict[str, Any]] = [record]
    if args.history and args.history.exists():
        prior = sorted(args.history.rglob("*.json"), reverse=True)
        for path in prior:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if (
                isinstance(payload, dict)
                and payload.get("schema") == "runner_saturation_record/v1"
            ):
                history.append(payload)

    alerts = evaluate_alerts(history, policy)
    print(json.dumps(record, indent=2, sort_keys=True))
    for alert in alerts:
        print(f"::warning title=Runner saturation ({alert.condition})::{alert.detail}")
    write_step_summary(alerts)
    if alerts and not args.dry_run:
        post_slack_alert(alerts)

    if not args.dry_run:
        args.out.write_text(
            json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
        )
        alert_path = args.out.with_name("runner-saturation-alerts.json")
        alert_path.write_text(
            json.dumps(
                [{"condition": a.condition, "detail": a.detail} for a in alerts],
                indent=2,
            ),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
