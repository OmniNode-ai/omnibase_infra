# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Incident replays for the per-run runner routing guards (OMN-18031).

Two cases, each driving the REAL guard over bytes captured from the surface that
produced the wrong answer -- not a reconstruction.

CASE 1 replays a defect measured in this repository's own live data on
2026-09-07T10:48:57Z. The first build of the routing decision read the workflow
path against ``hosted_runner_allowlist`` in config/runner_routing_policy.yaml.
That list means "this FILE contains a job that is deliberately bare
``runs-on: ubuntu-latest``", which is why ``.github/workflows/ci.yml`` is on it
-- for its one lightweight CI Summary aggregator. A dry run against the live
fleet returned ``policy_allowlist`` for ci.yml: all 54 of its jobs would have
been pinned to GitHub-hosted compute permanently, making the entire mechanism a
silent no-op on its largest consumer, with every unit test green. The artifact
is that policy file exactly as committed on dev.

CASE 2 replays the OMN-16030 lesson on the alerter this monitor rides. The fleet
canary used to fail on the org REST ``status`` field; measurement on 2026-08-14
showed offline-labelled runners serving ~80% of nominal throughput, so the
canary was persistently red on a signal that was not liveness -- which trains
operators to ignore it and halts landing sweeps on a false alarm. The saturation
monitor inherits the identical hazard from the other direction: while the
trusted seam reads ``["ubuntu-latest"]`` every run decides
``seam_ceiling_hosted``, so an alerter that counted that as a saturation
fallback would fire on EVERY sample from the moment it landed and be muted long
before the seam ever flips. The artifact is the live org runner registry.

Both cases carry an accept control in the same module, so a guard that simply
rejects everything cannot satisfy either one.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "omn18031"

POLICY_FIXTURE = FIXTURES / "runner_routing_policy.dev.yaml.captured"
POLICY_SHA256 = "015a35744de18e86d7abd5cdc3eef5acc530e562509d38e030b8b1ec23ffcbd7"

RUNNERS_FIXTURE = FIXTURES / "org-runners-omnibase-ci.json.captured"
RUNNERS_SHA256 = "9805b0a087535b29dcfd925a9fe3379d95cd98b99d1a93eaf44367be1eea37c7"


def _load(module_name: str, relative: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, REPO_ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


route = _load("replay_route_decision", "scripts/ci/runner_route_decision.py")
sat = _load("replay_saturation_record", "scripts/ci/runner_saturation_record.py")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_captured_artifacts_are_the_bytes_the_registry_records() -> None:
    """A fixture that drifted is no longer the artifact that failed."""
    assert _sha256(POLICY_FIXTURE) == POLICY_SHA256
    assert _sha256(RUNNERS_FIXTURE) == RUNNERS_SHA256


# --- CASE 1 ---------------------------------------------------------------


def _live_policy() -> dict[str, Any]:
    return route.load_route_policy(REPO_ROOT / "config" / "runner_routing_policy.yaml")


def _decide_ci_yml(hosted_list: list[str]) -> Any:
    """Drive the REAL decision function for ci.yml on an idle fleet."""
    return route.decide(
        event_name="push",
        head_repo="OmniNode-ai/omnibase_infra",
        repository="OmniNode-ai/omnibase_infra",
        workflow_path=".github/workflows/ci.yml",
        seam_json='["self-hosted","omnibase-ci"]',
        public_json='["ubuntu-latest"]',
        fleet={"ok": True, "online": 88, "busy": 9, "total": 88},
        lab={
            "ok": True,
            "age_seconds": 30,
            "hosts": [{"label": "omninode-pc", "ratio": 0.40, "free_mem_mib": 49000}],
        },
        policy=_live_policy(),
        allowlist=hosted_list,
    )


def test_the_captured_policy_still_carries_the_list_that_caused_the_defect() -> None:
    """Non-vacuity: the replay is only meaningful if ci.yml is really on that
    list in the captured bytes.
    """
    import yaml

    captured = yaml.safe_load(POLICY_FIXTURE.read_text(encoding="utf-8"))
    paths = [entry["path"] for entry in captured["hosted_runner_allowlist"]]
    assert ".github/workflows/ci.yml" in paths
    # And the captured bytes predate this ticket: no route section existed yet.
    assert "route" not in captured


def test_the_real_guard_pins_ci_yml_hosted_when_fed_the_audit_allowlist() -> None:
    """REPLAY OF THE DEFECT. Feeding the decision the audit's file-level
    allowlist -- the exact list in the captured artifact -- reproduces the wrong
    verdict: ci.yml routes hosted for `policy_allowlist` on a fully idle fleet.
    This is what the module did on live data before the lists were separated.
    """
    import yaml

    captured = yaml.safe_load(POLICY_FIXTURE.read_text(encoding="utf-8"))
    audit_allowlist = [entry["path"] for entry in captured["hosted_runner_allowlist"]]
    result = _decide_ci_yml(audit_allowlist)
    assert result.decision == "hosted"
    assert result.reason == "policy_allowlist"


def test_the_shipped_configuration_does_not_pin_ci_yml_hosted() -> None:
    """ACCEPT CONTROL, and the regression this case exists to hold. With the
    route-specific `hosted_workflows` list the shipped code actually reads,
    ci.yml is free to route on capacity. If someone re-points the module at
    `hosted_runner_allowlist`, this fails.
    """
    result = _decide_ci_yml(route.hosted_workflows(_live_policy()))
    assert result.decision == "self_hosted"
    assert result.reason == "capacity_available"


def test_the_route_hosted_list_and_the_audit_allowlist_are_not_the_same_list() -> None:
    """The two lists answer different questions; conflating them is the defect."""
    import yaml

    live = yaml.safe_load(
        (REPO_ROOT / "config" / "runner_routing_policy.yaml").read_text("utf-8")
    )
    audit_paths = {entry["path"] for entry in live["hosted_runner_allowlist"]}
    route_paths = set(live["route"]["hosted_workflows"])
    assert ".github/workflows/ci.yml" in audit_paths
    assert ".github/workflows/ci.yml" not in route_paths


# --- CASE 2 ---------------------------------------------------------------


def _window_from_capture(route_reason: str) -> list[dict[str, Any]]:
    """Build a sustain window from the REAL captured runner registry."""
    captured = json.loads(RUNNERS_FIXTURE.read_text(encoding="utf-8"))
    runners = captured["runners"]
    online = sum(1 for item in runners if item["status"] == "online")
    busy = sum(1 for item in runners if item["busy"])
    record = sat.build_record(
        fleet={"ok": True, "online": online, "busy": busy},
        lab={
            "ok": True,
            "hosts": [{"label": "omninode-pc", "ratio": 0.40, "free_mem_mib": 49000}],
        },
        route_reason=route_reason,
        sampled_at=captured["captured_at"],
    )
    # Twelve samples, ten minutes apart. Spacing is load-bearing: the sustain
    # rule is a DURATION as well as a count, so a window of identical
    # timestamps spans zero seconds and would alert on nothing regardless of
    # its contents -- which would make the positive control below vacuous.
    base = datetime(2026, 9, 7, 12, 0, 0, tzinfo=UTC)
    window = []
    for index in range(12):
        sample = json.loads(json.dumps(record))
        sample["sampled_at"] = (base - timedelta(seconds=index * 600)).isoformat()
        window.append(sample)
    return window


def _alert_policy() -> dict[str, Any]:
    return sat.load_alert_policy(REPO_ROOT / "config" / "runner_routing_policy.yaml")


def test_the_capture_is_a_real_fleet_and_not_an_empty_read() -> None:
    """Non-vacuity: an empty capture would make every assertion below trivial."""
    captured = json.loads(RUNNERS_FIXTURE.read_text(encoding="utf-8"))
    assert len(captured["runners"]) == 88
    assert all("status" in item and "busy" in item for item in captured["runners"])


def test_the_real_monitor_stays_silent_on_the_inert_seam_state() -> None:
    """REPLAY. Today every run decides `seam_ceiling_hosted`. Against the real
    fleet capture, a 12-sample window of that state must produce NO alert --
    otherwise the monitor fires continuously from the day it lands and gets
    muted, which is precisely the OMN-16030 failure this rides on top of.
    """
    alerts = sat.evaluate_alerts(
        _window_from_capture("seam_ceiling_hosted"), _alert_policy()
    )
    assert alerts == [], [a.condition for a in alerts]


def test_the_real_monitor_stays_silent_when_routing_is_working() -> None:
    """A fleet being USED is not a fleet being slammed."""
    alerts = sat.evaluate_alerts(
        _window_from_capture("capacity_available"), _alert_policy()
    )
    assert alerts == []


def test_the_same_monitor_does_fire_on_a_genuine_sustained_fallback() -> None:
    """POSITIVE CONTROL over the identical capture. Same fleet bytes, same
    window length, same policy -- only the route reason differs. Without this,
    the two silence assertions above would be satisfied by `return []`.
    """
    alerts = sat.evaluate_alerts(
        _window_from_capture("fleet_saturated"), _alert_policy()
    )
    assert [a.condition for a in alerts] == ["route_fallback_sustained"]
