# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for the lane-census-drift bus event builder (OMN-13011)."""

from __future__ import annotations

import importlib.util
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]
_EVENT_PATH = _REPO / "scripts" / "lane_census_event.py"


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("lane_census_event", _EVENT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EV = _load()

_FIXED_NOW = datetime(2026, 6, 11, 22, 0, 0, tzinfo=UTC)


def _prod_outage_plan() -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "lanes_checked": ["prod"],
        "has_drift": True,
        "findings": [
            {
                "lane": "prod",
                "kind": "network_detached",
                "container": "omnibase-infra-prod-network",
                "detail": "lane network is not present",
                "severity": "critical",
            },
            {
                "lane": "prod",
                "kind": "container_absent",
                "container": "omninode-prod-runtime",
                "detail": "required service not running",
                "severity": "critical",
            },
        ],
    }


def test_event_is_critical_when_any_finding_critical() -> None:
    event = EV.build_event(host="omninode-pc", plan=_prod_outage_plan(), now=_FIXED_NOW)
    assert event["severity"] == "critical"
    assert event["event_type"] == "lane-census-drift"
    assert event["topic"] == "onex.evt.infra.lane-census-drift.v1"
    assert event["drift_count"] == 2
    assert event["emitted_at"] == "2026-06-11T22:00:00+00:00"


def test_event_is_warning_when_no_critical() -> None:
    plan = {
        "lanes_checked": ["judge"],
        "findings": [
            {
                "lane": "judge",
                "kind": "image_tag_mismatch",
                "container": "omninode-judge-runtime",
                "detail": "stale tag",
                "severity": "warning",
            }
        ],
    }
    event = EV.build_event(host="h", plan=plan, now=_FIXED_NOW)
    assert event["severity"] == "warning"


def test_alert_key_is_stable_and_dedupes() -> None:
    """Same finding set => same alert_key (one ticket per outage, not per tick)."""
    plan = _prod_outage_plan()
    k1 = EV.build_event(host="omninode-pc", plan=plan, now=_FIXED_NOW)["alert_key"]
    k2 = EV.build_event(host="omninode-pc", plan=plan, now=datetime.now(UTC))[
        "alert_key"
    ]
    assert k1 == k2
    assert k1.startswith("lane-census-drift:omninode-pc:")


def test_alert_key_changes_with_finding_set() -> None:
    plan_a = _prod_outage_plan()
    plan_b = _prod_outage_plan()
    plan_b["findings"] = plan_b["findings"][:1]  # only the network finding
    ka = EV.build_event(host="h", plan=plan_a, now=_FIXED_NOW)["alert_key"]
    kb = EV.build_event(host="h", plan=plan_b, now=_FIXED_NOW)["alert_key"]
    assert ka != kb


def test_ticket_title_names_lane_and_kinds() -> None:
    title = EV.build_event(host="h", plan=_prod_outage_plan(), now=_FIXED_NOW)[
        "ticket_title"
    ]
    assert "prod" in title
    assert "network_detached" in title
    assert "container_absent" in title


def test_ticket_body_lists_every_finding() -> None:
    body = EV.build_event(host="h", plan=_prod_outage_plan(), now=_FIXED_NOW)[
        "ticket_body"
    ]
    assert "omnibase-infra-prod-network" in body
    assert "omninode-prod-runtime" in body
    assert "OMN-13011" in body
