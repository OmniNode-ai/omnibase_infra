# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Unit checks for migration 098 gateway_link_health + liveness-threshold drift.

OMN-15570 (G3) hardcodes the ``gateway_link_health_status`` view's health
thresholds (``max_silence_window_seconds`` / ``lag_threshold_messages`` /
``lag_threshold_seconds``) as SQL literals rather than reading them from the
contract at migration/config load time -- there is no existing repo pattern
for materializing a value into a static, already-applied SQL view at
migration time (unlike the runtime config loader's
``_materialize_contract_*`` functions, which run at process start against a
YAML file, a view body is baked in once at ``CREATE OR REPLACE VIEW`` time).
This is the 4th independent copy of these three numbers (contract.yaml,
``ModelGatewayForwarderConfig`` defaults, deploy YAML, and now this view) --
tracked as gateway config triplication under OMN-15762.

Per that ticket's disclosed scope, this test is the fail-closed guard: it
pins the SQL view's literals to the contract's declared values and fails if
they diverge, so an operator changing the contract's liveness block without
updating the migration finds out here instead of via a silent, wrong
``HEALTHY``/``UNHEALTHY`` verdict in production.
"""

import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
FORWARD = (
    REPO_ROOT
    / "docker"
    / "migrations"
    / "forward"
    / "098_create_gateway_link_health.sql"
)
ROLLBACK = (
    REPO_ROOT
    / "docker"
    / "migrations"
    / "rollback"
    / "rollback_098_create_gateway_link_health.sql"
)
CONTRACT = (
    REPO_ROOT
    / "src"
    / "omnibase_infra"
    / "nodes"
    / "node_bus_forwarder_effect"
    / "contract.yaml"
)


def _contract_liveness() -> dict[str, int]:
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    liveness = contract["config"]["gateway_forwarder"]["liveness"]
    return {
        "max_silence_window_seconds": int(liveness["max_silence_window_seconds"]),
        "lag_threshold_messages": int(liveness["lag_threshold_messages"]),
        "lag_threshold_seconds": int(liveness["lag_threshold_seconds"]),
    }


def test_098_forward_creates_table_and_view() -> None:
    sql = FORWARD.read_text(encoding="utf-8").lower()

    assert "create table if not exists public.gateway_link_health" in sql
    assert "create or replace view public.gateway_link_health_status" in sql
    assert "primary key (tenant_id)" in sql


def test_098_rollback_drops_view_and_table() -> None:
    sql = ROLLBACK.read_text(encoding="utf-8").lower()

    assert "drop view if exists public.gateway_link_health_status" in sql
    assert "drop table if exists public.gateway_link_health" in sql


def test_098_view_liveness_thresholds_match_contract_omn_15762() -> None:
    """Fail-closed drift guard (OMN-15762): the migration's hardcoded
    thresholds must match node_bus_forwarder_effect/contract.yaml's
    liveness block. This is the 4th copy of these numbers (contract,
    ModelGatewayForwarderConfig defaults, deploy YAML, this view) -- see
    module docstring. A mismatch here means the contract changed and the
    view was not updated to match, which would silently mis-classify
    gateway health.
    """
    sql = FORWARD.read_text(encoding="utf-8")
    contract = _contract_liveness()

    silence_match = re.search(
        r"NOW\(\)\s*-\s*last_seen_at\s*>\s*INTERVAL\s*'(\d+)\s*seconds'",
        sql,
    )
    assert silence_match is not None, (
        "expected an `INTERVAL '<n> seconds'` silence-window comparison in "
        "098_create_gateway_link_health.sql -- update this regex if the "
        "view's SQL shape changed"
    )
    assert int(silence_match.group(1)) == contract["max_silence_window_seconds"], (
        "098_create_gateway_link_health.sql's silence-window literal has "
        "drifted from node_bus_forwarder_effect/contract.yaml's "
        "gateway_forwarder.liveness.max_silence_window_seconds (OMN-15762 "
        "4th-copy class) -- update the migration's hardcoded literal (a new "
        "migration, since this one is already applied) to match the "
        "contract, or update the contract to match an intentional migration "
        "change"
    )

    lag_messages_match = re.search(
        r"lag_messages\s+IS\s+NOT\s+NULL\s+AND\s+lag_messages\s*>\s*(\d+)",
        sql,
    )
    assert lag_messages_match is not None
    assert int(lag_messages_match.group(1)) == contract["lag_threshold_messages"], (
        "098_create_gateway_link_health.sql's lag_threshold_messages "
        "literal has drifted from the contract (OMN-15762 4th-copy class)"
    )

    lag_seconds_match = re.search(
        r"lag_seconds\s+IS\s+NOT\s+NULL\s+AND\s+lag_seconds\s*>\s*(\d+)",
        sql,
    )
    assert lag_seconds_match is not None
    assert int(lag_seconds_match.group(1)) == contract["lag_threshold_seconds"], (
        "098_create_gateway_link_health.sql's lag_threshold_seconds literal "
        "has drifted from the contract (OMN-15762 4th-copy class)"
    )
