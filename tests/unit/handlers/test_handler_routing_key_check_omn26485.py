# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The routing key check, the refusal and the incident replay (OMN-26485).

omnimarket dev builds were routed to the second instance, whose agent's key check
(9b0c12edb01c7b61) differs from the dev runtime's (e30a85fc883bc113), so the agent dropped each command and
no job record answered it. The check reads each available instance's key check against
the runtime's; the router refuses a mismatched instance and falls back.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from omnibase_infra.handlers.handler_routing_key_check import HandlerRoutingKeyCheck
from omnibase_infra.models.model_deploy_lane_routing_table import (
    ModelDeployLaneRoutingTable,
)
from omnibase_infra.models.model_routing_key_check_request import (
    ModelRoutingKeyCheckRequest,
)

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[3]
_FIXTURES = _ROOT / "tests/fixtures/routing_key_check_incident_omn26485"
_RUNTIME = "e30a85fc883bc113"
_KEY_A = "e30a85fc883bc113"
_KEY_B = "9b0c12edb01c7b61"
_OMNIMARKET = "gha/omnimarket/pr-1"


def _table(
    *, available_b: bool = True, default: str = "inst-a"
) -> ModelDeployLaneRoutingTable:
    return ModelDeployLaneRoutingTable.model_validate(
        {
            "default_instance": default,
            "instances": {
                "inst-a": {},
                "inst-b": {"available": available_b},
            },
            "routes": [
                {
                    "runtime_lane": "dev",
                    "requester_repository": "omnimarket",
                    "instance": "inst-b",
                }
            ],
        }
    )


def _request(
    table: ModelDeployLaneRoutingTable,
    keys: dict[str, str],
    *,
    runtime: str = _RUNTIME,
    requested_by: str = "",
) -> ModelRoutingKeyCheckRequest:
    return ModelRoutingKeyCheckRequest(
        table=table,
        runtime_key_check=runtime,
        instance_key_checks=keys,
        runtime_lane="dev" if requested_by else "",
        requested_by=requested_by,
    )


def _handler() -> HandlerRoutingKeyCheck:
    return HandlerRoutingKeyCheck()


def test_routing_key_check_mismatch_names_both_values() -> None:
    """AC1: an available instance whose key check differs fails the check."""
    result = _handler().handle(_request(_table(), {"inst-a": _KEY_A, "inst-b": _KEY_B}))
    assert result.outcome == "mismatch"
    assert [m.instance for m in result.mismatches] == ["inst-b"]
    mismatch = result.mismatches[0]
    assert mismatch.instance_key_check == _KEY_B
    assert mismatch.runtime_key_check == _RUNTIME
    assert mismatch.reason == "key-check-differs"
    assert _KEY_B in mismatch.message
    assert _RUNTIME in mismatch.message


def test_routing_key_check_mismatch_passes_when_every_available_key_matches() -> None:
    """Positive control: the same table with matching keys, or with the instance withdrawn."""
    handler = _handler()
    matching = handler.handle(_request(_table(), {"inst-a": _KEY_A, "inst-b": _KEY_A}))
    assert matching.outcome == "ok"
    assert matching.mismatches == ()
    withdrawn = handler.handle(
        _request(_table(available_b=False), {"inst-a": _KEY_A, "inst-b": _KEY_B})
    )
    assert withdrawn.outcome == "ok"


def test_routing_key_check_mismatch_refuses_an_unread_key_check() -> None:
    """An available instance with no answer is refused, never assumed equal."""
    result = _handler().handle(_request(_table(), {"inst-a": _KEY_A}))
    assert result.outcome == "mismatch"
    assert result.mismatches[0].instance == "inst-b"
    assert result.mismatches[0].reason == "key-check-unread"
    assert result.mismatches[0].instance_key_check == ""


def test_routing_refuses_mismatched_instance_and_falls_back() -> None:
    """AC2: the routed instance is refused with a typed refusal; the command goes on."""
    result = _handler().handle(
        _request(
            _table(),
            {"inst-a": _KEY_A, "inst-b": _KEY_B},
            requested_by=_OMNIMARKET,
        )
    )
    assert result.route_outcome == "fell_back"
    assert result.routed_instance == "inst-a"
    assert len(result.refusals) == 1
    refusal = result.refusals[0]
    assert refusal.instance == "inst-b"
    assert (refusal.instance_key_check, refusal.runtime_key_check) == (
        _KEY_B,
        _RUNTIME,
    )


def test_routing_refuses_mismatched_instance_leaves_a_matching_route_alone() -> None:
    """Positive control: a routed instance whose key matches is routed, no refusal."""
    result = _handler().handle(
        _request(
            _table(),
            {"inst-a": _KEY_A, "inst-b": _KEY_A},
            requested_by=_OMNIMARKET,
        )
    )
    assert result.route_outcome == "routed"
    assert result.routed_instance == "inst-b"
    assert result.refusals == ()


def test_routing_refuses_mismatched_instance_sends_nothing_when_default_mismatches() -> (
    None
):
    """The default instance mismatching the runtime is a typed refusal; nothing is sent."""
    result = _handler().handle(
        _request(
            _table(),
            {"inst-a": "0000000000000000", "inst-b": _KEY_B},
            requested_by=_OMNIMARKET,
        )
    )
    assert result.route_outcome == "refused"
    assert result.routed_instance == ""
    assert [r.instance for r in result.refusals] == ["inst-b", "inst-a"]


def test_routing_refuses_mismatched_instance_keeps_the_availability_rule() -> None:
    """OMN-21402 is unchanged: a route naming an unavailable instance takes the default."""
    handler = _handler()
    unavailable = handler.handle(
        _request(
            _table(available_b=False),
            {"inst-a": _KEY_A, "inst-b": _KEY_B},
            requested_by=_OMNIMARKET,
        )
    )
    assert unavailable.route_outcome == "routed"
    assert unavailable.routed_instance == "inst-a"
    assert unavailable.refusals == ()
    other_requester = handler.handle(
        _request(
            _table(),
            {"inst-a": _KEY_A, "inst-b": _KEY_B},
            requested_by="gha/omnibase_infra/pr-9",
        )
    )
    assert other_requester.routed_instance == "inst-a"
    assert other_requester.refusals == ()


def _fixture_request(table_name: str) -> ModelRoutingKeyCheckRequest:
    table = ModelDeployLaneRoutingTable.model_validate(
        yaml.safe_load((_FIXTURES / table_name).read_text(encoding="utf-8"))
    )
    runtime = json.loads((_FIXTURES / "runtime_key_check.json").read_text())
    keys = {
        name: json.loads((_FIXTURES / f"health_instance_{name[-1]}.json").read_text())[
            "hmac_key_check"
        ]
        for name in ("inst-a", "inst-b")
    }
    return ModelRoutingKeyCheckRequest(
        table=table,
        runtime_key_check=runtime["hmac_key_check"],
        instance_key_checks=keys,
        runtime_lane="dev",
        requested_by=_OMNIMARKET,
    )


def test_routing_incident_replay_fails_on_the_broken_table_and_passes_on_the_fix() -> (
    None
):
    """AC4: the answers recorded at 2026-11-22T21:43Z, replayed through the check."""
    handler = _handler()
    broken = handler.handle(_fixture_request("routing_table_broken.yaml"))
    assert broken.outcome == "mismatch"
    assert [m.instance for m in broken.mismatches] == ["inst-b"]
    assert broken.mismatches[0].instance_key_check == "9b0c12edb01c7b61"
    assert broken.mismatches[0].runtime_key_check == "e30a85fc883bc113"
    # The router no longer sends the omnimarket build to the instance that drops it.
    assert broken.route_outcome == "fell_back"
    assert broken.routed_instance == "inst-a"

    fixed = handler.handle(_fixture_request("routing_table_fixed.yaml"))
    assert fixed.outcome == "ok"
    assert fixed.route_outcome == "routed"
    assert fixed.routed_instance == "inst-a"


def test_routing_incident_replay_reads_the_committed_table() -> None:
    """The committed table parses into the check's model, with the shape the replay relies on."""
    table = ModelDeployLaneRoutingTable.model_validate(
        yaml.safe_load(
            (_ROOT / "config/deploy_lane_routing.yaml").read_text(encoding="utf-8")
        )
    )
    assert table.instances[table.default_instance].available
    omnimarket = [
        r
        for r in table.routes
        if r.runtime_lane == "dev" and r.requester_repository == "omnimarket"
    ]
    assert len(omnimarket) == 1
    assert omnimarket[0].instance in table.instances
