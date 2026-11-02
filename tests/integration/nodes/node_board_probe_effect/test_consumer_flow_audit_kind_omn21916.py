# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""C28 grades an audit-purpose consumer that a lab lane without a gateway feeds.

On the lab dev lane the gateway phase is off (#5485), so nothing produces the
gateway heartbeat and its link-health projection consumer counts zero in every
window. C28 graded that consumer as its audit kind and read
``messages_in per row across the samples: [0, 0]`` on every run (OMN-21916).
The rows below are the two audit-purpose consumers as the dev lane's
consumer-flow exposure returned them at 2026-11-02T08:53:43Z, the second sample
being the next window of the same rows.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnibase_infra.nodes.node_board_probe_effect.handlers._consumer_flow_constants import (
    KINDS,
)
from omnibase_infra.nodes.node_board_probe_effect.handlers.handler_consumer_flow import (
    grade_kinds,
)
from omnibase_infra.nodes.node_board_probe_effect.models.model_consumer_flow_check import (
    ModelConsumerFlowCheck,
)
from scripts.ci import c28_consumer_flow_probe as script

pytestmark = pytest.mark.integration

NODES = Path(__file__).resolve().parents[4] / "src/omnibase_infra/nodes"
AUDIT_KIND = "audit_projection_consumer"


def _sample(window_start: str) -> list[dict[str, object]]:
    return [
        {
            "consumer_group": "local.omnibase_infra.node_ledger_projection_compute.consume.1.6.1",
            "topic": "onex.evt.platform.node-heartbeat.v1",
            "window_start": window_start,
            "messages_in": 4,
            "messages_out": 0,
            "flow_state": "CONSUMING",
        },
        {
            "consumer_group": "local.omnibase_infra.node_gateway_link_health_projection_compute.consume.1.0.0",
            "topic": "onex.evt.omnibase-infra.gateway-heartbeat.v1",
            "window_start": window_start,
            "messages_in": 0,
            "messages_out": 0,
            "flow_state": "IDLE",
        },
    ]


SAMPLES = [
    _sample("2026-11-02T08:50:01.865930+00:00"),
    _sample("2026-11-02T08:51:01.865930+00:00"),
]


def test_audit_kind_passes_on_the_gateway_off_dev_lane() -> None:
    rec: list[ModelConsumerFlowCheck] = []
    grade_kinds(rec, SAMPLES)
    audit = {c.name: c for c in rec if c.name.startswith(AUDIT_KIND)}
    assert audit, [c.name for c in rec]
    failed = {n: c.evidence for n, c in audit.items() if not c.ok}
    assert not failed, failed


def test_audit_kind_is_an_audit_consumer_of_its_declared_topic() -> None:
    _kind, prefix, topic, counter = next(k for k in KINDS if k[0] == AUDIT_KIND)
    node = prefix.removeprefix("local.omnibase_infra.").removesuffix(".consume.")
    contract = yaml.safe_load((NODES / node / "contract.yaml").read_text())
    event_bus = contract["event_bus"]
    assert event_bus["consumer_purpose"] == "audit"
    assert topic in event_bus["subscribe_topics"]
    assert counter == "messages_in"


def test_parity_grader_names_the_node_kinds() -> None:
    assert script.KINDS == KINDS
