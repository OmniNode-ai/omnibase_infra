# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""C28 wire topics and collection expectations, ported from the producer."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

import yaml


def _contract_topic(name: str) -> str:
    """Read an observed topic from this node's contract."""
    contract = Path(__file__).resolve().parents[1] / "contract.yaml"
    raw = yaml.safe_load(contract.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"contract {contract} must contain a mapping")
    topics = raw.get("consumer_flow_topics")
    if not isinstance(topics, dict):
        raise ValueError(f"contract {contract} must declare consumer_flow_topics")
    topic = topics.get(name)
    if not isinstance(topic, str) or not topic.startswith("onex.evt."):
        raise ValueError(f"contract {contract} has no event topic for {name}")
    return topic


# ---- the subject, as the readback named it ---------------------------------
EXPOSURE_TOPIC: Final[str] = "onex.snapshot.projection.consumer-flow.v1"
DEFAULT_BASE_URL: Final[str] = "http://host.docker.internal:3002"
LIVE_WINDOW_MINUTES: Final[int] = 10
LIVE_WINDOW_SQL: Final[str] = (
    "select distinct consumer_group from omninode_internal.consumer_flow_windows "
    "where window_start > now() - interval '10 minutes'"
)

#: (kind, consumer-group prefix, subscribed topic, the counter that must move)
KINDS: Final[tuple[tuple[str, str, str, str], ...]] = (
    (
        "audit_projection_consumer",
        "local.omnibase_infra.node_gateway_link_health_projection_compute.consume.",
        _contract_topic("audit_projection"),
        "messages_in",
    ),
    (
        "publishing_reducer",
        "local.omnimarket.projection_consumer_flow.consume.",
        _contract_topic("publishing_reducer"),
        "messages_out",
    ),
)

APPLIED_TOPIC: Final[str] = _contract_topic("applied")
GENERIC_DLQ: Final[str] = "onex.dlq.omnibase-infra.events.v1"
SEAM_DLQ: Final[str] = "onex.dlq.omnimarket.consumer-flow-stall-alert-malformed.v1"

RUNTIME_CONTAINERS: Final[tuple[str, ...]] = (
    "omninode-runtime",
    "omninode-runtime-effects",
)
BOOT_CONTAINERS: Final[tuple[str, ...]] = (
    *RUNTIME_CONTAINERS,
    "omnimarket-projection-api",
)
BROKER_CONTAINER: Final[str] = "omnibase-infra-redpanda"
PG_CONTAINER: Final[str] = "omnibase-infra-postgres"
ANALYTICS_DB: Final[str] = "omnidash_analytics"

#: Both names the stall-alert seam's input model has carried.
VALIDATION_ERROR_RE: Final[re.Pattern[str]] = re.compile(
    r"validation errors? for ModelConsumerFlowStallAlert(?:Request|Trigger)\b"
)
BOUNDARY_RE: Final[re.Pattern[str]] = re.compile(
    r"metric_name=boundary_swallow_prevented dlq_routed=true .*?"
    r"topic=" + re.escape(APPLIED_TOPIC) + r"\b.*?correlation_id=(?P<cid>[0-9a-f-]{36})"
)
#: How far after a validation error its boundary line may sit in the same log.
PAIRING_LOOKAHEAD: Final[int] = 60
PROBE_CID_PREFIX: Final[str] = "c28c28c2-c28c-"

# ---- clause 2 ---------------------------------------------------------------
NEGATIVE_TESTS: Final[tuple[str, ...]] = (
    "tests/ci/test_omn17214_subscription_flow_counter_gate.py",
    "tests/unit/runtime/auto_wiring/test_omn17214_raw_projection_flow_seam.py",
)
WIRING_MODULE: Final[str] = "src/omnibase_infra/runtime/auto_wiring/handler_wiring.py"
#: parametrize id -> the factory whose registration the mutation removes
BRANCHES: Final[dict[str, str]] = {
    "event_bus": "_make_event_bus_callback",
    "raw_event_projection": "_make_raw_event_projection_callback",
}
AST_GATE_TEST: Final[str] = (
    "test_every_selected_subscription_factory_registers_a_flow_counter"
)
