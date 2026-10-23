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
CURSOR_FIELD: Final[str] = "projection_cursor"
DEFAULT_BASE_URL: Final[str] = "http://host.docker.internal:3002"
LIVE_WINDOW_MINUTES: Final[int] = 10
# consumer_flow_windows keeps every window (48M rows, 32 GB on dev-201) and no
# index leads with window_start, so a plain distinct over it is a whole-table
# scan that outran the 120 s psql timeout under host load (C28 run 38968109234).
# Walk the distinct groups on (consumer_group, window_end DESC) instead, and
# check each group's recent windows on the same index; window_end >= window_start,
# so the extra window_end bound keeps every row the window_start predicate keeps.
LIVE_WINDOW_SQL: Final[str] = (
    "with recursive g(consumer_group) as ("
    "(select consumer_group from omninode_internal.consumer_flow_windows "
    "order by consumer_group limit 1) "
    "union all "
    "select (select c.consumer_group from omninode_internal.consumer_flow_windows c "
    "where c.consumer_group > g.consumer_group "
    "order by c.consumer_group limit 1) "
    "from g where g.consumer_group is not null) "
    "select g.consumer_group from g where g.consumer_group is not null and exists ("
    "select 1 from omninode_internal.consumer_flow_windows w "
    "where w.consumer_group = g.consumer_group "
    "and w.window_end > now() - interval '10 minutes' "
    "and w.window_start > now() - interval '10 minutes')"
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
# The broker's internal listener. Every rpk call pins it: an offset-range consume
# otherwise dials the external advertised listener, whose tailnet name does not
# resolve inside the container while the tailnet is down (C28 run 37140168388).
BROKER_INTERNAL_ADDRESS: Final[str] = "redpanda:9092"
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
#: Counted in lines, so it has to cover the runtime's log volume, not just the
#: handful of lines the seam itself writes. On the lab dev lane on 2026-10-22
#: the runtime logged about 6,000 lines a minute and the probe's own boundary
#: line sat 63 and 171 lines after its validation error; at 60 both read as
#: natural and C28 went red on the probe's own injection (OMN-17427). The
#: search still stops at the first applied-topic boundary line, so a natural
#: error pairs with its own boundary first and is never excused by a later one.
PAIRING_LOOKAHEAD: Final[int] = 5000
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
