# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for the shared ``derive_event_type_from_topic`` helper (OMN-14743).

The helper is the single canonical topic -> event_type derivation. These tests
pin its behavior AND prove it agrees field-for-field with the two runtime copies
it consolidates (the applier's staticmethod, which now delegates to it, and the
independent ``EventBusSubcontractWiring`` copy), so a future edit that drifts one
site is caught here rather than at runtime.
"""

from __future__ import annotations

import pytest

from omnibase_infra.runtime.event_bus_subcontract_wiring import (
    EventBusSubcontractWiring,
)
from omnibase_infra.runtime.service_dispatch_result_applier import DispatchResultApplier
from omnibase_infra.utils import derive_event_type_from_topic as pkg_export
from omnibase_infra.utils.util_topic_event_type import derive_event_type_from_topic


@pytest.mark.unit
@pytest.mark.parametrize(
    ("topic", "expected"),
    [
        (
            "onex.cmd.omnibase-infra.delegation-routing-request.v1",
            "omnibase-infra.delegation-routing-request",
        ),
        (
            "onex.evt.omnimarket.swarm-endpoint-health-completed.v1",
            "omnimarket.swarm-endpoint-health-completed",
        ),
        # >5 segments: only the {producer}.{event-name} pair is taken.
        ("onex.evt.producer.event-name.extra.v2", "producer.event-name"),
    ],
)
def test_derives_producer_event_name_for_valid_onex_topics(
    topic: str, expected: str
) -> None:
    assert derive_event_type_from_topic(topic) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    "topic",
    [
        "",
        "not-a-topic",
        "kafka.evt.producer.name.v1",  # does not start with 'onex'
        "onex.evt.producer.v1",  # only 4 segments
        "onex.evt.producer",  # only 3 segments
    ],
)
def test_returns_none_for_non_onex_or_short_topics(topic: str) -> None:
    assert derive_event_type_from_topic(topic) is None


@pytest.mark.unit
def test_package_export_is_the_same_callable() -> None:
    assert pkg_export is derive_event_type_from_topic


@pytest.mark.unit
@pytest.mark.parametrize(
    "topic",
    [
        "onex.cmd.omnibase-infra.delegation-routing-request.v1",
        "onex.evt.omnimarket.swarm-endpoint-health-completed.v1",
        "not-a-topic",
        "onex.evt.producer.v1",
    ],
)
def test_parity_with_runtime_copies(topic: str) -> None:
    """The applier and event-bus-wiring derivations must equal the shared helper."""
    canonical = derive_event_type_from_topic(topic)
    assert DispatchResultApplier._derive_event_type_from_topic(topic) == canonical
    assert EventBusSubcontractWiring._derive_event_type_from_topic(topic) == canonical
