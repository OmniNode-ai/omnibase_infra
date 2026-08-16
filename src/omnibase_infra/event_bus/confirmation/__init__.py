# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Durability confirmation strategies for published records (OMN-15861).

Canonical invariant 7: *a publish return is not durability.* This package is the
seam that makes that mechanically true -- it consumes the
``ModelPublishReceipt`` that ``EventBus.publish`` now returns and produces a
``ModelDurabilityConfirmation`` verdict that a durable outbox may act on.

Shipped strategies:

``BrokerReadbackStrategy``
    Reads the coordinate back off an authoritative surface. The one to use for
    duty-critical traffic.

``PublishReturnOnlyStrategy``
    Trusts the publish return. Explicitly named so this weak choice is
    attributable in an audit; valid only for lossy-tolerant traffic.

Shipped readback sources:

``InmemoryReadbackSource``
    The zero-infra proof surface -- an in-memory bus's own history.

``KafkaReadbackSource``
    Group-less ``assign``+``seek`` against the real broker.
"""

from __future__ import annotations

from omnibase_infra.event_bus.confirmation.readback_source_inmemory import (
    InmemoryReadbackSource,
    ProtocolInmemoryHistorySource,
)
from omnibase_infra.event_bus.confirmation.readback_source_kafka import (
    KafkaReadbackSource,
    ProtocolSeekableConsumer,
)
from omnibase_infra.event_bus.confirmation.strategy_broker_readback import (
    DEFAULT_READBACK_DEADLINE_SECONDS,
    STRATEGY_NAME_BROKER_READBACK,
    BrokerReadbackStrategy,
)
from omnibase_infra.event_bus.confirmation.strategy_publish_return_only import (
    STRATEGY_NAME_PUBLISH_RETURN_ONLY,
    PublishReturnOnlyStrategy,
)

__all__: list[str] = [
    "DEFAULT_READBACK_DEADLINE_SECONDS",
    "STRATEGY_NAME_BROKER_READBACK",
    "STRATEGY_NAME_PUBLISH_RETURN_ONLY",
    "BrokerReadbackStrategy",
    "InmemoryReadbackSource",
    "KafkaReadbackSource",
    "ProtocolInmemoryHistorySource",
    "ProtocolSeekableConsumer",
    "PublishReturnOnlyStrategy",
]
