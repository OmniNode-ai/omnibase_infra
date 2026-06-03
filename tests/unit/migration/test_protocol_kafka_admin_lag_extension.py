# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Structural test for the lag/offset extension of ProtocolKafkaAdminLike (OMN-12623).

Asserts the protocol exposes ``list_consumer_group_offsets`` + ``list_offsets``
so the consumer-lag observer can compute lag for the drain-proof gate.
"""

from __future__ import annotations

import pytest

from omnibase_infra.protocols.protocol_kafka_admin_like import ProtocolKafkaAdminLike

pytestmark = pytest.mark.unit


def test_protocol_declares_lag_offset_methods() -> None:
    for method in ("list_consumer_group_offsets", "list_offsets"):
        assert hasattr(ProtocolKafkaAdminLike, method), (
            f"ProtocolKafkaAdminLike must declare {method} for lag observation"
        )


def test_lag_capable_admin_satisfies_protocol() -> None:
    class _LagAdmin:
        async def start(self) -> None: ...
        async def stop(self) -> None: ...
        async def close(self) -> None: ...
        async def list_consumer_groups(self, broker_ids=None):
            return []

        async def describe_consumer_groups(self, group_ids, **kwargs):
            return []

        async def list_consumer_group_offsets(self, group_id, **kwargs):
            return {}

        async def list_offsets(self, topic_partitions):
            return {}

    admin: ProtocolKafkaAdminLike = _LagAdmin()
    assert admin is not None
