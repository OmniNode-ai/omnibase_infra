# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Public producer factory port for registration's event-bus effect backend."""

from typing import Protocol, runtime_checkable

from aiokafka import AIOKafkaProducer


@runtime_checkable
class ProtocolRegistrationSnapshotProducer(Protocol):
    """Construct a producer without starting it; the plugin owns its lifecycle."""

    def create(self, bootstrap_servers: str | None) -> AIOKafkaProducer: ...
