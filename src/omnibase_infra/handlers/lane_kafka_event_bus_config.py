# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A Kafka client config stated by a lane declaration, not by the process's auth names (OMN-26316).

The dev deploy agent holds ``KAFKA_SASL_MECHANISM`` and ``KAFKA_SECURITY_PROTOCOL`` and
keeps its SASL principal under the names ``KAFKA_SASL_ENV_PREFIX`` points at.
``ModelKafkaEventBusConfig.default()`` validates the environment it reads, so that
half-set pair raised ``ProtocolConfigurationError`` before the lane's stored identity
could apply: the post-roll ledger proof of dev jobs ae9ceb3d and 43b9fb72 exited 1 and
the boot guard restored the previous image (dev, 2026-11-23, reproduced on omni-201
with the agent's own environment).

A lane's declaration and the machine's stored identity state the protocol, mechanism
and principal, so the process's own auth names are left out of the defaults and every
other override (timeouts, retries, instance id) still applies.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

from omnibase_infra.cli.model_delegate_lane_selection import ModelDelegateLaneSelection
from omnibase_infra.event_bus.model_lane_client_transport import (
    ModelLaneClientTransport,
)
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig

__all__: list[str] = ["lane_event_bus_config"]

_AUTH_NAME_PREFIX = "KAFKA_SASL_"
_PROTOCOL_NAME = "KAFKA_SECURITY_PROTOCOL"


@contextmanager
def _without_auth_environment() -> Iterator[None]:
    """Hide the process's auth names while the defaults read the environment.

    The read is synchronous and the CLI that calls it is single threaded, so the
    names are back before anything else runs.
    """
    hidden = {
        name: os.environ.pop(name)
        for name in tuple(os.environ)
        if name == _PROTOCOL_NAME or name.startswith(_AUTH_NAME_PREFIX)
    }
    try:
        yield
    finally:
        os.environ.update(hidden)


def lane_event_bus_config(
    selection: ModelDelegateLaneSelection, transport: ModelLaneClientTransport | None
) -> ModelKafkaEventBusConfig:
    """Defaults plus the lane's broker address, protocol, mechanism and identity."""
    with _without_auth_environment():
        config = ModelKafkaEventBusConfig.default()
    overrides = {
        "bootstrap_servers": selection.bootstrap_servers,
        "security_protocol": selection.security_protocol,
        "sasl_mechanism": selection.sasl_mechanism,
    }
    if transport is not None:
        overrides.update(transport.as_client_config_overrides())
    return config.model_copy(update=overrides)
