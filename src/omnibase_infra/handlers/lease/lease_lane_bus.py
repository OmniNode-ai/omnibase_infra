# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The bus of a declared lane, as this machine's identity for it (OMN-21424).

The broker address comes from the workspace's lane declaration, never from the
ambient environment, and the identity is the one the lane's declaration binds
to this machine, exactly as ``onex delegate --lane`` resolves them.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, nullcontext
from pathlib import Path

from omnibase_infra.cli.delegate_lane import (
    DelegateLaneSelectionError,
    resolve_lane_selection_at,
)
from omnibase_infra.cli.delegate_lane_credentials import (
    DelegateLaneCredentialError,
    resolve_lane_client_transport_for,
)
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.lane_client_transport_binding import (
    bind_lane_client_transport,
)
from omnibase_infra.protocols.protocol_lease_bus import ProtocolLeaseBus

_CONNECT_TIMEOUT_S = 10.0


class LeaseLaneUnresolvedError(RuntimeError):
    """The lane has no usable broker or identity on this machine."""


@asynccontextmanager
async def open_lane_bus(
    lane: str, workspace_root: Path | None
) -> AsyncIterator[ProtocolLeaseBus]:
    """Connect to the lane's declared broker; close the connection on exit."""
    try:
        selection = resolve_lane_selection_at(lane, workspace_root)
        transport = resolve_lane_client_transport_for(
            lane_target=selection,
            onex_home=Path.home() / ".onex",
            environ=os.environ,
        )
    except (DelegateLaneSelectionError, DelegateLaneCredentialError, OSError) as exc:
        raise LeaseLaneUnresolvedError(f"cannot resolve lane {lane!r}: {exc}") from exc
    binding = (
        bind_lane_client_transport(transport)
        if transport is not None
        else nullcontext()
    )
    with binding:
        bus = EventBusKafka.from_bootstrap(selection.bootstrap_servers)
        await asyncio.wait_for(bus.start(), _CONNECT_TIMEOUT_S)
        try:
            yield bus
        finally:
            await asyncio.wait_for(bus.close(), _CONNECT_TIMEOUT_S)


__all__ = ["LeaseLaneUnresolvedError", "open_lane_bus"]
