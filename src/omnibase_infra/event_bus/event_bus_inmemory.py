# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Temporary re-export shim for EventBusInmemory.

EventBusInmemory was moved to omnibase_core as part of OMN-7062.
This module re-exports it so that existing omnibase_infra consumers
continue to work without import changes.

Remove this shim after the next release cycle when all consumers
have migrated to import from omnibase_core.event_bus directly.
"""

from omnibase_core.event_bus.event_bus_inmemory import EventBusInmemory

__all__: list[str] = ["EventBusInmemory"]
