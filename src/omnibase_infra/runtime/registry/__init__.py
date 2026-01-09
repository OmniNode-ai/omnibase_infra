# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""
Runtime Registry Module.

This module provides registry implementations for the ONEX runtime infrastructure:

Protocol Binding Registry:
    - **ProtocolBindingRegistry**: Handler registration and resolution
    - **RegistryError**: Error raised when registry operations fail

Event Bus Binding Registry:
    - **EventBusBindingRegistry**: Event bus implementation registration

Message Type Registry:
    - **MessageTypeRegistry**: Central registry mapping message types to handlers
    - **ModelMessageTypeEntry**: Registry entry model for message type registrations
    - **ModelDomainConstraint**: Domain constraint and ownership rules
    - **ProtocolMessageTypeRegistry**: Protocol definition for the registry interface

Key Features:
    - Protocol handler to handler type mapping
    - Event bus implementation registration
    - Message type to handler mapping with fan-out support
    - Topic category constraints (what message types can appear where)
    - Startup-time validation with fail-fast behavior
    - Domain ownership enforcement derived from topic names
    - Cross-domain consumption with explicit opt-in

Thread Safety:
    All registry operations follow the freeze-after-init pattern:
    1. Registration phase: Register handlers/types during startup (single-threaded)
    2. Freeze: Call freeze() to prevent further modifications
    3. Query phase: Thread-safe read access after freeze

Related:
    - OMN-937: Central Message Type Registry implementation
    - OMN-934: Message Dispatch Engine (prerequisite)
    - OMN-1271: ProtocolBindingRegistry and EventBusBindingRegistry extraction
    - MessageDispatchEngine: Uses this registry for handler resolution

.. versionadded:: 0.5.0
"""

from omnibase_infra.models.registry.model_domain_constraint import (
    ModelDomainConstraint,
)
from omnibase_infra.models.registry.model_message_type_entry import (
    ModelMessageTypeEntry,
)
from omnibase_infra.protocols.protocol_message_type_registry import (
    ProtocolMessageTypeRegistry,
)
from omnibase_infra.runtime.registry.registry_event_bus_binding import (
    EventBusBindingRegistry,
)
from omnibase_infra.runtime.registry.registry_message_type import (
    MessageTypeRegistry,
    MessageTypeRegistryError,
)
from omnibase_infra.runtime.registry.registry_protocol_binding import (
    ProtocolBindingRegistry,
    RegistryError,
)

__all__: list[str] = [
    # Event bus binding registry
    "EventBusBindingRegistry",
    # Registry implementation
    "MessageTypeRegistry",
    "MessageTypeRegistryError",
    "ModelDomainConstraint",
    # Models
    "ModelMessageTypeEntry",
    # Protocol binding registry
    "ProtocolBindingRegistry",
    # Protocol
    "ProtocolMessageTypeRegistry",
    # Registry error
    "RegistryError",
]
