# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""
Runtime Registry Module.

This module provides registry implementations for the ONEX runtime infrastructure:

Protocol Binding Registry:
    - **RegistryProtocolBinding**: Handler registration and resolution
    - **RegistryError**: Error raised when registry operations fail

Event Bus Binding Registry:
    - **RegistryEventBusBinding**: Event bus implementation registration

Message Type Registry:
    - **RegistryMessageType**: Central registry mapping message types to handlers
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
    - OMN-1271: RegistryProtocolBinding and RegistryEventBusBinding extraction
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
from omnibase_infra.runtime.registry.mixin_message_type_query import (
    MixinMessageTypeQuery,
)
from omnibase_infra.runtime.registry.mixin_message_type_registration import (
    MixinMessageTypeRegistration,
)
from omnibase_infra.runtime.registry.registry_event_bus_binding import (
    RegistryEventBusBinding,
)
from omnibase_infra.runtime.registry.registry_message_type import (
    MessageTypeRegistryError,
    RegistryMessageType,
    extract_domain_from_topic,
)
from omnibase_infra.runtime.registry.registry_protocol_binding import (
    RegistryError,
    RegistryProtocolBinding,
)

__all__: list[str] = [
    "MessageTypeRegistryError",
    # Mixins
    "MixinMessageTypeQuery",
    "MixinMessageTypeRegistration",
    "ModelDomainConstraint",
    # Models
    "ModelMessageTypeEntry",
    # Protocol
    "ProtocolMessageTypeRegistry",
    # Registry error
    "RegistryError",
    # Event bus binding registry
    "RegistryEventBusBinding",
    # Registry implementation
    "RegistryMessageType",
    # Protocol binding registry
    "RegistryProtocolBinding",
    # Utility functions
    "extract_domain_from_topic",
]
