# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""
Central Message Type Registry Module.

This module provides the Central Message Type Registry for the ONEX runtime dispatch
infrastructure. The registry maps message types to handler implementations and enforces
topic category constraints and domain ownership rules.

Core Components:
    - **MessageTypeRegistry**: Central registry mapping message types to handlers
    - **ModelMessageTypeEntry**: Registry entry model for message type registrations
    - **ModelDomainConstraint**: Domain constraint and ownership rules
    - **ProtocolMessageTypeRegistry**: Protocol definition for the registry interface

Key Features:
    - Message type to handler mapping with fan-out support
    - Topic category constraints (what message types can appear where)
    - Startup-time validation with fail-fast behavior
    - Domain ownership enforcement derived from topic names
    - Cross-domain consumption with explicit opt-in
    - Extensibility for new domains

Thread Safety:
    All registry operations follow the freeze-after-init pattern:
    1. Registration phase: Register message types during startup (single-threaded)
    2. Freeze: Call freeze() to prevent further modifications
    3. Query phase: Thread-safe read access after freeze

Related:
    - OMN-937: Central Message Type Registry implementation
    - OMN-934: Message Dispatch Engine (prerequisite)
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
from omnibase_infra.runtime.registry.registry_message_type import (
    MessageTypeRegistry,
    MessageTypeRegistryError,
)

__all__: list[str] = [
    # Models
    "ModelMessageTypeEntry",
    "ModelDomainConstraint",
    # Protocol
    "ProtocolMessageTypeRegistry",
    # Registry implementation
    "MessageTypeRegistry",
    "MessageTypeRegistryError",
]
