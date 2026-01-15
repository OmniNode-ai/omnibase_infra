# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Protocols for Registration Storage Effect Node.

This module exports protocols for the NodeRegistrationStorageEffect,
defining interfaces for pluggable storage backends.

Available Protocols:
    - ProtocolRegistrationPersistence: Interface for storage backends

Protocol Compliance:
    Per ONEX conventions, protocol compliance is verified via duck typing.
    The protocol is @runtime_checkable for isinstance() support.

Implementations:
    - HandlerRegistrationStoragePostgres: PostgreSQL backend
    - HandlerRegistrationStorageMock: Mock backend for testing
"""

from .protocol_registration_persistence import ProtocolRegistrationPersistence

__all__ = ["ProtocolRegistrationPersistence"]
