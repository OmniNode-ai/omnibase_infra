# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Service Discovery Operation Enum.

This module provides EnumServiceDiscoveryOperation, representing the type
of service discovery operation performed (register, deregister).

Architecture:
    EnumServiceDiscoveryOperation defines the possible operations:
    - REGISTER: Register a new service instance
    - DEREGISTER: Remove a service instance from discovery

    This enum is backend-agnostic and represents a normalized view
    of operations across Consul, Kubernetes, Etcd, etc.

Related:
    - ModelRegistrationResult: Uses this enum for operation field
    - OMN-1131: Capability-oriented node architecture
"""

from __future__ import annotations

from enum import Enum


class EnumServiceDiscoveryOperation(str, Enum):
    """Type of service discovery operation.

    Represents the possible operations that can be performed
    through service discovery backends.

    Values:
        REGISTER: Register a new service instance.
        DEREGISTER: Remove a service instance from discovery.

    Example:
        >>> op = EnumServiceDiscoveryOperation.REGISTER
        >>> op.value
        'register'
        >>> op.is_register
        True
    """

    REGISTER = "register"
    DEREGISTER = "deregister"

    @property
    def is_register(self) -> bool:
        """Return True if this is a register operation."""
        return self == EnumServiceDiscoveryOperation.REGISTER

    @property
    def is_deregister(self) -> bool:
        """Return True if this is a deregister operation."""
        return self == EnumServiceDiscoveryOperation.DEREGISTER


__all__ = ["EnumServiceDiscoveryOperation"]
