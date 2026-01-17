# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Protocols for Service Discovery Effect Node.

This module exports protocols for the service discovery effect node:

Protocols:
    ProtocolDiscoveryOperations: Protocol for pluggable service discovery
        backends (Consul, Kubernetes, Etcd).
"""

from .protocol_discovery_operations import ProtocolDiscoveryOperations

__all__ = ["ProtocolDiscoveryOperations"]
