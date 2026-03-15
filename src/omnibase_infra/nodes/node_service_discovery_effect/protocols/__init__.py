# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Protocols for Service Discovery Effect Node.

This module exports protocols for the service discovery effect node:

Protocols:
    ProtocolDiscoveryOperations: Protocol for pluggable service discovery
        backends (Consul, Kubernetes, Etcd).
"""

from .protocol_discovery_operations import ProtocolDiscoveryOperations

__all__ = ["ProtocolDiscoveryOperations"]
