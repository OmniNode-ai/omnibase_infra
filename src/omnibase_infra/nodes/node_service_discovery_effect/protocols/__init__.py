# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Protocols for Service Discovery Effect Node.

This module exports protocols for the service discovery effect node:

Protocols:
    ProtocolServiceDiscoveryHandler: Protocol for pluggable service discovery
        backends (Consul, Kubernetes, Etcd).
"""

from .protocol_service_discovery_handler import ProtocolServiceDiscoveryHandler

__all__ = ["ProtocolServiceDiscoveryHandler"]
