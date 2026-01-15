# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Protocols for Service Discovery Effect Node.

This module exports protocols for the service discovery effect node:

Protocols:
    ProtocolHandlerServiceDiscovery: Protocol for pluggable service discovery
        backends (Consul, Kubernetes, Etcd).
"""

from .protocol_handler_service_discovery import ProtocolHandlerServiceDiscovery

__all__ = ["ProtocolHandlerServiceDiscovery"]
