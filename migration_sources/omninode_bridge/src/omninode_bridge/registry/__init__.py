#!/usr/bin/env python3
"""
Node Registry package for omninode_bridge.

This package contains the node registry service for node discovery,
dual registration (Consul + PostgreSQL), and search capabilities.
"""

__all__ = [
    "NodeRegistryService",
    "SearchAPI",
]
