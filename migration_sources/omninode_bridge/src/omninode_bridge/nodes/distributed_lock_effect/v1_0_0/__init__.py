#!/usr/bin/env python3
"""
NodeDistributedLockEffect v1.0.0 - ONEX v2.0 Compliant.

PostgreSQL-backed distributed locking Effect node for multi-instance deployments.
"""

from .node import NodeDistributedLockEffect

__version__ = "1.0.0"

__all__ = ["NodeDistributedLockEffect"]
