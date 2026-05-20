# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Shared PostgreSQL adapter protocol marker for contract dependency metadata."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ProtocolPostgresAdapter(Protocol):
    """Marker protocol for PostgreSQL-backed effect adapters.

    Effect node contracts use this shared module path when the concrete handler
    dependency is an async PostgreSQL adapter or pool supplied by runtime DI.
    Node-specific protocols should live with their node package.
    """


__all__ = ["ProtocolPostgresAdapter"]
