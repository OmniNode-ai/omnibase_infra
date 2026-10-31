# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Binding site for NodeMigrationFreezeCheckCompute (OMN-21683).

``runtime_migration_freeze_check`` runs the migration freeze check
through ``NodeMigrationFreezeCheckCompute``. This module is the one place that builds it, so the
command-line entry point takes the handler from here and never constructs it by
hand.
"""

from __future__ import annotations

from omnibase_infra.nodes.node_migration_freeze_check_compute.handler import (
    NodeMigrationFreezeCheckCompute,
)

__all__: list[str] = ["bind_node_migration_freeze_check"]


def bind_node_migration_freeze_check() -> NodeMigrationFreezeCheckCompute:
    """Build the the migration freeze check COMPUTE handler."""
    return NodeMigrationFreezeCheckCompute()
