# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Binding site for NodeMigrationAppendOnlyCheckCompute (OMN-21683).

``runtime_migration_append_only_check`` runs the append-only migration check
through ``NodeMigrationAppendOnlyCheckCompute``. This module is the one place that builds it, so the
command-line entry point takes the handler from here and never constructs it by
hand.
"""

from __future__ import annotations

from omnibase_infra.nodes.node_migration_append_only_check_compute.handler import (
    NodeMigrationAppendOnlyCheckCompute,
)

__all__: list[str] = ["bind_node_migration_append_only_check"]


def bind_node_migration_append_only_check() -> NodeMigrationAppendOnlyCheckCompute:
    """Build the the append-only migration check COMPUTE handler."""
    return NodeMigrationAppendOnlyCheckCompute()
