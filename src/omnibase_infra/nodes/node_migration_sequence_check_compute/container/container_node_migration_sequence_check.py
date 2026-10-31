# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Binding site for NodeMigrationSequenceCheckCompute (OMN-21683).

``runtime_migration_sequence_check`` runs the migration sequence check
through ``NodeMigrationSequenceCheckCompute``. This module is the one place that builds it, so the
command-line entry point takes the handler from here and never constructs it by
hand.
"""

from __future__ import annotations

from omnibase_infra.nodes.node_migration_sequence_check_compute.handler import (
    NodeMigrationSequenceCheckCompute,
)

__all__: list[str] = ["bind_node_migration_sequence_check"]


def bind_node_migration_sequence_check() -> NodeMigrationSequenceCheckCompute:
    """Build the the migration sequence check COMPUTE handler."""
    return NodeMigrationSequenceCheckCompute()
