# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""migration_freeze_check COMPUTE node package (OMN-20568).

Exposes :class:`NodeMigrationFreezeCheckCompute`, which decides from explicit
inputs (the ``.migration_freeze`` text, today's date and the added or renamed
paths of a diff) whether the migration freeze is violated or expired, and
returns the canonical OMN-2362 ``ModelValidationReport``.
"""

from omnibase_infra.nodes.node_migration_freeze_check_compute.handler import (
    NodeMigrationFreezeCheckCompute,
)

__all__ = ["NodeMigrationFreezeCheckCompute"]
