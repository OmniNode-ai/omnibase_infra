# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Registry module for NodeLedgerProjectionCompute.

This module provides the RegistryInfraLedgerProjection class for
dependency injection registration and factory methods.

Usage:
    ```python
    from omnibase_infra.nodes.node_ledger_projection_compute.registry import (
        RegistryInfraLedgerProjection,
    )

    # Get node class for DI resolution
    node_class = RegistryInfraLedgerProjection.get_node_class()

    # Create node instance
    node = RegistryInfraLedgerProjection.create_node(container)
    ```

Related Tickets:
    - OMN-1648: Ledger Projection Compute Node
"""

from omnibase_infra.nodes.node_ledger_projection_compute.registry.registry_infra_ledger_projection import (
    RegistryInfraLedgerProjection,
)

__all__ = ["RegistryInfraLedgerProjection"]
