# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Registry for NodeReleaseIdentityCompute — DI bindings and exports.

Provides factory methods and dependency injection bindings for the
NodeReleaseIdentityCompute, following the ONEX registry pattern with the naming
convention ``RegistryInfra<NodeName>``.

Ticket: OMN-14471
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_infra.nodes.node_release_identity_compute.node import (
    NodeReleaseIdentityCompute,
)

if TYPE_CHECKING:
    from omnibase_core.container import ModelONEXContainer


class RegistryInfraReleaseIdentityCompute:
    """DI registry for the release-identity compute node.

    Provides factory methods and bindings for NodeReleaseIdentityCompute.
    """

    @staticmethod
    def get_node_class() -> type[NodeReleaseIdentityCompute]:
        """Return the node class for DI resolution."""
        return NodeReleaseIdentityCompute

    @staticmethod
    def create_node(
        container: ModelONEXContainer,
    ) -> NodeReleaseIdentityCompute:
        """Create a NodeReleaseIdentityCompute instance with the given container.

        Args:
            container: ONEX dependency injection container.

        Returns:
            Configured NodeReleaseIdentityCompute instance.
        """
        return NodeReleaseIdentityCompute(container)


__all__: list[str] = [
    "NodeReleaseIdentityCompute",
    "RegistryInfraReleaseIdentityCompute",
]
