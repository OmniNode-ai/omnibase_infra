# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Protocol for local provision effect node dependency injection.

Ticket: OMN-3495
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable
from uuid import UUID

if TYPE_CHECKING:
    from omnibase_core.models.core.model_deployment_topology import (
        ModelDeploymentTopology,
    )
    from omnibase_infra.nodes.node_setup_local_provision_effect.models.model_local_provision_effect_output import (
        ModelLocalProvisionEffectOutput,
    )


@runtime_checkable
class ProtocolProvisionEffect(Protocol):
    """Protocol for local service provisioning via Docker Compose.

    Implementations must start the local services defined in the topology
    and return an output indicating which services were started.
    """

    async def provision_local(
        self,
        topology: ModelDeploymentTopology,
        compose_file_path: str,
        correlation_id: UUID,
    ) -> ModelLocalProvisionEffectOutput:
        """Start local Docker Compose services.

        Args:
            topology: Deployment topology defining which services to start.
            compose_file_path: Path to the Docker Compose file.
            correlation_id: UUID for tracing across the setup workflow.

        Returns:
            ModelLocalProvisionEffectOutput with success flag and started services.
        """
        ...


__all__: list[str] = ["ProtocolProvisionEffect"]
