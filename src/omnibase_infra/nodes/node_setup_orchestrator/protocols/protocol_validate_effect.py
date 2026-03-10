# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Protocol for setup validate effect node dependency injection.

Ticket: OMN-3495
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from omnibase_core.models.core.model_deployment_topology import (
        ModelDeploymentTopology,
    )
    from omnibase_infra.nodes.node_setup_validate_effect.models.model_setup_validate_effect_output import (
        ModelSetupValidateEffectOutput,
    )


@runtime_checkable
class ProtocolValidateEffect(Protocol):
    """Protocol for post-provision service health validation.

    Implementations must perform health checks on all services defined
    in the topology and return aggregated pass/fail results.
    """

    async def validate_services(
        self,
        topology: ModelDeploymentTopology,
        correlation_id: object,
    ) -> ModelSetupValidateEffectOutput:
        """Run health checks on all deployed services.

        Args:
            topology: Deployment topology defining which services to validate.
            correlation_id: UUID for tracing across the setup workflow.

        Returns:
            ModelSetupValidateEffectOutput with all_healthy flag and per-node results.
        """
        ...


__all__: list[str] = ["ProtocolValidateEffect"]
