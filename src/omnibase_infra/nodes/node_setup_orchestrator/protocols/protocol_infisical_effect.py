# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Protocol for Infisical setup effect node dependency injection.

Ticket: OMN-3495
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from omnibase_core.models.core.model_deployment_topology import (
        ModelDeploymentTopology,
    )
    from omnibase_infra.nodes.node_setup_infisical_effect.models.model_infisical_setup_effect_output import (
        ModelInfisicalSetupEffectOutput,
    )


@runtime_checkable
class ProtocolInfisicalEffect(Protocol):
    """Protocol for Infisical secret store provisioning.

    Implementations must bootstrap the Infisical service defined in the
    topology and return an output indicating whether setup succeeded or was
    skipped.
    """

    async def setup_infisical(
        self,
        topology: ModelDeploymentTopology,
        correlation_id: object,
    ) -> ModelInfisicalSetupEffectOutput:
        """Bootstrap Infisical.

        Args:
            topology: Deployment topology; infisical service config is read from it.
            correlation_id: UUID for tracing across the setup workflow.

        Returns:
            ModelInfisicalSetupEffectOutput with status (completed, skipped, failed).
        """
        ...


__all__: list[str] = ["ProtocolInfisicalEffect"]
