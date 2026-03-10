# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Protocol for preflight effect node dependency injection.

Ticket: OMN-3495
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from omnibase_infra.nodes.node_setup_preflight_effect.models.model_preflight_effect_output import (
        ModelPreflightEffectOutput,
    )


@runtime_checkable
class ProtocolPreflightEffect(Protocol):
    """Protocol for running platform preflight checks.

    Implementations must run all required preflight checks and return
    a result indicating whether the system is ready for provisioning.
    """

    async def run_preflight(self, correlation_id: object) -> ModelPreflightEffectOutput:
        """Run all preflight checks.

        Args:
            correlation_id: UUID for tracing across the setup workflow.

        Returns:
            ModelPreflightEffectOutput with pass/fail status and per-check results.
        """
        ...


__all__: list[str] = ["ProtocolPreflightEffect"]
