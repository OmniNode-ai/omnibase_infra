# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Projector discovery result model.

Provides structured results for projector contract discovery operations,
containing both successfully loaded projectors and any validation errors.

Part of OMN-1168: ProjectorPluginLoader contract discovery loading.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omnibase_infra.models.projectors.model_projector_validation_error import (
        ModelProjectorValidationError,
    )
    from omnibase_infra.protocols.protocol_event_projector import (
        ProtocolEventProjector,
    )


class ModelProjectorDiscoveryResult:
    """Result of projector contract discovery.

    Contains successfully loaded projectors and any validation errors
    encountered during discovery.

    Attributes:
        projectors: List of successfully loaded projectors.
        validation_errors: List of errors for failed contracts.
    """

    def __init__(
        self,
        projectors: list[ProtocolEventProjector],
        validation_errors: list[ModelProjectorValidationError],
    ) -> None:
        """Initialize discovery result.

        Args:
            projectors: List of successfully loaded projectors.
            validation_errors: List of errors for failed contracts.
        """
        self.projectors = projectors
        self.validation_errors = validation_errors

    @property
    def success_count(self) -> int:
        """Number of successfully loaded projectors."""
        return len(self.projectors)

    @property
    def error_count(self) -> int:
        """Number of validation errors."""
        return len(self.validation_errors)


__all__ = ["ModelProjectorDiscoveryResult"]
