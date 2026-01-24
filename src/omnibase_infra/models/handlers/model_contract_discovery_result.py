# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Contract Discovery Result Model for Graceful Error Handling.

This module provides ModelContractDiscoveryResult, which holds the results of
contract discovery with graceful error handling. Part of OMN-1097
HandlerContractSource filesystem discovery.

When graceful_mode is enabled on HandlerContractSource, this model holds both
successfully discovered descriptors and any validation errors encountered.

See Also:
    - HandlerContractSource: Source that produces these results
    - ModelHandlerDescriptor: Individual descriptor for discovered handlers
    - ModelHandlerValidationError: Structured error model for validation failures

.. versionadded:: 0.6.2
    Created as part of OMN-1097 filesystem handler discovery.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.handlers.model_handler_descriptor import (
    ModelHandlerDescriptor,
)

if TYPE_CHECKING:
    from omnibase_infra.models.errors import ModelHandlerValidationError


class ModelContractDiscoveryResult(BaseModel):
    """Result of contract discovery with graceful error handling.

    When graceful_mode is enabled, this model holds both successfully
    discovered descriptors and any validation errors encountered.

    Attributes:
        descriptors: List of successfully discovered handler descriptors.
        validation_errors: List of validation errors for failed contracts.

    Example:
        >>> result = ModelContractDiscoveryResult(
        ...     descriptors=[descriptor1, descriptor2],
        ...     validation_errors=[error1],
        ... )
        >>> len(result.descriptors)
        2
        >>> len(result.validation_errors)
        1

    .. versionadded:: 0.6.2
        Created as part of OMN-1097 filesystem handler discovery.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
    )

    descriptors: list[ModelHandlerDescriptor] = Field(
        default_factory=list,
        description="Successfully discovered handler descriptors",
    )
    validation_errors: list[ModelHandlerValidationError] = Field(
        default_factory=list,
        description="Validation errors for failed contracts",
    )


# Forward Reference Resolution:
# This model uses TYPE_CHECKING to defer import of ModelHandlerValidationError.
# model_rebuild() is called in runtime modules that import ModelHandlerValidationError
# (e.g., handler_contract_source.py, handler_bootstrap_source.py, registry_contract_source.py).
# Each module calls model_rebuild() at module level after importing both the model
# and the forward-referenced type. This is safe because model_rebuild() is idempotent.

__all__ = ["ModelContractDiscoveryResult"]
