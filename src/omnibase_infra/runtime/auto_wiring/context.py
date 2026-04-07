# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Auto-wiring context model for lifecycle hooks.

Provides the structured context that lifecycle hooks receive when invoked.
The context carries references to the container, contract metadata, and
wiring state without exposing internal engine details.

.. versionadded:: 0.35.0
    Created as part of OMN-7655 (Contract lifecycle hooks).
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

# ONEX_EXCLUDE: dict_str_any - extensible contract metadata from YAML; schema-free by design
_MetadataDict = dict[str, object]


class ModelAutoWiringContext(BaseModel):
    """Context provided to lifecycle hook callables during auto-wiring.

    This model is passed as the sole argument to each lifecycle hook.
    It provides the hook with enough context to acquire resources,
    validate preconditions, or release resources without coupling to
    the wiring engine internals.

    Lifecycle hooks receive this context and must not:
        - Mutate the routing manifest
        - Register topics outside the contract
        - Access engine internals beyond this context

    Attributes:
        handler_id: The contract handler_id for the node being wired.
        node_kind: The node kind (COMPUTE, EFFECT, REDUCER, ORCHESTRATOR).
        contract_version: Semantic version of the contract.
        phase: Current lifecycle phase (on_start, validate_handshake, on_shutdown).
        services: Dict of named services available to the hook.
            Populated by the wiring engine from the container.
        metadata: Additional contract metadata passed through from YAML.
    """

    model_config = ConfigDict(extra="forbid")

    handler_id: str = Field(
        ...,
        min_length=1,
        description="Contract handler_id for the node being wired",
    )
    node_kind: str = Field(
        ...,
        min_length=1,
        description="Node kind (COMPUTE, EFFECT, REDUCER, ORCHESTRATOR)",
    )
    contract_version: str = Field(
        default="0.0.0",
        description="Semantic version of the contract",
    )
    phase: str = Field(
        ...,
        min_length=1,
        description="Current lifecycle phase",
    )
    services: _MetadataDict = Field(
        default_factory=dict,
        description="Named services available to the hook from the container",
    )
    metadata: _MetadataDict = Field(
        default_factory=dict,
        description="Additional contract metadata passed through from YAML",
    )


__all__ = ["ModelAutoWiringContext"]
