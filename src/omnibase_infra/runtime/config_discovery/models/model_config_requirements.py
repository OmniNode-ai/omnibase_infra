# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Aggregated configuration requirements from one or more ONEX contracts.

.. versionadded:: 0.10.0
    Created as part of OMN-2287.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.runtime.config_discovery.models.model_config_requirement import (
    ModelConfigRequirement,
)


class ModelConfigRequirements(BaseModel):
    """Aggregated configuration requirements from one or more contracts.

    Attributes:
        requirements: Individual config requirements.
        transport_types: Deduplicated transport types discovered.
        contract_paths: Paths of contracts that were scanned.
        errors: Errors encountered during extraction (non-fatal).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    requirements: tuple[ModelConfigRequirement, ...] = Field(
        default_factory=tuple,
        description="Individual config requirements.",
    )
    transport_types: tuple[EnumInfraTransportType, ...] = Field(
        default_factory=tuple,
        description="Deduplicated transport types discovered.",
    )
    contract_paths: tuple[Path, ...] = Field(
        default_factory=tuple,
        description="Contract paths that were scanned.",
    )
    errors: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Non-fatal extraction errors.",
    )

    def merge(self, other: ModelConfigRequirements) -> ModelConfigRequirements:
        """Merge another requirements set into this one.

        Returns a new frozen instance with combined requirements,
        deduplicated transport types and contract paths, and concatenated
        errors. Neither ``self`` nor ``other`` is mutated.

        Args:
            other: The requirements to merge into this set.

        Returns:
            A new ``ModelConfigRequirements`` containing the union of both
            sets' requirements, transport types (deduplicated, order-preserved),
            contract paths (deduplicated, order-preserved), and errors.
        """
        all_reqs = (*self.requirements, *other.requirements)
        all_types = tuple(
            dict.fromkeys((*self.transport_types, *other.transport_types))
        )
        all_paths = tuple(dict.fromkeys((*self.contract_paths, *other.contract_paths)))
        all_errors = (*self.errors, *other.errors)
        return ModelConfigRequirements(
            requirements=all_reqs,
            transport_types=all_types,
            contract_paths=all_paths,
            errors=all_errors,
        )
