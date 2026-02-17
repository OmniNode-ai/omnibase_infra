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
        requirements: List of individual config requirements.
        transport_types: Deduplicated set of transport types discovered.
        contract_paths: Paths of contracts that were scanned.
        errors: Errors encountered during extraction (non-fatal).
    """

    # Not frozen: uses mutable list fields for merge() aggregation during
    # contract scanning.  Other infra models are frozen for thread safety,
    # but this model is built up incrementally and never shared across threads.
    model_config = ConfigDict(extra="forbid", from_attributes=True)

    requirements: list[ModelConfigRequirement] = Field(
        default_factory=list,
        description="Individual config requirements.",
    )
    transport_types: list[EnumInfraTransportType] = Field(
        default_factory=list,
        description="Deduplicated transport types discovered.",
    )
    contract_paths: list[Path] = Field(
        default_factory=list,
        description="Contract paths that were scanned.",
    )
    errors: list[str] = Field(
        default_factory=list,
        description="Non-fatal extraction errors.",
    )

    def merge(self, other: ModelConfigRequirements) -> ModelConfigRequirements:
        """Merge another requirements set into this one.

        Returns a new instance with combined requirements, deduped
        transport types, and concatenated errors.
        """
        all_reqs = [*self.requirements, *other.requirements]
        all_types = list(dict.fromkeys([*self.transport_types, *other.transport_types]))
        all_paths = list(dict.fromkeys([*self.contract_paths, *other.contract_paths]))
        all_errors = [*self.errors, *other.errors]
        return ModelConfigRequirements(
            requirements=all_reqs,
            transport_types=all_types,
            contract_paths=all_paths,
            errors=all_errors,
        )
