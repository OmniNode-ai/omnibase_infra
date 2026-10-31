# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Contract-declared timeouts of the observability services (OMN-21623).

The values live in the ``timeouts`` block of
``services/observability/observability_timeouts.yaml``. The model has no field
defaults, so a block that omits a value fails to load rather than falling back to
a literal.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError

_TIMEOUTS_PATH = (
    Path(__file__).resolve().parents[2]
    / "services"
    / "observability"
    / "observability_timeouts.yaml"
)


class ModelObservabilityTimeouts(BaseModel):
    """Waits and intervals that the observability consumers and writers run under.

    Attributes:
        query_seconds: PostgreSQL statement timeout of the writers and
            the reader.
        batch_query_seconds: Statement timeout of the batch
            compute services.
        reset_seconds: Seconds an open writer circuit
            waits before it half-opens.
        stop_seconds: Wait for the LLM cost consumer to stop on
            shutdown.
        warn_every_polls: Consecutive empty polls between
            warnings in the LLM cost consumer.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    query_seconds: float = Field(gt=0)
    batch_query_seconds: float = Field(gt=0)
    reset_seconds: float = Field(gt=0)
    stop_seconds: float = Field(gt=0)
    warn_every_polls: int = Field(gt=0)

    @classmethod
    def from_file(cls, timeouts_path: Path) -> ModelObservabilityTimeouts:
        """Load the timeouts from the ``timeouts`` mapping of a YAML file.

        Raises:
            ProtocolConfigurationError: If the file has no ``timeouts`` mapping.
        """
        with timeouts_path.open(encoding="utf-8") as handle:
            document = yaml.safe_load(handle) or {}
        block = document.get("timeouts") if isinstance(document, dict) else None
        if not isinstance(block, dict):
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.FILESYSTEM,
                operation="observability_timeouts",
            )
            raise ProtocolConfigurationError(
                f"{timeouts_path.name} has no timeouts mapping",
                context=context,
            )
        return cls.model_validate(block)


@cache
def observability_timeouts() -> ModelObservabilityTimeouts:
    """Return the timeouts declared beside the observability services."""
    return ModelObservabilityTimeouts.from_file(_TIMEOUTS_PATH)


__all__: list[str] = ["ModelObservabilityTimeouts", "observability_timeouts"]
