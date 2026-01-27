# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Operation bindings subcontract model for contract.yaml section.

This model represents the full operation_bindings section from a contract.yaml file,
containing pre-parsed bindings for all operations plus optional global bindings.

.. versionadded:: 0.2.6
    Created as part of OMN-1518 - Declarative operation bindings.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_infra.models.bindings.model_parsed_binding import ModelParsedBinding


class ModelOperationBindingsSubcontract(BaseModel):
    """Full operation_bindings section from contract.yaml.

    Contains pre-parsed bindings for all operations, plus optional
    global bindings applied to every operation.

    Example YAML:
        operation_bindings:
          version: { major: 1, minor: 0, patch: 0 }
          global_bindings:
            - parameter_name: "correlation_id"
              expression: "${envelope.correlation_id}"
          bindings:
            "db.query":
              - parameter_name: "sql"
                expression: "${payload.sql}"

    Attributes:
        version: Schema version for evolution tracking.
        bindings: Mapping of operation name to list of parsed bindings.
        global_bindings: Optional bindings applied to all operations (can be overridden).

    .. versionadded:: 0.2.6
    """

    version: ModelSemVer = Field(
        default_factory=lambda: ModelSemVer(major=1, minor=0, patch=0),
        description="Schema version for evolution tracking",
    )
    bindings: dict[str, list[ModelParsedBinding]] = Field(
        default_factory=dict,
        description="Operation name -> list of parsed bindings",
    )
    global_bindings: list[ModelParsedBinding] | None = Field(
        default=None,
        description="Bindings applied to all operations (can be overridden)",
    )

    model_config = {"frozen": True, "extra": "forbid"}
