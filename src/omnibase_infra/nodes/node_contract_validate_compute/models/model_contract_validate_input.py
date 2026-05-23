# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Input model for contract validation compute."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelContractValidateInput(BaseModel):
    """Input for contract YAML, file, or model-compliance validation."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    contract_content: str | None = Field(
        default=None,
        description="YAML contract content to validate.",
    )
    contract_type: Literal["effect", "compute", "reducer", "orchestrator"] = Field(
        default="effect",
        description="Contract type used for schema validation.",
    )
    model_code: str | None = Field(
        default=None,
        description="Optional Pydantic model source for model compliance validation.",
    )
    file_path: str | Path | None = Field(
        default=None,
        description="Optional contract file path for file-based validation.",
    )
    base_dir: Path | None = Field(
        default=None,
        description="Optional base directory used to constrain file_path.",
    )

    @model_validator(mode="after")
    def validate_payload_shape(self) -> ModelContractValidateInput:
        """Require enough input for exactly one validation mode."""
        has_content = self.contract_content is not None
        has_file = self.file_path is not None
        if not has_content and not has_file:
            raise ValueError("contract_content or file_path is required")
        if self.model_code is not None and not has_content:
            raise ValueError("model_code validation requires contract_content")
        return self


__all__ = ["ModelContractValidateInput"]
