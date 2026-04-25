# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Top-level envelope for contracts/llm_endpoints.yaml + YAML loader.

First typed wrapper — runtime cutover (service_kernel.py) deferred.
See OMN-9750. DO NOT wire into service_kernel.py in this PR.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.contracts.model_llm_endpoint_entry import (
    ModelLlmEndpointEntry,
)


class ModelLlmEndpointContract(BaseModel):
    """Top-level envelope for contracts/llm_endpoints.yaml.

    First typed wrapper — runtime cutover deferred (OMN-9750).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    endpoints: list[ModelLlmEndpointEntry] = Field(
        default_factory=list,
        description="Ordered list of LLM inference endpoint slots.",
    )

    def running(self) -> list[ModelLlmEndpointEntry]:
        """Return only slots with status=running."""
        from omnibase_infra.models.contracts.enum_llm_endpoint_status import (
            EnumLlmEndpointStatus,
        )

        return [e for e in self.endpoints if e.status == EnumLlmEndpointStatus.RUNNING]

    def by_role(self, role: str) -> list[ModelLlmEndpointEntry]:
        """Return all slots matching a given role."""
        return [e for e in self.endpoints if e.role == role]


def load_llm_endpoint_contract(yaml_path: Path) -> ModelLlmEndpointContract:
    """Load and validate llm_endpoints.yaml into ModelLlmEndpointContract.

    Raises:
        FileNotFoundError: if yaml_path does not exist.
        pydantic.ValidationError: if the YAML content fails schema validation.
    """
    raw = yaml.safe_load(yaml_path.read_text())
    return ModelLlmEndpointContract.model_validate(raw)


__all__ = [
    "ModelLlmEndpointContract",
    "ModelLlmEndpointEntry",
    "load_llm_endpoint_contract",
]
