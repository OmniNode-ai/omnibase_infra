# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Contract-owned image maintenance policy (OMN-19998)."""

from pydantic import BaseModel, ConfigDict, Field


class ModelDeployAgentImagePrunePolicy(BaseModel):
    """One source-label predicate scopes both inventory and deletion."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    minimum_age_hours: int = Field(gt=0)
    source_label: str
    source_value: str
    docker_timeout_seconds: float = Field(gt=0)
    inspect_batch_size: int = Field(gt=0)
    unit_names: tuple[str, str]
    unit_source_directory: str
    user_home_path: str
    unit_directory: str
    environment_file: str
