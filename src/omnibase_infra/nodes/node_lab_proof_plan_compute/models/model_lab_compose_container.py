# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The Compose fields used by desired-state generation."""

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator

from omnibase_infra.nodes.node_lab_proof_plan_compute.models.model_lab_compose_build import (
    ModelLabComposeBuild,
)


class ModelLabComposeContainer(BaseModel):
    """Other Compose fields are covered by Compose's hash, never approximated."""

    model_config = ConfigDict(extra="ignore", frozen=True)
    container_name: str | None = None
    image: str = ""
    build: ModelLabComposeBuild | None = None
    command: list[str] = Field(default_factory=list)
    healthcheck: dict[str, JsonValue] = Field(default_factory=dict)
    environment: dict[str, str | None] = Field(default_factory=dict)

    @field_validator("command", mode="before")
    @classmethod
    def normalize_command(cls, value: object) -> object:
        """Compose null inherits the image command; scalar commands stay intact."""
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return value

    @field_validator("build", mode="before")
    @classmethod
    def normalize_build(cls, value: object) -> object:
        """Compose's short build form is a context path."""
        return {"context": value} if isinstance(value, str) else value
