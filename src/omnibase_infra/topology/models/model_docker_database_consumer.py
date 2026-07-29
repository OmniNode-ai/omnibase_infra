# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Typed Docker consumer of topology-owned database semantics."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = ["ModelDockerDatabaseConsumer"]


class ModelDockerDatabaseConsumer(BaseModel):
    """One Docker catalog consumer of topology-owned database semantics."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    bindings: tuple[str, ...] = Field(default=())
    physical_database_envs: tuple[str, ...] = Field(default=())

    @model_validator(mode="after")
    def has_a_database_projection(self) -> ModelDockerDatabaseConsumer:
        """Reject inert entries which validate no binding or physical name."""
        if not self.bindings and not self.physical_database_envs:
            raise ValueError(
                "Docker database consumers require a binding or physical_database_env"
            )
        return self
