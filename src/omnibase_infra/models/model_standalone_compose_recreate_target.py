# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One compose file and service the standalone recreate route may touch."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelStandaloneComposeRecreateTarget(BaseModel):
    """A declared target of ``node_standalone_compose_recreate_effect``.

    ``container_key`` is the compose service's fixed container name, the one container the
    recreate may change.
    ``env_file_variable`` names the environment variable, on the host that serves the target,
    that holds the path of the host's own compose env file. An agent whose environment lacks it
    does not serve the target.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    compose_file: str = Field(min_length=1)
    service: str = Field(min_length=1)
    container_key: str = Field(min_length=1)
    env_file_variable: str = Field(min_length=1)
