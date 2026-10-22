# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Request to read one dev lane pin from the operator env store (OMN-18475)."""

from pydantic import BaseModel, ConfigDict


class ModelDevLanePinReadRequest(BaseModel):
    """Name the store and the key. The store's path has no default (rule 8)."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    env_file: str
    key: str
    docker_cmd: str = "docker"
