# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Private retained dev runtime configuration."""

from pydantic import BaseModel, ConfigDict, JsonValue


class ModelDevBootPoint(BaseModel):
    """Private, project-bound recovery configuration and retained image ids."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    project: str
    revision: str
    container_ids: tuple[str, ...]
    images: dict[str, str]
    compose: dict[str, JsonValue]
