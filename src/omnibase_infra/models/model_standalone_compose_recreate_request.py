# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One signed request to recreate one service of one standalone compose file."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelStandaloneComposeRecreateRequest(BaseModel):
    """The wire body of the standalone compose recreate command.

    The deploy agent checks ``compose_file`` and ``service`` against the node contract's
    declared targets and ``runtime_lane`` against its allowed lanes before it touches docker.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: UUID
    requested_by: str = Field(min_length=1)
    runtime_lane: str = Field(min_length=1)
    compose_file: str = Field(min_length=1)
    service: str = Field(min_length=1)
    git_ref: str = Field(min_length=1)
