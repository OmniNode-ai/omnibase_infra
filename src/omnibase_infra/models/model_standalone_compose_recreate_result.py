# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The outcome of one standalone compose recreate, with the container ids before and after."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums.enum_standalone_compose_recreate_outcome import (
    EnumStandaloneComposeRecreateOutcome,
)


class ModelStandaloneComposeRecreateResult(BaseModel):
    """Result of one request.

    ``containers_before`` and ``containers_after`` map every container name on the host to its
    id, read around the recreate, so the readback shows that only the one service changed.
    ``changed`` lists the container names whose id differs, appeared or vanished.
    ``carried_files`` lists the repo-relative files whose bytes the recreate rewrote in the host's
    tree from the requested git ref before it ran compose.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: UUID
    outcome: EnumStandaloneComposeRecreateOutcome
    message: str
    service_container_before: str = ""
    service_container_after: str = ""
    containers_before: dict[str, str] = Field(default_factory=dict)
    containers_after: dict[str, str] = Field(default_factory=dict)
    changed: tuple[str, ...] = ()
    carried_files: tuple[str, ...] = ()
