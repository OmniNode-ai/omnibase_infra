# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""What the stability integration refresh is asked to do (OMN-26715)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelStabilityIntegrationRefreshRequest(BaseModel):
    """The integration date and the stability lane the build is recreated on.

    The branch of each repository is ``integration/launch-<integration_date>``.
    ``health_window_seconds`` is the time ``health_url`` has to read 200 after
    the recreate; past it the recorded images are restored. The repositories,
    services, URL and window come from the caller (the contract's overlay), never
    from a default.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    integration_date: str = Field(pattern=r"^[0-9]{8}$")
    repositories: tuple[str, ...]
    compose_project: str
    services: tuple[str, ...]
    health_url: str
    health_window_seconds: int = Field(gt=0)
    record_dir: str
