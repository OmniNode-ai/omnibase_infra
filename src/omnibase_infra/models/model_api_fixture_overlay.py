# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Private DI overlay: sanctioned targets, request recipes and lifecycle operations."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.model_api_fixture_count_probe import (
    ModelApiFixtureCountProbe,
)
from omnibase_infra.models.model_api_fixture_http_step import ModelApiFixtureHttpStep
from omnibase_infra.models.model_api_fixture_route import ModelApiFixtureRoute


class ModelApiFixtureOverlay(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    target_kind: Literal["dev", "throwaway_slot"]
    sanctioned: bool = False
    targets: dict[str, str]
    # Ticket-scoped throwaway name, the shape the target's throwaway-account
    # reaper matches, so a resource left by a failed cleanup is still reaped.
    scope_prefix: str = Field(pattern=r"^omn\d{4,6}-[a-z0-9][a-z0-9-]{7,63}$")
    private_output_directory: str = Field(min_length=1)
    routes: tuple[ModelApiFixtureRoute, ...]
    counts: tuple[ModelApiFixtureCountProbe, ...]
    seeds: tuple[ModelApiFixtureHttpStep, ...] = ()
    cleanup: tuple[ModelApiFixtureHttpStep, ...]
    approved_body_keys: tuple[str, ...]
    timeout_seconds: float = Field(default=20, gt=0, le=120)
