# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The containers the C28 check reads on the lane it grades (OMN-21485)."""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field

#: The compose project every lab lane's name is formed from: the base project
#: itself, or the base followed by ``-<lane>``.
BASE_COMPOSE_PROJECT = "omnibase-infra"
_PROJECT_RE = re.compile(rf"^{BASE_COMPOSE_PROJECT}(?P<suffix>(?:-[a-z0-9]+)*)$")


class ModelConsumerFlowLane(BaseModel):
    """One lane's runtime, projection-api, broker and Postgres containers.

    The names follow the lane naming the compose overlays share: a lane whose
    compose project is ``omnibase-infra-<lane>`` names its runtime
    ``omninode-<lane>-runtime``, its projection API
    ``omnimarket-<lane>-projection-api`` and its infrastructure
    ``omnibase-infra-<lane>-<service>``. The compose project comes from the lane
    the C28 workflow resolved, so the check reads that lane's containers and no
    other's.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    compose_project: str = Field(min_length=1)
    runtime_containers: tuple[str, ...] = Field(min_length=1)
    projection_api_container: str = Field(min_length=1)
    broker_container: str = Field(min_length=1)
    postgres_container: str = Field(min_length=1)

    @classmethod
    def from_compose_project(cls, compose_project: str) -> ModelConsumerFlowLane:
        """Derive the lane's containers from its compose project; refuse any other shape."""
        match = _PROJECT_RE.match(compose_project)
        if match is None:
            raise ValueError(
                f"compose project {compose_project!r} is not {BASE_COMPOSE_PROJECT} "
                f"or {BASE_COMPOSE_PROJECT}-<lane>"
            )
        suffix = match.group("suffix")
        return cls(
            compose_project=compose_project,
            runtime_containers=(
                f"omninode{suffix}-runtime",
                f"omninode{suffix}-runtime-effects",
            ),
            projection_api_container=f"omnimarket{suffix}-projection-api",
            broker_container=f"{compose_project}-redpanda",
            postgres_container=f"{compose_project}-postgres",
        )

    @property
    def boot_containers(self) -> tuple[str, ...]:
        """The containers whose identity marks one boot of the lane."""
        return (*self.runtime_containers, self.projection_api_container)


__all__ = ["BASE_COMPOSE_PROJECT", "ModelConsumerFlowLane"]
