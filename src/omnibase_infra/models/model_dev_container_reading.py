# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One docker inspect reading of the dev runtime container (OMN-25912)."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelDevContainerReading(BaseModel):
    """The container as ``docker inspect`` reports it at one tick.

    ``container_digest`` changes when compose recreates the container; ``started_at`` and
    ``restart_count`` change when the same container starts again. ``image_digest`` is the
    image the container came up on and ``image_ref`` the reference its config names.
    ``first_healthy_at`` is read from docker's health log, which keeps only the last five
    checks, so it is a reading of a container seen while still young.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    container: str = Field(min_length=1)
    container_digest: str = Field(min_length=1)
    started_at: datetime
    restart_count: int = Field(ge=0)
    exit_code: int | None = None
    image_digest: str = ""
    image_ref: str = ""
    #: The image's ``org.opencontainers.image.revision`` label: the omnibase_infra commit it was built from (OMN-26703).
    image_revision: str = ""
    #: Docker's health status (``starting``, ``healthy``, ``unhealthy``) and the end of the first passing check in its log.
    health_status: str = ""
    first_healthy_at: datetime | None = None

    @field_validator("started_at", "first_healthy_at")
    @classmethod
    def _aware(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.utcoffset() is None:
            raise ValueError("started_at and first_healthy_at must carry a UTC offset")
        return value
