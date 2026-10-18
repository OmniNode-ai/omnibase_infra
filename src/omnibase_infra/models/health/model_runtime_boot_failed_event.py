# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runtime boot-failed event model — OMN-18550.

Published when a runtime process exits with a failing code before it ever
served, and written to the durable boot-failure ledger beside the counter that
bounds the restart loop. The event is the record of the 2026-09-16 incident
shape: a runtime that died and was relaunched by its restart policy every forty
seconds for hours, with the cause readable only in a log nobody was reading.

Topic: ``onex.evt.omnibase-infra.runtime-boot-failed.v1``
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ModelRuntimeBootFailedEvent(BaseModel):
    """One failing runtime exit, with its cause and the running streak count."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    service: str = Field(
        ...,
        description="Runtime identity the streak is counted for (the consumer group id).",
    )
    container_ref: str = Field(
        ...,
        description="Container hostname the failing process ran in.",
    )
    runtime_source_hash: str = Field(
        ...,
        description=(
            "RUNTIME_SOURCE_HASH baked into the image. A new image starts a new "
            "streak, so a redeploy of a fixed build is never held by the old count."
        ),
    )
    exit_code: int = Field(..., description="Exit code of the failing process.")
    cause: str = Field(
        ...,
        description="Sanitized cause: the last error the kernel logged, or the preflight exit.",
    )
    uptime_seconds: float = Field(
        ..., description="Seconds the process ran before it exited."
    )
    consecutive_failures: int = Field(
        ...,
        ge=1,
        description="Failing exits in a row, this one included.",
    )
    max_consecutive_failures: int = Field(
        ...,
        ge=1,
        description="Declared bound: the streak length at which retrying stops.",
    )
    retries_exhausted: bool = Field(
        ...,
        description="True when this failure reached the bound and the runtime stopped retrying.",
    )
    failed_at: datetime = Field(..., description="UTC time of the failing exit.")


__all__: list[str] = ["ModelRuntimeBootFailedEvent"]
