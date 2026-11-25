# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The outcome of one overdue dev merge build publish (OMN-26316)."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class ModelDevMergeBuildPublishResult(BaseModel):
    """``requested`` after the event and the build request were sent; ``skipped`` or ``failed`` otherwise."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: Literal["requested", "skipped", "failed"]
    reason: Literal[
        "published",
        "not_overdue",
        "already_requested",
        "no_pull_request_number",
        "send_refused",
        "send_failed",
    ]
    request_id: UUID | None = None
