# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""What the build repository held for a request's git ref (OMN-26724)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ModelBuildRepositoryRefRequest(BaseModel):
    """Facts the caller read from the clone; the handler decides from them."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    git_ref: str
    #: The clone held the commit before any fetch.
    held: bool
    #: Exit code of the one fetch run on a miss; None when none ran.
    fetch_exit_code: int | None = None
    #: The clone held the commit after that fetch.
    held_after_fetch: bool = False
