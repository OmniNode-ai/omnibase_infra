# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One action archive ref that could not be filled."""

from pydantic import BaseModel, ConfigDict


class ModelActionArchiveCacheFailure(BaseModel):
    """A ref as written in the workflow, why it failed, and the detail."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ref: str
    reason_code: str
    detail: str = ""
