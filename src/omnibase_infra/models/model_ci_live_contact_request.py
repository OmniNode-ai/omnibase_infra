# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Repository admission request for CI-tooling changes (OMN-18648)."""

from pydantic import BaseModel, ConfigDict


class ModelCILiveContactRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    repository: str = "."
    base_ref: str | None = None
    head_ref: str = "HEAD"
    pr_body: str = ""
