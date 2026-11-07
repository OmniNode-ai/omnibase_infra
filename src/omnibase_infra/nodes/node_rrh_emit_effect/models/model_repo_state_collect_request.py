# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Request to collect git repository state."""

from pydantic import BaseModel, ConfigDict


class ModelRepoStateCollectRequest(BaseModel):
    """Repository path to inspect."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    repo_path: str
