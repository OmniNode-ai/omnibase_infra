# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Request to fill the action archive cache (OMN-24299)."""

from pydantic import BaseModel, ConfigDict


class ModelActionArchiveCacheFillRequest(BaseModel):
    """Where the config and clones live, and whether to write anything."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    config_path: str
    clone_root: str
    dry_run: bool = False
