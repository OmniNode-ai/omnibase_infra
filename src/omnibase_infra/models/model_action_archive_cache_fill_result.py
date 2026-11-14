# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Outcome of one action archive cache fill run (OMN-24299)."""

from pydantic import BaseModel, ConfigDict

from omnibase_infra.models.model_action_archive_cache_failure import (
    ModelActionArchiveCacheFailure,
)


class ModelActionArchiveCacheFillResult(BaseModel):
    """success False when the config is absent or any ref failed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    success: bool
    staging_root: str
    filled: tuple[str, ...] = ()
    present: tuple[str, ...] = ()
    failed: tuple[ModelActionArchiveCacheFailure, ...] = ()
    error_message: str = ""
