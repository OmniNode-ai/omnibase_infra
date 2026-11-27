# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runner TMPDIR ownership policy."""

import re

from pydantic import BaseModel, ConfigDict, Field


class ModelRunnerTmpPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    uid: int = Field(default=1001, ge=0)
    mode: str = "1777"
    pause_dir_mode: str = "755"
    job_dir_mode: str = "755"
    timeout_seconds: int = Field(default=60, gt=0)

    cache_target_pattern: str = r"/\.(?:npm|cache)(?:/|$)"

    def is_cache_target(self, target: str) -> bool:
        """OMN-26113: whether a container path is a cache mount (``~/.npm`` or under ``~/.cache``)."""
        return re.search(self.cache_target_pattern, target) is not None
