# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Terminal result for the lab host table copy of one runner host."""

from pydantic import BaseModel, ConfigDict


class ModelRunnerLabHostTableResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    success: bool
    # Absolute path on the runner host that RUNNER_LAB_HOST_TABLE now names; empty on a refusal.
    path: str = ""
    # RUNNER-LAB-HOST-TABLE-<kind> host=<host> [path=<path>]: <reason>
    error_message: str | None = None
