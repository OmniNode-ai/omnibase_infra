# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One start that shows a dev recreate violation (OMN-25912)."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict

from omnibase_infra.models.model_dev_runtime_started import JobId


class ModelDevRecreateViolationRow(BaseModel):
    """One start that shows the violation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    job_id: JobId
    at: datetime
    requested_by: str
    announced: bool
