# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Local verification spec model for standalone use.

When omnibase_core >= 0.29.0 is available, prefer
ModelStepVerification from omnibase_core.models.orchestrator.
"""

from pydantic import BaseModel, Field


class ModelVerificationSpec(BaseModel):
    """Local verification spec for use without omnibase_core dependency.

    Duck-typed compatible with ModelStepVerification from omnibase_core.
    """

    check_type: str = Field(description="Type of verification check to perform")
    target: str = Field(description="Check-specific target value")
    timeout_seconds: int = Field(
        default=10, description="Maximum time for the check in seconds"
    )


__all__ = ["ModelVerificationSpec"]
