# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""How far the code the dev deploy agent runs is behind the branch it tracks (OMN-23368)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

#: Journalled with the report so a log reader can count stale-pin deploys.
STALE_PIN_FRICTION = "code_ref_pin_stale"


class ModelDevLanePinStaleness(BaseModel):
    """Commits of the tracking ref the agent's running code does not contain."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pin_label: str = Field(min_length=1)
    dev_ref: str = Field(min_length=1)
    behind_by: int = Field(ge=0)

    def describe(self) -> str:
        noun = "commit" if self.behind_by == 1 else "commits"
        return (
            f"{self.pin_label} is behind {self.dev_ref} by {self.behind_by} {noun}; "
            f"the deploy agent's code lacks them while the compose it builds comes "
            f"from the deploy clone friction_type={STALE_PIN_FRICTION}"
        )
