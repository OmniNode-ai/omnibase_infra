# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""What one check or apply of the dev-lane branch installs found and did (OMN-17427)."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict

from omnibase_infra.enums.enum_dev_lane_branch_install_state import (
    EnumDevLaneBranchInstallState,
)
from omnibase_infra.models.model_dev_lane_branch_install_readback import (
    ModelDevLaneBranchInstallReadback,
)
from omnibase_infra.models.model_dev_lane_branch_install_release import (
    ModelDevLaneBranchInstallRelease,
)


class ModelDevLaneBranchInstallReceipt(BaseModel):
    """The readbacks of one pass, the installs it released and the containers it restarted."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    checked_at: datetime
    compose_project: str
    readbacks: tuple[ModelDevLaneBranchInstallReadback, ...] = ()
    released: tuple[ModelDevLaneBranchInstallRelease, ...] = ()
    restarted: tuple[str, ...] = ()

    @property
    def missing(self) -> tuple[ModelDevLaneBranchInstallReadback, ...]:
        """Installs that are not whole in the running containers."""
        return tuple(
            r
            for r in self.readbacks
            if r.state
            in (
                EnumDevLaneBranchInstallState.MISSING,
                EnumDevLaneBranchInstallState.FAILED,
            )
        )

    @property
    def ok(self) -> bool:
        """True when every declared install is whole."""
        return not self.missing
