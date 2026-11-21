# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""What one check or apply of the dev-lane branch installs found and did (OMN-17427)."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, model_validator

from omnibase_infra.enums.enum_dev_lane_branch_install_hold import (
    EnumDevLaneBranchInstallHold,
)
from omnibase_infra.enums.enum_dev_lane_branch_install_state import (
    EnumDevLaneBranchInstallState,
)
from omnibase_infra.models.model_dev_lane_branch_install_held import (
    ModelDevLaneBranchInstallHeld,
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
    held: tuple[ModelDevLaneBranchInstallHeld, ...] = ()
    """Each install that is not whole, with the reason ``status.json`` shows (OMN-26300)."""

    @model_validator(mode="before")
    @classmethod
    def _derive_held(cls, data: object) -> object:
        """Name every install the readbacks show as not whole, unless the caller named them."""
        if not isinstance(data, dict) or "held" in data:
            return data
        readbacks = (
            ModelDevLaneBranchInstallReadback.model_validate(r)
            for r in data.get("readbacks", ())
        )
        return {**data, "held": tuple(_held_of(r) for r in readbacks if not _whole(r))}

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


def _whole(readback: ModelDevLaneBranchInstallReadback) -> bool:
    return readback.state in (
        EnumDevLaneBranchInstallState.PRESENT,
        EnumDevLaneBranchInstallState.REAPPLIED,
    )


def _held_of(
    readback: ModelDevLaneBranchInstallReadback,
) -> ModelDevLaneBranchInstallHeld:
    detail = next((f.detail for f in readback.files if f.detail), "")
    failed = readback.state is EnumDevLaneBranchInstallState.FAILED
    what = "apply failed" if failed else "not in the running containers"
    return ModelDevLaneBranchInstallHeld(
        id=readback.install_key,
        code=(
            EnumDevLaneBranchInstallHold.APPLY_FAILED
            if failed
            else EnumDevLaneBranchInstallHold.NOT_APPLIED
        ),
        reason=f"{what} ({readback.state.value})" + (f": {detail}" if detail else ""),
    )
