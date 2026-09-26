# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""How a profile shows its own checks can fail.

Ticket: OMN-19565
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, model_validator

from omnibase_infra.lab_proof.enum_lab_proof_negative_control_kind import (
    EnumLabProofNegativeControlKind,
)


class ModelLabProofNegativeControl(BaseModel):
    """A proof that cannot fail proves nothing (plan section 6).

    ``sabotage_import`` runs the same steps with one line appended to
    ``sabotage_path`` in the subject tree that raises on import; the run must
    come back FAIL. ``base_rerun`` runs the PR's added tests at the merge base,
    where they must fail. ``none`` needs a reason.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: EnumLabProofNegativeControlKind
    sabotage_path: str | None = None
    reason: str = ""

    @model_validator(mode="after")
    def _shape(self) -> ModelLabProofNegativeControl:
        if self.kind is EnumLabProofNegativeControlKind.SABOTAGE_IMPORT:
            if not self.sabotage_path:
                raise ValueError("negative_control sabotage_import needs sabotage_path")
            if self.sabotage_path.startswith("/") or ".." in self.sabotage_path:
                raise ValueError(
                    "negative_control sabotage_path must be relative to the subject "
                    f"tree, got {self.sabotage_path!r}"
                )
        elif self.sabotage_path is not None:
            raise ValueError(
                f"sabotage_path is only valid for sabotage_import, not {self.kind}"
            )
        if self.kind is EnumLabProofNegativeControlKind.NONE and not self.reason:
            raise ValueError("negative_control none needs a reason")
        return self


__all__ = ["ModelLabProofNegativeControl"]
