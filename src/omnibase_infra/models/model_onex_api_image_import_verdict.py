# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""onex-api pin import check: what the check found and did (OMN-23039)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ModelOnexApiImageImportVerdict(BaseModel):
    """The verdict on one candidate image.

    ``importable`` is the only pass. ``rolled_back`` says the advanced pin was
    written back; it is False on a pass, and False on a failure whose rollback
    could not be done, which ``reason`` then says in words.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    importable: bool
    rolled_back: bool
    reason: str
    candidate: str
    pin_after: str

    def record_fields(self) -> dict[str, object]:
        """The fields a refused delivery adds to the executor's record.

        ``result`` is ``REFUSED``, the repoint script's own word for "nothing
        was delivered", so the typed job record reads it as the failure it is.
        """
        return {
            "result": "REFUSED",
            "reason": self.reason,
            "rejected_candidate": self.candidate,
            "pin_after": self.pin_after,
            "tag_advanced": False,
            "recreated": False,
        }
