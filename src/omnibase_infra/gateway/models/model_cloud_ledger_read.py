# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""ModelCloudLedgerRead -- one operator read of the cloud ledger (OMN-17205).

The typed answer ``onex ledger read`` prints and a goal row cites. Every field
is present on every verdict: a caller that has to branch on which keys exist
before it can report anything is a caller that will report the wrong thing when
the shape it did not expect arrives.

``exit_code`` is a derived property rather than a stored field so the mapping
from verdict to process status has exactly one authority. Only ``FOUND`` is 0 --
"the row is not there" is a probe failure, not a successful empty read, which is
the whole reason the row exists.

Ticket: OMN-17205
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.types import JsonType
from omnibase_infra.enums.enum_cloud_ledger_verdict import EnumCloudLedgerVerdict

__all__ = ["ModelCloudLedgerRead"]

# Distinct non-zero codes so a shell caller can branch without parsing JSON.
_EXIT_CODES: dict[EnumCloudLedgerVerdict, int] = {
    EnumCloudLedgerVerdict.FOUND: 0,
    EnumCloudLedgerVerdict.NOT_FOUND: 1,
    EnumCloudLedgerVerdict.PROJECTION_ABSENT: 2,
    EnumCloudLedgerVerdict.UNAUTHENTICATED: 3,
    EnumCloudLedgerVerdict.UNAVAILABLE: 4,
}


class ModelCloudLedgerRead(BaseModel):
    """The result of asking the cloud ledger about one correlation id."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    verdict: EnumCloudLedgerVerdict
    correlation_id: str = Field(min_length=1)
    projection: str = Field(default="", description="Projection the server named.")
    url: str = Field(
        default="",
        description="The route actually called, resolved from the stored base URL.",
    )
    http_status: int = Field(
        default=0, description="0 when the server was never reached."
    )
    count: int = Field(default=0, ge=0)
    rows: list[dict[str, JsonType]] = Field(default_factory=list)
    detail: str = Field(
        default="",
        description=(
            "Operator-facing explanation and remediation. Never carries a "
            "credential, a token, or a response body that could hold one."
        ),
    )

    @property
    def exit_code(self) -> int:
        """Process exit status for this verdict. Only FOUND is 0."""
        return _EXIT_CODES[self.verdict]
