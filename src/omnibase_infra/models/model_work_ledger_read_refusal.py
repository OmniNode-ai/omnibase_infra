# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""What a refused work-ledger read adds to the CLI's error line."""

from pydantic import BaseModel, ConfigDict

from omnibase_infra.cli.delegate_lane import DelegateLaneSelectionError
from omnibase_infra.errors.error_infra import WorkLedgerQueryError


class ModelWorkLedgerReadRefusal(BaseModel):
    """The message a refused read may print after its exception type.

    A ``WorkLedgerQueryError`` message is written by the client or is the node's own
    refusal sentence (naming its bound and the bounded form of the request); none
    carries a driver detail, because the client turns a database error into a fixed
    sentence. A ``DelegateLaneSelectionError`` message names only the lane
    declaration path and the variables to set (ONEX_WORKSPACE_CONFIG_ROOT,
    OMNIBASE_PATH); it never carries a credential. Both are safe to print. Any
    other exception adds nothing, since a broker or driver message may hold a
    credential.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    message: str | None = None

    @classmethod
    def of(cls, exc: Exception) -> "ModelWorkLedgerReadRefusal":
        if isinstance(exc, WorkLedgerQueryError):
            return cls(message=exc.message)
        if isinstance(exc, DelegateLaneSelectionError):
            return cls(message=str(exc))
        return cls(message=None)

    def __str__(self) -> str:
        return "" if self.message is None else f": {self.message}"
