# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""What a refused work-ledger read adds to the CLI's error line."""

from pydantic import BaseModel, ConfigDict

from omnibase_infra.errors.error_infra import WorkLedgerQueryError


class ModelWorkLedgerReadRefusal(BaseModel):
    """The message a refused read may print after its exception type.

    Every ``WorkLedgerQueryError`` message is written by the client or is the node's
    own refusal sentence (naming its bound and the bounded form of the request);
    none carries a driver detail, because the client turns a database error into a
    fixed sentence. So it is safe to print. Any other exception adds nothing, since a
    broker or driver message may hold a credential.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    message: str | None = None

    @classmethod
    def of(cls, exc: Exception) -> "ModelWorkLedgerReadRefusal":
        return cls(
            message=exc.message if isinstance(exc, WorkLedgerQueryError) else None
        )

    def __str__(self) -> str:
        return "" if self.message is None else f": {self.message}"
