# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The holder of a ``lane-work`` lease: the lane a ledger CLAIM names (OMN-21428)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

_CELL = r"^[^|\s=]+$"


class ModelLaneWorkHolder(BaseModel):
    """``{host, lane, run, principal}`` plus the ticket; serialized into the lease holder.

    Every value is one ledger cell value: no ``|``, ``=`` or whitespace.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    host: str = Field(pattern=_CELL)
    lane: str = Field(pattern=_CELL)
    run: str = Field(pattern=_CELL)
    principal: str = Field(pattern=_CELL)
    ticket: str = Field(pattern=_CELL)
    parent: str = Field(default="", pattern=r"^[^|\s=]*$")

    @property
    def resource(self) -> str:
        """The leased resource: one ``lane-work`` grant per lane."""
        return f"lane:{self.lane}"

    def holder(self) -> str:
        """The lease ``holder`` string this lane acquires with."""
        return self.model_dump_json()

    @classmethod
    def parse(cls, holder: str) -> ModelLaneWorkHolder:
        return cls.model_validate_json(holder)


__all__ = ["ModelLaneWorkHolder"]
