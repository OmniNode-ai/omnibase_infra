# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One contract's measured dispatch cost over one reading window (OMN-26250).

A runtime member attaches many contracts and shares one event loop between
them. To split the contracts across members by what they cost, each contract
needs a number: how long its dispatches held the member, how many there were
and how many bytes they carried. Every value on the reading is a wall-clock or
counted measurement taken at the dispatch seam of the member that attached the
contract. The reading names the window it covers; a reading with no window, or
a window that ends before it starts, is refused.

A contract that was attached and dispatched nothing in the window still has a
reading, with count and times of zero: a zero that was counted is different
from a contract nobody read.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelContractCostReading(BaseModel):
    """Dispatch count, wall time on and off the event loop, and bytes of one contract."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    contract_name: str = Field(
        ..., min_length=1, description="Name of the attached contract."
    )
    window_start: datetime = Field(
        ..., description="Start of the window the reading covers (UTC)."
    )
    window_end: datetime = Field(
        ..., description="End of the window the reading covers (UTC)."
    )
    dispatch_count: int = Field(
        ...,
        ge=0,
        description="Dispatches of this contract's handlers that finished in the window.",
    )
    on_loop_wall_ms: float = Field(
        ...,
        ge=0.0,
        description=(
            "Wall time of those dispatches spent on the event loop, in "
            "milliseconds: the time from the dispatcher's start to its return "
            "minus the time its plain-function entry method ran on a worker "
            "thread. An async handler's cooperative awaits count here."
        ),
    )
    off_loop_wall_ms: float = Field(
        ...,
        ge=0.0,
        description=(
            "Wall time those dispatches ran a plain-function entry method on a "
            "worker thread (OMN-24491), in milliseconds."
        ),
    )
    payload_bytes: int = Field(
        ...,
        ge=0,
        description=(
            "Bytes of the bus messages those dispatches consumed. A dispatch "
            "that did not arrive as a bus message adds none."
        ),
    )

    @model_validator(mode="after")
    def _validate_window(self) -> ModelContractCostReading:
        if self.window_end <= self.window_start:
            raise ValueError("window_end must be after window_start")
        return self


__all__: list[str] = ["ModelContractCostReading"]
