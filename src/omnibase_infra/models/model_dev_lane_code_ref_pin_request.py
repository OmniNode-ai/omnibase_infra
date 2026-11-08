# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One operation on the dev deploy agent's code-ref pin (OMN-23039)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from omnibase_infra.models.model_dev_lane_code_ref_pin import ModelDevLaneCodeRefPin


class ModelDevLaneCodeRefPinRequest(BaseModel):
    """Register, read or release the pin kept in ``state_dir``.

    ``register`` carries the pin to write; ``active`` carries the instant to judge
    expiry against, injected by the caller; ``release`` carries neither.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    operation: Literal["register", "active", "release"]
    state_dir: Path
    pin: ModelDevLaneCodeRefPin | None = None
    now: datetime | None = None

    @model_validator(mode="after")
    def _operation_inputs(self) -> ModelDevLaneCodeRefPinRequest:
        if self.operation == "register" and self.pin is None:
            raise ValueError("register needs a pin")
        if self.operation == "active" and self.now is None:
            raise ValueError("active needs the instant to judge expiry against")
        if self.operation == "active" and self.now is not None:
            if self.now.tzinfo is None:
                raise ValueError("now must carry a timezone")
        return self
