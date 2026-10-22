# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Outcome of one dev lane pin read or set (OMN-18475)."""

from pydantic import BaseModel, ConfigDict

from omnibase_infra.enums.enum_dev_lane_pin_outcome import EnumDevLanePinOutcome


class ModelDevLanePinResult(BaseModel):
    """What one read or set found and did.

    A refusal carries its reason and leaves the store and the ledger untouched.
    ``ledger_row`` is the exact row a write appended.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    outcome: EnumDevLanePinOutcome
    key: str
    env_file: str
    value: str | None = None
    pin_before: str | None = None
    pin_after: str | None = None
    resident: bool | None = None
    image_id: str | None = None
    backup_path: str | None = None
    ledger_row: str | None = None
    reason: str | None = None
