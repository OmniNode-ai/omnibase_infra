# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A recorded sha256 of one systemd unit file, to read the file as changed or untouched (OMN-26106)."""

from __future__ import annotations

import hashlib
import re

from pydantic import BaseModel, ConfigDict, Field, field_validator

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ModelUnitFileDigest(BaseModel):
    """The digest a unit file had when it was recorded.

    The prod deploy agent's unit file and drop-in must not move when a fix to the
    dev agent installs. This record is the "before" read; :meth:`reads_as_changed`
    is the "after" read against the file's bytes.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    name: str = Field(min_length=1)
    sha256: str

    @field_validator("sha256")
    @classmethod
    def _sha256_shape(cls, value: str) -> str:
        if not _SHA256.match(value):
            raise ValueError("sha256 must be 64 lowercase hex characters")
        return value

    def reads_as_changed(self, content: bytes) -> bool:
        """Whether ``content`` is not the bytes this digest was recorded from."""
        return hashlib.sha256(content).hexdigest() != self.sha256
