# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One file of a dev-lane branch install (OMN-17427)."""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, field_validator

#: Where a file may be installed: the application tree of the runtime image.
CONTAINER_ROOT = "/app/"

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ModelDevLaneBranchInstallFile(BaseModel):
    """Where one file lands in the container and what it must hash to."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str
    sha256: str

    @field_validator("path")
    @classmethod
    def _path_in_app_tree(cls, value: str) -> str:
        if (
            not value.startswith(CONTAINER_ROOT)
            or len(value) <= len(CONTAINER_ROOT)
            or value.endswith("/")
            or "\x00" in value
            or any(part in ("", ".", "..") for part in value.split("/")[1:])
        ):
            raise ValueError(
                f"container path {value!r} must be a file under {CONTAINER_ROOT} "
                "with no empty, '.' or '..' segment"
            )
        return value

    @field_validator("sha256")
    @classmethod
    def _sha256_hex(cls, value: str) -> str:
        if not SHA256_PATTERN.match(value):
            raise ValueError("sha256 must be 64 lowercase hex characters")
        return value
