# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One directory prefix a dev-lane branch install may write under (OMN-25536)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, field_validator


class ModelDevLaneBranchInstallRoot(BaseModel):
    """One directory prefix a branch install may write under."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str
    overlay_root: bool

    @field_validator("path")
    @classmethod
    def _absolute_directory_prefix(cls, value: str) -> str:
        if (
            not value.startswith("/")
            or not value.endswith("/")
            or len(value) < 2
            or "\x00" in value
            or any(part in ("", ".", "..") for part in value[1:-1].split("/"))
        ):
            raise ValueError(
                f"root {value!r} must be an absolute directory ending in '/' "
                "with no empty, '.' or '..' segment"
            )
        return value


__all__: list[str] = ["ModelDevLaneBranchInstallRoot"]
