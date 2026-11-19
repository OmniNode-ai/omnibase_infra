# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One file of a dev-lane branch install (OMN-17427)."""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, field_validator

from omnibase_infra.models.model_dev_lane_branch_install_roots import (
    load_dev_lane_branch_install_roots,
)

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ModelDevLaneBranchInstallFile(BaseModel):
    """Where one file lands in the container and what it must hash to."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str
    sha256: str

    @field_validator("path")
    @classmethod
    def _path_under_a_declared_root(cls, value: str) -> str:
        """A file under a root ``dev_lane_branch_install_roots.yaml`` declares."""
        root = load_dev_lane_branch_install_roots().root_of(value)
        if (
            root is None
            or len(value) <= len(root.path)
            or value.endswith("/")
            or "\x00" in value
            or any(part in ("", ".", "..") for part in value.split("/")[1:])
        ):
            declared = ", ".join(
                r.path for r in load_dev_lane_branch_install_roots().roots
            )
            raise ValueError(
                f"container path {value!r} must be a file under one of {declared} "
                "with no empty, '.' or '..' segment"
            )
        return value

    def prepare_argv(self, container: str) -> tuple[list[str], ...]:
        """Commands to run in the container before the copy: the directory of an overlay root.

        ``docker cp`` needs the parent directory, and an overlay root is not in the image.
        """
        root = load_dev_lane_branch_install_roots().root_of(self.path)
        if root is None or not root.overlay_root:
            return ()
        parent = self.path.rpartition("/")[0]
        return (["docker", "exec", "-u", "0", container, "mkdir", "-p", "--", parent],)

    @field_validator("sha256")
    @classmethod
    def _sha256_hex(cls, value: str) -> str:
        if not SHA256_PATTERN.match(value):
            raise ValueError("sha256 must be 64 lowercase hex characters")
        return value
