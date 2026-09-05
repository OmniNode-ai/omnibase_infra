# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed declaration for an optional host-directory bind mount."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)  # internal-dataclass-ok: docker-catalog-internal
class ModelOptionalDirectoryBindMount:
    """A read-only host directory mounted only when its source env var is set.

    The catalog generator validates a configured source as an existing absolute
    directory before it renders the compose interpolation.  An unset source is
    intentionally absent from the rendered service instead of being replaced by
    a file sentinel, which prevents Docker from binding a file onto a directory
    target.
    """

    source_env: str
    container_path: str
    read_only: bool = True

    def __post_init__(self) -> None:
        if not self.source_env:
            raise ValueError(
                "optional directory bind mount source_env must not be empty"
            )
        if not self.container_path.startswith("/"):
            raise ValueError(
                "optional directory bind mount container_path must be absolute"
            )
