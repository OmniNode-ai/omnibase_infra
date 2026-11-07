# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Declared stack policy and source snapshots supplied by the validation caller."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelStackShapeSnapshot(BaseModel):
    """Pure input for the command extension of T1's stack-shape check.

    The caller gathers files; the compute handler performs no filesystem or broker I/O.
    Base snapshots let the first command baseline be checked against the base's actual
    commands, and later command baselines can only shrink.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    policy_yaml: str
    compose_files: dict[str, str]
    base_policy_yaml: str | None = None
    base_compose_files: dict[str, str] = Field(default_factory=dict)
    package_scripts: dict[str, dict[str, str]] = Field(default_factory=dict)
