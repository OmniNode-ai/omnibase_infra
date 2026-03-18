# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Env var validator for the infrastructure catalog."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ValidationResult:
    """Result of env var validation."""

    ok: bool
    missing: list[str] = field(default_factory=list)


def validate_env(required: set[str]) -> ValidationResult:
    """Check that all required env vars are set in the current environment."""
    missing = [var for var in sorted(required) if not os.environ.get(var)]
    return ValidationResult(ok=len(missing) == 0, missing=missing)
