# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Infisical single secret result model.

.. versionadded:: 0.9.0
    Initial implementation for OMN-2286.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import SecretStr


@dataclass(frozen=True)
class ModelInfisicalSecretResult:
    """Result of a single secret fetch from Infisical.

    Attributes:
        key: The secret name / key.
        value: The secret value wrapped in ``SecretStr``.
        version: The secret version (if available).
        secret_path: The path where the secret was found.
        environment: The environment slug.
    """

    key: str
    value: SecretStr
    version: int | None = None
    secret_path: str = "/"
    environment: str = ""
