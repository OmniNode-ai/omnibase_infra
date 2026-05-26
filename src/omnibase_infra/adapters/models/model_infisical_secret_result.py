# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Infisical single secret result model.

.. versionadded:: 0.9.0
    Initial implementation for OMN-2286.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, SecretStr


class ModelInfisicalSecretResult(BaseModel):
    """Result of a single secret fetch from Infisical.

    Attributes:
        key: The secret name / key.
        value: The secret value wrapped in ``SecretStr``.
        version: The secret version (if available).
        secret_path: The path where the secret was found.
        environment: The environment slug.
    """

    model_config = ConfigDict(frozen=True)

    key: str
    value: SecretStr
    version: int | None = None
    secret_path: str = "/"
    environment: str = ""
