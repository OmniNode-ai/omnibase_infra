# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Infisical batch secret result model.

.. versionadded:: 0.9.0
    Initial implementation for OMN-2286.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.adapters.models.model_infisical_secret_result import (
    ModelInfisicalSecretResult,
)


class ModelInfisicalBatchResult(BaseModel):
    """Result of a batch secret fetch.

    Note:
        This model is intentionally **not** frozen because batch results are
        accumulated incrementally: each secret fetch appends to ``secrets``
        or ``errors`` during the batch loop in
        ``AdapterInfisical.get_secrets_batch``.

    Attributes:
        secrets: Mapping of secret name to result.
        errors: Mapping of secret name to error message for failed fetches.
    """

    model_config = ConfigDict(frozen=False)

    secrets: dict[str, ModelInfisicalSecretResult] = Field(default_factory=dict)
    errors: dict[str, str] = Field(default_factory=dict)
