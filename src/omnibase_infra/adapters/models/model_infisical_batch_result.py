# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Infisical batch secret result model.

.. versionadded:: 0.9.0
    Initial implementation for OMN-2286.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from omnibase_infra.adapters.models.model_infisical_secret_result import (
    ModelInfisicalSecretResult,
)


@dataclass
class ModelInfisicalBatchResult:
    """Result of a batch secret fetch.

    Note:
        This dataclass is intentionally **mutable** (not ``frozen=True``)
        because batch results are accumulated incrementally: each secret
        fetch appends to ``secrets`` or ``errors`` during the batch loop
        in ``AdapterInfisical.get_secrets_batch``.

    Attributes:
        secrets: Mapping of secret name to result.
        errors: Mapping of secret name to error message for failed fetches.
    """

    secrets: dict[str, ModelInfisicalSecretResult] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
