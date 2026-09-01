# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One direct-message claim against a pinned staged output."""

from __future__ import annotations

from pydantic import Field

from omnibase_infra.runtime.first_effect_ledger.model_first_effect_output_identity import (
    ModelFirstEffectOutputIdentity,
)


class ModelFirstEffectConsumerClaim(ModelFirstEffectOutputIdentity):
    """Exact direct-message identity plus the required optimistic version."""

    expected_ledger_version: int = Field(ge=0)


__all__ = ["ModelFirstEffectConsumerClaim"]
