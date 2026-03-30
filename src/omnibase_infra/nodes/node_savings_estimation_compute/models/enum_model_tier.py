# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Model tier enum and pricing constants for savings estimation.

Pricing constants reflect Anthropic's published rates as of 2026-03:
    - Claude Opus 4: $15/M input tokens, $75/M output tokens
    - Claude Sonnet 4: $3/M input tokens, $15/M output tokens

Related Tickets:
    - OMN-6964: Token savings emitter
"""

from __future__ import annotations

from enum import StrEnum


class EnumModelTier(StrEnum):
    """Supported model tiers for pricing."""

    OPUS = "opus"
    SONNET = "sonnet"


# Prices per million tokens
MODEL_PRICING_INPUT: dict[EnumModelTier, float] = {
    EnumModelTier.OPUS: 15.0,  # $15 / 1M input tokens
    EnumModelTier.SONNET: 3.0,  # $3 / 1M input tokens
}

MODEL_PRICING_OUTPUT: dict[EnumModelTier, float] = {
    EnumModelTier.OPUS: 75.0,  # $75 / 1M output tokens
    EnumModelTier.SONNET: 15.0,  # $15 / 1M output tokens
}

TOKENS_PER_MILLION: float = 1_000_000.0

# Current pricing manifest version for reproducibility
PRICING_MANIFEST_VERSION: str = "anthropic-2026-03"


__all__: list[str] = [
    "EnumModelTier",
    "MODEL_PRICING_INPUT",
    "MODEL_PRICING_OUTPUT",
    "TOKENS_PER_MILLION",
    "PRICING_MANIFEST_VERSION",
]
