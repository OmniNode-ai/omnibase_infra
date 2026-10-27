# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Dev broker bounds authorized by OPERATOR-CONSENT 2026-10-27T06:37:48Z."""

from __future__ import annotations

import re
from fractions import Fraction

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelDevBrokerMemoryPolicy(BaseModel):
    """A private runtime-policy overlay may change only these dev broker values."""

    model_config = ConfigDict(extra="forbid", frozen=True, from_attributes=True)

    dev_redpanda_memory: str = Field(min_length=1, max_length=32)
    dev_redpanda_mem_limit: str = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def _consented_memory_bounds(self) -> ModelDevBrokerMemoryPolicy:
        sizes: list[Fraction] = []
        for value in (self.dev_redpanda_memory, self.dev_redpanda_mem_limit):
            match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([kmgt]?)i?b?", value, re.I)
            if not match:
                raise ValueError("malformed dev broker memory size")
            sizes.append(
                Fraction(match[1]) * (1 << (10 * " kmgt".index(match[2].lower())))
            )
        arena, limit = sizes
        if not (arena == 12 * 1024**3 and limit == 16 * 1024**3):
            raise ValueError(
                "dev broker policy violates OPERATOR-CONSENT 2026-10-27T06:37:48Z: "
                "retain the measured 12 GiB arena floor and the authorized 16 GiB container cap"
            )
        return self
