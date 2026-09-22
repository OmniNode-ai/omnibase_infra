# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Typed external and internal identities for a bounded lane broker."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelBoundedLaneBrokerTopology(BaseModel):
    """Explicit external/internal identities for one declared broker lane."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    external_bootstrap_servers: str = Field(min_length=1)
    internal_bootstrap_servers: str = Field(min_length=1)

    @model_validator(mode="after")
    def _require_distinct_exact_identities(self) -> ModelBoundedLaneBrokerTopology:
        if self.external_bootstrap_servers == self.internal_bootstrap_servers:
            raise ValueError("broker topology identities must be distinct")
        return self
