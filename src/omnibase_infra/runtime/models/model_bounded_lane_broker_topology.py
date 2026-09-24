# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Declared listener pair and runtime environment of a bounded lane (OMN-18933).

Carried from the Codex draft omnibase_infra#3951, plus ``runtime_environment``:
the .201 dev lane's runtime reports ``ONEX_ENVIRONMENT=local`` on the compose
listener ``redpanda:9092`` (read live 2026-09-24), so the lane key ``dev`` alone
never identifies it. The declared pair does.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelBoundedLaneBrokerTopology(BaseModel):
    """Explicit external/internal broker identities for one declared lane."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    external_bootstrap_servers: str = Field(min_length=1)
    internal_bootstrap_servers: str = Field(min_length=1)
    runtime_environment: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _require_distinct_exact_identities(self) -> ModelBoundedLaneBrokerTopology:
        if self.external_bootstrap_servers == self.internal_bootstrap_servers:
            raise ValueError("broker topology identities must be distinct")
        return self


__all__ = ["ModelBoundedLaneBrokerTopology"]
