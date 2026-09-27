# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""A resolved bounded delegation route (OMN-18933).

Carried from the Codex draft omnibase_infra#3951. The resolved route adds the identities the pre-dispatch decision actually used,
so the K6 evidence can bind them: the runtime identity it matched, the selected
contract's topics, and the declaration's vendor and manifest hashes.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelBoundedDelegationRoute(BaseModel):
    """A route the pre-dispatch validator accepted, with every identity it read."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    lane: str = Field(min_length=1)
    broker: str = Field(min_length=1)
    runtime_environment: str = Field(min_length=1)
    runtime_bootstrap_servers: str = Field(min_length=1)
    consumer: str = Field(min_length=1)
    terminal_route: str = Field(min_length=1)
    repository_owner: str = Field(min_length=1)
    command_topic: str = Field(min_length=1)
    terminal_events: tuple[str, ...] = Field(min_length=2)
    declaration_sha256: str = Field(min_length=64, max_length=64)
    manifest_sha256: str = Field(min_length=64, max_length=64)
    declaration_source: str = Field(min_length=1)


__all__ = ["ModelBoundedDelegationRoute"]
