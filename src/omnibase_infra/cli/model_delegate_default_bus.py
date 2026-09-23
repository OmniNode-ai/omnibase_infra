# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The transport ``onex delegate`` resolves when ``--bus`` is omitted.

OMN-19193. The embedded runtime's configuration answers two questions at once:
which transport, and -- for a shared bus -- which declared lane that bus is.
They are answered by one configuration, so they travel as one value. Split,
the CLI could take the transport from the workspace's tier-1 config and then
refuse for want of a lane the same file already named, which is the defect
that left every default workspace delegation on the in-memory bus.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["ModelDelegateDefaultBus"]


class ModelDelegateDefaultBus(BaseModel):
    """The configured transport, its lane, and which authority answered."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    bus: str = Field(description="Resolved transport: 'kafka' or 'inmemory'")
    reason: str = Field(description="Provenance naming the configuration that answered")
    lane: str | None = Field(
        default=None,
        description=(
            "The declared lane the configuration binds a shared bus to; None "
            "when the configuration names none"
        ),
    )
