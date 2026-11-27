# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A runtime that answered /health running with its monitor degraded (OMN-20864)."""

from pydantic import BaseModel, ConfigDict


class ModelAnsweringDegraded(BaseModel):
    """A runtime that answered ``/health`` running, with its monitor degraded.

    OMN-20864. ``observed_at`` is the health monitor's own timestamp for the
    reading, so two probes inside one monitor cycle read as ONE observation.
    ``dimensions`` names every monitor dimension that is not ``HEALTHY``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    observed_at: str
    dimensions: frozenset[str]

    def describe(self) -> str:
        return (
            f"monitor DEGRADED on {', '.join(sorted(self.dimensions))} "
            f"(observed {self.observed_at or 'at an unread time'})"
        )
