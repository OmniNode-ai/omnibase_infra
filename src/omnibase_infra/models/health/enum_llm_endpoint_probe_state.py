# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Classified probe state for a single LLM endpoint.

Distinguishes *transient* unavailability (retry at full cadence) from
*terminal* authentication problems (a wrong or missing credential is not an
outage and must not be retried at outage cadence).

Ticket: OMN-16900
"""

from __future__ import annotations

from enum import StrEnum


class EnumLlmEndpointProbeState(StrEnum):
    """Outcome classification for an LLM endpoint health probe.

    Members:
        HEALTHY: The endpoint answered a probe with a 2xx status.
        UNAVAILABLE: Transient failure — connection error, timeout, or a
            non-auth non-2xx status. Retried at the normal probe cadence and
            tracked by the per-endpoint circuit breaker.
        AUTH_FAILED: Sustained 401/403. The credential is wrong or revoked;
            this is terminal, not transient, so the endpoint is moved to
            exponential backoff-to-idle instead of the fixed probe cadence.
        SKIPPED_NO_AUTH: The endpoint declares an auth secret that is absent
            or unresolvable in this environment. Classified once at
            construction and **never probed**.
        CIRCUIT_OPEN: The per-endpoint circuit breaker is open, so the probe
            was short-circuited this cycle.
    """

    HEALTHY = "HEALTHY"
    UNAVAILABLE = "UNAVAILABLE"
    AUTH_FAILED = "AUTH_FAILED"
    SKIPPED_NO_AUTH = "SKIPPED_NO_AUTH"
    CIRCUIT_OPEN = "CIRCUIT_OPEN"


__all__ = ["EnumLlmEndpointProbeState"]
