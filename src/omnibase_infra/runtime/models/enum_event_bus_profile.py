# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Event-bus validation profile for per-runtime configuration (OMN-17304).

The OMN-17304 operator ruling replaced the blanket "INMEMORY is forbidden in
omnibase_infra" rule with a PROFILE AXIS on ``ModelEventBusConfig``: rejecting
the in-memory bus is correct for lane-profile runtimes (the deployed
dev/stability/judge/prod containers, where an in-memory bus silently discards
shared-substrate evidence) and wrong for local-profile runtimes (a laptop CLI's
embedded runtime, an offline/standalone install), where the in-memory bus is a
first-class configured value AND the shipped tier-0 default.

This is deliberately NOT ``RUNTIME_PROFILE`` / ``ModelRuntimeProfile``
(``runtime_profile.py``): that axis encodes topic-ownership ROLE identity
(main / effects / workers / ...) plus secret-prefetch policy and must never be
repurposed to carry transport-validation semantics. This enum is scoped to one
question only: which transports may this runtime's ``event_bus.type`` legally
declare?
"""

from __future__ import annotations

from enum import Enum, unique

__all__ = ["EnumEventBusProfile"]


@unique
class EnumEventBusProfile(str, Enum):
    """Validation profile for a runtime's declared event-bus transport.

    Values:
        LANE: A deployed lane runtime (dev / stability / judge / prod
            containers). Only production-safe transports (``kafka`` /
            ``cloud``) are accepted — an in-memory bus in a lane silently
            strands evidence outside the shared projections. This is the
            DEFAULT: an ``event_bus`` block that does not declare a profile
            validates exactly as strictly as it did before the axis existed
            (fail-closed).
        LOCAL: A local runtime (the ``onex delegate`` CLI's embedded runtime,
            an offline/standalone install, a developer laptop). Any supported
            transport is accepted, including ``inmemory`` — the shipped
            tier-0 default configuration is exactly this profile.
    """

    LANE = "lane"
    LOCAL = "local"

    def __str__(self) -> str:
        """Return the plain value so log lines read ``lane``, not the repr."""
        return self.value
