# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Test-only fixture handler that reports its injected event bus (OMN-20381).

This module is NOT production code. It extends ``HandlerCorrelatedNoop`` to
expose the concrete bus type supplied by the real timed runtime in the terminal
response. Production handlers MUST NOT import from or depend on this module.
"""

from __future__ import annotations

from tests.fixtures.handler_correlated_noop import (
    HandlerCorrelatedNoop,
    ModelCorrelatedNoopRequest,
    ModelDelegateSkillFixtureTerminal,
)

__all__ = ["HandlerEventBusProbe", "ModelCorrelatedNoopRequest"]


class HandlerEventBusProbe(HandlerCorrelatedNoop):
    """Report the injected bus while preserving the correlated terminal."""

    def __init__(self, event_bus: object) -> None:
        self._event_bus = event_bus

    def handle(
        self, request: ModelCorrelatedNoopRequest
    ) -> ModelDelegateSkillFixtureTerminal:
        return (
            super()
            .handle(request)
            .model_copy(update={"response": type(self._event_bus).__name__})
        )
