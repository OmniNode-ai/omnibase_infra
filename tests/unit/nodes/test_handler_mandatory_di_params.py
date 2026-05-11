# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Mandatory DI param enforcement tests for infra handlers (OMN-10278).

Asserts that injectable handler params are NOT optional — constructing
handlers without their required injectable params must raise TypeError,
not silently accept None and degrade.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit]


def test_handler_ledger_projection_requires_container() -> None:
    """HandlerLedgerProjection must require container, not default to None."""
    from omnibase_infra.nodes.node_ledger_projection_compute.handlers.handler_ledger_projection import (
        HandlerLedgerProjection,
    )

    with pytest.raises(TypeError):
        HandlerLedgerProjection()  # type: ignore[call-arg]


def test_handler_build_loop_projection_requires_container() -> None:
    """HandlerBuildLoopProjection must require container, not default to None."""
    from omnibase_infra.nodes.node_build_loop_projection_compute.handlers.handler_build_loop_projection import (
        HandlerBuildLoopProjection,
    )

    with pytest.raises(TypeError):
        HandlerBuildLoopProjection()  # type: ignore[call-arg]


def test_handler_runtime_lifecycle_requires_container() -> None:
    """HandlerRuntimeLifecycle must require container, not default to None."""
    from omnibase_infra.nodes.node_runtime_orchestrator.handlers.handler_runtime_lifecycle import (
        HandlerRuntimeLifecycle,
    )

    with pytest.raises(TypeError):
        HandlerRuntimeLifecycle()  # type: ignore[call-arg]


def test_handler_runtime_lifecycle_still_accepts_steps() -> None:
    """HandlerRuntimeLifecycle steps kwarg must remain optional (domain param)."""
    from unittest.mock import MagicMock

    from omnibase_infra.nodes.node_runtime_orchestrator.handlers.handler_runtime_lifecycle import (
        HandlerRuntimeLifecycle,
    )

    container = MagicMock()
    # container required, steps still optional
    handler = HandlerRuntimeLifecycle(container=container)
    assert handler is not None


def test_handler_runtime_error_triage_requires_event_bus() -> None:
    """HandlerRuntimeErrorTriage must require event_bus, not default to None."""
    from omnibase_infra.nodes.node_runtime_error_triage_effect.handlers.handler_runtime_error_triage import (
        HandlerRuntimeErrorTriage,
    )

    with pytest.raises(TypeError):
        HandlerRuntimeErrorTriage()  # type: ignore[call-arg]
