# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Test NodeNodeGraphReducer FSM transitions (pure-function pattern) [OMN-6349]."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.mark.unit
def test_fsm_initial_state() -> None:
    """Default state must be 'initializing'."""
    from omnibase_infra.nodes.node_node_graph_reducer.models.model_node_graph_state import (
        ModelNodeGraphState,
    )

    state = ModelNodeGraphState()
    assert state.fsm_state == "initializing"


@pytest.mark.unit
def test_fsm_transition_initializing_to_wiring() -> None:
    """registry_ready event transitions initializing -> wiring."""
    from omnibase_infra.nodes.node_node_graph_reducer.models.model_node_graph_state import (
        ModelNodeGraphState,
    )
    from omnibase_infra.nodes.node_node_graph_reducer.reducer import (
        NodeGraphReducer,
    )

    reducer = NodeGraphReducer()
    state = ModelNodeGraphState(fsm_state="initializing")
    event = MagicMock()
    event.event_type = "registry_ready"
    output = reducer.reduce(state, event, event_metadata={})
    assert output.result.fsm_state == "wiring"


@pytest.mark.unit
def test_fsm_transition_wiring_to_running() -> None:
    """wiring_complete event transitions wiring -> running."""
    from omnibase_infra.nodes.node_node_graph_reducer.models.model_node_graph_state import (
        ModelNodeGraphState,
    )
    from omnibase_infra.nodes.node_node_graph_reducer.reducer import (
        NodeGraphReducer,
    )

    reducer = NodeGraphReducer()
    state = ModelNodeGraphState(fsm_state="wiring")
    event = MagicMock()
    event.event_type = "wiring_complete"
    output = reducer.reduce(state, event, event_metadata={})
    assert output.result.fsm_state == "running"


@pytest.mark.unit
def test_fsm_fatal_error_from_any_state() -> None:
    """fatal_error is a wildcard transition -- works from any state."""
    from omnibase_infra.nodes.node_node_graph_reducer.models.model_node_graph_state import (
        ModelNodeGraphState,
    )
    from omnibase_infra.nodes.node_node_graph_reducer.reducer import (
        NodeGraphReducer,
    )

    reducer = NodeGraphReducer()
    for start_state in ("initializing", "wiring", "running", "draining"):
        state = ModelNodeGraphState(fsm_state=start_state)
        event = MagicMock()
        event.event_type = "fatal_error"
        output = reducer.reduce(state, event, event_metadata={})
        assert output.result.fsm_state == "stopped"


@pytest.mark.unit
def test_reducer_is_stateless() -> None:
    """Reducer must not track state internally."""
    from omnibase_infra.nodes.node_node_graph_reducer.reducer import (
        NodeGraphReducer,
    )

    reducer = NodeGraphReducer()
    assert not hasattr(reducer, "current_state")
    assert not hasattr(reducer, "_state")
    assert not hasattr(reducer, "state")


@pytest.mark.unit
def test_invalid_transition_raises() -> None:
    """Invalid transitions must raise ValueError."""
    from omnibase_infra.nodes.node_node_graph_reducer.models.model_node_graph_state import (
        ModelNodeGraphState,
    )
    from omnibase_infra.nodes.node_node_graph_reducer.reducer import (
        NodeGraphReducer,
    )

    reducer = NodeGraphReducer()
    state = ModelNodeGraphState(fsm_state="running")
    event = MagicMock()
    event.event_type = "registry_ready"  # Invalid from running state
    with pytest.raises(ValueError, match="Invalid FSM transition"):
        reducer.reduce(state, event, event_metadata={})
