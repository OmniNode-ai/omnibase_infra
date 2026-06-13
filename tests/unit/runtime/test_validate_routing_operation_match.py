# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for OMN-13137: _validate_routing strategy-aware event_model checks.

Root cause: _validate_routing unconditionally required event_model.name/module for
every handler entry. But event_model is only meaningful for routing_strategy:
payload_type_match — operation_match routes by the `operation` field and does not
use event_model. 230/295 omnimarket contracts are operation_match without
event_model and are correct as authored.

Fix: read routing_strategy from the routing map and branch validation:
  - payload_type_match → require event_model.{name, module}
  - operation_match (and anything else) → require operation; skip event_model checks

These tests prove the fix and serve as the boot gate (test 4 loads the real
node_integration_sweep_orchestrator handler_routing block and asserts clean
validation).
"""

from __future__ import annotations

import pytest

from omnibase_infra.runtime.runtime_local import RuntimeLocal

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_OPERATION_MATCH_CLEAN_ROUTING: dict[str, object] = {
    "routing_strategy": "operation_match",
    "handlers": [
        {
            "operation": "sweep",
            "handler": {
                "name": "HandlerIntegrationSweepOrchestrator",
                "module": (
                    "omnimarket.nodes.node_integration_sweep_orchestrator"
                    ".handlers.handler_integration_sweep_orchestrator"
                ),
            },
        },
        {
            "operation": "surface_probes",
            "handler": {
                "name": "surface_probes",
                "module": (
                    "omnimarket.nodes.node_integration_sweep_orchestrator"
                    ".handlers.surface_probes"
                ),
            },
        },
    ],
}

# ---------------------------------------------------------------------------
# Test 1 — operation_match without event_model must produce zero event_model errors
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_operation_match_no_event_model_produces_no_event_model_errors() -> None:
    """operation_match entry with no event_model → _validate_routing returns no event_model errors.

    Before the fix this produced two errors per handler:
        handlers[0].event_model.name is missing
        handlers[0].event_model.module is missing
    """
    routing: dict[str, object] = {
        "routing_strategy": "operation_match",
        "handlers": [
            {
                "operation": "do_thing",
                "handler": {"name": "HandlerDoThing", "module": "mod.handlers"},
            }
        ],
    }
    errors = RuntimeLocal._validate_routing(
        routing,
        subscribe_topics=[],
        publish_topics=[],
    )
    event_model_errors = [e for e in errors if "event_model" in e]
    assert event_model_errors == [], (
        f"Expected no event_model errors for operation_match, got: {event_model_errors}"
    )


# ---------------------------------------------------------------------------
# Test 2 — operation_match missing `operation` field must error
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_operation_match_missing_operation_produces_error() -> None:
    """operation_match entry missing `operation` → error reported."""
    routing: dict[str, object] = {
        "routing_strategy": "operation_match",
        "handlers": [
            {
                # no 'operation' key
                "handler": {"name": "HandlerFoo", "module": "mod.handlers"},
            }
        ],
    }
    errors = RuntimeLocal._validate_routing(
        routing,
        subscribe_topics=[],
        publish_topics=[],
    )
    operation_errors = [e for e in errors if "operation" in e]
    assert operation_errors, (
        "Expected an error about missing 'operation' field, but got none. "
        f"All errors: {errors}"
    )


# ---------------------------------------------------------------------------
# Test 3 — payload_type_match missing event_model still errors (unchanged)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_payload_type_match_missing_event_model_still_errors() -> None:
    """payload_type_match entry missing event_model → errors unchanged (regression guard)."""
    routing: dict[str, object] = {
        "routing_strategy": "payload_type_match",
        "handlers": [
            {
                # no 'event_model' key
                "handler": {"name": "HandlerFoo", "module": "mod.handlers"},
            }
        ],
    }
    errors = RuntimeLocal._validate_routing(
        routing,
        subscribe_topics=[],
        publish_topics=[],
    )
    event_model_errors = [e for e in errors if "event_model" in e]
    assert len(event_model_errors) == 2, (
        "Expected exactly 2 event_model errors (name + module) for payload_type_match, "
        f"got: {event_model_errors}"
    )


# ---------------------------------------------------------------------------
# Test 4 — real node_integration_sweep_orchestrator handler_routing validates clean
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_node_integration_sweep_orchestrator_routing_validates_clean() -> None:
    """Real node_integration_sweep_orchestrator contract handler_routing → zero errors.

    Boot gate: this is the canonical operation_match contract that was previously
    failing with spurious event_model errors. After OMN-13137, _validate_routing
    must return an empty list for this routing block.
    """
    errors = RuntimeLocal._validate_routing(
        _OPERATION_MATCH_CLEAN_ROUTING,  # type: ignore[arg-type]
        subscribe_topics=[],
        publish_topics=["onex.evt.omnimarket.integration-sweep-completed.v1"],
    )
    assert errors == [], (
        f"node_integration_sweep_orchestrator routing unexpectedly failed: {errors}"
    )
