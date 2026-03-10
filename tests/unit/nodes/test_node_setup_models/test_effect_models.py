# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Unit tests for setup node I/O models.

Tests cover Invariant I5 (orchestrator has no result field),
Invariant I6 (16 event type constants), and key validation constraints.

Ticket: OMN-3491
"""

from __future__ import annotations

import uuid

import pytest
from pydantic import ValidationError

from omnibase_core.models.core.model_deployment_topology import ModelDeploymentTopology
from omnibase_infra.nodes.node_setup_orchestrator.constants.setup_event_types import (
    SETUP_EVENT_TYPES,
)
from omnibase_infra.nodes.node_setup_orchestrator.models.model_setup_orchestrator_output import (
    ModelSetupOrchestratorOutput,
)
from omnibase_infra.nodes.node_setup_preflight_effect.models.model_preflight_check_result import (
    ModelPreflightCheckResult,
)
from omnibase_infra.nodes.node_setup_preflight_effect.models.model_preflight_effect_input import (
    ModelPreflightEffectInput,
)
from omnibase_infra.nodes.node_setup_preflight_effect.models.model_preflight_effect_output import (
    ModelPreflightEffectOutput,
)


@pytest.mark.unit
class TestEffectModels:
    """Unit tests for setup effect node I/O models."""

    def test_preflight_input_requires_topology(self) -> None:
        """ModelPreflightEffectInput must reject construction without a topology."""
        with pytest.raises(ValidationError):
            ModelPreflightEffectInput(  # type: ignore[call-arg]
                correlation_id=uuid.uuid4(),
            )

    def test_preflight_output_passed_false_when_check_fails(self) -> None:
        """A failing check result should propagate through the output model correctly."""
        failing_check = ModelPreflightCheckResult(
            check_key="postgres_reachable",
            passed=False,
            message="Connection refused",
            detail="Could not connect to localhost:5436",
        )
        output = ModelPreflightEffectOutput(
            passed=False,
            checks=(failing_check,),
            correlation_id=uuid.uuid4(),
            duration_ms=42.5,
        )
        assert output.passed is False
        assert len(output.checks) == 1
        assert output.checks[0].passed is False

    def test_duration_ms_must_be_non_negative(self) -> None:
        """ModelPreflightEffectOutput must reject negative duration_ms values."""
        topology = ModelDeploymentTopology.default_minimal()
        with pytest.raises(ValidationError):
            ModelPreflightEffectOutput(
                passed=True,
                checks=(),
                correlation_id=uuid.uuid4(),
                duration_ms=-1.0,
            )
        # Verify zero is accepted (boundary)
        output = ModelPreflightEffectOutput(
            passed=True,
            checks=(),
            correlation_id=uuid.uuid4(),
            duration_ms=0.0,
        )
        assert output.duration_ms == 0.0
        # suppress unused variable warning
        _ = topology

    def test_orchestrator_output_has_no_result_field(self) -> None:
        """ModelSetupOrchestratorOutput must not have a 'result' field (Invariant I5)."""
        assert "result" not in ModelSetupOrchestratorOutput.model_fields

    def test_setup_event_types_defines_exactly_16_events(self) -> None:
        """SETUP_EVENT_TYPES frozenset must contain exactly 16 entries (Invariant I6)."""
        assert len(SETUP_EVENT_TYPES) == 16
