# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Unit tests for HandlerSetupOrchestrator.

TDD coverage: 8 tests covering the full sequential workflow, failure paths,
invariants I5 (no result field), and I6 (event types are constants).

Ticket: OMN-3495
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from omnibase_core.enums import EnumDeploymentMode
from omnibase_core.models.core.model_deployment_topology import ModelDeploymentTopology
from omnibase_core.models.core.model_deployment_topology_local_config import (
    ModelDeploymentTopologyLocalConfig,
)
from omnibase_core.models.core.model_deployment_topology_service import (
    ModelDeploymentTopologyService,
)
from omnibase_infra.nodes.node_setup_infisical_effect.models.model_infisical_setup_effect_output import (
    ModelInfisicalSetupEffectOutput,
)
from omnibase_infra.nodes.node_setup_local_provision_effect.models.model_local_provision_effect_output import (
    ModelLocalProvisionEffectOutput,
)
from omnibase_infra.nodes.node_setup_orchestrator.constants.setup_event_types import (
    SETUP_EVENT_TYPES,
)
from omnibase_infra.nodes.node_setup_orchestrator.handlers.handler_setup_orchestrator import (
    HandlerSetupOrchestrator,
)
from omnibase_infra.nodes.node_setup_preflight_effect.models.model_preflight_check_result import (
    ModelPreflightCheckResult,
)
from omnibase_infra.nodes.node_setup_preflight_effect.models.model_preflight_effect_output import (
    ModelPreflightEffectOutput,
)
from omnibase_infra.nodes.node_setup_validate_effect.models.model_setup_validate_effect_output import (
    ModelSetupValidateEffectOutput,
)

# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


def _make_container() -> MagicMock:
    return MagicMock()


def _make_minimal_topology() -> ModelDeploymentTopology:
    """Topology with postgres, redpanda, valkey — no infisical."""
    return ModelDeploymentTopology.default_minimal()


def _make_standard_topology() -> ModelDeploymentTopology:
    """Topology with infisical enabled (LOCAL mode)."""
    return ModelDeploymentTopology.default_standard()


def _make_cloud_topology(cloud_service: str = "postgres") -> ModelDeploymentTopology:
    """Topology where one service has mode=CLOUD."""
    services = {
        cloud_service: ModelDeploymentTopologyService(
            mode=EnumDeploymentMode.CLOUD,
            local=None,
        ),
        "redpanda": ModelDeploymentTopologyService(
            mode=EnumDeploymentMode.LOCAL,
            local=ModelDeploymentTopologyLocalConfig(
                compose_service="omnibase-infra-redpanda",
                host_port=19092,
                health_check_path=None,
            ),
        ),
    }
    return ModelDeploymentTopology(
        schema_version="1.0",
        services=services,
        presets={},
        active_preset=None,
    )


def _passing_preflight(corr_id: Any) -> ModelPreflightEffectOutput:
    return ModelPreflightEffectOutput(
        passed=True,
        checks=(
            ModelPreflightCheckResult(
                check_key="docker_version", passed=True, message="ok"
            ),
        ),
        correlation_id=corr_id,
        duration_ms=1.0,
    )


def _failing_preflight(
    corr_id: Any, failed_key: str = "docker_version"
) -> ModelPreflightEffectOutput:
    return ModelPreflightEffectOutput(
        passed=False,
        checks=(
            ModelPreflightCheckResult(
                check_key=failed_key, passed=False, message="fail"
            ),
        ),
        correlation_id=corr_id,
        duration_ms=1.0,
    )


def _passing_provision(corr_id: Any) -> ModelLocalProvisionEffectOutput:
    return ModelLocalProvisionEffectOutput(
        success=True,
        correlation_id=corr_id,
        services_started=("postgres", "redpanda", "valkey"),
    )


def _failing_provision(corr_id: Any) -> ModelLocalProvisionEffectOutput:
    return ModelLocalProvisionEffectOutput(
        success=False,
        correlation_id=corr_id,
        error="compose up failed",
    )


def _completed_infisical(corr_id: Any) -> ModelInfisicalSetupEffectOutput:
    return ModelInfisicalSetupEffectOutput(
        success=True,
        correlation_id=corr_id,
        status="completed",
        infisical_addr="http://localhost:8880",
    )


def _passing_validate(corr_id: Any) -> ModelSetupValidateEffectOutput:
    return ModelSetupValidateEffectOutput(
        all_healthy=True,
        results=(),
        correlation_id=corr_id,
    )


def _failing_validate(corr_id: Any) -> ModelSetupValidateEffectOutput:
    return ModelSetupValidateEffectOutput(
        all_healthy=False,
        results=(),
        correlation_id=corr_id,
        error="postgres unhealthy",
    )


def _make_handler(
    preflight_result: ModelPreflightEffectOutput | None = None,
    provision_result: ModelLocalProvisionEffectOutput | None = None,
    infisical_result: ModelInfisicalSetupEffectOutput | None = None,
    validate_result: ModelSetupValidateEffectOutput | None = None,
) -> HandlerSetupOrchestrator:
    """Build a HandlerSetupOrchestrator with all effects mocked."""
    container = _make_container()
    corr_id = uuid4()

    preflight_mock = AsyncMock()
    preflight_mock.run_preflight = AsyncMock(
        return_value=preflight_result or _passing_preflight(corr_id)
    )

    provision_mock = AsyncMock()
    provision_mock.provision_local = AsyncMock(
        return_value=provision_result or _passing_provision(corr_id)
    )

    infisical_mock = AsyncMock()
    infisical_mock.setup_infisical = AsyncMock(
        return_value=infisical_result or _completed_infisical(corr_id)
    )

    validate_mock = AsyncMock()
    validate_mock.validate_services = AsyncMock(
        return_value=validate_result or _passing_validate(corr_id)
    )

    return HandlerSetupOrchestrator(
        container=container,
        preflight=preflight_mock,
        provision=provision_mock,
        infisical=infisical_mock,
        validate=validate_mock,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHandlerSetupOrchestrator:
    """Unit tests for HandlerSetupOrchestrator sequential workflow."""

    async def test_cloud_service_emits_cloud_unavailable_stops_before_preflight(
        self,
    ) -> None:
        """I8: Cloud gate fires first. Preflight must NOT be called."""
        corr_id = uuid4()
        container = _make_container()

        preflight_mock = AsyncMock()
        provision_mock = AsyncMock()
        infisical_mock = AsyncMock()
        validate_mock = AsyncMock()

        handler = HandlerSetupOrchestrator(
            container=container,
            preflight=preflight_mock,
            provision=provision_mock,
            infisical=infisical_mock,
            validate=validate_mock,
        )

        topology = _make_cloud_topology("postgres")
        result = await handler.handle(topology, corr_id, "docker-compose.yml")

        event_types = [e.event_type for e in result.result.events]
        assert "setup.cloud.unavailable" in event_types
        assert "setup.preflight.started" not in event_types

        # Preflight effect must NOT have been called
        preflight_mock.run_preflight.assert_not_called()

    async def test_preflight_failure_halts_emits_preflight_failed_not_provision(
        self,
    ) -> None:
        """Preflight failure must stop workflow before provision."""
        corr_id = uuid4()
        handler = _make_handler(
            preflight_result=_failing_preflight(corr_id, "docker_version"),
        )

        topology = _make_minimal_topology()
        result = await handler.handle(topology, corr_id, "docker-compose.yml")

        event_types = [e.event_type for e in result.result.events]
        assert "setup.preflight.started" in event_types
        assert "setup.preflight.failed" in event_types
        assert "setup.provision.started" not in event_types

    async def test_provision_failure_halts_before_infisical_and_validate(
        self,
    ) -> None:
        """Provision failure must stop workflow before infisical and validate."""
        corr_id = uuid4()
        handler = _make_handler(
            provision_result=_failing_provision(corr_id),
        )

        topology = _make_standard_topology()  # has infisical
        result = await handler.handle(topology, corr_id, "docker-compose.yml")

        event_types = [e.event_type for e in result.result.events]
        assert "setup.provision.started" in event_types
        assert "setup.provision.failed" in event_types
        assert "setup.infisical.started" not in event_types
        assert "setup.validate.started" not in event_types

    async def test_minimal_topology_skips_infisical_emits_skipped(
        self,
    ) -> None:
        """Minimal topology (no infisical service) must emit setup.infisical.skipped."""
        corr_id = uuid4()
        handler = _make_handler()

        topology = _make_minimal_topology()  # no infisical
        result = await handler.handle(topology, corr_id, "docker-compose.yml")

        event_types = [e.event_type for e in result.result.events]
        assert "setup.infisical.skipped" in event_types
        assert "setup.infisical.started" not in event_types
        assert "setup.infisical.completed" not in event_types

    async def test_standard_topology_runs_infisical_step(
        self,
    ) -> None:
        """Standard topology (infisical LOCAL) must emit infisical.started and completed."""
        corr_id = uuid4()
        handler = _make_handler()

        topology = _make_standard_topology()  # has infisical
        result = await handler.handle(topology, corr_id, "docker-compose.yml")

        event_types = [e.event_type for e in result.result.events]
        assert "setup.infisical.started" in event_types
        assert "setup.infisical.completed" in event_types

    async def test_full_success_emits_setup_completed(
        self,
    ) -> None:
        """Full success path must end with setup.completed."""
        corr_id = uuid4()
        handler = _make_handler()

        topology = _make_minimal_topology()
        result = await handler.handle(topology, corr_id, "docker-compose.yml")

        event_types = [e.event_type for e in result.result.events]
        assert "setup.completed" in event_types
        assert event_types[-1] == "setup.completed"

    async def test_orchestrator_output_has_no_result_field(
        self,
    ) -> None:
        """I5: ModelSetupOrchestratorOutput.model_dump() must NOT contain 'result'."""
        corr_id = uuid4()
        handler = _make_handler()

        topology = _make_minimal_topology()
        handler_output = await handler.handle(topology, corr_id, "docker-compose.yml")

        dumped = handler_output.result.model_dump()
        assert "result" not in dumped, (
            f"I5 violated: 'result' key found in ModelSetupOrchestratorOutput.model_dump().\n"
            f"Keys present: {list(dumped.keys())}"
        )

    async def test_all_emitted_event_types_are_in_constants(
        self,
    ) -> None:
        """I6: All emitted event types must be in SETUP_EVENT_TYPES frozenset."""
        corr_id = uuid4()
        handler = _make_handler()

        topology = _make_standard_topology()
        result = await handler.handle(topology, corr_id, "docker-compose.yml")

        for event in result.result.events:
            assert event.event_type in SETUP_EVENT_TYPES, (
                f"I6 violated: event_type '{event.event_type}' is not in SETUP_EVENT_TYPES"
            )
