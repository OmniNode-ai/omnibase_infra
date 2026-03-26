# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Unit tests for the cloud gate invariant (I8).

TDD coverage: 4 tests covering single cloud service gate, gated_services payload,
all-local topology pass-through, and disabled service pass-through.

Ticket: OMN-3495
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from omnibase_core.enums import EnumDeploymentMode
from omnibase_core.models.core.model_deployment_topology import ModelDeploymentTopology
from omnibase_core.models.core.model_deployment_topology_service import (
    ModelDeploymentTopologyService,
)
from omnibase_infra.nodes.node_setup_orchestrator.handlers.handler_setup_orchestrator import (
    HandlerSetupOrchestrator,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_handler_no_effects() -> HandlerSetupOrchestrator:
    """Build a handler with all effects as unregistered mocks (never called)."""
    return HandlerSetupOrchestrator(
        container=MagicMock(),
        preflight=AsyncMock(),
        provision=AsyncMock(),
        infisical=AsyncMock(),
        validate=AsyncMock(),
    )


def _topology_with_cloud(
    cloud_service: str,
    other_services: dict[str, ModelDeploymentTopologyService] | None = None,
) -> ModelDeploymentTopology:
    services: dict[str, ModelDeploymentTopologyService] = {
        cloud_service: ModelDeploymentTopologyService(
            mode=EnumDeploymentMode.CLOUD,
            local=None,
        ),
    }
    if other_services:
        services.update(other_services)
    return ModelDeploymentTopology(
        schema_version="1.0",
        services=services,
        presets={},
        active_preset=None,
    )


def _topology_all_local() -> ModelDeploymentTopology:
    return ModelDeploymentTopology.default_minimal()


def _topology_with_disabled(
    disabled_service: str = "infisical",
) -> ModelDeploymentTopology:
    """Minimal topology with one additional DISABLED service."""
    base = ModelDeploymentTopology.default_minimal()
    new_services = dict(base.services)
    new_services[disabled_service] = ModelDeploymentTopologyService(
        mode=EnumDeploymentMode.DISABLED,
        local=None,
    )
    return ModelDeploymentTopology(
        schema_version=base.schema_version,
        services=new_services,
        presets=base.presets,
        active_preset=base.active_preset,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCloudGate:
    """Unit tests for I8 — cloud gate is a hard stop."""

    async def test_single_cloud_service_gates_entire_run(self) -> None:
        """A single CLOUD-mode service triggers setup.cloud.unavailable and halts."""
        handler = _make_handler_no_effects()
        corr_id = uuid4()
        topology = _topology_with_cloud("postgres")

        result = await handler.handle(topology, corr_id, "docker-compose.yml")

        event_types = [e.event_type for e in result.result.events]
        assert "setup.cloud.unavailable" in event_types
        # Must not advance past the gate
        assert "setup.preflight.started" not in event_types
        assert "setup.provision.started" not in event_types
        assert "setup.completed" not in event_types

    async def test_cloud_unavailable_event_carries_gated_service_names(self) -> None:
        """setup.cloud.unavailable event payload must include gated_services list."""
        handler = _make_handler_no_effects()
        corr_id = uuid4()
        topology = _topology_with_cloud(
            "postgres",
            other_services={
                "keycloak": ModelDeploymentTopologyService(
                    mode=EnumDeploymentMode.CLOUD,
                    local=None,
                ),
            },
        )

        result = await handler.handle(topology, corr_id, "docker-compose.yml")

        cloud_event = next(
            e for e in result.result.events if e.event_type == "setup.cloud.unavailable"
        )
        gated = cloud_event.payload.get("gated_services")
        assert gated is not None, "gated_services must be present in payload"
        gated_set = set(gated) if isinstance(gated, list) else set()
        assert "postgres" in gated_set
        assert "keycloak" in gated_set

    async def test_all_local_topology_does_not_trigger_gate(self) -> None:
        """A topology with all LOCAL services must NOT trigger the cloud gate."""
        corr_id = uuid4()

        # We need the effects to return valid outputs so the workflow can complete.
        from omnibase_infra.nodes.node_setup_infisical_effect.models.model_infisical_setup_effect_output import (
            ModelInfisicalSetupEffectOutput,
        )
        from omnibase_infra.nodes.node_setup_local_provision_effect.models.model_local_provision_effect_output import (
            ModelLocalProvisionEffectOutput,
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

        preflight_mock = AsyncMock()
        preflight_mock.run_preflight = AsyncMock(
            return_value=ModelPreflightEffectOutput(
                passed=True,
                checks=(
                    ModelPreflightCheckResult(
                        check_key="docker_version", passed=True, message="ok"
                    ),
                ),
                correlation_id=corr_id,
                duration_ms=1.0,
            )
        )

        provision_mock = AsyncMock()
        provision_mock.provision_local = AsyncMock(
            return_value=ModelLocalProvisionEffectOutput(
                success=True,
                correlation_id=corr_id,
                services_started=("postgres",),
            )
        )

        infisical_mock = AsyncMock()
        infisical_mock.setup_infisical = AsyncMock(
            return_value=ModelInfisicalSetupEffectOutput(
                success=True,
                correlation_id=corr_id,
                status="completed",
            )
        )

        validate_mock = AsyncMock()
        validate_mock.validate_services = AsyncMock(
            return_value=ModelSetupValidateEffectOutput(
                all_healthy=True,
                results=(),
                correlation_id=corr_id,
            )
        )

        handler = HandlerSetupOrchestrator(
            container=MagicMock(),
            preflight=preflight_mock,
            provision=provision_mock,
            infisical=infisical_mock,
            validate=validate_mock,
        )

        topology = _topology_all_local()
        result = await handler.handle(topology, corr_id, "docker-compose.yml")

        event_types = [e.event_type for e in result.result.events]
        assert "setup.cloud.unavailable" not in event_types
        assert "setup.preflight.started" in event_types

    async def test_disabled_service_does_not_trigger_cloud_gate(self) -> None:
        """A DISABLED service must not trigger the cloud gate (I8 is CLOUD only)."""
        corr_id = uuid4()

        from omnibase_infra.nodes.node_setup_local_provision_effect.models.model_local_provision_effect_output import (
            ModelLocalProvisionEffectOutput,
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

        preflight_mock = AsyncMock()
        preflight_mock.run_preflight = AsyncMock(
            return_value=ModelPreflightEffectOutput(
                passed=True,
                checks=(
                    ModelPreflightCheckResult(
                        check_key="docker_version", passed=True, message="ok"
                    ),
                ),
                correlation_id=corr_id,
                duration_ms=1.0,
            )
        )

        provision_mock = AsyncMock()
        provision_mock.provision_local = AsyncMock(
            return_value=ModelLocalProvisionEffectOutput(
                success=True,
                correlation_id=corr_id,
                services_started=("postgres",),
            )
        )

        infisical_mock = AsyncMock()
        # infisical is DISABLED in this topology — should not be called

        validate_mock = AsyncMock()
        validate_mock.validate_services = AsyncMock(
            return_value=ModelSetupValidateEffectOutput(
                all_healthy=True,
                results=(),
                correlation_id=corr_id,
            )
        )

        handler = HandlerSetupOrchestrator(
            container=MagicMock(),
            preflight=preflight_mock,
            provision=provision_mock,
            infisical=infisical_mock,
            validate=validate_mock,
        )

        topology = _topology_with_disabled("infisical")
        result = await handler.handle(topology, corr_id, "docker-compose.yml")

        event_types = [e.event_type for e in result.result.events]
        # DISABLED service must NOT trigger cloud gate
        assert "setup.cloud.unavailable" not in event_types
        # Workflow should proceed normally
        assert "setup.preflight.started" in event_types
