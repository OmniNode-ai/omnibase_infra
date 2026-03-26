# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Integration tests for NodeSetupOrchestrator end-to-end flow.

Exercises the full sequential setup path
(HandlerSetupOrchestrator → 4 effect handlers) with mocked Docker subprocess
and socket I/O. Verifies:

  I4 — Port semantics: preflight checks ports FREE; validate checks ports OPEN.
       Tracked in separate call lists to confirm phase separation.
  I8 — Cloud gate fires before any Docker commands run.

Test matrix:
  Test 1 — Full minimal setup with mocked subprocess → setup.completed emitted.
  Test 2 — Cloud topology → setup.cloud.unavailable, zero docker calls.

Ticket: OMN-3497
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

import omnibase_infra.nodes.node_setup_preflight_effect.handlers.handler_preflight_check as preflight_handler_mod
import omnibase_infra.nodes.node_setup_validate_effect.handlers.handler_service_validate as validate_handler_mod
from omnibase_core.enums import EnumDeploymentMode
from omnibase_core.models.container.model_onex_container import ModelONEXContainer
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
from omnibase_infra.nodes.node_setup_local_provision_effect.handlers.handler_local_provision import (
    HandlerLocalProvision,
)
from omnibase_infra.nodes.node_setup_local_provision_effect.models.model_local_provision_effect_output import (
    ModelLocalProvisionEffectOutput,
)
from omnibase_infra.nodes.node_setup_orchestrator.handlers.handler_setup_orchestrator import (
    HandlerSetupOrchestrator,
)
from omnibase_infra.nodes.node_setup_orchestrator.models.model_setup_orchestrator_output import (
    ModelSetupOrchestratorOutput,
)
from omnibase_infra.nodes.node_setup_preflight_effect.handlers.handler_preflight_check import (
    HandlerPreflightCheck,
)
from omnibase_infra.nodes.node_setup_preflight_effect.models.model_preflight_check_result import (
    ModelPreflightCheckResult,
)
from omnibase_infra.nodes.node_setup_preflight_effect.models.model_preflight_effect_output import (
    ModelPreflightEffectOutput,
)
from omnibase_infra.nodes.node_setup_validate_effect.handlers.handler_service_validate import (
    HandlerServiceValidate,
)
from omnibase_infra.nodes.node_setup_validate_effect.models.model_setup_validate_effect_output import (
    ModelSetupValidateEffectOutput,
)

# ---------------------------------------------------------------------------
# Protocol adapter wrappers
# ---------------------------------------------------------------------------
# HandlerPreflightCheck, HandlerLocalProvision, HandlerServiceValidate expose
# ``execute(envelope)`` rather than the protocol method names.
# These thin adapters bridge the gap so that HandlerSetupOrchestrator (which
# calls run_preflight / provision_local / validate_services) can use the real
# handler implementations while module-level I/O is patched in tests.
# ---------------------------------------------------------------------------


class _PreflightAdapter:
    """Wraps HandlerPreflightCheck as ProtocolPreflightEffect."""

    def __init__(self, handler: HandlerPreflightCheck) -> None:
        self._handler = handler

    async def run_preflight(self, correlation_id: object) -> ModelPreflightEffectOutput:
        envelope: dict[str, object] = {}
        if isinstance(correlation_id, UUID):
            envelope["correlation_id"] = correlation_id
        result = await self._handler.execute(envelope)
        return result.result  # type: ignore[return-value]


class _ProvisionAdapter:
    """Wraps HandlerLocalProvision as ProtocolProvisionEffect."""

    def __init__(self, handler: HandlerLocalProvision, compose_file_path: str) -> None:
        self._handler = handler
        self._compose_file_path = compose_file_path

    async def provision_local(
        self,
        topology: ModelDeploymentTopology,
        compose_file_path: str,
        correlation_id: UUID,
    ) -> ModelLocalProvisionEffectOutput:
        envelope: dict[str, object] = {
            "topology": topology,
            "compose_file_path": compose_file_path or self._compose_file_path,
            "correlation_id": correlation_id,
        }
        result = await self._handler.execute(envelope)
        return result.result  # type: ignore[return-value]


class _ValidateAdapter:
    """Wraps HandlerServiceValidate as ProtocolValidateEffect."""

    def __init__(self, handler: HandlerServiceValidate) -> None:
        self._handler = handler

    async def validate_services(
        self,
        topology: ModelDeploymentTopology,
        correlation_id: object,
    ) -> ModelSetupValidateEffectOutput:
        envelope: dict[str, object] = {"topology": topology}
        if isinstance(correlation_id, UUID):
            envelope["correlation_id"] = correlation_id
        result = await self._handler.execute(envelope)
        return result.result  # type: ignore[return-value]


class _InfisicalSkipAdapter:
    """Stub infisical effect that always returns 'skipped'."""

    async def setup_infisical(
        self,
        topology: ModelDeploymentTopology,
        correlation_id: object,
    ) -> ModelInfisicalSetupEffectOutput:
        corr_id = correlation_id if isinstance(correlation_id, UUID) else uuid4()
        return ModelInfisicalSetupEffectOutput(
            success=True,
            correlation_id=corr_id,
            status="skipped",
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_container() -> ModelONEXContainer:
    return ModelONEXContainer()


def _build_orchestrator_handler(
    container: ModelONEXContainer,
    compose_file_path: str = "/dev/null",
) -> HandlerSetupOrchestrator:
    """Build HandlerSetupOrchestrator wired to real (patchable) effect handlers."""
    preflight_handler = HandlerPreflightCheck(container=container)
    provision_handler = HandlerLocalProvision(container=container)
    validate_handler = HandlerServiceValidate(container=container)

    return HandlerSetupOrchestrator(
        container=container,
        preflight=_PreflightAdapter(preflight_handler),
        provision=_ProvisionAdapter(provision_handler, compose_file_path),
        infisical=_InfisicalSkipAdapter(),
        validate=_ValidateAdapter(validate_handler),
    )


def _fake_compose_up(returncode: int = 0) -> Any:
    """Return a factory that creates an AsyncMock simulating docker compose up."""

    async def _fake(*args: Any, **kwargs: Any) -> MagicMock:
        proc = MagicMock()
        proc.communicate = AsyncMock(return_value=(b"", b""))
        proc.returncode = returncode
        return proc

    return _fake


def _topology_with_cloud_postgres() -> ModelDeploymentTopology:
    """Topology where postgres has mode=CLOUD (triggers I8 hard stop)."""
    return ModelDeploymentTopology(
        schema_version="1.0",
        services={
            "postgres": ModelDeploymentTopologyService(
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
            "valkey": ModelDeploymentTopologyService(
                mode=EnumDeploymentMode.LOCAL,
                local=ModelDeploymentTopologyLocalConfig(
                    compose_service="omnibase-infra-valkey",
                    host_port=16379,
                    health_check_path=None,
                ),
            ),
        },
        presets={"minimal": ["postgres", "redpanda", "valkey"]},
        active_preset="minimal",
    )


async def _run_handler(
    handler: HandlerSetupOrchestrator,
    topology: ModelDeploymentTopology,
    compose_file_path: str = "/dev/null",
) -> ModelSetupOrchestratorOutput:
    """Run the orchestrator handler and return the unwrapped output."""
    corr_id = uuid4()
    result = await handler.handle(topology, corr_id, compose_file_path)
    return result.result  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestSetupOrchestratorE2E:
    """End-to-end integration tests for the full setup orchestration path."""

    async def test_full_minimal_setup_with_mocked_subprocess(self) -> None:
        """Full orchestrator flow: preflight (ports FREE) → provision → validate (ports OPEN).

        I4: Preflight calls _check_port_free (returns True = FREE).
            Validate calls _check_tcp_health (returns (True, 3.0) = OPEN).
            Tracked in separate lists to confirm phase separation.
        """
        topology = ModelDeploymentTopology.default_minimal()

        # Separate call trackers for preflight (FREE) vs validate (OPEN)
        preflight_calls: list[int] = []
        validate_calls: list[int] = []

        def _mock_port_free(host: str, port: int) -> bool:
            """I4: preflight checks ports are FREE (not in use)."""
            preflight_calls.append(port)
            return True  # FREE = True = port not occupied

        async def _mock_tcp_health(
            host: str, port: int, timeout_s: float
        ) -> tuple[bool, float]:
            """I4: validate checks ports are OPEN (service reachable)."""
            validate_calls.append(port)
            return True, 3.0  # OPEN = True = service healthy

        container = _build_container()
        handler = _build_orchestrator_handler(container, compose_file_path="/dev/null")

        _provision_mod = (
            "omnibase_infra.nodes.node_setup_local_provision_effect"
            ".handlers.handler_local_provision"
        )

        async def _instant_poll(host: str, port: int, max_wait: float) -> bool:
            return True

        def _mock_postgres_check() -> ModelPreflightCheckResult:
            return ModelPreflightCheckResult(
                check_key="postgres_password_set",
                passed=True,
                message="POSTGRES_PASSWORD is set (mocked)",
                detail=None,
            )

        def _mock_omnibase_dir_check() -> ModelPreflightCheckResult:
            return ModelPreflightCheckResult(
                check_key="omnibase_dir",
                passed=True,
                message="omnibase dir ok (mocked)",
                detail=None,
            )

        with (
            patch.object(preflight_handler_mod, "_check_port_free", _mock_port_free),
            patch.object(
                preflight_handler_mod, "_check_postgres_password", _mock_postgres_check
            ),
            patch.object(
                preflight_handler_mod, "_check_omnibase_dir", _mock_omnibase_dir_check
            ),
            patch(
                f"{_provision_mod}.asyncio.create_subprocess_exec",
                side_effect=_fake_compose_up(returncode=0),
            ),
            patch(f"{_provision_mod}._poll_port_open", _instant_poll),
            patch.object(validate_handler_mod, "_check_tcp_health", _mock_tcp_health),
        ):
            output = await _run_handler(handler, topology)

        event_types = [e.event_type for e in output.events]

        # Full happy-path events must all be present
        assert "setup.completed" in event_types, (
            f"Expected setup.completed in {event_types}"
        )
        assert "setup.cloud.unavailable" not in event_types
        assert "setup.preflight.completed" in event_types
        assert "setup.provision.completed" in event_types
        assert "setup.validate.completed" in event_types

        # Minimal topology has no infisical service → skipped (not started)
        assert "setup.infisical.skipped" in event_types
        assert "setup.infisical.started" not in event_types

        # I4: Verify phase separation — preflight and validate use separate checks
        # Minimal topology has 3 LOCAL services: postgres (5436), redpanda (19092), valkey (16379)
        assert len(preflight_calls) == 3, (
            f"Expected 3 preflight FREE checks (one per service), got {preflight_calls}"
        )
        assert len(validate_calls) == 3, (
            f"Expected 3 validate OPEN checks (one per service), got {validate_calls}"
        )

        # The two call sets must be disjoint in semantics (both cover the same ports,
        # but via different functions — _check_port_free vs _check_tcp_health).
        # The key invariant is that preflight_calls and validate_calls are populated
        # by different functions in different handlers, confirming I4 phase separation.
        assert set(preflight_calls) == set(validate_calls), (
            "Preflight and validate must check the same set of ports "
            f"(preflight={preflight_calls}, validate={validate_calls})"
        )

    async def test_cloud_gate_stops_before_any_docker_ops(self) -> None:
        """I8: Cloud topology → setup.cloud.unavailable before any docker calls.

        No docker compose subprocess must be spawned when any service is CLOUD.
        """
        topology = _topology_with_cloud_postgres()
        docker_calls: list[Any] = []

        async def _capture_docker(*args: Any, **kwargs: Any) -> MagicMock:
            docker_calls.append(args)
            proc = MagicMock()
            proc.communicate = AsyncMock(return_value=(b"", b""))
            proc.returncode = 0
            return proc

        container = _build_container()
        handler = _build_orchestrator_handler(container)

        _provision_mod = (
            "omnibase_infra.nodes.node_setup_local_provision_effect"
            ".handlers.handler_local_provision"
        )
        with patch(
            f"{_provision_mod}.asyncio.create_subprocess_exec",
            side_effect=_capture_docker,
        ):
            output = await _run_handler(handler, topology)

        event_types = [e.event_type for e in output.events]

        # I8: Cloud gate fires, setup stops immediately
        assert "setup.cloud.unavailable" in event_types, (
            f"Expected setup.cloud.unavailable in {event_types}"
        )

        # No docker subprocess must have been called (I8 hard stop)
        assert len(docker_calls) == 0, (
            f"I8 violated: docker compose was called {len(docker_calls)} time(s) "
            "even though a CLOUD service was present"
        )

        # None of the later phases should have started
        assert "setup.preflight.started" not in event_types
        assert "setup.provision.started" not in event_types
        assert "setup.validate.started" not in event_types
