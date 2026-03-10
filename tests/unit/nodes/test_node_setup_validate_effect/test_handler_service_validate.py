# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Unit tests for HandlerServiceValidate.

Tests:
    - test_all_local_services_healthy_returns_all_healthy_true
    - test_one_unhealthy_service_returns_all_healthy_false
    - test_disabled_services_not_in_results
    - test_service_with_health_check_path_uses_http_not_tcp
    - test_response_time_ms_is_non_negative

Invariant I3 — Monkeypatch discipline:
    Patches via ``monkeypatch.setattr(validate_mod, "_check_http_health", ...)``.

Invariant I4 — Port semantics:
    TCP checks use asyncio.open_connection (OPEN checks only).

Ticket: OMN-3494
"""

from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4

import pytest

import omnibase_infra.nodes.node_setup_validate_effect.handlers.handler_service_validate as validate_mod
from omnibase_core.models.core.model_deployment_topology import (
    EnumDeploymentMode,
    ModelDeploymentTopology,
    ModelDeploymentTopologyLocalConfig,
    ModelDeploymentTopologyService,
)
from omnibase_infra.nodes.node_setup_validate_effect.handlers.handler_service_validate import (
    HandlerServiceValidate,
)

# =============================================================================
# Helpers
# =============================================================================


def _make_container() -> MagicMock:
    container = MagicMock()
    container.config = MagicMock()
    return container


def _local_svc(
    host_port: int, health_check_path: str | None = None
) -> ModelDeploymentTopologyService:
    return ModelDeploymentTopologyService(
        mode=EnumDeploymentMode.LOCAL,
        local=ModelDeploymentTopologyLocalConfig(
            compose_service=f"svc-{host_port}",
            host_port=host_port,
            health_check_path=health_check_path,
        ),
    )


def _disabled_svc() -> ModelDeploymentTopologyService:
    return ModelDeploymentTopologyService(
        mode=EnumDeploymentMode.DISABLED,
        local=None,
    )


def _topology(
    *services: tuple[str, ModelDeploymentTopologyService],
) -> ModelDeploymentTopology:
    return ModelDeploymentTopology(
        schema_version="1.0",
        services=dict(services),
    )


async def _tcp_healthy(_host: str, _port: int, _timeout_s: float) -> tuple[bool, float]:
    return True, 5.0


async def _tcp_unhealthy(
    _host: str, _port: int, _timeout_s: float
) -> tuple[bool, float]:
    return False, 5.0


async def _http_healthy(_url: str, _timeout_s: float) -> tuple[bool, float]:
    return True, 10.0


async def _http_unhealthy(_url: str, _timeout_s: float) -> tuple[bool, float]:
    return False, 10.0


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.unit
class TestHandlerServiceValidate:
    """Unit tests for HandlerServiceValidate."""

    @pytest.mark.asyncio
    async def test_all_local_services_healthy_returns_all_healthy_true(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When all LOCAL services pass TCP health check, all_healthy is True."""
        monkeypatch.setattr(validate_mod, "_check_tcp_health", _tcp_healthy)
        monkeypatch.setattr(validate_mod, "_check_http_health", _http_healthy)

        topology = _topology(
            ("postgres", _local_svc(5436)),
            ("redpanda", _local_svc(19092)),
        )

        container = _make_container()
        handler = HandlerServiceValidate(container)
        await handler.initialize({})

        output = await handler.execute(
            {
                "topology": topology,
                "correlation_id": uuid4(),
                "timeout_seconds": 5,
            }
        )

        result = output.result
        assert result.all_healthy is True
        assert len(result.results) == 2
        assert all(r.healthy for r in result.results)

    @pytest.mark.asyncio
    async def test_one_unhealthy_service_returns_all_healthy_false(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When one LOCAL service fails health check, all_healthy is False."""
        call_count: list[int] = [0]

        async def mixed_tcp(
            host: str, port: int, timeout_s: float
        ) -> tuple[bool, float]:
            call_count[0] += 1
            # First service healthy, second unhealthy
            return (call_count[0] == 1, 3.0)

        monkeypatch.setattr(validate_mod, "_check_tcp_health", mixed_tcp)
        monkeypatch.setattr(validate_mod, "_check_http_health", _http_healthy)

        topology = _topology(
            ("postgres", _local_svc(5436)),
            ("redpanda", _local_svc(19092)),
        )

        container = _make_container()
        handler = HandlerServiceValidate(container)
        await handler.initialize({})

        output = await handler.execute(
            {
                "topology": topology,
                "correlation_id": uuid4(),
                "timeout_seconds": 5,
            }
        )

        result = output.result
        assert result.all_healthy is False
        assert len(result.results) == 2
        # At least one unhealthy
        assert any(not r.healthy for r in result.results)

    @pytest.mark.asyncio
    async def test_disabled_services_not_in_results(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """DISABLED-mode services are excluded from health check results."""
        monkeypatch.setattr(validate_mod, "_check_tcp_health", _tcp_healthy)
        monkeypatch.setattr(validate_mod, "_check_http_health", _http_healthy)

        topology = _topology(
            ("postgres", _local_svc(5436)),
            ("disabled_svc", _disabled_svc()),
        )

        container = _make_container()
        handler = HandlerServiceValidate(container)
        await handler.initialize({})

        output = await handler.execute(
            {
                "topology": topology,
                "correlation_id": uuid4(),
                "timeout_seconds": 5,
            }
        )

        result = output.result
        node_labels = {r.node_label for r in result.results}
        assert "postgres" in node_labels
        assert "disabled_svc" not in node_labels
        assert len(result.results) == 1

    @pytest.mark.asyncio
    async def test_service_with_health_check_path_uses_http_not_tcp(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Services with health_check_path use HTTP check, not TCP."""
        http_called: list[str] = []
        tcp_called: list[int] = []

        async def record_http(url: str, timeout_s: float) -> tuple[bool, float]:
            http_called.append(url)
            return True, 5.0

        async def record_tcp(
            host: str, port: int, timeout_s: float
        ) -> tuple[bool, float]:
            tcp_called.append(port)
            return True, 5.0

        monkeypatch.setattr(validate_mod, "_check_http_health", record_http)
        monkeypatch.setattr(validate_mod, "_check_tcp_health", record_tcp)

        # One service with health_check_path (should use HTTP)
        # One service without (should use TCP)
        topology = _topology(
            ("consul", _local_svc(28500, health_check_path="/v1/health")),
            ("postgres", _local_svc(5436)),
        )

        container = _make_container()
        handler = HandlerServiceValidate(container)
        await handler.initialize({})

        output = await handler.execute(
            {
                "topology": topology,
                "correlation_id": uuid4(),
                "timeout_seconds": 5,
            }
        )

        result = output.result
        # consul → HTTP; postgres → TCP
        assert len(http_called) == 1
        assert "28500" in http_called[0]
        assert "/v1/health" in http_called[0]
        assert len(tcp_called) == 1
        assert 5436 in tcp_called
        assert result.all_healthy is True

    @pytest.mark.asyncio
    async def test_response_time_ms_is_non_negative(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """All health result entries must have response_time_ms >= 0."""
        monkeypatch.setattr(validate_mod, "_check_tcp_health", _tcp_healthy)
        monkeypatch.setattr(validate_mod, "_check_http_health", _http_healthy)

        topology = _topology(
            ("postgres", _local_svc(5436)),
            ("valkey", _local_svc(16379)),
        )

        container = _make_container()
        handler = HandlerServiceValidate(container)
        await handler.initialize({})

        output = await handler.execute(
            {
                "topology": topology,
                "correlation_id": uuid4(),
                "timeout_seconds": 5,
            }
        )

        result = output.result
        assert len(result.results) == 2
        for r in result.results:
            assert r.response_time_ms >= 0.0


__all__: list[str] = ["TestHandlerServiceValidate"]
