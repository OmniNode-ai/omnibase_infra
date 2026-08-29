# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Auth-state classification tests for ServiceLlmEndpointHealth (OMN-16900).

The health service used to probe every configured endpoint on a fixed ~30s
cadence with no notion of authentication state.  An endpoint whose API key was
absent, revoked, or simply wrong was retried forever at outage cadence — on
.201 the GLM endpoints returned 401 on every probe for 5+ days (544-4525 hits
per container).  A bad credential is not a transient outage.

This suite pins the two behaviours that make that structurally impossible:

1. **Fail-fast classification.** An endpoint that declares an auth secret which
   is absent or unresolvable is classified ``SKIPPED_NO_AUTH`` once and is
   **never probed** — zero HTTP requests, one status, no re-evaluation.
2. **Terminal auth failure.** Sustained 401/403 becomes ``AUTH_FAILED`` after a
   small consecutive threshold and moves the endpoint to exponential
   backoff-to-idle instead of the fixed probe interval.

Genuine transient failures (5xx, timeouts, connection errors) must keep their
existing full-cadence probing and circuit-breaker behaviour — pinned here by
negative controls so the backoff cannot silently swallow real outages.

Related Tickets:
    - OMN-16900: this fix
    - OMN-2255: original LLM endpoint health checker
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from omnibase_infra.models.health.enum_llm_endpoint_probe_state import (
    EnumLlmEndpointProbeState,
)
from omnibase_infra.protocols import ProtocolEventBusLike
from omnibase_infra.services.service_llm_endpoint_health import (
    ModelLlmEndpointHealthConfig,
    ServiceLlmEndpointHealth,
)

# =============================================================================
# Helpers
# =============================================================================


class FakeClock:
    """Deterministic monotonic clock so backoff windows are testable."""

    def __init__(self) -> None:
        self._now = 1_000.0

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


@pytest.fixture
def mock_event_bus() -> AsyncMock:
    """Return a mock ProtocolEventBusLike."""
    bus = AsyncMock(spec=ProtocolEventBusLike)
    bus.publish_envelope = AsyncMock()
    return bus


@pytest.fixture
def registry_with_auth_gated_model(tmp_path: Path) -> Path:
    """Write a registry mirroring the live GLM entries (url env + api key env)."""
    registry = tmp_path / "model_registry.yaml"
    registry.write_text(
        textwrap.dedent(
            """
            models:
              - model_key: qwen3-coder-30b
                provider: local
                transport: http
                base_url_env: LLM_CODER_URL
                health_path: /health
              - model_key: glm-4.5
                provider: zhipu
                transport: http
                base_url_env: LLM_GLM_URL
                api_key_env: LLM_GLM_API_KEY
              - model_key: glm-5
                provider: zhipu
                transport: http
                base_url_env: LLM_GLM_URL
                api_key_env: LLM_GLM_API_KEY
            """
        ).strip(),
        encoding="utf-8",
    )
    return registry


# =============================================================================
# Config-level partitioning
# =============================================================================


class TestConfigAuthPartition:
    """``from_model_registry`` must split probeable from auth-dead endpoints."""

    @pytest.mark.unit
    def test_missing_api_key_env_partitions_out_of_probe_set(
        self,
        registry_with_auth_gated_model: Path,
    ) -> None:
        """A declared api_key_env that resolves to None must not be probeable."""
        env = {
            "LLM_CODER_URL": "http://192.168.86.201:8000",
            "LLM_GLM_URL": "https://api.z.ai/api/coding/paas/v4",
            # LLM_GLM_API_KEY deliberately absent — operator removed the key.
        }

        cfg = ModelLlmEndpointHealthConfig.from_model_registry(
            registry_path=registry_with_auth_gated_model,
            env_resolver=env.get,
        )

        assert cfg.endpoints == {"qwen3-coder-30b": "http://192.168.86.201:8000"}
        assert cfg.unauthenticated_endpoints == {
            "glm-4.5": "https://api.z.ai/api/coding/paas/v4",
            "glm-5": "https://api.z.ai/api/coding/paas/v4",
        }

    @pytest.mark.unit
    def test_empty_api_key_env_is_also_unresolvable(
        self,
        registry_with_auth_gated_model: Path,
    ) -> None:
        """An api_key_env set to the empty string counts as absent, not valid."""
        env = {
            "LLM_GLM_URL": "https://api.z.ai/api/coding/paas/v4",
            "LLM_GLM_API_KEY": "",
        }

        cfg = ModelLlmEndpointHealthConfig.from_model_registry(
            registry_path=registry_with_auth_gated_model,
            env_resolver=env.get,
        )

        assert cfg.endpoints == {}
        assert set(cfg.unauthenticated_endpoints) == {"glm-4.5", "glm-5"}

    @pytest.mark.unit
    def test_present_api_key_env_keeps_endpoint_probeable(
        self,
        registry_with_auth_gated_model: Path,
    ) -> None:
        """Negative control: a resolvable key leaves the endpoint in the probe set."""
        env = {
            "LLM_GLM_URL": "https://api.z.ai/api/coding/paas/v4",
            "LLM_GLM_API_KEY": "a-real-key",
        }

        cfg = ModelLlmEndpointHealthConfig.from_model_registry(
            registry_path=registry_with_auth_gated_model,
            env_resolver=env.get,
        )

        assert set(cfg.endpoints) == {"glm-4.5", "glm-5"}
        assert cfg.unauthenticated_endpoints == {}

    @pytest.mark.unit
    def test_endpoint_cannot_be_both_probeable_and_unauthenticated(self) -> None:
        """The two maps must be disjoint — a name in both is a wiring bug."""
        with pytest.raises(ValueError, match="both 'endpoints' and"):
            ModelLlmEndpointHealthConfig(
                endpoints={"glm": "https://api.z.ai/v4"},
                unauthenticated_endpoints={"glm": "https://api.z.ai/v4"},
            )


# =============================================================================
# SKIPPED_NO_AUTH — zero recurring probes, one classified status
# =============================================================================


class TestSkippedNoAuth:
    """An endpoint with an unresolvable auth secret is classified, never probed."""

    @pytest.mark.asyncio
    async def test_missing_auth_env_produces_zero_probes_and_one_status(
        self,
        mock_event_bus: AsyncMock,
    ) -> None:
        """THE red test: no recurring probes, exactly one classified status."""
        cfg = ModelLlmEndpointHealthConfig(
            endpoints={},
            unauthenticated_endpoints={
                "glm-4.5": "https://api.z.ai/api/coding/paas/v4"
            },
            probe_interval_seconds=30.0,
        )
        service = ServiceLlmEndpointHealth(config=cfg, event_bus=mock_event_bus)

        seen_urls: list[str] = []

        async def mock_get(url: str, **kwargs: object) -> httpx.Response:
            seen_urls.append(url)
            return httpx.Response(401, request=httpx.Request("GET", url))

        with patch.object(httpx.AsyncClient, "get", side_effect=mock_get):
            for _ in range(3):
                await service.probe_all()

        # Zero probes, ever.
        assert seen_urls == []

        # Exactly one status, classified, and stable across cycles.
        status_map = service.get_status()
        assert list(status_map) == ["glm-4.5"]
        status = status_map["glm-4.5"]
        assert status.probe_state is EnumLlmEndpointProbeState.SKIPPED_NO_AUTH
        assert status.available is False
        assert status.latency_ms == -1.0
        assert "auth" in status.error.lower()

    @pytest.mark.asyncio
    async def test_skipped_status_timestamp_does_not_churn(
        self,
        mock_event_bus: AsyncMock,
    ) -> None:
        """The classification is made once; later cycles must not restamp it."""
        cfg = ModelLlmEndpointHealthConfig(
            endpoints={},
            unauthenticated_endpoints={"glm-4.5": "https://api.z.ai/v4"},
        )
        service = ServiceLlmEndpointHealth(config=cfg, event_bus=mock_event_bus)

        first = service.get_status()["glm-4.5"].last_check
        for _ in range(3):
            await service.probe_all()

        assert service.get_status()["glm-4.5"].last_check == first

    @pytest.mark.asyncio
    async def test_skipped_endpoint_emitted_once_not_every_cycle(
        self,
        mock_event_bus: AsyncMock,
    ) -> None:
        """The classification reaches the bus once, not on every 30s tick."""
        cfg = ModelLlmEndpointHealthConfig(
            endpoints={},
            unauthenticated_endpoints={"glm-4.5": "https://api.z.ai/v4"},
        )
        service = ServiceLlmEndpointHealth(config=cfg, event_bus=mock_event_bus)

        for _ in range(5):
            await service.probe_all()

        assert mock_event_bus.publish_envelope.await_count == 1

    @pytest.mark.asyncio
    async def test_skipped_endpoint_gets_no_circuit_breaker(
        self,
        mock_event_bus: AsyncMock,
    ) -> None:
        """No probe means no circuit breaker state to maintain."""
        cfg = ModelLlmEndpointHealthConfig(
            endpoints={"coder": "http://192.168.86.201:8000"},
            unauthenticated_endpoints={"glm-4.5": "https://api.z.ai/v4"},
        )
        service = ServiceLlmEndpointHealth(config=cfg, event_bus=mock_event_bus)

        assert set(service.circuit_breaker_names) == {"coder"}


# =============================================================================
# AUTH_FAILED — sustained 401/403 is terminal, backed off to idle
# =============================================================================


class TestTerminalAuthFailure:
    """Sustained 401/403 must back off, not hammer at the probe interval."""

    @pytest.mark.asyncio
    async def test_sustained_401_becomes_terminal_and_stops_probing(
        self,
        mock_event_bus: AsyncMock,
    ) -> None:
        """After the threshold, further cycles issue no HTTP until backoff elapses."""
        clock = FakeClock()
        cfg = ModelLlmEndpointHealthConfig(
            endpoints={"glm": "https://api.z.ai/api/coding/paas/v4/chat/completions"},
            probe_interval_seconds=30.0,
            auth_failure_threshold=2,
        )
        service = ServiceLlmEndpointHealth(
            config=cfg, event_bus=mock_event_bus, monotonic=clock
        )

        health_hits = 0
        hits_by_cycle: list[int] = []

        async def mock_get(url: str, **kwargs: object) -> httpx.Response:
            nonlocal health_hits
            if url.endswith("/health"):
                health_hits += 1
            return httpx.Response(401, request=httpx.Request("GET", url))

        with patch.object(httpx.AsyncClient, "get", side_effect=mock_get):
            # Cycles 1 and 2 (t=1000) reach the threshold and arm a 60s window.
            await service.probe_all()
            await service.probe_all()

            # Six further 30s ticks: t = 1030, 1060, 1090, 1120, 1150, 1180.
            for _ in range(6):
                clock.advance(30.0)
                await service.probe_all()
                hits_by_cycle.append(health_hits)

        # At fixed cadence those 8 cycles would have produced 8 probes.
        # With backoff-to-idle only t=1060 (window 60s) and t=1180 (window
        # doubled to 120s) probe; the other four ticks are silent.
        assert hits_by_cycle == [2, 3, 3, 3, 3, 4], (
            "auth-dead endpoint was re-probed inside its backoff window"
        )
        assert health_hits == 4
        status = service.get_status()["glm"]
        assert status.probe_state is EnumLlmEndpointProbeState.AUTH_FAILED
        assert status.available is False
        assert "401" in status.error

    @pytest.mark.asyncio
    async def test_auth_failure_does_not_open_the_circuit_breaker(
        self,
        mock_event_bus: AsyncMock,
    ) -> None:
        """Auth state is tracked separately — it must not poison route health.

        Preserves the intent of the original OMN-13699-era test: a 401 from a
        cloud provider proves the route is reachable.  What changes is that the
        endpoint is no longer reported *available*.
        """
        clock = FakeClock()
        cfg = ModelLlmEndpointHealthConfig(
            endpoints={"glm": "https://api.z.ai/api/coding/paas/v4/chat/completions"},
            circuit_breaker_threshold=1,
            auth_failure_threshold=1,
        )
        service = ServiceLlmEndpointHealth(
            config=cfg, event_bus=mock_event_bus, monotonic=clock
        )

        async def mock_get(url: str, **kwargs: object) -> httpx.Response:
            return httpx.Response(401, request=httpx.Request("GET", url))

        with patch.object(httpx.AsyncClient, "get", side_effect=mock_get):
            status_map = await service.probe_all()

        status = status_map["glm"]
        assert status.circuit_state == "closed"
        assert status.probe_state is EnumLlmEndpointProbeState.AUTH_FAILED

    @pytest.mark.asyncio
    async def test_backoff_grows_and_endpoint_is_retried_after_the_window(
        self,
        mock_event_bus: AsyncMock,
    ) -> None:
        """Backoff-to-idle, not backoff-to-never: the endpoint is retried eventually."""
        clock = FakeClock()
        cfg = ModelLlmEndpointHealthConfig(
            endpoints={"glm": "https://api.z.ai/v4"},
            probe_interval_seconds=30.0,
            auth_failure_threshold=1,
            auth_failure_backoff_max_seconds=3600.0,
        )
        service = ServiceLlmEndpointHealth(
            config=cfg, event_bus=mock_event_bus, monotonic=clock
        )

        probe_cycles = 0

        async def mock_get(url: str, **kwargs: object) -> httpx.Response:
            nonlocal probe_cycles
            if url.endswith("/health"):
                probe_cycles += 1
            return httpx.Response(403, request=httpx.Request("GET", url))

        with patch.object(httpx.AsyncClient, "get", side_effect=mock_get):
            await service.probe_all()
            assert probe_cycles == 1

            # First backoff window is 60s (2 x the 30s interval).
            clock.advance(59.0)
            await service.probe_all()
            assert probe_cycles == 1

            clock.advance(2.0)
            await service.probe_all()
            assert probe_cycles == 2

            # Second failure doubles the window to 120s.
            clock.advance(61.0)
            await service.probe_all()
            assert probe_cycles == 2

            clock.advance(60.0)
            await service.probe_all()
            assert probe_cycles == 3

    @pytest.mark.asyncio
    async def test_recovery_clears_auth_backoff(
        self,
        mock_event_bus: AsyncMock,
    ) -> None:
        """A restored key must return the endpoint to full cadence immediately."""
        clock = FakeClock()
        cfg = ModelLlmEndpointHealthConfig(
            endpoints={"glm": "https://api.z.ai/v4"},
            probe_interval_seconds=30.0,
            auth_failure_threshold=1,
        )
        service = ServiceLlmEndpointHealth(
            config=cfg, event_bus=mock_event_bus, monotonic=clock
        )

        status_code = 401

        async def mock_get(url: str, **kwargs: object) -> httpx.Response:
            return httpx.Response(status_code, request=httpx.Request("GET", url))

        with patch.object(httpx.AsyncClient, "get", side_effect=mock_get):
            await service.probe_all()
            assert (
                service.get_status()["glm"].probe_state
                is EnumLlmEndpointProbeState.AUTH_FAILED
            )

            status_code = 200
            clock.advance(61.0)
            await service.probe_all()
            assert (
                service.get_status()["glm"].probe_state
                is EnumLlmEndpointProbeState.HEALTHY
            )

            # Back at full cadence — no residual backoff.
            clock.advance(1.0)
            await service.probe_all()
            assert service.get_status()["glm"].available is True


# =============================================================================
# Negative controls — transient failure behaviour must NOT change
# =============================================================================


class TestTransientProbingUnchanged:
    """5xx / timeouts keep full-cadence probing and circuit-breaker handling."""

    @pytest.mark.asyncio
    async def test_sustained_5xx_is_probed_every_cycle(
        self,
        mock_event_bus: AsyncMock,
    ) -> None:
        """A real outage must not be silently backed off into invisibility."""
        clock = FakeClock()
        cfg = ModelLlmEndpointHealthConfig(
            endpoints={"coder": "http://192.168.86.201:8000"},
            probe_interval_seconds=30.0,
            circuit_breaker_threshold=100,
        )
        service = ServiceLlmEndpointHealth(
            config=cfg, event_bus=mock_event_bus, monotonic=clock
        )

        health_probes = 0

        async def mock_get(url: str, **kwargs: object) -> httpx.Response:
            nonlocal health_probes
            if url.endswith("/health"):
                health_probes += 1
            return httpx.Response(503, request=httpx.Request("GET", url))

        with patch.object(httpx.AsyncClient, "get", side_effect=mock_get):
            for _ in range(4):
                clock.advance(30.0)
                await service.probe_all()

        assert health_probes == 4
        status = service.get_status()["coder"]
        assert status.probe_state is EnumLlmEndpointProbeState.UNAVAILABLE
        assert status.available is False

    @pytest.mark.asyncio
    async def test_connection_error_is_probed_every_cycle(
        self,
        mock_event_bus: AsyncMock,
    ) -> None:
        """Timeouts/connection errors stay transient, at full cadence."""
        clock = FakeClock()
        cfg = ModelLlmEndpointHealthConfig(
            endpoints={"coder": "http://192.168.86.201:8000"},
            probe_interval_seconds=30.0,
            circuit_breaker_threshold=100,
        )
        service = ServiceLlmEndpointHealth(
            config=cfg, event_bus=mock_event_bus, monotonic=clock
        )

        health_probes = 0

        async def mock_get(url: str, **kwargs: object) -> httpx.Response:
            nonlocal health_probes
            if url.endswith("/health"):
                health_probes += 1
            raise httpx.ConnectTimeout("timed out")

        with patch.object(httpx.AsyncClient, "get", side_effect=mock_get):
            for _ in range(4):
                clock.advance(30.0)
                await service.probe_all()

        assert health_probes == 4
        assert (
            service.get_status()["coder"].probe_state
            is EnumLlmEndpointProbeState.UNAVAILABLE
        )

    @pytest.mark.asyncio
    async def test_healthy_endpoint_is_probed_every_cycle(
        self,
        mock_event_bus: AsyncMock,
    ) -> None:
        """Negative control: nothing about the happy path changes."""
        clock = FakeClock()
        cfg = ModelLlmEndpointHealthConfig(
            endpoints={"coder": "http://192.168.86.201:8000"},
            probe_interval_seconds=30.0,
        )
        service = ServiceLlmEndpointHealth(
            config=cfg, event_bus=mock_event_bus, monotonic=clock
        )

        health_probes = 0

        async def mock_get(url: str, **kwargs: object) -> httpx.Response:
            nonlocal health_probes
            health_probes += 1
            return httpx.Response(200, request=httpx.Request("GET", url))

        with patch.object(httpx.AsyncClient, "get", side_effect=mock_get):
            for _ in range(3):
                clock.advance(30.0)
                await service.probe_all()

        assert health_probes == 3
        status = service.get_status()["coder"]
        assert status.probe_state is EnumLlmEndpointProbeState.HEALTHY
        assert status.available is True
