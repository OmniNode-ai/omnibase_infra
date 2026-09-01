# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Shared pytest fixtures for runtime unit tests.

This conftest.py provides fixtures commonly used across runtime tests,
consolidating shared mocks to avoid code duplication.

Fixtures:
    mock_wire_infrastructure: Mocks wire_infrastructure_services and
        ModelONEXContainer to avoid wiring errors in tests.
    mock_runtime_handler: Auto-discovered from root tests/conftest.py via
        pytest's conftest hierarchy (not re-exported here).

Functions:
    seed_mock_handlers: Imported from tests.helpers.runtime_helpers for
        fail-fast bypass in RuntimeHostProcess tests.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnibase_infra.runtime.models.model_runtime_node_graph_config import (
    ModelRuntimeNodeGraphConfig,
)
from omnibase_infra.runtime.registry import RegistryProtocolBinding

# Import handler seeding utilities from canonical location.
# mock_runtime_handler is a pytest fixture defined in root conftest.py and
# is automatically available to all tests via pytest's conftest discovery.
# seed_mock_handlers is a regular function from runtime_helpers.
from tests.helpers.runtime_helpers import seed_mock_handlers

__all__ = ["seed_mock_handlers"]

if TYPE_CHECKING:
    from collections.abc import Generator


_TERMINAL_CORRELATOR_THREAD_PREFIX = "terminal-correlator-"


@pytest.fixture(autouse=True)
def reap_terminal_correlator_threads() -> Generator[None, None, None]:
    """Reap leaked terminal-event correlator daemon threads after each test (OMN-14708).

    Tests that construct the terminal-event consumer -- directly as
    ``LongLivedTerminalCorrelator`` or through the auto-wiring path that builds a
    ``TerminalEventConsumer`` (which owns a correlator) -- start a daemon thread
    named ``terminal-correlator-<handler>`` running a dedicated asyncio event
    loop for the process lifetime. When such a test does not close the consumer,
    that thread survives into later tests in the shared single-process slice and,
    under CI load, starves the event loop that
    ``test_bootstrap_uses_config_grace_period`` depends on, pushing it past its
    60s timeout (a nondeterministic hang that greens on a clean dev slice).

    This autouse fixture tracks every correlator constructed during the test,
    closes each at teardown (``close()`` is idempotent, so a test that already
    closed its own consumer is unaffected), then fails the test if any
    ``terminal-correlator`` daemon thread survives -- turning a silent
    cross-test leak into a deterministic, locally-attributable failure.
    """
    import omnibase_infra.runtime.service_terminal_event_consumer as _stec

    correlator_cls = _stec.LongLivedTerminalCorrelator
    created: list[object] = []
    original_init = correlator_cls.__init__

    def _tracking_init(self: object, *args: object, **kwargs: object) -> None:
        original_init(self, *args, **kwargs)  # type: ignore[arg-type]
        created.append(self)

    with patch.object(correlator_cls, "__init__", _tracking_init):
        yield

    for correlator in created:
        close = getattr(correlator, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass

    survivors = sorted(
        thread.name
        for thread in threading.enumerate()
        if thread.name.startswith(_TERMINAL_CORRELATOR_THREAD_PREFIX)
    )
    assert not survivors, (
        "terminal-event correlator daemon threads survived teardown: "
        f"{survivors}. A test constructed the terminal-event consumer without "
        "closing it; the leaked thread starves later tests in the shared slice "
        "(OMN-14708 test-isolation leak)."
    )


@pytest.fixture
def mock_wire_infrastructure() -> Generator[MagicMock, None, None]:
    """Mock wire_infrastructure_services and container to avoid wiring errors in tests.

    This fixture mocks:
    1. wire_infrastructure_services - to be a no-op async function
    2. ModelONEXContainer - to have a mock service_registry with resolve_service
    3. wire_from_manifest - to return a clean no-wiring report (OMN-8735:
       auto-wiring now raises on failure; kernel tests that don't test
       auto-wiring must skip it). The stand-in still honours the wiring-report
       totality contract (OMN-15474) — see ``noop_wire_from_manifest``.

    Note: Returns a real RegistryProtocolBinding for handler registration to work.
    """
    from omnibase_infra.runtime.auto_wiring import (
        build_unwired_contract_results,
    )
    from omnibase_infra.runtime.auto_wiring.models import ModelAutoWiringManifest
    from omnibase_infra.runtime.auto_wiring.report import (
        ModelAutoWiringReport,
    )

    # Create a shared registry instance that will be used throughout the test
    shared_registry = RegistryProtocolBinding()

    async def noop_wire(container: object) -> dict[str, list[str]]:
        """Async no-op for wire_infrastructure_services."""
        return {"services": []}

    async def mock_resolve_service(
        service_class: type,
    ) -> MagicMock | RegistryProtocolBinding:
        """Mock resolve_service to return appropriate instances.

        Returns a real RegistryProtocolBinding for handler registration,
        and MagicMock for other service types.
        """
        if service_class == RegistryProtocolBinding:
            return shared_registry
        return MagicMock()

    async def noop_wire_from_manifest(**kwargs: object) -> ModelAutoWiringReport:
        """Return a TOTAL no-wiring report for the manifest the kernel holds.

        Kernel tests that don't test auto-wiring should not hit the real
        wire_from_manifest which now raises on any failure (OMN-8735).

        OMN-15474: this stand-in must still satisfy the wiring-report totality
        contract. ``discover_contracts`` is NOT mocked by this fixture, so
        bootstrap holds a real manifest (>100 contracts); returning
        ``results=()`` asserted the impossible — that discovery found those
        contracts and wiring produced no verdict on any of them. The real
        ``wire_from_manifest`` never does that (verified: 131/131 rows over the
        live discovered manifest), and the initial-subscription identity check
        correctly rejected the fixture's claim. The truthful encoding of "the
        wiring engine was stubbed out and nothing wired" is one explicit
        SKIPPED row per manifest contract, built from the manifest the kernel
        actually passed via the product's own totality constructor — never a
        hand-mirrored name list, which would just re-create the matched-pair
        fixture that hid this in the first place.
        """
        manifest = kwargs.get("manifest")
        if not isinstance(manifest, ModelAutoWiringManifest):
            return ModelAutoWiringReport(results=(), duplicates=())
        return ModelAutoWiringReport(
            results=build_unwired_contract_results(
                manifest,
                reason=(
                    "auto-wiring engine stubbed out by the "
                    "mock_wire_infrastructure test fixture"
                ),
            ),
            duplicates=(),
        )

    with patch(
        "omnibase_infra.runtime.service_kernel.wire_infrastructure_services"
    ) as mock_wire:
        mock_wire.side_effect = noop_wire

        with patch(
            "omnibase_infra.runtime.service_kernel.ModelONEXContainer"
        ) as mock_container_cls:
            mock_container = MagicMock()
            mock_service_registry = MagicMock()
            mock_service_registry.resolve_service = AsyncMock(
                side_effect=mock_resolve_service
            )
            # Also mock register_instance as AsyncMock to avoid
            # "object MagicMock can't be used in 'await' expression" errors
            # when wire_registration_handlers calls await register_instance(...)
            mock_service_registry.register_instance = AsyncMock(
                return_value="mock-uuid"
            )
            mock_container.service_registry = mock_service_registry
            mock_container_cls.return_value = mock_container

            # OMN-8735: wire_from_manifest now raises on any wiring failure.
            # Mock it at the source module so the deferred import inside
            # bootstrap() picks up the mock (kernel imports lazily).
            with patch(
                "omnibase_infra.runtime.auto_wiring.wire_from_manifest",
                side_effect=noop_wire_from_manifest,
            ):
                yield mock_wire


def force_inmemory_runtime_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Path:
    """Point the kernel at a LOCAL-PROFILE inmemory runtime config (OMN-17304).

    ONEX_EVENT_BUS_TYPE holds no tier in transport resolution any more, so
    forcing the in-memory bus is done the ruled way: a per-runtime config
    declaring ``event_bus.type: inmemory`` under ``event_bus.profile: local``
    (the first-class configured form of the shipped tier-0 default), reached
    through the ``ONEX_CONTRACTS_DIR`` bootstrap pointer — an env var may name
    WHERE config lives, never WHAT the transport is.

    Also clears ``KAFKA_BOOTSTRAP_SERVERS`` and the dead
    ``ONEX_EVENT_BUS_TYPE`` so an ambient developer shell cannot leak either a
    broker or a set-and-ignored warning into the test.

    Returns:
        The contracts directory the pointer names (for tests that need it).
    """
    contracts_dir = tmp_path / "inmemory-contracts"
    (contracts_dir / "runtime").mkdir(parents=True, exist_ok=True)
    (contracts_dir / "runtime" / "runtime_config.yaml").write_text(
        (
            # name is load-bearing: bootstrap resolves service_name from it
            # and fails closed when absent.
            'name: "runtime_config"\n'
            'description: "Test-local inmemory runtime config (OMN-17304)"\n'
            'input_topic: "requests"\n'
            'output_topic: "responses"\n'
            'group_id: "onex-runtime"\n'
            "event_bus:\n"
            '  type: "inmemory"\n'
            '  profile: "local"\n'
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ONEX_CONTRACTS_DIR", str(contracts_dir))
    monkeypatch.delenv("ONEX_EVENT_BUS_TYPE", raising=False)
    monkeypatch.delenv("KAFKA_BOOTSTRAP_SERVERS", raising=False)
    return contracts_dir


@pytest.fixture
def mock_inmemory_runtime_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Generator[MagicMock, None, None]:
    """Force the inmemory event bus via a configured local-profile runtime config.

    OMN-17304: the previous mechanism (``ONEX_EVENT_BUS_TYPE=inmemory``) is
    dead — that env var holds no tier in the shared resolution order any more.
    The ruled mechanism is per-runtime configuration: this fixture writes a
    runtime config declaring ``event_bus.type: inmemory`` with
    ``event_bus.profile: local`` into a tmp contracts dir and points the
    ``ONEX_CONTRACTS_DIR`` bootstrap pointer at it, so the kernel resolves the
    in-memory bus from its OWN configured authority — the same way every other
    runtime resolves its transport.

    Yields:
        MagicMock (for backwards compatibility with tests expecting a MagicMock).
    """
    force_inmemory_runtime_config(monkeypatch, tmp_path)

    # Return MagicMock for backwards compatibility with tests that
    # reference the fixture but don't actually use the mock object
    return MagicMock()


def _default_node_graph_config() -> ModelRuntimeNodeGraphConfig:
    """Build a sensible default config for tests that don't need real contract YAMLs."""
    return ModelRuntimeNodeGraphConfig(
        startup_timeout_ms=120000,
        step_timeout_ms=30000,
        max_step_retries=3,
        retry_backoff_ms=2000,
        retry_backoff_multiplier=2.0,
        drain_timeout_ms=30000,
        max_concurrent_handlers=10,
        handler_pool_size=10,
        health_check_timeout_ms=5000,
        batch_response_size=100,
        batch_flush_interval_ms=1000,
        topic_validation_pattern=r"^[a-z][a-z0-9._-]*$",
        topic_deny_patterns=("__consumer_offsets", "_schemas"),
        max_topic_length=255,
        max_subscriptions_per_node=100,
        subscription_timeout_ms=5000,
        circuit_breaker_failure_threshold=5,
        circuit_breaker_timeout_ms=30000,
        wiring_retry_max=3,
        wiring_retry_base_delay_ms=1000,
        wiring_retry_max_delay_ms=10000,
        scan_exclude_patterns=("__pycache__", ".git"),
        scan_deny_paths=("/etc", "/var"),
        scan_timeout_ms=60000,
    )


@pytest.fixture(autouse=True)
def mock_load_node_graph_config() -> Generator[MagicMock, None, None]:
    """Mock _load_node_graph_config to avoid FileNotFoundError in CI.

    The real function navigates to omnibase_core's contracts/runtime/ directory
    on disk, which doesn't exist when core is installed from PyPI. This fixture
    returns a sensible default config for all runtime tests.
    """
    with patch(
        "omnibase_infra.runtime.service_kernel._load_node_graph_config",
        return_value=_default_node_graph_config(),
    ) as mock_fn:
        yield mock_fn
