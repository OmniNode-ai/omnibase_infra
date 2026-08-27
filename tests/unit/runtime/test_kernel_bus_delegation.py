# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The kernel delegates bus resolution wholly to the authority (OMN-16693).

OMN-16678 landed ``backends/auto_configure.py::resolve_bus_type`` as the single
event-bus resolution authority, but left ``service_kernel.py`` holding a SECOND,
independent ``ONEX_EVENT_BUS_TYPE`` read with its own precedence ladder. Two
defects followed from that duplication:

1. **A dead branch whose warning was false.** The kernel logged
   ``Invalid ONEX_EVENT_BUS_TYPE value '<x>' ... Falling back to
   config.event_bus.type='<y>'`` and then continued into ``select_event_bus``,
   which routes to ``resolve_bus_type``, which RAISES on that same value. The
   fallback was unreachable, so a typo hard-failed kernel boot *after* telling
   the operator, in the log, that the runtime had recovered. Verified with
   ``ONEX_EVENT_BUS_TYPE=kakfa``.

2. **The declared config intent reached no decision.** ``config.event_bus.type``
   was used only to decide whether to forward ``KAFKA_BOOTSTRAP_SERVERS``; the
   live broker probe then chose the transport. A contract declaring
   ``event_bus.type: kafka`` could still resolve to in-memory when the broker
   was down (the OMN-14376 failure class), and a transient metadata timeout
   could fail boot outright even though the contract was unambiguous.

These tests pin the delegation: the kernel owns the fail-fast
bootstrap-servers check and nothing else about *which* transport is chosen.

The probe is always stubbed. No test in this module opens a socket.
"""

from __future__ import annotations

import logging
from pathlib import Path
from uuid import uuid4

import pytest

from omnibase_infra.backends import auto_configure
from omnibase_infra.backends.auto_configure import (
    BUS_INMEMORY,
    BUS_KAFKA,
    BUS_TYPE_OVERRIDE_ENV,
    EventBusResolutionAmbiguousError,
)
from omnibase_infra.backends.enum_probe_state import EnumProbeState
from omnibase_infra.backends.model_probe_result import ModelProbeResult
from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.runtime import service_kernel
from omnibase_infra.runtime.service_kernel import _resolve_event_bus_transport

pytestmark = pytest.mark.unit

_BROKER = "broker.example:9092"


def _stub_probe(
    monkeypatch: pytest.MonkeyPatch,
    state: EnumProbeState,
    *,
    reason: str = "stub probe",
) -> list[dict[str, object]]:
    """Replace ``probe_kafka`` with a stub and return its recorded call log."""
    calls: list[dict[str, object]] = []

    def _probe(
        *, bootstrap_servers: str | None = None, authority_topic: str | None = None
    ) -> ModelProbeResult:
        calls.append(
            {"bootstrap_servers": bootstrap_servers, "authority_topic": authority_topic}
        )
        return ModelProbeResult(
            state=state, reason=reason, backend_label="event_bus_kafka"
        )

    monkeypatch.setattr(auto_configure, "probe_kafka", _probe)
    return calls


class TestTheMisleadingFallbackIsGone:
    """AC1: no second env read, no dead branch, no warning that contradicts the outcome."""

    def test_typo_override_surfaces_the_authority_error_without_promising_a_fallback(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A typo'd override fails loud — and says nothing about falling back.

        Pre-OMN-16693 this exact input logged WARNING "Falling back to
        config.event_bus.type='kafka'" and then raised anyway one call later.
        The log recorded a recovery that never happened.
        """
        monkeypatch.setenv(BUS_TYPE_OVERRIDE_ENV, "kakfa")
        _stub_probe(monkeypatch, EnumProbeState.AUTHORITATIVE)

        with caplog.at_level(logging.DEBUG):
            with pytest.raises(ValueError) as excinfo:
                _resolve_event_bus_transport(
                    config_bus_type="kafka",
                    kafka_bootstrap_servers=_BROKER,
                    correlation_id=uuid4(),
                )

        message = str(excinfo.value)
        assert BUS_TYPE_OVERRIDE_ENV in message
        assert "kakfa" in message

        emitted = "\n".join(record.getMessage() for record in caplog.records)
        assert "Falling back" not in emitted
        assert "Invalid ONEX_EVENT_BUS_TYPE" not in emitted

    def test_kernel_source_holds_no_second_resolution_ladder(self) -> None:
        """The duplication itself is the defect — pin its absence, not just its effect.

        A behavioral test alone would pass again the moment someone reintroduces
        a local ``os.getenv("ONEX_EVENT_BUS_TYPE")`` that happens to agree with
        the authority today and drifts from it tomorrow. That drift is exactly
        what OMN-16678 fixed once already.
        """
        source = Path(service_kernel.__file__).read_text(encoding="utf-8")

        for read_form in (
            f'os.getenv("{BUS_TYPE_OVERRIDE_ENV}"',
            f'os.environ["{BUS_TYPE_OVERRIDE_ENV}"',
            f'os.environ.get("{BUS_TYPE_OVERRIDE_ENV}"',
        ):
            assert read_form not in source, (
                f"service_kernel must not read {BUS_TYPE_OVERRIDE_ENV}; resolution "
                f"belongs to backends/auto_configure.py::resolve_bus_type. "
                f"Found: {read_form}"
            )

        # The dead branch and the local precedence ladder it belonged to.
        assert "Falling back to config.event_bus.type" not in source
        assert "_broker_required_types" not in source


class TestConfigIntentReachesTheDecision:
    """AC2: config.event_bus.type is a resolution tier, not a bootstrap-servers hint."""

    def test_declared_config_beats_a_disagreeing_probe(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """DISCOVERED (broker down) no longer silently downgrades a kafka contract.

        This is the OMN-14376 failure class: a runtime that declares kafka but
        boots in-memory publishes to nothing, and no caller can tell.
        """
        monkeypatch.delenv(BUS_TYPE_OVERRIDE_ENV, raising=False)
        calls = _stub_probe(monkeypatch, EnumProbeState.DISCOVERED)

        bus, reason = _resolve_event_bus_transport(
            config_bus_type="kafka",
            kafka_bootstrap_servers=_BROKER,
            correlation_id=uuid4(),
        )

        assert bus == BUS_KAFKA
        assert "config.event_bus.type" in reason
        assert calls == [], "a declared transport must not be probed against"

    def test_declared_config_makes_an_indeterminate_probe_irrelevant(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A flaky metadata timeout must not fail boot when the contract is explicit."""
        monkeypatch.delenv(BUS_TYPE_OVERRIDE_ENV, raising=False)
        _stub_probe(monkeypatch, EnumProbeState.REACHABLE)

        bus, _reason = _resolve_event_bus_transport(
            config_bus_type="kafka",
            kafka_bootstrap_servers=_BROKER,
            correlation_id=uuid4(),
        )

        assert bus == BUS_KAFKA

    def test_cloud_config_resolves_to_the_broker_backed_transport(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``EnumEventBusType.CLOUD`` is production-safe and broker-backed.

        Before the vocabulary was unified, passing ``cloud`` as an explicit
        selection raised while the same word set in the env var resolved fine.
        """
        monkeypatch.delenv(BUS_TYPE_OVERRIDE_ENV, raising=False)
        _stub_probe(monkeypatch, EnumProbeState.DISCOVERED)

        bus, _reason = _resolve_event_bus_transport(
            config_bus_type="cloud",
            kafka_bootstrap_servers=_BROKER,
            correlation_id=uuid4(),
        )

        assert bus == BUS_KAFKA

    def test_env_override_still_beats_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The CI contract, pinned.

        ``contracts/runtime/runtime_config.yaml`` ships ``event_bus.type: kafka``
        and documents ``ONEX_EVENT_BUS_TYPE`` as its override; eight workflows
        set ``ONEX_EVENT_BUS_TYPE: inmemory`` against those same contracts.
        Config outranking the env var would break every one of them on the
        missing-KAFKA_BOOTSTRAP_SERVERS guard below.
        """
        monkeypatch.setenv(BUS_TYPE_OVERRIDE_ENV, "inmemory")
        _stub_probe(monkeypatch, EnumProbeState.AUTHORITATIVE)

        bus, reason = _resolve_event_bus_transport(
            config_bus_type="kafka",
            kafka_bootstrap_servers=None,
            correlation_id=uuid4(),
        )

        assert bus == BUS_INMEMORY
        assert BUS_TYPE_OVERRIDE_ENV in reason


class TestTheKernelKeepsItsFailFastGuard:
    """Resolution moved to the authority; the bootstrap-servers check did not."""

    def test_kafka_without_bootstrap_servers_fails_fast(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Preserves the pre-existing guard against an implicit localhost:9092."""
        monkeypatch.delenv(BUS_TYPE_OVERRIDE_ENV, raising=False)
        _stub_probe(monkeypatch, EnumProbeState.DISCOVERED)

        with pytest.raises(ProtocolConfigurationError) as excinfo:
            _resolve_event_bus_transport(
                config_bus_type="kafka",
                kafka_bootstrap_servers=None,
                correlation_id=uuid4(),
            )

        message = str(excinfo.value)
        assert "KAFKA_BOOTSTRAP_SERVERS" in message
        # The operator must be told WHY kafka was selected, not just that it was.
        assert "config.event_bus.type" in message

    def test_inmemory_needs_no_broker(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(BUS_TYPE_OVERRIDE_ENV, "inmemory")
        _stub_probe(monkeypatch, EnumProbeState.DISCOVERED)

        bus, _reason = _resolve_event_bus_transport(
            config_bus_type="kafka",
            kafka_bootstrap_servers=None,
            correlation_id=uuid4(),
        )

        assert bus == BUS_INMEMORY

    def test_indeterminate_probe_still_refuses_when_nothing_declares_intent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The OMN-16678 anti-coin-flip behavior survives the delegation."""
        monkeypatch.delenv(BUS_TYPE_OVERRIDE_ENV, raising=False)
        _stub_probe(monkeypatch, EnumProbeState.REACHABLE)

        with pytest.raises(EventBusResolutionAmbiguousError):
            _resolve_event_bus_transport(
                config_bus_type=None,
                kafka_bootstrap_servers=_BROKER,
                correlation_id=uuid4(),
            )
