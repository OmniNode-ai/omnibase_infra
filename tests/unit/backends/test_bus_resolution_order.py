# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Resolution-order and indeterminate-probe pins for the event-bus authority (OMN-16678).

``resolve_bus_type`` is the SINGLE authority that decides which transport a
call site gets. Before OMN-16678 there were two divergent implementations:

* ``backends/auto_configure.py::select_event_bus`` honoured
  ``ONEX_EVENT_BUS_TYPE`` and mapped ``REACHABLE`` -> kafka.
* ``cli/cli_delegate.py::resolve_default_bus`` ignored ``ONEX_EVENT_BUS_TYPE``
  entirely and mapped ``REACHABLE`` -> inmemory.

Because ``probe_kafka`` degrades ANY Stage-2 metadata failure (including a
plain 2s ``AdminClient.list_topics`` timeout against a healthy broker) to
``REACHABLE``, that second mapping made the resolved transport a coin flip:
20 consecutive ``resolve_default_bus()`` calls with an unchanged environment
and a healthy broker returned kafka 14x / inmemory 6x
(``OmniNode-ai/knowledge-base#59``).

These tests pin the fix, as amended by OMN-17304:

1. the resolution ORDER — explicit argument > ``config.event_bus.type`` >
   probe — identically for every call site;
2. ``ONEX_EVENT_BUS_TYPE`` holds NO tier (OMN-17304): a set value is ignored
   with a WARNING naming the removal — it never decides the transport and an
   unrecognised value never fails a resolution (a dead variable must not
   retain the power to fail a boot); and
3. the INDETERMINATE-probe behavior — ``REACHABLE`` raises
   :class:`EventBusResolutionAmbiguousError` instead of silently picking a
   transport that varies run to run.

The probe is always mocked. No unit test in this module opens a socket.
"""

from __future__ import annotations

import logging

import pytest

from omnibase_infra.backends import auto_configure
from omnibase_infra.backends.auto_configure import (
    BUS_INMEMORY,
    BUS_KAFKA,
    BUS_TYPE_OVERRIDE_ENV,
    SUPPORTED_BUS_TYPES,
    EventBusResolutionAmbiguousError,
    resolve_bus_type,
)
from omnibase_infra.backends.enum_probe_state import EnumProbeState
from omnibase_infra.backends.model_probe_result import ModelProbeResult

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
            {
                "bootstrap_servers": bootstrap_servers,
                "authority_topic": authority_topic,
            }
        )
        return ModelProbeResult(
            state=state, reason=reason, backend_label="event_bus_kafka"
        )

    monkeypatch.setattr(auto_configure, "probe_kafka", _probe)
    return calls


class TestResolutionOrder:
    """explicit argument > config > probe. One order, every caller (OMN-17304)."""

    def test_explicit_bus_beats_config_env_and_probe(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(BUS_TYPE_OVERRIDE_ENV, "inmemory")
        calls = _stub_probe(monkeypatch, EnumProbeState.DISCOVERED)

        bus, reason = resolve_bus_type(explicit_bus=BUS_KAFKA, config_bus=BUS_INMEMORY)

        assert bus == BUS_KAFKA
        assert "explicit" in reason
        assert calls == [], "explicit selection must not probe the network"

    def test_set_env_var_is_ignored_and_the_probe_decides(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """OMN-17304: ONEX_EVENT_BUS_TYPE holds NO tier.

        Pre-ruling, env=inmemory here returned inmemory without probing. Now
        the var is inert: with no explicit argument and no config, the probe
        decides — and the set-and-ignored state is WARNED about so the export
        gets deleted rather than silently going dead.
        """
        monkeypatch.setenv(BUS_TYPE_OVERRIDE_ENV, "inmemory")
        calls = _stub_probe(monkeypatch, EnumProbeState.AUTHORITATIVE)

        with caplog.at_level(
            logging.WARNING, logger="omnibase_infra.backends.auto_configure"
        ):
            bus, reason = resolve_bus_type(kafka_bootstrap=_BROKER)

        assert bus == BUS_KAFKA
        assert BUS_TYPE_OVERRIDE_ENV not in reason
        assert len(calls) == 1
        warnings = [
            r.getMessage()
            for r in caplog.records
            if BUS_TYPE_OVERRIDE_ENV in r.getMessage()
        ]
        assert warnings and any("ignored" in m for m in warnings)

    def test_unset_or_empty_env_var_produces_no_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Only a SET value warns — clean environments stay silent."""
        monkeypatch.setenv(BUS_TYPE_OVERRIDE_ENV, "")
        _stub_probe(monkeypatch, EnumProbeState.AUTHORITATIVE)

        with caplog.at_level(
            logging.WARNING, logger="omnibase_infra.backends.auto_configure"
        ):
            bus, _reason = resolve_bus_type(kafka_bootstrap=_BROKER)

        assert bus == BUS_KAFKA
        assert not [
            r for r in caplog.records if BUS_TYPE_OVERRIDE_ENV in r.getMessage()
        ]

    def test_unrecognised_env_value_cannot_fail_resolution(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A dead variable must not retain the power to fail a boot.

        Pre-OMN-17304 an unrecognised override raised. Now the var holds no
        tier, so its value is not even validated — resolution proceeds from
        the real ladder and the warning is the only trace.
        """
        monkeypatch.setenv(BUS_TYPE_OVERRIDE_ENV, "redpanda")
        _stub_probe(monkeypatch, EnumProbeState.AUTHORITATIVE)

        with caplog.at_level(
            logging.WARNING, logger="omnibase_infra.backends.auto_configure"
        ):
            bus, _reason = resolve_bus_type(kafka_bootstrap=_BROKER)

        assert bus == BUS_KAFKA
        assert [r for r in caplog.records if "redpanda" in r.getMessage()]

    def test_unsupported_explicit_bus_fails_loud(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_probe(monkeypatch, EnumProbeState.AUTHORITATIVE)

        with pytest.raises(ValueError) as excinfo:
            resolve_bus_type(explicit_bus="redpanda")

        assert "redpanda" in str(excinfo.value)

    def test_bootstrap_and_authority_topic_reach_the_probe(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(BUS_TYPE_OVERRIDE_ENV, raising=False)
        calls = _stub_probe(monkeypatch, EnumProbeState.AUTHORITATIVE)

        resolve_bus_type(kafka_bootstrap=_BROKER, authority_topic="some.topic")

        assert calls == [
            {"bootstrap_servers": _BROKER, "authority_topic": "some.topic"}
        ]


class TestDeclaredConfigTier:
    """The config tier is the top non-explicit tier (OMN-16693, OMN-17304).

    OMN-16678 shipped the tiers, but the runtime kernel supplied none of them
    — it never passed ``explicit_bus``, so ``config.event_bus.type`` reached no
    decision and the live probe chose the transport for every deployed runtime.

    Order is ``explicit argument > config > probe`` (OMN-17304). The env var
    that used to sit between explicit and config holds no tier at all any
    more: a checked-in (or shipped tier-0) runtime config is THE authority for
    every runtime that has one, and every runtime has one.
    """

    def test_config_beats_the_probe(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(BUS_TYPE_OVERRIDE_ENV, raising=False)
        calls = _stub_probe(monkeypatch, EnumProbeState.DISCOVERED)

        bus, reason = resolve_bus_type(config_bus=BUS_KAFKA, kafka_bootstrap=_BROKER)

        assert bus == BUS_KAFKA
        assert "config.event_bus.type" in reason
        assert calls == [], "a declared transport must not be probed against"

    def test_config_beats_the_ignored_env_var(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """OMN-17304 inversion of the old ``test_env_override_beats_config``.

        The pre-ruling ladder ranked the env var ABOVE config, which made one
        line in a shell profile the transport authority for every delegation
        on the machine. Now the configured binding outranks env everywhere —
        because env stopped being a peer authority at all.
        """
        monkeypatch.setenv(BUS_TYPE_OVERRIDE_ENV, "inmemory")
        _stub_probe(monkeypatch, EnumProbeState.AUTHORITATIVE)

        bus, reason = resolve_bus_type(config_bus=BUS_KAFKA, kafka_bootstrap=_BROKER)

        assert bus == BUS_KAFKA
        assert "config.event_bus.type" in reason
        assert BUS_TYPE_OVERRIDE_ENV not in reason

    def test_config_source_is_appended_to_the_reason(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Provenance names WHICH authority answered, not merely that one did."""
        monkeypatch.delenv(BUS_TYPE_OVERRIDE_ENV, raising=False)
        _stub_probe(monkeypatch, EnumProbeState.DISCOVERED)

        _bus, reason = resolve_bus_type(
            config_bus=BUS_KAFKA,
            config_source="per-runtime config at /some/contracts/runtime/runtime_config.yaml",
        )

        assert "config.event_bus.type=kafka" in reason
        assert "/some/contracts/runtime/runtime_config.yaml" in reason

    def test_explicit_argument_beats_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_probe(monkeypatch, EnumProbeState.AUTHORITATIVE)

        bus, reason = resolve_bus_type(explicit_bus=BUS_INMEMORY, config_bus=BUS_KAFKA)

        assert bus == BUS_INMEMORY
        assert "explicit" in reason

    def test_absent_config_falls_through_to_the_probe(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(BUS_TYPE_OVERRIDE_ENV, raising=False)
        calls = _stub_probe(monkeypatch, EnumProbeState.AUTHORITATIVE)

        bus, _reason = resolve_bus_type(config_bus=None, kafka_bootstrap=_BROKER)

        assert bus == BUS_KAFKA
        assert len(calls) == 1

    def test_unrecognised_config_value_fails_loud_naming_its_source(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(BUS_TYPE_OVERRIDE_ENV, raising=False)
        _stub_probe(monkeypatch, EnumProbeState.AUTHORITATIVE)

        with pytest.raises(ValueError) as excinfo:
            resolve_bus_type(config_bus="redpanda", kafka_bootstrap=_BROKER)

        message = str(excinfo.value)
        assert "config.event_bus.type" in message
        assert "redpanda" in message


class TestOneVocabularyAcrossTiers:
    """Every tier accepts the same spellings. A word valid in one is valid in all.

    Pre-OMN-16693, ``cloud`` resolved fine through ``ONEX_EVENT_BUS_TYPE`` and
    raised through ``explicit_bus`` — the same class of divergence OMN-16678
    was opened to remove. ``config.event_bus.type`` can legally hold
    ``EnumEventBusType.CLOUD``, which made the asymmetry reachable. (The env
    var itself holds no tier since OMN-17304, so the shared vocabulary now
    spans exactly two value-carrying tiers: explicit and config.)
    """

    @pytest.mark.parametrize("raw", ["cloud", "CLOUD", " Cloud "])
    def test_cloud_is_accepted_by_every_tier(
        self, raw: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_probe(monkeypatch, EnumProbeState.DISCOVERED)

        monkeypatch.delenv(BUS_TYPE_OVERRIDE_ENV, raising=False)
        assert resolve_bus_type(explicit_bus=raw)[0] == BUS_KAFKA
        assert resolve_bus_type(config_bus=raw)[0] == BUS_KAFKA

    @pytest.mark.parametrize("raw", [" KAFKA ", "InMemory"])
    def test_config_tier_is_case_and_whitespace_insensitive(
        self, raw: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(BUS_TYPE_OVERRIDE_ENV, raising=False)
        _stub_probe(monkeypatch, EnumProbeState.DISCOVERED)

        bus, _reason = resolve_bus_type(config_bus=raw, kafka_bootstrap=_BROKER)

        assert bus == (BUS_INMEMORY if raw.strip().lower() == "inmemory" else BUS_KAFKA)


class TestProbeTierIsTotalAndDeterministic:
    """Every probe state maps to exactly one outcome, and it never varies."""

    @pytest.mark.parametrize(
        "state", [EnumProbeState.HEALTHY, EnumProbeState.AUTHORITATIVE]
    )
    def test_determinate_positive_states_resolve_kafka(
        self, state: EnumProbeState, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(BUS_TYPE_OVERRIDE_ENV, raising=False)
        _stub_probe(monkeypatch, state, reason="broker serves traffic")

        bus, reason = resolve_bus_type(kafka_bootstrap=_BROKER)

        assert bus == BUS_KAFKA
        assert "broker serves traffic" in reason

    def test_discovered_is_a_determinate_negative_and_resolves_inmemory(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """DISCOVERED is a *conclusive* "no usable broker" verdict.

        It is reached only via branches that decided something definite with no
        indeterminacy: no bootstrap configured, an unparseable broker address,
        or a refused TCP connect. Resolving it to inmemory is repeatable, so it
        stays a fallback rather than an error.
        """
        monkeypatch.delenv(BUS_TYPE_OVERRIDE_ENV, raising=False)
        _stub_probe(monkeypatch, EnumProbeState.DISCOVERED, reason="TCP connect failed")

        bus, reason = resolve_bus_type(kafka_bootstrap=_BROKER)

        assert bus == BUS_INMEMORY
        assert "DISCOVERED" in reason
        assert "TCP connect failed" in reason

    def test_reachable_is_indeterminate_and_refuses_to_guess(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The OMN-16678 core fix: no coin flip on an unknown broker state.

        REACHABLE means "TCP connected, but the metadata call did not complete"
        — a timeout, an auth failure, or a missing client library. The broker's
        ability to serve this caller is genuinely UNKNOWN, and the same input
        produced kafka on one call and inmemory on the next. Refusing, with the
        ambiguity and both remedies named, is the only repeatable answer.
        """
        monkeypatch.delenv(BUS_TYPE_OVERRIDE_ENV, raising=False)
        _stub_probe(
            monkeypatch,
            EnumProbeState.REACHABLE,
            reason="TCP reachable but topic list failed: Broker: Request timed out",
        )

        with pytest.raises(EventBusResolutionAmbiguousError) as excinfo:
            resolve_bus_type(kafka_bootstrap=_BROKER)

        message = str(excinfo.value)
        # The error must name the ambiguity...
        assert "REACHABLE" in message
        assert "Request timed out" in message
        # ...both deterministic remedies (explicit argument / declared
        # config), and the fact that the old env-var remedy is gone.
        assert "runtime config" in message
        assert BUS_TYPE_OVERRIDE_ENV in message  # named as removed, not as a remedy
        assert "no longer" in message
        for bus_name in SUPPORTED_BUS_TYPES:
            assert bus_name in message

    def test_indeterminate_probe_is_recoverable_by_declared_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The documented remedy in the error message actually works (OMN-17304).

        A declared config outranks the probe entirely, so the indeterminate
        state is never even reached.
        """
        calls = _stub_probe(monkeypatch, EnumProbeState.REACHABLE)
        monkeypatch.delenv(BUS_TYPE_OVERRIDE_ENV, raising=False)

        bus, _reason = resolve_bus_type(config_bus=BUS_KAFKA, kafka_bootstrap=_BROKER)

        assert bus == BUS_KAFKA
        assert calls == []

    def test_repeated_calls_on_identical_input_return_identical_output(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Direct counterpart to the measured 14-kafka/6-inmemory split.

        With the probe pinned to one state, N consecutive calls must produce
        ONE distinct answer. The pre-fix defect was never that the mapping was
        wrong for a given state — it was that a single unchanged environment
        produced two different states, and both were silently accepted as
        transport decisions.
        """
        monkeypatch.delenv(BUS_TYPE_OVERRIDE_ENV, raising=False)
        _stub_probe(monkeypatch, EnumProbeState.AUTHORITATIVE)

        observed = {resolve_bus_type(kafka_bootstrap=_BROKER)[0] for _ in range(20)}

        assert observed == {BUS_KAFKA}
