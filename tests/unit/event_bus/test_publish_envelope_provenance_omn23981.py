# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-23981: the runtime stamps where it runs from onto the envelopes it publishes.

The deployment's overlay (the ``ONEX_PROVENANCE_*`` variables) is the only
source of the stamp. These tests run the real resolver
and the real ``MixinKafkaBroadcast.publish_envelope`` seam, and read the bytes
that would reach the broker back through ``ModelEventEnvelope``, the way a
consumer does. An overlay that is absent, partial or invalid must leave the
envelope without provenance so a reader reports ``unknown``, never ``dev``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping

import pytest

from omnibase_core.enums.enum_provenance_install_source import (
    EnumProvenanceInstallSource,
)
from omnibase_core.enums.enum_provenance_lane import EnumProvenanceLane
from omnibase_core.enums.enum_provenance_lane_kind import EnumProvenanceLaneKind
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_core.models.events.model_event_provenance import ModelEventProvenance
from omnibase_infra.event_bus.mixin_kafka_broadcast import MixinKafkaBroadcast
from omnibase_infra.handlers.handler_event_provenance_stamp.handler import (
    provenance_from_overlay,
    read_provenance_overlay,
)
from omnibase_infra.models.event_bus.model_event_provenance_overlay import (
    ModelEventProvenanceOverlay,
)

_REVISION = "a09066caf945cd79e13447d92ad42549f1019f78"
_VARIABLES = dict(ModelEventProvenanceOverlay.VARIABLES)
ENV_PROVENANCE_LANE = _VARIABLES["lane"]
ENV_PROVENANCE_LANE_KIND = _VARIABLES["lane_kind"]
ENV_PROVENANCE_HOST = _VARIABLES["host"]
ENV_PROVENANCE_INSTALL_SOURCE = _VARIABLES["install_source"]
ENV_PROVENANCE_REVISION = _VARIABLES["revision"]


def resolve_event_provenance(
    environ: Mapping[str, str],
) -> ModelEventProvenance | None:
    """The overlay as the runtime reads it, validated by the stamp handler."""
    return provenance_from_overlay(read_provenance_overlay(environ))[0]


def _overlay(lane: str, **extra: str) -> dict[str, str]:
    env = {
        ENV_PROVENANCE_LANE: lane,
        ENV_PROVENANCE_HOST: "lab-host-201",
        ENV_PROVENANCE_INSTALL_SOURCE: "branch",
        ENV_PROVENANCE_REVISION: _REVISION,
    }
    env.update(extra)
    return env


class _Host(MixinKafkaBroadcast):
    """Smallest host the mixin's publish_envelope needs: a publish that records."""

    def __init__(self) -> None:
        self._environment = "dev"
        self.sent: list[bytes] = []

    async def publish(
        self, topic: str, key: bytes | None, value: bytes, headers: object
    ) -> None:
        self.sent.append(value)


def _clear_provenance_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        ENV_PROVENANCE_LANE,
        ENV_PROVENANCE_HOST,
        ENV_PROVENANCE_INSTALL_SOURCE,
        ENV_PROVENANCE_REVISION,
        ENV_PROVENANCE_LANE_KIND,
    ):
        monkeypatch.delenv(name, raising=False)


def _set_env(monkeypatch: pytest.MonkeyPatch, env: Mapping[str, str]) -> None:
    _clear_provenance_env(monkeypatch)
    for name, value in env.items():
        monkeypatch.setenv(name, value)


@pytest.mark.unit
class TestResolveEventProvenance:
    def test_dev_lane_stamps_dev(self) -> None:
        stamp = resolve_event_provenance(_overlay("dev"))
        assert stamp is not None
        assert stamp.lane == EnumProvenanceLane.DEV
        assert stamp.lane_kind is None
        assert stamp.host == "lab-host-201"
        assert stamp.install_source == EnumProvenanceInstallSource.BRANCH
        assert stamp.revision == _REVISION

    def test_stability_lane_stamps_stability(self) -> None:
        stamp = resolve_event_provenance(_overlay("stability"))
        assert stamp is not None
        assert stamp.lane == EnumProvenanceLane.STABILITY

    def test_other_lane_stamps_other_with_its_kind(self) -> None:
        stamp = resolve_event_provenance(
            _overlay("other", **{ENV_PROVENANCE_LANE_KIND: "hand_run"})
        )
        assert stamp is not None
        assert stamp.lane == EnumProvenanceLane.OTHER
        assert stamp.lane_kind == EnumProvenanceLaneKind.HAND_RUN

    def test_other_lane_without_a_kind_stamps_nothing(self) -> None:
        # Seam: core refuses lane OTHER without lane_kind, and the resolver
        # must turn that refusal into an omitted stamp, not an exception.
        assert resolve_event_provenance(_overlay("other")) is None

    @pytest.mark.parametrize(
        "missing",
        [
            ENV_PROVENANCE_LANE,
            ENV_PROVENANCE_HOST,
            ENV_PROVENANCE_INSTALL_SOURCE,
            ENV_PROVENANCE_REVISION,
        ],
    )
    def test_half_stamped_overlay_stamps_nothing(self, missing: str) -> None:
        env = _overlay("dev")
        del env[missing]
        assert resolve_event_provenance(env) is None

    def test_blank_host_stamps_nothing(self) -> None:
        env = _overlay("dev", **{ENV_PROVENANCE_HOST: "   "})
        assert resolve_event_provenance(env) is None

    def test_unknown_install_source_stamps_nothing(self) -> None:
        env = _overlay("dev", **{ENV_PROVENANCE_INSTALL_SOURCE: "usb-stick"})
        assert resolve_event_provenance(env) is None

    @pytest.mark.parametrize("lane", ["no-such-lane", "unknown", ""])
    def test_unrecognised_or_unknown_lane_stamps_nothing(self, lane: str) -> None:
        # `unknown` is the absence of a stamp, so a deployment cannot declare it.
        assert resolve_event_provenance(_overlay(lane)) is None

    def test_empty_environment_stamps_nothing(self) -> None:
        assert resolve_event_provenance({}) is None


@pytest.mark.unit
class TestPublishEnvelopeStampsProvenance:
    async def test_dev_runtime_event_carries_provenance_on_the_wire(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_env(monkeypatch, _overlay("dev"))
        host = _Host()
        await host.publish_envelope(ModelEventEnvelope(payload={"k": "v"}), "t.topic")

        wire = json.loads(host.sent[0])
        assert wire["provenance"]["lane"] == "dev"
        assert wire["provenance"]["host"] == "lab-host-201"
        assert wire["provenance"]["install_source"] == "branch"
        assert wire["provenance"]["revision"] == _REVISION
        # A consumer on the same model reads the same lane back.
        assert ModelEventEnvelope.model_validate(wire).provenance_lane == (
            EnumProvenanceLane.DEV
        )

    async def test_no_overlay_event_reads_unknown_never_dev(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _clear_provenance_env(monkeypatch)
        host = _Host()
        await host.publish_envelope(ModelEventEnvelope(payload={"k": "v"}), "t.topic")

        wire = json.loads(host.sent[0])
        assert wire["provenance"] is None
        assert ModelEventEnvelope.model_validate(wire).provenance_lane == (
            EnumProvenanceLane.UNKNOWN
        )

    async def test_half_stamped_overlay_publishes_without_breaking_the_wire(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env = _overlay("dev")
        del env[ENV_PROVENANCE_REVISION]
        _set_env(monkeypatch, env)
        host = _Host()
        await host.publish_envelope(ModelEventEnvelope(payload={"k": "v"}), "t.topic")

        wire = json.loads(host.sent[0])
        assert wire["provenance"] is None
        assert ModelEventEnvelope.model_validate(wire).provenance_lane == (
            EnumProvenanceLane.UNKNOWN
        )

    async def test_an_envelope_that_already_carries_provenance_keeps_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_env(monkeypatch, _overlay("dev"))
        own = ModelEventProvenance(
            lane=EnumProvenanceLane.OTHER,
            lane_kind=EnumProvenanceLaneKind.MAC_UNIT,
            host="mac-1",
            install_source=EnumProvenanceInstallSource.TAG,
            revision="v1",
        )
        host = _Host()
        await host.publish_envelope(
            ModelEventEnvelope(payload={"k": "v"}, provenance=own), "t.topic"
        )

        wire = json.loads(host.sent[0])
        assert wire["provenance"]["lane"] == "other"
        assert wire["provenance"]["host"] == "mac-1"

    async def test_a_dict_payload_is_published_unchanged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_env(monkeypatch, _overlay("dev"))
        host = _Host()
        await host.publish_envelope({"already": "a dict"}, "t.topic")
        assert json.loads(host.sent[0]) == {"already": "a dict"}
