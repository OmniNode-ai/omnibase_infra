# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20590: the lane's runtime tick producer is started by the kernel.

RED on the parent: ``RuntimeScheduler`` had no instantiation site in ``src``,
so no lane published ``onex.intent.platform.runtime-tick.v1``, and there was no
lane switch to turn one on.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
import yaml

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.protocols import ProtocolEventBusLike
from omnibase_infra.runtime.models import ModelRuntimeTick
from omnibase_infra.runtime.runtime_profile import resolve_runtime_scheduler_enabled
from omnibase_infra.runtime.runtime_scheduler import (
    RUNTIME_TICK_EVENT_TYPE,
    start_lane_runtime_scheduler,
)
from omnibase_infra.topics import SUFFIX_RUNTIME_TICK

pytestmark = pytest.mark.unit

_FLAG = "ONEX_RUNTIME_SCHEDULER_ENABLED"
_DOCKER_DIR = Path(__file__).resolve().parents[3] / "docker"


@pytest.fixture
def bus() -> AsyncMock:
    return AsyncMock(spec=ProtocolEventBusLike)


@pytest.fixture(autouse=True)
def _no_valkey(monkeypatch: pytest.MonkeyPatch) -> None:
    # No sequence persistence in unit tests: the scheduler would try Valkey.
    monkeypatch.setenv("ONEX_RUNTIME_SCHEDULER_PERSIST_SEQUENCE", "false")
    monkeypatch.setenv("ONEX_RUNTIME_SCHEDULER_TICK_INTERVAL_MS", "60000")


class TestFlag:
    @pytest.mark.parametrize("raw", ["", "  "])
    def test_unset_or_blank_is_off(
        self, monkeypatch: pytest.MonkeyPatch, raw: str
    ) -> None:
        monkeypatch.setenv(_FLAG, raw)
        assert resolve_runtime_scheduler_enabled() is False

    def test_absent_is_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(_FLAG, raising=False)
        assert resolve_runtime_scheduler_enabled() is False

    @pytest.mark.parametrize("raw", ["true", "TRUE", "1", "yes", "on"])
    def test_true_spellings(self, monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
        monkeypatch.setenv(_FLAG, raw)
        assert resolve_runtime_scheduler_enabled() is True

    @pytest.mark.parametrize("raw", ["false", "0", "no", "off"])
    def test_false_spellings(self, monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
        monkeypatch.setenv(_FLAG, raw)
        assert resolve_runtime_scheduler_enabled() is False

    def test_unparseable_value_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(_FLAG, "ture")
        with pytest.raises(ProtocolConfigurationError, match=_FLAG):
            resolve_runtime_scheduler_enabled()


@pytest.mark.asyncio
class TestStartLaneRuntimeScheduler:
    async def test_main_profile_with_flag_starts_and_publishes_envelope(
        self, monkeypatch: pytest.MonkeyPatch, bus: AsyncMock
    ) -> None:
        monkeypatch.setenv(_FLAG, "true")
        scheduler = await start_lane_runtime_scheduler(bus, "main")
        assert scheduler is not None
        try:
            assert scheduler.is_running
            await scheduler.emit_tick()
        finally:
            await scheduler.stop()

        call = bus.publish_envelope.call_args
        assert call.kwargs["topic"] == SUFFIX_RUNTIME_TICK
        envelope = call.kwargs["envelope"]
        assert envelope.event_type == RUNTIME_TICK_EVENT_TYPE
        assert isinstance(envelope.payload, ModelRuntimeTick)

    @pytest.mark.parametrize("profile", ["effects", "workers", "projection-api"])
    async def test_secondary_role_never_starts_one(
        self, monkeypatch: pytest.MonkeyPatch, bus: AsyncMock, profile: str
    ) -> None:
        monkeypatch.setenv(_FLAG, "true")
        assert await start_lane_runtime_scheduler(bus, profile) is None
        bus.publish_envelope.assert_not_called()

    async def test_main_profile_without_flag_starts_none_and_says_so(
        self,
        monkeypatch: pytest.MonkeyPatch,
        bus: AsyncMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.delenv(_FLAG, raising=False)
        with caplog.at_level(logging.WARNING):
            assert await start_lane_runtime_scheduler(bus, "main") is None
        bus.publish_envelope.assert_not_called()
        assert any(
            SUFFIX_RUNTIME_TICK in r.getMessage() and _FLAG in r.getMessage()
            for r in caplog.records
            if r.levelno == logging.WARNING
        )

    async def test_unparseable_flag_refuses_boot(
        self, monkeypatch: pytest.MonkeyPatch, bus: AsyncMock
    ) -> None:
        monkeypatch.setenv(_FLAG, "ture")
        with pytest.raises(ProtocolConfigurationError):
            await start_lane_runtime_scheduler(bus, "main")


def _load_compose(name: str) -> dict[str, Any]:
    # `!override` is a compose-only tag SafeLoader refuses.
    text = (_DOCKER_DIR / name).read_text(encoding="utf-8")
    document = yaml.safe_load(text.replace("!override", ""))
    assert isinstance(document, dict)
    return document


def _service_env(document: dict[str, Any], service: str) -> dict[str, str]:
    env = (document.get("services") or {}).get(service, {}).get("environment") or {}
    if isinstance(env, list):
        return dict(item.split("=", 1) for item in env if "=" in item)
    return {str(k): str(v) for k, v in env.items()}


# The lane overlays that publish the runtime tick. dev-202 (OMN-20590) and the
# h201 stability-test lane (OMN-20593, the backup delegation lane beside the
# h201 dev lane, which stays off so its demo runtime is never restarted).
_OPTED_IN_OVERLAYS = (
    "docker-compose.dev-202.yml",
    "docker-compose.stability-test.yml",
)


class TestLaneOverlays:
    @pytest.mark.parametrize("overlay", _OPTED_IN_OVERLAYS)
    def test_overlay_turns_on_the_main_runtime_producer(self, overlay: str) -> None:
        document = _load_compose(overlay)
        assert _service_env(document, "omninode-runtime").get(_FLAG) == "true"

    @pytest.mark.parametrize("overlay", _OPTED_IN_OVERLAYS)
    def test_overlay_sets_the_flag_on_no_secondary_role(self, overlay: str) -> None:
        # One producer per lane: only the main role may carry the switch.
        services = _load_compose(overlay).get("services") or {}
        carriers = sorted(
            name
            for name in services
            if _FLAG in _service_env({"services": services}, name)
        )
        assert carriers == ["omninode-runtime"]

    @pytest.mark.parametrize("overlay", _OPTED_IN_OVERLAYS)
    def test_overlay_gives_the_tick_driven_prunes_their_archive(
        self, overlay: str
    ) -> None:
        # The two prune effects on runtime-effects resolve these on every tick;
        # an opted-in lane without them dead-letters every tick (read live on
        # dev-202, 2026-10-05: KeyError 'ONEX_CONSUMER_FLOW_ARCHIVE_DIR').
        env = _service_env(_load_compose(overlay), "runtime-effects")
        for name in ("ONEX_DEAD_LETTER_ARCHIVE_DIR", "ONEX_CONSUMER_FLOW_ARCHIVE_DIR"):
            assert env.get(name, "").startswith("/app/data/"), name

    def test_overlay_flag_is_set_by_the_opted_in_lanes_alone(self) -> None:
        # Every other lane changes only by an explicit overlay edit; this test
        # names the lanes that opted in so the next one is a reviewed diff.
        opted_in = sorted(
            path.name
            for path in _DOCKER_DIR.glob("docker-compose*.yml")
            if _FLAG in path.read_text(encoding="utf-8")
        )
        assert opted_in == sorted(_OPTED_IN_OVERLAYS)


def test_envelope_survives_the_consumer_deserializer_shape() -> None:
    """The bytes a Kafka bus would publish validate as the consumer expects."""
    from datetime import UTC, datetime
    from uuid import uuid4

    now = datetime.now(UTC)
    tick = ModelRuntimeTick(
        now=now,
        tick_id=uuid4(),
        sequence_number=1,
        scheduled_at=now,
        correlation_id=uuid4(),
        scheduler_id="runtime-scheduler-default",
        tick_interval_ms=1000,
    )
    envelope = ModelEventEnvelope(
        payload=tick,
        correlation_id=tick.correlation_id,
        event_type=RUNTIME_TICK_EVENT_TYPE,
        source_tool="runtime-scheduler.runtime-scheduler-default",
        tenant_id=None,
    )
    wire = json.loads(json.dumps(envelope.model_dump(mode="json")))
    decoded = ModelEventEnvelope[object].model_validate(wire)
    assert decoded.event_type == RUNTIME_TICK_EVENT_TYPE
    assert ModelRuntimeTick.model_validate(decoded.payload) == tick
