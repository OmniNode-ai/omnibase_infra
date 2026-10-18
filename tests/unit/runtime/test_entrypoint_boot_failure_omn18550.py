# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A runtime that cannot boot stops retrying and says so (OMN-18550).

2026-09-16: ``omninode-runtime`` died on a malformed contract and its restart
policy relaunched it every forty seconds for hours, the cause readable only in a
log. The boot supervisor in ``entrypoint_preflight`` counts consecutive failing
exits in a durable ledger, emits a boot-failed event for each (cause and count),
and parks once the declared bound is reached.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import pytest

from omnibase_infra.handlers.handler_boot_failure_ledger import (
    HandlerBootFailureLedger,
)
from omnibase_infra.models.health.model_boot_attempt_outcome import (
    ModelBootAttemptOutcome,
)
from omnibase_infra.models.health.model_boot_identity import ModelBootIdentity
from omnibase_infra.models.health.model_runtime_boot_failed_event import (
    ModelRuntimeBootFailedEvent,
)
from omnibase_infra.runtime import entrypoint_preflight

pytestmark = [pytest.mark.unit]

_KERNEL_LOGGER = "omnibase_infra.runtime.service_kernel"


class _ParkedError(Exception):
    """Raised in place of the endless idle loop so a test can observe parking."""


def _identity(
    state_dir: Path,
    *,
    source_hash: str = "abc123",
    container_ref: str = "c1",
    max_failures: int = 3,
) -> ModelBootIdentity:
    return ModelBootIdentity(
        state_dir=str(state_dir),
        service="onex-runtime-main",
        container_ref=container_ref,
        runtime_source_hash=source_hash,
        max_consecutive_failures=max_failures,
    )


def _outcome(
    state_dir: Path,
    *,
    exit_code: int = 1,
    uptime: float = 40.0,
    source_hash: str = "abc123",
    container_ref: str = "c1",
    max_failures: int = 3,
    healthy_after: float = 600.0,
) -> ModelBootAttemptOutcome:
    return ModelBootAttemptOutcome(
        identity=_identity(
            state_dir,
            source_hash=source_hash,
            container_ref=container_ref,
            max_failures=max_failures,
        ),
        exit_code=exit_code,
        cause="Duplicate local ingress route alias",
        uptime_seconds=uptime,
        healthy_after_seconds=healthy_after,
        observed_at=datetime(2026, 10, 17, tzinfo=UTC),
    )


# ---------------------------------------------------------------------------
# The ledger handler: counting, streak boundaries, durability
# ---------------------------------------------------------------------------


def test_consecutive_failures_are_counted_and_the_bound_is_flagged(
    tmp_path: Path,
) -> None:
    handler = HandlerBootFailureLedger()

    events = [handler.handle(_outcome(tmp_path)) for _ in range(3)]

    assert [e.consecutive_failures for e in events] == [1, 2, 3]
    assert [e.retries_exhausted for e in events] == [False, False, True]
    assert events[-1].max_consecutive_failures == 3


def test_every_failure_is_appended_to_the_durable_event_log(tmp_path: Path) -> None:
    handler = HandlerBootFailureLedger()
    handler.handle(_outcome(tmp_path))
    handler.handle(_outcome(tmp_path))

    lines = (tmp_path / "boot_failures" / "events.jsonl").read_text().splitlines()

    parsed = [ModelRuntimeBootFailedEvent.model_validate(json.loads(x)) for x in lines]
    assert [e.consecutive_failures for e in parsed] == [1, 2]
    assert parsed[0].cause == "Duplicate local ingress route alias"


def test_a_new_image_starts_a_new_streak(tmp_path: Path) -> None:
    handler = HandlerBootFailureLedger()
    handler.handle(_outcome(tmp_path, source_hash="old"))
    handler.handle(_outcome(tmp_path, source_hash="old"))

    event = handler.handle(_outcome(tmp_path, source_hash="fixed"))

    assert event.consecutive_failures == 1


def test_a_recreated_container_starts_a_new_streak(tmp_path: Path) -> None:
    """A redeploy recreates the container even when every image's hash is 'unknown'."""
    handler = HandlerBootFailureLedger()
    handler.handle(_outcome(tmp_path))
    handler.handle(_outcome(tmp_path))

    event = handler.handle(_outcome(tmp_path, container_ref="c2"))

    assert event.consecutive_failures == 1


def test_a_failure_after_a_healthy_run_is_a_new_incident(tmp_path: Path) -> None:
    handler = HandlerBootFailureLedger()
    handler.handle(_outcome(tmp_path))
    handler.handle(_outcome(tmp_path))

    event = handler.handle(_outcome(tmp_path, uptime=3600.0))

    assert event.consecutive_failures == 1


def test_a_clean_exit_ends_the_streak(tmp_path: Path) -> None:
    handler = HandlerBootFailureLedger()
    handler.handle(_outcome(tmp_path))
    handler.record_clean_exit(_identity(tmp_path))

    assert handler.handle(_outcome(tmp_path)).consecutive_failures == 1


def test_exhaustion_is_read_back_for_the_same_image_and_container_only(
    tmp_path: Path,
) -> None:
    handler = HandlerBootFailureLedger()
    for _ in range(3):
        handler.handle(_outcome(tmp_path))

    same = handler.exhausted_event(_identity(tmp_path))
    other_image = handler.exhausted_event(_identity(tmp_path, source_hash="new"))
    other_container = handler.exhausted_event(_identity(tmp_path, container_ref="c2"))
    below_bound = handler.exhausted_event(_identity(tmp_path, max_failures=4))

    assert same is not None
    assert same.consecutive_failures == 3
    assert other_image is None
    assert other_container is None
    assert below_bound is None


def test_a_clean_exit_is_not_a_failure_to_record(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="clean exit"):
        HandlerBootFailureLedger().handle(_outcome(tmp_path, exit_code=0))


def test_a_corrupt_streak_file_restarts_the_count_instead_of_wedging(
    tmp_path: Path,
) -> None:
    ledger_dir = tmp_path / "boot_failures"
    ledger_dir.mkdir()
    (ledger_dir / "onex-runtime-main.json").write_text("{not json")

    event = HandlerBootFailureLedger().handle(_outcome(tmp_path))

    assert event.consecutive_failures == 1


# ---------------------------------------------------------------------------
# The supervisor: what the entrypoint does around the kernel
# ---------------------------------------------------------------------------


@pytest.fixture
def boot_env(tmp_path: Path) -> dict[str, str]:
    return {
        "ONEX_STATE_DIR": str(tmp_path),
        "ONEX_GROUP_ID": "onex-runtime-main",
        "RUNTIME_SOURCE_HASH": "abc123",
        "HOSTNAME": "c1",
        "ONEX_BOOT_FAILURE_MAX_CONSECUTIVE": "3",
    }


@pytest.fixture
def published(monkeypatch: pytest.MonkeyPatch) -> list[ModelRuntimeBootFailedEvent]:
    seen: list[ModelRuntimeBootFailedEvent] = []
    monkeypatch.setattr(
        entrypoint_preflight,
        "_publish_boot_failed",
        lambda event, _env: seen.append(event),
    )
    return seen


def _failing_kernel(monkeypatch: pytest.MonkeyPatch, message: str) -> None:
    """A kernel that logs its cause the way bootstrap() does, then exits 1."""
    monkeypatch.setattr(entrypoint_preflight, "run_preflight", lambda _env: 0)

    def handoff(_argv: list[str]) -> int:
        try:
            raise ValueError(message)
        except ValueError:
            logging.getLogger(_KERNEL_LOGGER).exception("ONEX runtime failed")
        raise SystemExit(1)

    monkeypatch.setattr(entrypoint_preflight, "handoff", handoff)


def _parking(monkeypatch: pytest.MonkeyPatch) -> None:
    def sleep(_seconds: float) -> None:
        raise _ParkedError

    monkeypatch.setattr(entrypoint_preflight.time, "sleep", sleep)


def test_a_failing_boot_exits_with_its_code_and_publishes_cause_and_count(
    monkeypatch: pytest.MonkeyPatch,
    boot_env: dict[str, str],
    published: list[ModelRuntimeBootFailedEvent],
    capsys: pytest.CaptureFixture[str],
) -> None:
    _failing_kernel(monkeypatch, "Duplicate local ingress route alias a.b.c")

    rc = entrypoint_preflight._supervised_start(["onex-runtime"], boot_env)

    assert rc == 1
    assert len(published) == 1
    assert published[0].consecutive_failures == 1
    assert "Duplicate local ingress route alias a.b.c" in published[0].cause
    assert "BOOT FAILED 1/3" in capsys.readouterr().err


def test_the_bound_stops_the_retry_and_records_the_exhausted_event(
    monkeypatch: pytest.MonkeyPatch,
    boot_env: dict[str, str],
    published: list[ModelRuntimeBootFailedEvent],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _failing_kernel(monkeypatch, "bad contract")
    _parking(monkeypatch)

    assert entrypoint_preflight._supervised_start(["onex-runtime"], boot_env) == 1
    assert entrypoint_preflight._supervised_start(["onex-runtime"], boot_env) == 1
    with pytest.raises(_ParkedError):
        entrypoint_preflight._supervised_start(["onex-runtime"], boot_env)

    assert [e.consecutive_failures for e in published] == [1, 2, 3]
    assert published[-1].retries_exhausted is True
    assert "RETRIES EXHAUSTED" in capsys.readouterr().err
    log = (tmp_path / "boot_failures" / "events.jsonl").read_text().splitlines()
    assert len(log) == 3


def test_a_restarted_container_past_the_bound_parks_without_booting(
    monkeypatch: pytest.MonkeyPatch,
    boot_env: dict[str, str],
    published: list[ModelRuntimeBootFailedEvent],
) -> None:
    _failing_kernel(monkeypatch, "bad contract")
    _parking(monkeypatch)
    for _ in range(2):
        entrypoint_preflight._supervised_start(["onex-runtime"], boot_env)
    with pytest.raises(_ParkedError):
        entrypoint_preflight._supervised_start(["onex-runtime"], boot_env)

    booted: list[str] = []

    def preflight(_env: object) -> int:
        booted.append("preflight")
        return 0

    monkeypatch.setattr(entrypoint_preflight, "run_preflight", preflight)
    with pytest.raises(_ParkedError):
        entrypoint_preflight._supervised_start(["onex-runtime"], boot_env)

    assert booted == []
    assert len(published) == 3


def test_a_redeployed_image_boots_again_after_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
    boot_env: dict[str, str],
    published: list[ModelRuntimeBootFailedEvent],
) -> None:
    _failing_kernel(monkeypatch, "bad contract")
    _parking(monkeypatch)
    for _ in range(2):
        entrypoint_preflight._supervised_start(["onex-runtime"], boot_env)
    with pytest.raises(_ParkedError):
        entrypoint_preflight._supervised_start(["onex-runtime"], boot_env)

    monkeypatch.setattr(entrypoint_preflight, "handoff", lambda _argv: 0)
    rc = entrypoint_preflight._supervised_start(
        ["onex-runtime"], {**boot_env, "RUNTIME_SOURCE_HASH": "fixed"}
    )

    assert rc == 0


def test_a_clean_run_leaves_no_streak(
    monkeypatch: pytest.MonkeyPatch,
    boot_env: dict[str, str],
    published: list[ModelRuntimeBootFailedEvent],
    tmp_path: Path,
) -> None:
    _failing_kernel(monkeypatch, "bad contract")
    entrypoint_preflight._supervised_start(["onex-runtime"], boot_env)

    monkeypatch.setattr(entrypoint_preflight, "handoff", lambda _argv: 0)
    assert entrypoint_preflight._supervised_start(["onex-runtime"], boot_env) == 0

    assert not (tmp_path / "boot_failures" / "onex-runtime-main.json").exists()


def test_a_failed_preflight_is_counted_with_its_exit_code(
    monkeypatch: pytest.MonkeyPatch,
    boot_env: dict[str, str],
    published: list[ModelRuntimeBootFailedEvent],
) -> None:
    monkeypatch.setattr(entrypoint_preflight, "run_preflight", lambda _env: 2)
    monkeypatch.setattr(
        entrypoint_preflight,
        "handoff",
        lambda _argv: pytest.fail("the kernel must not start after a failed preflight"),
    )

    rc = entrypoint_preflight._supervised_start(["onex-runtime"], boot_env)

    assert rc == 2
    assert published[0].exit_code == 2
    assert "preflight aborted with exit 2" in published[0].cause


def test_an_unhandled_boot_crash_is_counted_and_still_propagates(
    monkeypatch: pytest.MonkeyPatch,
    boot_env: dict[str, str],
    published: list[ModelRuntimeBootFailedEvent],
) -> None:
    monkeypatch.setattr(entrypoint_preflight, "run_preflight", lambda _env: 0)

    def crash(_argv: list[str]) -> int:
        raise ValueError("unhandled")

    monkeypatch.setattr(entrypoint_preflight, "handoff", crash)

    with pytest.raises(ValueError, match="unhandled"):
        entrypoint_preflight._supervised_start(["onex-runtime"], boot_env)

    assert "unhandled" in published[0].cause
    assert published[0].consecutive_failures == 1


def test_without_a_state_directory_the_boot_is_unsupervised(
    monkeypatch: pytest.MonkeyPatch,
    published: list[ModelRuntimeBootFailedEvent],
) -> None:
    monkeypatch.setattr(entrypoint_preflight, "run_preflight", lambda _env: 0)
    monkeypatch.setattr(entrypoint_preflight, "handoff", lambda _argv: 0)

    assert entrypoint_preflight._supervised_start(["onex-runtime"], {}) == 0
    assert published == []


def test_the_capture_handler_is_removed_after_the_boot(
    monkeypatch: pytest.MonkeyPatch,
    boot_env: dict[str, str],
    published: list[ModelRuntimeBootFailedEvent],
) -> None:
    kernel_logger = logging.getLogger(_KERNEL_LOGGER)
    before = list(kernel_logger.handlers)
    _failing_kernel(monkeypatch, "bad contract")

    entrypoint_preflight._supervised_start(["onex-runtime"], boot_env)

    assert kernel_logger.handlers == before


def test_the_event_is_published_on_the_contract_declared_topic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import aiokafka

    from omnibase_infra.runtime import gateway_forwarder

    sent: list[tuple[str, bytes, bytes]] = []
    ensured: list[tuple[str, ...]] = []

    class _FakeProducer:
        def __init__(self, **_kwargs: object) -> None: ...

        async def start(self) -> None: ...

        async def stop(self) -> None: ...

        async def send_and_wait(self, topic: str, *, value: bytes, key: bytes) -> None:
            sent.append((topic, value, key))

    async def fake_ensure(
        _config: object, topics: tuple[str, ...], **_kw: object
    ) -> None:
        ensured.append(topics)

    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "bus.invalid:9092")
    monkeypatch.setattr(aiokafka, "AIOKafkaProducer", _FakeProducer)
    monkeypatch.setattr(gateway_forwarder, "ensure_topics_exist", fake_ensure)
    model = ModelRuntimeBootFailedEvent(
        service="onex-runtime-main",
        container_ref="c",
        runtime_source_hash="h",
        exit_code=1,
        cause="bad contract",
        uptime_seconds=1.0,
        consecutive_failures=2,
        max_consecutive_failures=3,
        retries_exhausted=False,
        failed_at=datetime(2026, 10, 17, tzinfo=UTC),
    )

    entrypoint_preflight._publish_boot_failed(
        model, {"KAFKA_BOOTSTRAP_SERVERS": "bus.invalid:9092"}
    )

    topic = "onex.evt.omnibase-infra.runtime-boot-failed.v1"
    assert ensured == [(topic,)]
    assert [(t, k) for t, _v, k in sent] == [(topic, b"onex-runtime-main")]
    assert json.loads(sent[0][1])["consecutive_failures"] == 2
    assert json.loads(sent[0][1])["cause"] == "bad contract"


def test_a_publish_failure_never_masks_the_exit_code(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model = ModelRuntimeBootFailedEvent(
        service="s",
        container_ref="c",
        runtime_source_hash="h",
        exit_code=1,
        cause="x",
        uptime_seconds=1.0,
        consecutive_failures=1,
        max_consecutive_failures=3,
        retries_exhausted=False,
        failed_at=datetime(2026, 10, 17, tzinfo=UTC),
    )

    entrypoint_preflight._publish_boot_failed(
        model, {"KAFKA_BOOTSTRAP_SERVERS": "127.0.0.1:1"}
    )
    entrypoint_preflight._publish_boot_failed(model, {})

    err = capsys.readouterr().err
    assert "boot-failed event not published" in err


__all__: list[str] = []
