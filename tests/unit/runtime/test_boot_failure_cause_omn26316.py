# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A boot failure's cause is why the kernel ended, never the last message that failed (OMN-26316).

2026-11-23: the dev runtime built from omnibase_infra fafe00001 exited 1 after 823 seconds and
the boot ledger recorded ``Dispatch completed with errors`` as its cause. That line is the
dispatch engine's record of one failed message, logged on the kernel's logger for every
handler failure while the kernel carries on, so the ledger named whichever message failed last
and the lane could not tell a defect in the head from a host that could not start consumers.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from omnibase_infra.models.health.model_runtime_boot_failed_event import (
    ModelRuntimeBootFailedEvent,
)
from omnibase_infra.runtime import entrypoint_preflight, message_dispatch_engine

pytestmark = [pytest.mark.unit]

_KERNEL_LOGGER = "omnibase_infra.runtime.service_kernel"
_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures/boot_failure_cause_omn26316/dev_boot_ledger_event_20261123T2052Z.json"
)
_ENGINE_FILE = message_dispatch_engine.__file__


def _record(message: str, *, pathname: str) -> logging.LogRecord:
    return logging.LogRecord(
        name=_KERNEL_LOGGER,
        level=logging.ERROR,
        pathname=pathname,
        lineno=1785,
        msg=message,
        args=(),
        exc_info=None,
    )


def _log_from_engine(message: str) -> None:
    logging.getLogger(_KERNEL_LOGGER).handle(_record(message, pathname=_ENGINE_FILE))


@pytest.fixture
def boot_env(tmp_path: Path) -> dict[str, str]:
    return {
        "ONEX_STATE_DIR": str(tmp_path),
        "ONEX_GROUP_ID": "onex-runtime-main",
        "RUNTIME_SOURCE_HASH": "abc123",
        "HOSTNAME": "c1",
        "ONEX_BOOT_FAILURE_MAX_CONSECUTIVE": "5",
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


def _kernel(monkeypatch: pytest.MonkeyPatch, *, kernel_error: str | None) -> None:
    """A kernel whose last message failed in the dispatch engine, then exits 1."""
    monkeypatch.setattr(entrypoint_preflight, "run_preflight", lambda _env: 0)

    def handoff(_argv: list[str]) -> int:
        if kernel_error:
            logging.getLogger(_KERNEL_LOGGER).error(kernel_error)
        _log_from_engine("Dispatch completed with errors")
        raise SystemExit(1)

    monkeypatch.setattr(entrypoint_preflight, "handoff", handoff)


def test_the_recorded_incident_cause_is_the_dispatch_engines_own_message() -> None:
    recorded = json.loads(_FIXTURE.read_text())["event"]
    assert recorded["exit_code"] == 1
    assert round(recorded["uptime_seconds"]) == 823
    assert f'"{recorded["cause"]}"' in Path(_ENGINE_FILE).read_text()


def test_a_dispatch_error_is_not_the_cause_of_an_exit_after_it(
    monkeypatch: pytest.MonkeyPatch,
    boot_env: dict[str, str],
    published: list[ModelRuntimeBootFailedEvent],
) -> None:
    """startup_boot_cause: the ledger said 'Dispatch completed with errors' for the 823 s exit."""
    cause = json.loads(_FIXTURE.read_text())["event"]["cause"]
    _kernel(monkeypatch, kernel_error=None)

    assert entrypoint_preflight._supervised_start(["onex-runtime"], boot_env) == 1

    assert len(published) == 1
    assert published[0].cause != cause
    assert published[0].cause.startswith("no kernel error was logged before the exit")
    assert cause in published[0].cause


def test_a_kernel_error_stays_the_cause_when_a_dispatch_error_follows_it(
    monkeypatch: pytest.MonkeyPatch,
    boot_env: dict[str, str],
    published: list[ModelRuntimeBootFailedEvent],
) -> None:
    _kernel(
        monkeypatch, kernel_error="Runtime heartbeat is required and none was published"
    )

    assert entrypoint_preflight._supervised_start(["onex-runtime"], boot_env) == 1

    assert published[0].cause == "Runtime heartbeat is required and none was published"


def test_an_exit_with_no_error_at_all_keeps_the_existing_wording(
    monkeypatch: pytest.MonkeyPatch,
    boot_env: dict[str, str],
    published: list[ModelRuntimeBootFailedEvent],
) -> None:
    monkeypatch.setattr(entrypoint_preflight, "run_preflight", lambda _env: 0)

    def handoff(_argv: list[str]) -> int:
        raise SystemExit(1)

    monkeypatch.setattr(entrypoint_preflight, "handoff", handoff)

    assert entrypoint_preflight._supervised_start(["onex-runtime"], boot_env) == 1

    assert published[0].cause == "runtime exited 1 without a logged error"


def test_only_the_dispatch_engines_records_are_marked_and_a_kernel_error_wins() -> None:
    capture = entrypoint_preflight.KernelErrorCapture()
    capture.emit(_record("Dispatch completed with errors", pathname=_ENGINE_FILE))
    assert capture.cause == (
        "no kernel error was logged before the exit; last dispatch error: "
        "Dispatch completed with errors"
    )
    capture.emit(
        _record("Topic provisioning failed", pathname="/app/service_kernel.py")
    )
    assert capture.cause == "Topic provisioning failed"
    capture.emit(_record("Dispatch completed with errors", pathname=_ENGINE_FILE))
    assert capture.cause == "Topic provisioning failed"
