# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A routed command with no job record raises one alarm row (OMN-26485).

The handler is built from the node's own contract settings and appends through a real
stand-in append executable that, like the work ledger's append, answers a repeated
request id as a duplicate and writes nothing.
"""

from __future__ import annotations

import functools
import stat
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

import pytest
import yaml
from pydantic import JsonValue

from omnibase_infra.handlers.handler_deploy_job_reading import append_ledger_row
from omnibase_infra.handlers.handler_routed_command_alarm import (
    HandlerRoutedCommandAlarm,
)
from omnibase_infra.models.model_job_record import ModelJobRecord
from omnibase_infra.models.model_routed_command import ModelRoutedCommand
from omnibase_infra.models.model_routed_command_alarm_request import (
    ModelRoutedCommandAlarmRequest,
)

pytestmark = pytest.mark.unit

_CONTRACT = (
    Path(__file__).resolve().parents[3]
    / "src/omnibase_infra/nodes/node_deploy_job_reading_effect/contract.yaml"
)
_ROUTED_AT = datetime(2026, 11, 22, 21, 50, 0, tzinfo=UTC)


def _settings() -> dict[str, JsonValue]:
    settings: dict[str, JsonValue] = yaml.safe_load(
        _CONTRACT.read_text(encoding="utf-8")
    )["settings"]
    return settings


def _append_command(tmp_path: Path) -> tuple[str, Path]:
    """An append executable that keeps one line per request id and ignores a repeat."""
    sink = tmp_path / "rows.txt"
    script = tmp_path / "append.sh"
    script.write_text(
        "#!/bin/sh\n"
        'while [ "$#" -gt 0 ]; do [ "$1" = "--request-id" ] && id="$2"; shift; done\n'
        f'row="$(cat)"\ngrep -qs "^$id " {sink} && exit 0\n'
        f'echo "$id $row" >> {sink}\n'
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return str(script), sink


def _handler(tmp_path: Path) -> tuple[HandlerRoutedCommandAlarm, Path]:
    settings = _settings()
    command, sink = _append_command(tmp_path)
    environ = {
        str(settings["append_command_env"]): command,
        str(settings["host_name_env"]): "host-a",
    }
    handler = HandlerRoutedCommandAlarm(
        settings=settings,
        append=functools.partial(append_ledger_row, settings, environ),
    )
    return handler, sink


def _command(
    instance: str = "inst-b", correlation: UUID | None = None
) -> ModelRoutedCommand:
    return ModelRoutedCommand(
        correlation_id=correlation or uuid4(),
        instance=instance,
        runtime_lane="dev",
        requested_by="gha/omnimarket/pr-1",
        routed_at=_ROUTED_AT,
    )


def _record(correlation: UUID) -> ModelJobRecord:
    return ModelJobRecord.model_validate(
        {
            "correlation_id": str(correlation),
            "command": {"runtime_lane": "dev", "requested_by": "gha/omnimarket/pr-1"},
            "status": "failed",
        }
    )


def _poll(
    commands: tuple[ModelRoutedCommand, ...],
    minutes: float,
    records: dict[str, tuple[ModelJobRecord, ...]] | None = None,
) -> ModelRoutedCommandAlarmRequest:
    return ModelRoutedCommandAlarmRequest(
        commands=commands,
        job_records=records or {},
        now=_ROUTED_AT + timedelta(minutes=minutes),
    )


def test_routed_command_no_job_alarm_names_the_instance_and_the_command(
    tmp_path: Path,
) -> None:
    """AC3: past the declared window with no record, one alarm row names both."""
    handler, sink = _handler(tmp_path)
    command = _command()
    result = handler.handle(_poll((command,), minutes=10))
    assert [a.outcome for a in result.alarms] == ["appended"]
    row = result.alarms[0].row
    assert "src=alarm" in row
    assert "alarm=routed-command-no-job-record" in row
    assert "instance=inst-b" in row
    assert f"command={command.correlation_id}" in row
    assert len(sink.read_text().splitlines()) == 1


def test_routed_command_no_job_alarm_window_is_the_contracts() -> None:
    """The ticket's ten minutes is data in the node contract, never a literal in code."""
    assert _settings().get("routed_command_no_job_window_seconds") == 600


def test_routed_command_no_job_alarm_waits_inside_the_window(tmp_path: Path) -> None:
    handler, sink = _handler(tmp_path)
    command = _command()
    result = handler.handle(_poll((command,), minutes=9.9))
    assert result.alarms == ()
    assert result.waiting == (command.correlation_id,)
    assert not sink.exists()


def test_routed_command_no_job_alarm_fires_once_per_command_not_per_poll(
    tmp_path: Path,
) -> None:
    handler, sink = _handler(tmp_path)
    command = _command()
    for minutes in (10, 11, 25, 60):
        handler.handle(_poll((command,), minutes=minutes))
    assert len(sink.read_text().splitlines()) == 1


def test_routed_command_no_job_alarm_is_answered_by_a_record_on_its_instance(
    tmp_path: Path,
) -> None:
    """A record of any status on the routed instance answers; one on the other does not."""
    handler, sink = _handler(tmp_path)
    answered = _command("inst-b")
    wrong_instance = _command("inst-b")
    result = handler.handle(
        _poll(
            (answered, wrong_instance),
            minutes=30,
            records={
                "inst-b": (_record(answered.correlation_id),),
                "inst-a": (_record(wrong_instance.correlation_id),),
            },
        )
    )
    assert [a.correlation_id for a in result.alarms] == [wrong_instance.correlation_id]
    assert len(sink.read_text().splitlines()) == 1


def test_routed_command_no_job_alarm_reports_an_unwritten_row(tmp_path: Path) -> None:
    settings = _settings()
    handler = HandlerRoutedCommandAlarm(
        settings=settings, append=functools.partial(append_ledger_row, settings, {})
    )
    result = handler.handle(_poll((_command(),), minutes=12))
    assert [a.outcome for a in result.alarms] == ["unappended"]
    assert "must both be set" in result.alarms[0].detail
