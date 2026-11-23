# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A routed command with no job record raises one alarm row (OMN-26485).

THE FAILURE
-----------
The second instance's job record held 546 jobs and none answered the five held commands routed to it:
the agent dropped each command without a record, and nothing noticed the silence.

THE ALARM
---------
A command is answered by a job record with its ``correlation_id`` on the instance it was
routed to, in any status. A command with none once the contract's
``routed_command_no_job_window_seconds`` have passed since it was routed writes one ALARM
row (``src=alarm``, ``alarm=routed-command-no-job-record``) naming the instance and the
command, through the append the node's other rows use, handed in as ``append(row, request_id)``
like the build decision handler's. The row is a function of the command alone (its stamp is the
end of the window), and the append's request id is derived from the instance and the
command, so a command that is polled again every minute writes one row, not one per poll.
"""

from __future__ import annotations

import logging
import re
import uuid
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta

from pydantic import JsonValue

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.model_routed_command import ModelRoutedCommand
from omnibase_infra.models.model_routed_command_alarm_request import (
    ModelRoutedCommandAlarmRequest,
)
from omnibase_infra.models.model_routed_command_alarm_result import (
    ModelRoutedCommandAlarmResult,
)
from omnibase_infra.models.model_routed_command_alarm_row import (
    ModelRoutedCommandAlarmRow,
)

logger = logging.getLogger(__name__)

_ALARM = "routed-command-no-job-record"
_TOKEN_UNSAFE = re.compile(r"[^A-Za-z0-9._:-]+")


def _text(settings: Mapping[str, JsonValue], key: str) -> str:
    value = settings[key]
    if not isinstance(value, str):
        raise RuntimeError(f"the deploy job reading contract's {key} is not text")
    return value


def _token(text: str) -> str:
    return _TOKEN_UNSAFE.sub("-", text).strip("-") or "unknown"


def _stamp(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class HandlerRoutedCommandAlarm:
    """Alarm on each routed command that no job record answers inside the window."""

    def __init__(
        self,
        *,
        settings: Mapping[str, JsonValue],
        append: Callable[[str, str], str],
    ) -> None:
        """``append(row, request_id)`` returns ``""`` when the ledger took the row, else why not."""
        self._settings = settings
        self._append = append

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    def _window(self) -> timedelta:
        seconds = self._settings["routed_command_no_job_window_seconds"]
        if not isinstance(seconds, int) or isinstance(seconds, bool) or seconds <= 0:
            raise RuntimeError(
                "the deploy job reading contract's routed_command_no_job_window_seconds "
                "is not a positive whole number"
            )
        return timedelta(seconds=seconds)

    def build_row(self, command: ModelRoutedCommand) -> str:
        """The command's alarm row; identical on every poll after the window ends."""
        window = self._window()
        cells = [
            f"{_stamp(command.routed_at + window)} | STATUS",
            f"lane={_text(self._settings, 'row_lane')}",
            "src=alarm",
            f"actor={_text(self._settings, 'row_actor')}",
            "model=none",
            f"alarm={_ALARM}",
            f"deploy_lane={_token(command.runtime_lane)}",
            f"instance={_token(command.instance)}",
            f"command={command.correlation_id}",
            f"requested_by={_token(command.requested_by)}",
            f"routed_at={_stamp(command.routed_at)}",
            f"window_seconds={int(window.total_seconds())}",
            f"owner={_text(self._settings, 'alarm_routed_command_owner')}",
            f"owner_line={_text(self._settings, 'alarm_routed_command_owner_line')}",
        ]
        return " | ".join(cells)

    def handle(
        self, request: ModelRoutedCommandAlarmRequest
    ) -> ModelRoutedCommandAlarmResult:
        window = self._window()
        answered = {
            (instance, record.correlation_id)
            for instance, records in request.job_records.items()
            for record in records
        }
        alarms: list[ModelRoutedCommandAlarmRow] = []
        waiting: list[uuid.UUID] = []
        for command in request.commands:
            if (command.instance, command.correlation_id) in answered:
                continue
            if request.now - command.routed_at < window:
                waiting.append(command.correlation_id)
                continue
            row = self.build_row(command)
            key = str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"onex:routed-command-alarm:{command.instance}:{command.correlation_id}",
                )
            )
            error = self._append(row, key)
            if error:
                logger.warning(
                    "routed command alarm for %s not appended: %s",
                    command.correlation_id,
                    error,
                )
            alarms.append(
                ModelRoutedCommandAlarmRow(
                    correlation_id=command.correlation_id,
                    instance=command.instance,
                    row=row,
                    append_key=key,
                    outcome="unappended" if error else "appended",
                    detail=error,
                )
            )
        return ModelRoutedCommandAlarmResult(
            alarms=tuple(alarms), waiting=tuple(waiting)
        )
