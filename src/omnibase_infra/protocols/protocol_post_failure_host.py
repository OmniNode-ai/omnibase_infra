# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
from __future__ import annotations

from collections.abc import Callable, Collection
from datetime import datetime
from typing import Protocol

from omnibase_infra.protocols.protocol_command import ProtocolCommand
from omnibase_infra.protocols.protocol_job_store import ProtocolJobStore
from omnibase_infra.protocols.protocol_lane_flags import ProtocolLaneFlags


class ProtocolPostFailureHost(Protocol):
    """The deploy agent, as far as the rule reads it; looked up at call time."""

    job_store: ProtocolJobStore
    _router: object | None
    _allowed_lanes: Collection[str]
    _idle_converge_now: Callable[[], datetime]
    _idle_read_running_ref: Callable[[str], str | None]
    _idle_read_head_ref: Callable[[str], str | None]
    _idle_probe_blocker: Callable[[datetime], str | None]
    _lane_flags: ProtocolLaneFlags

    def _idle_converge_gate_open(self, who: str) -> bool: ...

    def _execute_command(self, cmd: ProtocolCommand) -> None: ...
