# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The two sends an overdue dev merge makes: the event and the signed build request (OMN-26316)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from omnibase_infra.models.model_dev_merge_build_command import (
    ModelDevMergeBuildCommand,
)
from omnibase_infra.models.model_dev_merge_build_overdue_event import (
    ModelDevMergeBuildOverdueEvent,
)


@runtime_checkable
class ProtocolDevMergeBuildSender(Protocol):
    """``send_command`` goes through the signed build request path and returns its status."""

    def publish_event(self, event: ModelDevMergeBuildOverdueEvent) -> None: ...

    def send_command(self, command: ModelDevMergeBuildCommand) -> str: ...
