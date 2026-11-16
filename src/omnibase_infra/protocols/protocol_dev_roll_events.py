# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The events seam the dev boot guard announces a roll through."""

from collections.abc import Mapping
from datetime import datetime
from typing import Protocol

from pydantic import JsonValue


class ProtocolDevRollEvents(Protocol):
    """Publishes dev roll events; every method swallows its own failure."""

    def manifest(self, command: Mapping[str, JsonValue]) -> None: ...

    def started(self, job_id: str, started_at: datetime, cause: str) -> None: ...

    def incident(
        self, guard_incident: Mapping[str, JsonValue], *, issuer: str
    ) -> None: ...

    def preflight_failure(
        self,
        command: Mapping[str, JsonValue],
        *,
        error: str,
        phase: str,
        project: str,
    ) -> None: ...
