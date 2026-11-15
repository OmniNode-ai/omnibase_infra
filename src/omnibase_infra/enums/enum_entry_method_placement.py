# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Where a handler's dispatch entry method is invoked (OMN-24492)."""

from __future__ import annotations

from enum import StrEnum


class EnumEntryMethodPlacement(StrEnum):
    """Placement of one handler entry method invocation.

    ``ON_LOOP`` — the entry method is an ``async def``; it cooperates with
    the loop and is awaited exactly where it is called today.

    ``WORKER_THREAD`` — the entry method is a plain ``def``; invoking it on
    the loop would freeze the loop (heartbeats, readiness endpoint) for as
    long as it runs, so it is invoked on a worker thread instead.
    """

    ON_LOOP = "on_loop"
    WORKER_THREAD = "worker_thread"


__all__ = ["EnumEntryMethodPlacement"]
