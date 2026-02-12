# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Session lifecycle FSM states.

Defines the state machine states for session lifecycle management.
Used by the session lifecycle reducer for concurrent pipeline isolation.

FSM Diagram::

    +------+   create_run   +-------------+   activate_run   +------------+
    | idle | -------------> | run_created | ---------------> | run_active |
    +------+                +-------------+                  +------------+
       ^                                                          |
       |                    +----------+                          |
       +--------------------| run_ended | <-----------------------+
                            +----------+       end_run
"""

from enum import Enum


class EnumSessionLifecycleState(str, Enum):
    """Session lifecycle FSM states.

    Attributes:
        IDLE: No active run — waiting for a pipeline to start.
        RUN_CREATED: Run document created, not yet active.
        RUN_ACTIVE: Run is actively executing.
        RUN_ENDED: Run has completed or been terminated.
    """

    IDLE = "idle"
    RUN_CREATED = "run_created"
    RUN_ACTIVE = "run_active"
    RUN_ENDED = "run_ended"


__all__: list[str] = ["EnumSessionLifecycleState"]
