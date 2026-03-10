# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Update plan FSM states.

Defines the state machine states for artifact update plan lifecycle management.
Used by the update plan reducer for tracking plan progression from creation
through comment posting, YAML emission, and closure or waiver.

FSM Diagram::

    +------+  create_plan  +---------+  post_comment  +----------------+
    | idle | ------------> | created | -------------> | comment_posted |
    +------+               +---------+                +----------------+
                                                              |
                                           emit_yaml          |  waive
                                              |               |
                                              v               v
                                       +-----------+    +--------+
                                       |yaml_emitted|   | waived |
                                       +-----------+    +--------+
                                              |
                                          close
                                              |
                                              v
                                         +--------+
                                         | closed |
                                         +--------+

Tracking:
    - OMN-3943: Task 6 — Update Plan REDUCER Node
    - OMN-3925: Artifact Reconciliation + Update Planning MVP
"""

from enum import Enum


class EnumUpdatePlanState(str, Enum):
    """Update plan lifecycle FSM states.

    Attributes:
        IDLE: No active plan — reducer ready for new trigger.
        CREATED: Plan created from impact analysis result, tasks assigned.
        COMMENT_POSTED: PR comment with impact summary has been posted.
        YAML_EMITTED: YAML plan emitted as structured artifact.
        CLOSED: Plan fully processed and closed.
        WAIVED: Plan skipped via explicit waiver (e.g. no-op merge policy).
    """

    IDLE = "idle"
    CREATED = "created"
    COMMENT_POSTED = "comment_posted"
    YAML_EMITTED = "yaml_emitted"
    CLOSED = "closed"
    WAIVED = "waived"


__all__: list[str] = ["EnumUpdatePlanState"]
