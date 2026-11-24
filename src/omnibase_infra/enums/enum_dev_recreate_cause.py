# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Why the dev runtime container's start happened, as the observer could prove it."""

from enum import StrEnum


class EnumDevRecreateCause(StrEnum):
    """The cause named on a ``dev_container_recreated`` event.

    ``UNKNOWN`` is a real answer: nothing read ties the start to a deploy job or to the
    restart policy, and no cause is guessed.
    """

    DEPLOY_AGENT_JOB = "deploy_agent_job"
    RESTART_POLICY = "restart_policy"
    UNKNOWN = "unknown"
