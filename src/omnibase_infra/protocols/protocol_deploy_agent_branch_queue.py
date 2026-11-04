# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The existing agent's durable acknowledgement port."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from deploy_agent.job_state import JobStore


class ProtocolDeployAgentBranchQueue(Protocol):
    _state: str
    job_store: JobStore

    def _publish_trial_result(self, payload: dict[str, object]) -> None: ...
