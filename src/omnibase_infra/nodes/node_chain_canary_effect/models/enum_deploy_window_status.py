# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""What the deploy agent said about the probe's window (OMN-19811)."""

from __future__ import annotations

from enum import StrEnum


class EnumDeployWindowStatus(StrEnum):
    """Why this run did, or did not, retry after a deploy.

    OMN-19811: chain-canary run 36202173467 (2026-09-25T23:44Z) reported
    ``terminal_missing`` because deploy-agent job 193cfeda recreated
    ``omninode-runtime`` inside the 120 s budget. The chain was fine; the
    lane was being replaced underneath the probe. Nothing in the receipt said
    so, because the canary never asked the deploy agent.

    Every member except ``DEPLOY_IN_WINDOW_RETRIED`` leaves the first
    attempt's verdict standing. In particular ``AGENT_UNREADABLE`` and
    ``NOT_CONFIGURED`` never retry: a deploy nobody could see is not evidence
    that a deploy happened, and retrying on its absence would let an
    intermittently dead chain report green.
    """

    # No deploy-agent URL was configured. The run behaves exactly as it did
    # before OMN-19811, and says so.
    NOT_CONFIGURED = "not_configured"
    # The first attempt ended in a verdict a redeploy does not explain (only
    # TERMINAL_MISSING and INGRESS_UNREACHABLE are), so the window was not
    # examined for a retry. The pre-fire wait (if any) is still recorded.
    NOT_NEEDED = "not_needed"
    # The agent was read and no deploy job was accepted, running or completed
    # inside the window. The RED is a real one and stays RED.
    NO_DEPLOY_IN_WINDOW = "no_deploy_in_window"
    # The agent's surface could not be read after the first attempt, so there
    # is no evidence either way. Fails closed: no retry, the RED stands.
    AGENT_UNREADABLE = "agent_unreadable"
    # A deploy overlapped the window, and the lane did not converge (agent
    # idle, readiness 200) inside the remaining wait budget. No retry: firing
    # into a lane still being replaced would only repeat the first attempt.
    DEPLOY_IN_WINDOW_NOT_CONVERGED = "deploy_in_window_not_converged"
    # A deploy overlapped the window, the lane converged, and the probe was
    # fired ONCE more. The run's verdict is the retry's verdict, whatever it is.
    DEPLOY_IN_WINDOW_RETRIED = "deploy_in_window_retried"


__all__ = ["EnumDeployWindowStatus"]
