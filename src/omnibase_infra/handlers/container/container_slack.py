# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Binding site for HandlerSlackWebhook (OMN-21683).

The waitlist notifier and the runner-health alert CLI send Slack messages through
``HandlerSlackWebhook``. This module is the one place that builds it, so a caller
takes the handler from here and never constructs it by hand.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_infra.handlers.handler_slack_webhook import HandlerSlackWebhook

if TYPE_CHECKING:
    import aiohttp

__all__: list[str] = ["bind_slack_webhook"]


def bind_slack_webhook(
    *,
    http_session: aiohttp.ClientSession | None = None,
    bot_token: str | None = None,
    default_channel: str | None = None,
) -> HandlerSlackWebhook:
    """Build the Slack webhook handler for the given session, token and channel."""
    return HandlerSlackWebhook(
        http_session=http_session,
        bot_token=bot_token,
        default_channel=default_channel,
    )
