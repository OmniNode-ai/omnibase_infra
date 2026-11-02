# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The packaged endpoints file loads and feeds the real consumer modules (OMN-21623)."""

from __future__ import annotations

import pytest

from omnibase_infra.handlers import handler_gmail_api, handler_slack_webhook
from omnibase_infra.models.third_party_endpoints import load_third_party_endpoints

pytestmark = pytest.mark.integration


def test_packaged_endpoints_file_loads_and_consumers_hold_its_values() -> None:
    endpoints = load_third_party_endpoints()
    assert endpoints.slack_post_message_url == handler_slack_webhook._SLACK_WEB_API_URL
    assert endpoints.gmail_api_base_url == handler_gmail_api._GMAIL_API_BASE
    assert endpoints.github_pull_url("org/repo", 7).startswith(
        endpoints.github_web_base_url
    )
