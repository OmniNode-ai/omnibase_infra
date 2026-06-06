# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Signed transport envelopes should validate as deploy commands."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from deploy_agent.consumer import DeployConsumer


@pytest.mark.unit
def test_signed_payload_strips_signature_before_command_validation() -> None:
    consumer = DeployConsumer.__new__(DeployConsumer)
    consumer.consumer = Mock()
    consumer.job_store = Mock()
    consumer.job_store.has_active_job.return_value = False
    consumer.job_store.is_duplicate.return_value = False

    payload = {
        "correlation_id": "aaaaaaaa-0000-0000-0000-000000000001",
        "git_ref": "origin/main",
        "requested_by": "operator-manual",
        "scope": "runtime",
        "runtime_lane": "dev",
        "services": [],
        "_signature": "a" * 64,
    }

    with patch("deploy_agent.consumer.verify_command", return_value=True):
        cmd, reason = consumer._process_message(SimpleNamespace(value=payload))

    assert reason is None
    assert cmd is not None
    consumer.job_store.accept.assert_called_once()
    accepted = consumer.job_store.accept.call_args.kwargs["command"]
    assert "_signature" not in accepted
