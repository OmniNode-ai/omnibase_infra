# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Signed transport envelopes should validate as deploy commands."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from deploy_agent.consumer import DeployConsumer
from deploy_agent.events import EnumRuntimeLane


@pytest.mark.unit
def test_signed_payload_strips_signature_before_command_validation() -> None:
    consumer = DeployConsumer.__new__(DeployConsumer)
    consumer.consumer = Mock()
    consumer.job_store = Mock()
    consumer.job_store.has_active_job.return_value = False
    consumer.job_store.is_duplicate.return_value = False
    # OMN-16939: the lane fence is required; this payload is lane=dev.
    consumer.allowed_lanes = frozenset({EnumRuntimeLane.DEV})
    # OMN-16442: the pre-accept self-update boundary. These tests are not
    # testing self-update, so the hook is an explicit no-op rather than absent
    # -- an absent attribute would be swallowed by the boundary's own
    # error rail and read as a pass.
    consumer.self_update_hook = lambda rewind: None

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
        cmd, reason = consumer._process_message(
            SimpleNamespace(
                value=payload,
                topic="onex.cmd.deploy.rebuild-requested.v1",
                partition=0,
                offset=3,
            )
        )

    assert reason is None
    assert cmd is not None
    consumer.job_store.accept.assert_called_once()
    accepted = consumer.job_store.accept.call_args.kwargs["command"]
    assert "_signature" not in accepted
