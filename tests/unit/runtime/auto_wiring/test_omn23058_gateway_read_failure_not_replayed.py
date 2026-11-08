# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-23058 — a failed GitHub gateway read is answered once and never replayed.

Measured on the lab dev lane from the gateway's terminal topic, 2026-11-08
03:33Z to 03:46Z: 1,256 boundary failure terminals for 133 distinct commands
(10 to 15 terminals per command, in bursts of five), 742 of them
``job log not cached: HTTP 404``. The gateway topic has one partition and one
active consumer, so each replay queued behind the backlog it was part of; the
same minutes were the ones in which ``onex scm`` reads from lab hosts timed out
at 60 s and retried. A read has a caller that already holds the failure
terminal: a replay answers a request nobody waits for, and spends GitHub quota
the live readers need.

``GitHubTransportError`` reaches the consume boundary only as a flattened
class name inside the failure text, so ``EnumNonRetryableErrorCategory`` must
name it for ``should_replay`` and ``classify_boundary_failure`` to read it.
"""

from __future__ import annotations

from uuid import UUID

import pytest

from omnibase_infra.enums import EnumNonRetryableErrorCategory
from omnibase_infra.nodes.node_dlq_replay_effect.engine_dlq_replay import (
    ModelDlqReplayEngineConfig,
    should_replay,
)
from omnibase_infra.nodes.node_dlq_replay_effect.models.model_dlq_message import (
    ModelDlqMessage,
)
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    HandlerDispatchFailureError,
)
from omnibase_infra.runtime.boundary_failure_terminal import classify_boundary_failure
from omnibase_infra.utils.util_error_sanitization import sanitize_error_message

_COMMAND_TOPIC = "onex.cmd.omnimarket.github-repo-gateway-read-requested.v1"
_CORRELATION = UUID("cb9580b5-b45d-45bd-bcad-99bb5ded8b00")
_CONFIG = ModelDlqReplayEngineConfig(
    bootstrap_servers="localhost:9092",
    dlq_topic="test-dlq-topic",
)

# The failure texts the gateway recorded in the window above, one per cause.
_GATEWAY_FAILURES = [
    "job log not cached: HTTP 404, read_ceiling_hit=False",
    "GitHub request failed: [{'type': 'RATE_LIMITED', 'message': 'API rate limit exceeded'}]",
    "GitHub GET /repos/OmniNode-ai/omnimarket/actions/runs/1/jobs failed: HTTP 502",
]


class GitHubTransportError(Exception):
    """Stands for omnimarket's class of the same name; infra matches it by name."""


def _flattened(reason: str) -> HandlerDispatchFailureError:
    return HandlerDispatchFailureError(
        "dispatch to topic=onex.cmd.omnimarket.github-repo-gateway-read-requested.v1 "
        f"returned status=handler_error: GitHubTransportError: {reason}"
    )


def _dlq_message(reason: str, retry_count: int = 0) -> ModelDlqMessage:
    return ModelDlqMessage(
        original_topic=_COMMAND_TOPIC,
        original_value='{"operation": "job_log_tail"}',
        failure_reason=sanitize_error_message(_flattened(reason)),
        correlation_id=_CORRELATION,
        retry_count=retry_count,
        error_type="HandlerDispatchFailureError",
        dlq_offset=1,
        dlq_partition=0,
    )


@pytest.mark.unit
def test_github_transport_error_is_a_named_non_retryable_category() -> None:
    assert EnumNonRetryableErrorCategory.is_non_retryable(GitHubTransportError.__name__)


@pytest.mark.unit
@pytest.mark.parametrize("reason", _GATEWAY_FAILURES)
def test_failed_gateway_read_is_not_replayed(reason: str) -> None:
    eligible, why = should_replay(_dlq_message(reason), _CONFIG)
    assert eligible is False, why
    assert "non-retryable" in why.lower()


@pytest.mark.unit
@pytest.mark.parametrize("reason", _GATEWAY_FAILURES)
def test_failed_gateway_read_terminal_states_not_retryable(reason: str) -> None:
    terminal = classify_boundary_failure(
        _flattened(reason),
        topic=_COMMAND_TOPIC,
        correlation_id=_CORRELATION,
        failure_reason=sanitize_error_message(_flattened(reason)),
    )
    assert terminal.failure_class == "GitHubTransportError"
    assert terminal.retryable is False


@pytest.mark.unit
def test_transient_failure_of_another_class_is_still_replayed() -> None:
    """Positive control: the fix names one class; it does not turn replay off."""
    message = ModelDlqMessage(
        original_topic=_COMMAND_TOPIC,
        original_value='{"operation": "pr_text"}',
        failure_reason=sanitize_error_message(
            RuntimeError("dispatch failed: InfraConnectionError: broker unreachable")
        ),
        correlation_id=_CORRELATION,
        retry_count=0,
        error_type="HandlerDispatchFailureError",
        dlq_offset=1,
        dlq_partition=0,
    )
    eligible, why = should_replay(message, _CONFIG)
    assert eligible is True, why
