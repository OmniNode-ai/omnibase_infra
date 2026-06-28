# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for HandlerLlmCliSubprocess.execute_cli_inference.

OMN-13743: Raise coverage of handler_llm_cli_subprocess.py.

Covers execute_cli_inference branches (previously uncovered):
- UNAVAILABLE when self._cli is None (no CLI binary specified)
- UNAVAILABLE when shutil.which returns None for an explicitly-set binary
- INVALID_REQUEST when no user-role message in request.messages
- INVALID_REQUEST when user message exists but content is None / empty
- SUBPROCESS_ERROR when subprocess.run returns non-zero exit code
- SUBPROCESS_ERROR stderr truncation and empty-stderr placeholder
- EMPTY_RESPONSE when stdout is empty after a successful exit
- SUCCESS with fully-constructed ModelLlmInferenceResponse
- SUCCESS token estimates scale with word count
- SUCCESS uses last user message when multiple messages present
- TIMEOUT when subprocess.TimeoutExpired is raised
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import pytest

from omnibase_infra.models.llm.model_llm_inference_request import (
    ModelLlmInferenceRequest,
)
from omnibase_infra.models.llm.model_llm_message import ModelLlmMessage
from omnibase_infra.nodes.node_llm_inference_effect.handlers.handler_llm_cli_subprocess import (
    EnumCliBackendStatus,
    HandlerLlmCliSubprocess,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _req(
    model: str = "gemini-cli",
    user_content: str = "hello",
    messages: tuple[ModelLlmMessage, ...] | None = None,
    correlation_id: UUID | None = None,
) -> ModelLlmInferenceRequest:
    """Build a minimal CHAT_COMPLETION request."""
    if messages is None:
        messages = (ModelLlmMessage(role="user", content=user_content),)
    kwargs: dict = {
        "base_url": "http://localhost:1",
        "model": model,
        "messages": messages,
    }
    if correlation_id is not None:
        kwargs["correlation_id"] = correlation_id
    return ModelLlmInferenceRequest(**kwargs)


def _ok(
    stdout: str = "the answer\n", returncode: int = 0, stderr: str = ""
) -> subprocess.CompletedProcess:  # type: ignore[type-arg]
    return subprocess.CompletedProcess(
        args=["gemini", "-p", "hello"],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


# ---------------------------------------------------------------------------
# UNAVAILABLE: cli is None
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_unavailable_when_cli_is_none() -> None:
    """UNAVAILABLE when cli=None and the request model has no CLI mapping."""
    handler = HandlerLlmCliSubprocess()  # cli=None (default)
    req = _req(model="unmapped-cli")

    with patch("subprocess.run") as mock_run:
        response, status, detail = handler.execute_cli_inference(req)

    assert response is None
    assert status == EnumCliBackendStatus.UNAVAILABLE
    assert "cli not configured" in detail
    # subprocess.run must never be reached
    mock_run.assert_not_called()


@pytest.mark.unit
def test_unavailable_detail_mentions_no_binary() -> None:
    """UNAVAILABLE detail message is descriptive."""
    handler = HandlerLlmCliSubprocess()
    req = _req(model="unmapped-cli")
    _, _, detail = handler.execute_cli_inference(req)
    assert detail != ""


# ---------------------------------------------------------------------------
# UNAVAILABLE: binary not found on PATH
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_unavailable_when_which_returns_none() -> None:
    """UNAVAILABLE when shutil.which cannot find the CLI binary."""
    handler = HandlerLlmCliSubprocess(cli="gemini", cli_args=["-p"])
    req = _req()

    with (
        patch("shutil.which", return_value=None),
        patch("subprocess.run") as mock_run,
    ):
        response, status, detail = handler.execute_cli_inference(req)

    assert response is None
    assert status == EnumCliBackendStatus.UNAVAILABLE
    assert "gemini" in detail
    assert "PATH" in detail
    mock_run.assert_not_called()


@pytest.mark.unit
def test_unavailable_detail_contains_binary_name() -> None:
    """UNAVAILABLE detail names the missing binary."""
    handler = HandlerLlmCliSubprocess(cli="my-llm-tool", cli_args=["--prompt"])
    req = _req()

    with patch("shutil.which", return_value=None):
        _, _, detail = handler.execute_cli_inference(req)

    assert "my-llm-tool" in detail


# ---------------------------------------------------------------------------
# INVALID_REQUEST: no user-role message / empty content
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_invalid_request_when_no_user_role_message() -> None:
    """INVALID_REQUEST when messages contain no role='user' entry."""
    handler = HandlerLlmCliSubprocess(cli="gemini", cli_args=["-p"])

    # Use a mock to bypass Pydantic validation and inject a non-user message
    mock_req = MagicMock(spec=ModelLlmInferenceRequest)
    mock_req.model = "gemini-cli"
    assistant_msg = MagicMock()
    assistant_msg.role = "assistant"
    assistant_msg.content = "I am the model"
    mock_req.messages = [assistant_msg]

    with (
        patch("shutil.which", return_value="/usr/bin/gemini"),
        patch("subprocess.run") as mock_run,
    ):
        response, status, detail = handler.execute_cli_inference(mock_req)

    assert response is None
    assert status == EnumCliBackendStatus.INVALID_REQUEST
    assert "no user message" in detail
    mock_run.assert_not_called()


@pytest.mark.unit
def test_invalid_request_when_user_message_content_is_none() -> None:
    """INVALID_REQUEST when user message exists but content is None."""
    handler = HandlerLlmCliSubprocess(cli="gemini", cli_args=["-p"])

    mock_req = MagicMock(spec=ModelLlmInferenceRequest)
    mock_req.model = "gemini-cli"
    user_msg = MagicMock()
    user_msg.role = "user"
    user_msg.content = None
    mock_req.messages = [user_msg]

    with (
        patch("shutil.which", return_value="/usr/bin/gemini"),
        patch("subprocess.run") as mock_run,
    ):
        response, status, _detail = handler.execute_cli_inference(mock_req)

    assert response is None
    assert status == EnumCliBackendStatus.INVALID_REQUEST
    mock_run.assert_not_called()


@pytest.mark.unit
def test_invalid_request_when_user_message_content_is_empty_string() -> None:
    """INVALID_REQUEST when user message content is an empty string."""
    handler = HandlerLlmCliSubprocess(cli="gemini", cli_args=["-p"])

    mock_req = MagicMock(spec=ModelLlmInferenceRequest)
    mock_req.model = "gemini-cli"
    user_msg = MagicMock()
    user_msg.role = "user"
    user_msg.content = ""
    mock_req.messages = [user_msg]

    with (
        patch("shutil.which", return_value="/usr/bin/gemini"),
        patch("subprocess.run") as mock_run,
    ):
        response, status, _detail = handler.execute_cli_inference(mock_req)

    assert response is None
    assert status == EnumCliBackendStatus.INVALID_REQUEST
    mock_run.assert_not_called()


# ---------------------------------------------------------------------------
# SUBPROCESS_ERROR: non-zero exit code
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_subprocess_error_on_nonzero_exit() -> None:
    """SUBPROCESS_ERROR when subprocess exits with code != 0."""
    handler = HandlerLlmCliSubprocess(cli="gemini", cli_args=["-p"])
    req = _req()

    failed = _ok(returncode=1, stdout="", stderr="authentication error: bad token")

    with (
        patch("shutil.which", return_value="/usr/bin/gemini"),
        patch("subprocess.run", return_value=failed),
    ):
        response, status, detail = handler.execute_cli_inference(req)

    assert response is None
    assert status == EnumCliBackendStatus.SUBPROCESS_ERROR
    assert "exit 1" in detail
    assert "authentication error" in detail


@pytest.mark.unit
def test_subprocess_error_exit_code_appears_in_detail() -> None:
    """Non-zero exit code value is embedded in the SUBPROCESS_ERROR detail."""
    handler = HandlerLlmCliSubprocess(cli="gemini", cli_args=["-p"])
    req = _req()

    with (
        patch("shutil.which", return_value="/usr/bin/gemini"),
        patch("subprocess.run", return_value=_ok(returncode=42, stderr="oops")),
    ):
        _, status, detail = handler.execute_cli_inference(req)

    assert status == EnumCliBackendStatus.SUBPROCESS_ERROR
    assert "42" in detail


@pytest.mark.unit
def test_subprocess_error_empty_stderr_shows_placeholder() -> None:
    """When stderr is empty, SUBPROCESS_ERROR detail contains '(no stderr)'."""
    handler = HandlerLlmCliSubprocess(cli="gemini", cli_args=["-p"])
    req = _req()

    with (
        patch("shutil.which", return_value="/usr/bin/gemini"),
        patch("subprocess.run", return_value=_ok(returncode=1, stderr="")),
    ):
        _, status, detail = handler.execute_cli_inference(req)

    assert status == EnumCliBackendStatus.SUBPROCESS_ERROR
    assert "(no stderr)" in detail


@pytest.mark.unit
def test_subprocess_error_stderr_truncated_to_200_chars() -> None:
    """SUBPROCESS_ERROR detail truncates stderr preview to 200 characters."""
    handler = HandlerLlmCliSubprocess(cli="gemini", cli_args=["-p"])
    req = _req()

    long_stderr = "X" * 500

    with (
        patch("shutil.which", return_value="/usr/bin/gemini"),
        patch("subprocess.run", return_value=_ok(returncode=1, stderr=long_stderr)),
    ):
        _, status, detail = handler.execute_cli_inference(req)

    assert status == EnumCliBackendStatus.SUBPROCESS_ERROR
    # The 500-char stderr should be truncated to <=200 in the detail
    assert long_stderr not in detail  # full string not present
    assert "X" * 200 in detail  # first 200 chars are present


# ---------------------------------------------------------------------------
# EMPTY_RESPONSE: exit 0, stdout empty
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_empty_response_when_stdout_is_empty_string() -> None:
    """EMPTY_RESPONSE returned when subprocess exits 0 but stdout is empty."""
    handler = HandlerLlmCliSubprocess(cli="gemini", cli_args=["-p"])
    req = _req()

    with (
        patch("shutil.which", return_value="/usr/bin/gemini"),
        patch("subprocess.run", return_value=_ok(stdout="")),
    ):
        response, status, detail = handler.execute_cli_inference(req)

    assert response is None
    assert status == EnumCliBackendStatus.EMPTY_RESPONSE
    assert "empty" in detail


@pytest.mark.unit
def test_empty_response_when_stdout_is_whitespace_only() -> None:
    """EMPTY_RESPONSE when stdout contains only whitespace (strip() → empty)."""
    handler = HandlerLlmCliSubprocess(cli="gemini", cli_args=["-p"])
    req = _req()

    with (
        patch("shutil.which", return_value="/usr/bin/gemini"),
        patch("subprocess.run", return_value=_ok(stdout="   \n\t  ")),
    ):
        response, status, _detail = handler.execute_cli_inference(req)

    assert response is None
    assert status == EnumCliBackendStatus.EMPTY_RESPONSE


# ---------------------------------------------------------------------------
# SUCCESS: fully-constructed ModelLlmInferenceResponse
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_success_returns_response_with_generated_text() -> None:
    """SUCCESS returns ModelLlmInferenceResponse with stripped stdout as text."""
    from omnibase_infra.models.llm.model_llm_inference_response import (
        ModelLlmInferenceResponse,
    )

    handler = HandlerLlmCliSubprocess(cli="gemini", cli_args=["-p"])
    req = _req(user_content="What is 2+2?")

    with (
        patch("shutil.which", return_value="/usr/bin/gemini"),
        patch("subprocess.run", return_value=_ok(stdout="  Four  \n")),
    ):
        response, status, detail = handler.execute_cli_inference(req)

    assert status == EnumCliBackendStatus.SUCCESS
    assert detail == ""
    assert isinstance(response, ModelLlmInferenceResponse)
    assert response.generated_text == "Four"  # stripped


@pytest.mark.unit
def test_success_model_used_reflects_cli_binary() -> None:
    """SUCCESS response model_used encodes the CLI binary name."""
    handler = HandlerLlmCliSubprocess(cli="gemini", cli_args=["-p"])
    req = _req()

    with (
        patch("shutil.which", return_value="/usr/bin/gemini"),
        patch("subprocess.run", return_value=_ok()),
    ):
        response, status, _ = handler.execute_cli_inference(req)

    assert status == EnumCliBackendStatus.SUCCESS
    assert response is not None
    assert "gemini" in response.model_used


@pytest.mark.unit
def test_success_operation_type_and_finish_reason() -> None:
    """SUCCESS response has CHAT_COMPLETION operation and STOP finish reason."""
    from omnibase_infra.enums import EnumLlmFinishReason, EnumLlmOperationType

    handler = HandlerLlmCliSubprocess(cli="gemini", cli_args=["-p"])
    req = _req()

    with (
        patch("shutil.which", return_value="/usr/bin/gemini"),
        patch("subprocess.run", return_value=_ok()),
    ):
        response, status, _ = handler.execute_cli_inference(req)

    assert status == EnumCliBackendStatus.SUCCESS
    assert response is not None
    assert response.operation_type == EnumLlmOperationType.CHAT_COMPLETION
    assert response.finish_reason == EnumLlmFinishReason.STOP


@pytest.mark.unit
def test_success_usage_fields_are_positive_integers() -> None:
    """SUCCESS response token estimates are positive and internally consistent."""
    handler = HandlerLlmCliSubprocess(cli="gemini", cli_args=["-p"])
    req = _req(user_content="Explain quantum entanglement")

    with (
        patch("shutil.which", return_value="/usr/bin/gemini"),
        patch(
            "subprocess.run",
            return_value=_ok(stdout="It is a quantum phenomenon\n"),
        ),
    ):
        response, _, _ = handler.execute_cli_inference(req)

    assert response is not None
    assert response.usage.tokens_input > 0
    assert response.usage.tokens_output > 0
    assert response.usage.tokens_total == (
        response.usage.tokens_input + response.usage.tokens_output
    )


@pytest.mark.unit
def test_success_latency_ms_is_non_negative() -> None:
    """SUCCESS response latency_ms field is >= 0."""
    handler = HandlerLlmCliSubprocess(cli="gemini", cli_args=["-p"])
    req = _req()

    with (
        patch("shutil.which", return_value="/usr/bin/gemini"),
        patch("subprocess.run", return_value=_ok()),
    ):
        response, _, _ = handler.execute_cli_inference(req)

    assert response is not None
    assert response.latency_ms >= 0.0


@pytest.mark.unit
def test_success_backend_result_reports_success() -> None:
    """SUCCESS response backend_result.success is True."""
    handler = HandlerLlmCliSubprocess(cli="gemini", cli_args=["-p"])
    req = _req()

    with (
        patch("shutil.which", return_value="/usr/bin/gemini"),
        patch("subprocess.run", return_value=_ok()),
    ):
        response, _, _ = handler.execute_cli_inference(req)

    assert response is not None
    assert response.backend_result.success is True


@pytest.mark.unit
def test_success_correlation_id_propagated_from_request() -> None:
    """SUCCESS response correlation_id matches the request's correlation_id."""
    caller_cid = uuid4()
    handler = HandlerLlmCliSubprocess(cli="gemini", cli_args=["-p"])
    req = _req(correlation_id=caller_cid)

    with (
        patch("shutil.which", return_value="/usr/bin/gemini"),
        patch("subprocess.run", return_value=_ok()),
    ):
        response, _, _ = handler.execute_cli_inference(req)

    assert response is not None
    assert response.correlation_id == caller_cid


@pytest.mark.unit
def test_success_execution_id_is_fresh_uuid() -> None:
    """SUCCESS response execution_id is a valid UUID (auto-generated per call)."""
    handler = HandlerLlmCliSubprocess(cli="gemini", cli_args=["-p"])
    req = _req()

    with (
        patch("shutil.which", return_value="/usr/bin/gemini"),
        patch("subprocess.run", return_value=_ok()),
    ):
        response, _, _ = handler.execute_cli_inference(req)

    assert response is not None
    assert isinstance(response.execution_id, UUID)


@pytest.mark.unit
def test_success_timestamp_is_timezone_aware() -> None:
    """SUCCESS response timestamp is timezone-aware."""
    handler = HandlerLlmCliSubprocess(cli="gemini", cli_args=["-p"])
    req = _req()

    with (
        patch("shutil.which", return_value="/usr/bin/gemini"),
        patch("subprocess.run", return_value=_ok()),
    ):
        response, _, _ = handler.execute_cli_inference(req)

    assert response is not None
    assert response.timestamp.tzinfo is not None


@pytest.mark.unit
def test_success_uses_last_user_message_as_prompt() -> None:
    """When multiple user messages exist, the LAST one is used as the prompt."""
    handler = HandlerLlmCliSubprocess(cli="gemini", cli_args=["-p"])
    req = _req(
        messages=(
            ModelLlmMessage(role="user", content="first message"),
            ModelLlmMessage(role="user", content="last message"),
        )
    )

    with (
        patch("shutil.which", return_value="/usr/bin/gemini"),
        patch("subprocess.run", return_value=_ok()) as mock_run,
    ):
        _, status, _ = handler.execute_cli_inference(req)

    assert status == EnumCliBackendStatus.SUCCESS
    call_argv = mock_run.call_args.args[0]
    assert call_argv[-1] == "last message"


@pytest.mark.unit
def test_success_token_estimates_scale_with_length() -> None:
    """Longer prompts and responses produce higher token estimates."""
    handler = HandlerLlmCliSubprocess(cli="gemini", cli_args=["-p"])

    short_req = _req(user_content="hi")
    long_req = _req(
        user_content="Please give me a very comprehensive and detailed explanation of all aspects of quantum mechanics"
    )

    short_out = "OK\n"
    long_out = " ".join(["word"] * 40) + "\n"

    with (
        patch("shutil.which", return_value="/usr/bin/gemini"),
        patch("subprocess.run", return_value=_ok(stdout=short_out)),
    ):
        short_resp, _, _ = handler.execute_cli_inference(short_req)

    with (
        patch("shutil.which", return_value="/usr/bin/gemini"),
        patch("subprocess.run", return_value=_ok(stdout=long_out)),
    ):
        long_resp, _, _ = handler.execute_cli_inference(long_req)

    assert short_resp is not None and long_resp is not None
    assert long_resp.usage.tokens_input > short_resp.usage.tokens_input
    assert long_resp.usage.tokens_output > short_resp.usage.tokens_output


@pytest.mark.unit
def test_success_cli_args_forwarded_to_subprocess() -> None:
    """CLI args configured on the handler are forwarded verbatim to subprocess.run."""
    custom_args = ["--mode", "headless", "--no-color"]
    handler = HandlerLlmCliSubprocess(cli="gemini", cli_args=custom_args)
    req = _req(user_content="ping")

    with (
        patch("shutil.which", return_value="/usr/bin/gemini"),
        patch("subprocess.run", return_value=_ok()) as mock_run,
    ):
        handler.execute_cli_inference(req)

    call_argv = mock_run.call_args.args[0]
    # argv: [binary, *cli_args, prompt]
    assert call_argv[0] == "gemini"
    assert call_argv[1:4] == custom_args
    assert call_argv[-1] == "ping"


# ---------------------------------------------------------------------------
# TIMEOUT: subprocess.TimeoutExpired
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_timeout_returned_on_timeout_expired() -> None:
    """TIMEOUT returned when subprocess.run raises TimeoutExpired."""
    handler = HandlerLlmCliSubprocess(cli="gemini", cli_args=["-p"], timeout=5)
    req = _req()

    with (
        patch("shutil.which", return_value="/usr/bin/gemini"),
        patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="gemini", timeout=5),
        ),
    ):
        response, status, detail = handler.execute_cli_inference(req)

    assert response is None
    assert status == EnumCliBackendStatus.TIMEOUT
    assert "gemini" in detail
    assert "5" in detail


@pytest.mark.unit
def test_timeout_detail_contains_configured_deadline() -> None:
    """TIMEOUT detail message mentions the configured timeout value."""
    timeout_val = 77
    handler = HandlerLlmCliSubprocess(
        cli="gemini", cli_args=["-p"], timeout=timeout_val
    )
    req = _req()

    with (
        patch("shutil.which", return_value="/usr/bin/gemini"),
        patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="gemini", timeout=timeout_val),
        ),
    ):
        _, status, detail = handler.execute_cli_inference(req)

    assert status == EnumCliBackendStatus.TIMEOUT
    assert str(timeout_val) in detail


@pytest.mark.unit
def test_timeout_response_is_none() -> None:
    """TIMEOUT tuple has None as the response element."""
    handler = HandlerLlmCliSubprocess(cli="gemini", cli_args=["-p"], timeout=1)
    req = _req()

    with (
        patch("shutil.which", return_value="/usr/bin/gemini"),
        patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="gemini", timeout=1),
        ),
    ):
        response, _, _ = handler.execute_cli_inference(req)

    assert response is None
