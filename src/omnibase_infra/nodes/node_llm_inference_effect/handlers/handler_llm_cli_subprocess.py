# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""CLI subprocess handler for LLM inference via Gemini / Claude / opencode CLI.

Generalizes the proven subprocess dispatch pattern from the hostile reviewer
aggregator (aggregate_reviews.py) into an ONEX handler that accepts
ModelLlmInferenceRequest and returns ModelLlmInferenceResponse.

CLI subprocess handlers distinguish these failure classes:
- UNAVAILABLE: CLI binary not found on PATH
- INVALID_REQUEST: empty prompt or malformed input
- TIMEOUT: subprocess exceeded deadline
- SUBPROCESS_ERROR: non-zero exit code with stderr
- EMPTY_RESPONSE: process succeeded but stdout was empty
- SUCCESS: valid response returned

Related:
    - OMN-7106: Add Gemini CLI and Codex CLI as subprocess LLM handlers
    - OMN-7103: Node-Based LLM Delegation Workflow
    - OMN-13215: codex-cli REMOVED — the delegation ceiling executes over the
      canonical HTTP inference path; no codex subprocess config remains here.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
from datetime import UTC
from enum import Enum
from uuid import uuid4

from omnibase_core.models.dispatch.model_handler_output import ModelHandlerOutput
from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
    EnumInfraTransportType,
)
from omnibase_infra.errors import InfraUnavailableError, ModelInfraErrorContext
from omnibase_infra.models.llm.model_llm_inference_request import (
    ModelLlmInferenceRequest,
)
from omnibase_infra.models.llm.model_llm_inference_response import (
    ModelLlmInferenceResponse,
)

logger = logging.getLogger(__name__)


class EnumCliBackendStatus(str, Enum):
    """Structured failure classes for CLI subprocess handlers."""

    SUCCESS = "success"
    UNAVAILABLE = "unavailable"
    INVALID_REQUEST = "invalid_request"
    TIMEOUT = "timeout"
    SUBPROCESS_ERROR = "subprocess_error"
    EMPTY_RESPONSE = "empty_response"


# OMN-13215: the ``codex-cli`` model config was REMOVED. The delegation ceiling
# executes over the canonical HTTP inference path — no codex shell-out remains.
_CLI_CONFIG_BY_MODEL: dict[str, tuple[str, list[str]]] = {
    "gemini-cli": ("gemini", ["-p"]),
    "claude-cli": (
        "claude",
        [
            "-p",
            "--output-format",
            "text",
            "--permission-mode",
            "dontAsk",
            "--no-session-persistence",
        ],
    ),
    "opencode-cli": ("opencode", ["run", "--format", "json", "--pure"]),
}


class HandlerLlmCliSubprocess:
    """Dispatch LLM inference to a CLI tool (gemini, codex) via subprocess.

    This handler spawns the CLI in headless/non-interactive mode,
    passes the user prompt, captures stdout as the response, and wraps it
    in a ModelLlmInferenceResponse.

    ``handle`` is the canonical def-B dispatch entrypoint the auto-wired runtime
    binds; it returns the response as a ``ModelHandlerOutput`` effect event or
    fails fast on a non-SUCCESS backend status. ``execute_cli_inference`` exposes
    the same owned core as the ``(response, status, detail)`` triple, preserving
    the structured ``EnumCliBackendStatus`` classification for fallback routing and
    metrics callers.

    Example:
        >>> handler = HandlerLlmCliSubprocess(cli="gemini", cli_args=["-p"])
        >>> output = await handler.handle(request)  # canonical dispatch entrypoint
        >>> response, status, detail = handler.execute_cli_inference(request)
        >>> if status == EnumCliBackendStatus.SUCCESS:
        ...     print(response.generated_text)
    """

    def __init__(
        self,
        cli: str | None = None,
        cli_args: list[str] | None = None,
        timeout: int = 120,
    ) -> None:
        self._cli = cli
        self._cli_args = cli_args or ["-p"]
        self._timeout = timeout

    @property
    def cli_name(self) -> str | None:
        """Return the CLI binary name, or None if not configured."""
        return self._cli

    @property
    def handler_type(self) -> EnumHandlerType:
        """Architectural role classification."""
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Behavioral classification."""
        return EnumHandlerTypeCategory.EFFECT

    async def handle(
        self,
        request: ModelLlmInferenceRequest,
    ) -> ModelHandlerOutput[None]:
        """Canonical dispatch entrypoint (def B) for the auto-wired runtime.

        OMN-14804 (child of OMN-14510 missing-handle burn-down): before this
        method existed, ``HandlerLlmCliSubprocess`` was contract-declared on three
        CLI operations (``inference.gemini_cli`` / ``inference.claude_cli`` /
        ``inference.opencode_cli``) yet exposed only ``execute_cli_inference()``.
        Auto-wiring's ``_make_dispatch_callback`` then bound ``_missing_handle``,
        which raised ``ModelOnexError`` on the FIRST dispatch while the contract
        validated, the node booted, and CI stayed green.

        ``handle`` owns the runtime dispatch contract for this EFFECT node: it runs
        CLI-subprocess inference through the owned ``_run_cli_inference`` core and
        returns the produced ``ModelLlmInferenceResponse`` as the node's output
        event, or fails fast with ``InfraUnavailableError`` when the CLI backend
        cannot produce a usable response (mirroring the fail-fast transport error in
        the OMN-14489 reference ``HandlerA2ATask.submit``). The structured
        ``(response, status, detail)`` triple — carrying the ``EnumCliBackendStatus``
        classification for fallback-routing callers — stays available via
        ``execute_cli_inference``; both public methods consume the same core.
        """
        response, status, detail = self._run_cli_inference(request)
        if response is None or status is not EnumCliBackendStatus.SUCCESS:
            raise InfraUnavailableError(
                f"CLI inference produced no usable response [{status.value}]: {detail}",
                context=ModelInfraErrorContext.with_correlation(
                    correlation_id=request.correlation_id,
                    transport_type=EnumInfraTransportType.RUNTIME,
                    operation="cli_inference",
                ),
            )
        return ModelHandlerOutput.for_effect(
            input_envelope_id=uuid4(),
            correlation_id=response.correlation_id,
            handler_id=type(self).__name__,
            events=(response,),
            processing_time_ms=response.latency_ms,
        )

    def execute_cli_inference(
        self,
        request: ModelLlmInferenceRequest,
    ) -> tuple[ModelLlmInferenceResponse | None, EnumCliBackendStatus, str]:
        """Structured-status view over the owned CLI-subprocess inference core.

        Returns the ``(response, status, detail)`` triple so fallback-routing and
        metrics callers can branch on the ``EnumCliBackendStatus`` classification.
        ``handle`` is the canonical auto-wired dispatch entrypoint; this method and
        ``handle`` both delegate to the same ``_run_cli_inference`` core, so their
        behavior can never diverge.

        Returns:
            Tuple of (response, status, detail). Status is always set even
            on failure, preserving structured failure information.
        """
        return self._run_cli_inference(request)

    def _run_cli_inference(
        self,
        request: ModelLlmInferenceRequest,
    ) -> tuple[ModelLlmInferenceResponse | None, EnumCliBackendStatus, str]:
        """Owned CLI-subprocess inference core (spawn + structured classification).

        Resolves the CLI backend from the handler config or the request model,
        spawns the subprocess, and classifies the outcome into one of the
        ``EnumCliBackendStatus`` failure classes or SUCCESS. Consumed by both
        ``handle`` (dispatch entrypoint) and ``execute_cli_inference``
        (structured-status accessor).
        """
        cli = self._cli
        cli_args = self._cli_args
        if cli is None:
            resolved = _CLI_CONFIG_BY_MODEL.get(request.model)
            if resolved is not None:
                cli, cli_args = resolved

        if cli is None:
            return (
                None,
                EnumCliBackendStatus.UNAVAILABLE,
                "cli not configured (no CLI binary specified)",
            )

        if not shutil.which(cli):
            return (
                None,
                EnumCliBackendStatus.UNAVAILABLE,
                f"{cli} not found on PATH",
            )

        # Extract last user message as prompt
        prompt = ""
        for msg in reversed(request.messages):
            if msg.role == "user":
                prompt = msg.content or ""
                break

        if not prompt:
            return (
                None,
                EnumCliBackendStatus.INVALID_REQUEST,
                "no user message in request",
            )

        try:
            start = time.monotonic()
            result = subprocess.run(
                [cli, *cli_args, prompt],
                capture_output=True,
                text=True,
                timeout=self._timeout,
                check=False,
            )
            latency_ms = (time.monotonic() - start) * 1000

            if result.returncode != 0:
                stderr_preview = result.stderr[:200] if result.stderr else "(no stderr)"
                logger.debug("%s exited %d: %s", cli, result.returncode, stderr_preview)
                return (
                    None,
                    EnumCliBackendStatus.SUBPROCESS_ERROR,
                    f"exit {result.returncode}: {stderr_preview}",
                )

            content = result.stdout.strip()
            if not content:
                return (
                    None,
                    EnumCliBackendStatus.EMPTY_RESPONSE,
                    "stdout empty after successful exit",
                )

            # Build response with all required fields
            from datetime import datetime

            from omnibase_infra.enums import (
                EnumLlmFinishReason,
                EnumLlmOperationType,
            )
            from omnibase_infra.models.llm.model_llm_usage import ModelLlmUsage
            from omnibase_infra.models.model_backend_result import ModelBackendResult

            # Rough token estimate: ~1.3 tokens per word
            prompt_tokens = int(len(prompt.split()) * 1.3)
            completion_tokens = int(len(content.split()) * 1.3)

            response = ModelLlmInferenceResponse(
                status="success",
                generated_text=content,
                model_used=f"{cli}-cli",
                operation_type=EnumLlmOperationType.CHAT_COMPLETION,
                finish_reason=EnumLlmFinishReason.STOP,
                usage=ModelLlmUsage(
                    tokens_input=prompt_tokens,
                    tokens_output=completion_tokens,
                    tokens_total=prompt_tokens + completion_tokens,
                ),
                latency_ms=latency_ms,
                backend_result=ModelBackendResult(success=True, duration_ms=latency_ms),
                correlation_id=getattr(request, "correlation_id", uuid4()),
                execution_id=uuid4(),
                timestamp=datetime.now(tz=UTC),
            )

            logger.info(
                "%s-cli: completed in %.0fms (~%d tokens)",
                cli,
                latency_ms,
                completion_tokens,
            )

            return (response, EnumCliBackendStatus.SUCCESS, "")

        except subprocess.TimeoutExpired:
            return (
                None,
                EnumCliBackendStatus.TIMEOUT,
                f"{cli} exceeded {self._timeout}s deadline",
            )


__all__ = [
    "EnumCliBackendStatus",
    "HandlerLlmCliSubprocess",
]
